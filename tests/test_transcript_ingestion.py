import asyncio
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from kairos.api.server import create_app
from kairos.engine import SessionEngine
from kairos.streaming.transcript import TranscriptIngestionCoordinator, TranscriptSnapshot
from kairos.telemetry.validator import validate


def snap(utterance_id: str, revision: int, text: str, *, asr_final: bool = False,
         end_of_turn: bool = False) -> TranscriptSnapshot:
    return TranscriptSnapshot(type="transcript_snapshot", utterance_id=utterance_id, revision=revision,
                              text=text, asr_final=asr_final, end_of_turn=end_of_turn,
                              source="microphone")


async def coordinator(ctx, session_id="voice-test", tail=1):
    engine = SessionEngine(ctx, session_id)
    engine.start()
    return engine, TranscriptIngestionCoordinator(engine, {
        "mutable_tail_tokens": tail,
        "max_utterance_chars": 4000,
    })


def test_cumulative_growth_emits_stable_prefix_before_finalization(ctx):
    async def run():
        engine, ingest = await coordinator(ctx)
        texts = [
            "I need to",
            "I need to book a",
            "I need to book a workshop venue in Pune",
            "I need to book a workshop venue in Pune for 30 people,",
            "I need to book a workshop venue in Pune for 30 people, tell me the cancellation policy",
        ]
        for revision, text in enumerate(texts, 1):
            ack = await ingest.ingest(snap("mic-1", revision, text))
            assert ack.status == "accepted"
        assert not engine.versions
        assert any(e.event_type == "TranscriptChunkReceived" for e in engine.events)
        assert any(e.event_type == "RetrievalStarted" for e in engine.events)

        final_text = f"{texts[-1]} and whether catering is available."
        ack = await ingest.ingest(snap("mic-1", 6, final_text, asr_final=True, end_of_turn=True))
        return engine, ack

    engine, ack = asyncio.run(run())
    assert ack.status == "finalized"
    assert engine.utterances["mic-1"].text.endswith("and whether catering is available.")
    assert len(engine.versions) == 1
    started = [e for e in engine.active_events() if e.event_type == "RetrievalStarted"]
    last_chunk = [e for e in engine.active_events() if e.event_type == "TranscriptChunkReceived"][-1]
    assert min(e.timestamp for e in started) < last_chunk.timestamp
    assert validate(engine.active_events())["coverage"] == 1.0


def test_duplicate_stale_and_conflicting_revisions_are_safe(ctx):
    async def run():
        engine, ingest = await coordinator(ctx)
        first = snap("mic-1", 1, "I need a workshop")
        assert (await ingest.ingest(first)).status == "accepted"
        before = len(engine.events)
        assert (await ingest.ingest(first)).status == "duplicate"
        assert len(engine.events) == before
        assert (await ingest.ingest(snap("mic-1", 0 + 1, "I need a workshop"))).status == "duplicate"
        await ingest.ingest(snap("mic-1", 3, "I need a workshop venue"))
        assert (await ingest.ingest(snap("mic-1", 2, "I need a"))).status == "stale"
        with pytest.raises(ValueError, match="already used"):
            await ingest.ingest(snap("mic-1", 3, "different text"))
        return engine

    asyncio.run(run())


@pytest.mark.parametrize("revisions, expected", [
    (["book a workshop in Pune", "book a workshop venue in Pune"], "book a workshop venue in Pune"),
    (["book a large workshop venue in Pune", "book a workshop venue in Pune"], "book a workshop venue in Pune"),
    (["book a workshop venue in Poona", "book a workshop venue in Pune"], "book a workshop venue in Pune"),
])
def test_insert_delete_and_mutable_tail_corrections(revisions, expected, ctx):
    async def run():
        engine, ingest = await coordinator(ctx, tail=2)
        for revision, text in enumerate(revisions, 1):
            await ingest.ingest(snap("mic-1", revision, text))
        await ingest.ingest(snap("mic-1", len(revisions) + 1, expected, asr_final=True, end_of_turn=True))
        return engine

    engine = asyncio.run(run())
    assert engine.utterances["mic-1"].text == expected
    assert "  " not in engine.utterances["mic-1"].text


def test_correction_across_committed_prefix_reconciles_stale_evidence(ctx):
    async def run():
        engine, ingest = await coordinator(ctx, tail=0)
        await ingest.ingest(snap("mic-1", 1, "Which rooms can seat 100 people in Pune", asr_final=True))
        old_ids = {e.event_id for e in engine.events if e.utterance_id == "mic-1"}
        ack = await ingest.ingest(snap("mic-1", 2, "Which rooms can seat 100 people in Mumbai"))
        await ingest.ingest(snap("mic-1", 3, "Which rooms can seat 100 people in Mumbai",
                                 asr_final=True, end_of_turn=True))
        return engine, old_ids, ack

    engine, old_ids, ack = asyncio.run(run())
    assert ack.reconciled
    assert engine.utterances["mic-1"].text == "Which rooms can seat 100 people in Mumbai"
    reconciled = [e for e in engine.active_events() if e.event_type == "TranscriptReconciled"]
    assert reconciled and old_ids <= set(reconciled[-1].payload["invalidated_event_ids"])
    active_ids = {e.event_id for e in engine.active_events()}
    assert not (old_ids & active_ids)
    latest_frame = [e for e in engine.active_events() if e.event_type == "QueryFrameUpdated"][-1]
    assert "Mumbai" in latest_frame.payload["raw_text"] and "Pune" not in latest_frame.payload["raw_text"]
    assert validate(engine.active_events())["coverage"] == 1.0


def test_spoken_self_repair_remains_real_speech(ctx):
    async def run():
        engine, ingest = await coordinator(ctx)
        text = "Which rooms can seat 100 people in Pune, sorry, Mumbai?"
        await ingest.ingest(snap("mic-1", 1, text))
        await ingest.ingest(snap("mic-1", 2, text, asr_final=True, end_of_turn=True))
        return engine

    engine = asyncio.run(run())
    frame = [e for e in engine.active_events() if e.event_type == "QueryFrameUpdated"][-1]
    assert frame.payload["repairs"] == [{"kind": "place", "from": "Pune", "to": "Mumbai"}]
    assert not any(e.event_type == "TranscriptReconciled" for e in engine.events)


def test_empty_turn_duplicate_finalization_and_session_isolation(ctx):
    async def run():
        a, ia = await coordinator(ctx, "voice-a")
        b, ib = await coordinator(ctx, "voice-b")
        empty = snap("mic-1", 1, "   ", asr_final=True, end_of_turn=True)
        assert (await ia.ingest(empty)).status == "finalized"
        assert (await ia.ingest(empty)).status == "duplicate"
        await ib.ingest(snap("mic-1", 1, "Is alcohol served at company venues?",
                             asr_final=True, end_of_turn=True))
        return a, b

    a, b = asyncio.run(run())
    assert not a.utterances and not a.versions
    assert len(b.versions) == 1
    assert all(event.session_id == "voice-b" for event in b.events)


def test_live_snapshots_preserve_v1_v2_v3_delta_and_suppression(ctx):
    async def run():
        engine, ingest = await coordinator(ctx, tail=1)
        first_turn = [
            "I need to",
            "I need to book a workshop venue in Pune",
            "I need to book a workshop venue in Pune for 30 people,",
            "I need to book a workshop venue in Pune for 30 people, tell me the cancellation policy",
        ]
        for revision, text in enumerate(first_turn, 1):
            await ingest.ingest(snap("mic-1", revision, text))
        await ingest.ingest(snap("mic-1", 5, first_turn[-1] + " and whether catering is available.",
                                 asr_final=True, end_of_turn=True))
        await ingest.ingest(snap("mic-2", 1, "Actually, this is for an"))
        await ingest.ingest(snap("mic-2", 2, "Actually, this is for an international event.",
                                 asr_final=True, end_of_turn=True))
        await ingest.ingest(snap("mic-3", 1, "Make that shorter.", asr_final=True, end_of_turn=True))
        return engine

    engine = asyncio.run(run())
    assert [version.kind for version in engine.versions] == ["SEMANTIC", "SEMANTIC", "PRESENTATION"]
    turns = {uid: state.turn.turn_type for uid, state in engine.utterances.items()}
    assert turns == {"mic-1": "NEW_INFO", "mic-2": "CONSTRAINT", "mic-3": "TRANSFORM"}
    assert any(e.event_type == "DeltaPlanComputed" and e.utterance_id == "mic-2" for e in engine.active_events())
    assert not any(e.event_type == "RetrievalStarted" and e.utterance_id == "mic-3"
                   for e in engine.active_events())
    assert any(e.event_type == "RetrievalSuppressed" and e.utterance_id == "mic-3"
               for e in engine.active_events())


def test_websocket_snapshot_protocol_and_resume(cfg, tmp_path):
    cfg = cfg.with_overrides({
        "telemetry.trace_dir": str(tmp_path / "traces"),
        "telemetry.sqlite_path": str(tmp_path / "telemetry.sqlite"),
        "live.mutable_tail_tokens": 1,
    })
    client = TestClient(create_app(cfg))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["resumed"] is False
        session_id = hello["session_id"]
        resume_token = hello["resume_token"]
        ws.send_json(snap("mic-1", 1, "I need a workshop venue").model_dump())
        ack = ws.receive_json()
        while ack["type"] != "transcript_ack":
            ack = ws.receive_json()
        assert ack["status"] == "accepted"
        ws.send_json(snap("mic-1", 2, "I need a workshop venue in Pune").model_dump())
        last_event_id = None
        while True:
            message = ws.receive_json()
            if message["type"] == "event":
                last_event_id = message["event"]["event_id"]
            if message["type"] == "transcript_ack" and message["revision"] == 2:
                break

    with client.websocket_connect(
        f"/ws?session_id={session_id}&resume_token={resume_token}&after_event_id={last_event_id}"
    ) as ws:
        hello = ws.receive_json()
        assert hello["session_id"] == session_id and hello["resumed"] is True
        ws.send_json(snap("mic-1", 3, "I need a workshop venue in Pune for 30 people",
                          asr_final=True, end_of_turn=True).model_dump())
        event_types = []
        finalized = False
        while not (finalized and "TelemetryEmitted" in event_types):
            message = ws.receive_json()
            if message["type"] == "event":
                event_types.append(message["event"]["event_type"])
            elif message["type"] == "transcript_ack":
                finalized = message["status"] == "finalized"
        assert {"TranscriptChunkReceived", "QueryFrameUpdated", "RetrievalDecision",
                "RetrievalStarted", "RetrievalCompleted", "EvidenceFused", "GroundingVerified",
                "AnswerVersionCreated", "TelemetryEmitted"} <= set(event_types)

    trace = client.get(f"/api/sessions/{session_id}/trace",
                       headers={"X-Kairos-Session-Token": resume_token}).json()
    assert trace["validation"]["coverage"] == 1.0

    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws?session_id={session_id}&resume_token=wrong-token") as ws:
            ws.receive_json()


def test_trace_content_false_redacts_jsonl_and_sqlite(cfg, tmp_path):
    trace_dir = tmp_path / "traces"
    sqlite_path = tmp_path / "telemetry.sqlite"
    cfg = cfg.with_overrides({
        "telemetry.trace_dir": str(trace_dir),
        "telemetry.sqlite_path": str(sqlite_path),
        "telemetry.trace_content": False,
    })
    client = TestClient(create_app(cfg))
    secret_phrase = "private workshop question in Pune"
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        session_id = hello["session_id"]
        ws.send_json(snap("mic-1", 1, secret_phrase, asr_final=True, end_of_turn=True).model_dump())
        while True:
            message = ws.receive_json()
            if message["type"] == "event" and message["event"]["event_type"] == "TelemetryEmitted":
                break

    jsonl = (trace_dir / f"{session_id}.jsonl").read_text(encoding="utf-8")
    assert secret_phrase not in jsonl and "<redacted:" in jsonl
    with sqlite3.connect(sqlite_path) as connection:
        bodies = [row[0] for row in connection.execute(
            "SELECT body FROM events WHERE session_id = ?", (session_id,)
        )]
    serialized = json.dumps(bodies)
    assert secret_phrase not in serialized and "<redacted:" in serialized
