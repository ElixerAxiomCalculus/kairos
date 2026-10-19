import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from benchmark.controller.counterfactual import label_optimal_stopping
from benchmark.controller.audit_v2_4 import docs_consistency_audit
from benchmark.controller.modeling_v2_3 import _select_threshold
from kairos.api.server import create_app
from kairos.controller.evidence_cache import EvidenceCache
from kairos.engine import ChunkIn, EngineContext, SessionEngine
from kairos.retrieval.index import RetrievalQuery
from kairos.telemetry.bus import _redact


def test_v2_4_documentation_artifacts_and_runtime_claims_are_consistent():
    result = docs_consistency_audit()
    assert result["status"] == "PASS", result["failures"]


@pytest.mark.asyncio
async def test_caper_prepares_then_commits_only_after_grounding(cfg):
    caper_cfg = cfg.with_overrides({
        "controller.mode": "eou",
        "controller.caper.enabled": True,
        "controller.caper.debounce_ms": 0,
        "controller.caper.finalization_wait_ms": 1000,
    })
    ctx = EngineContext(caper_cfg)
    engine = SessionEngine(ctx, "caper-lifecycle")
    engine.start()
    await engine.on_chunk(ChunkIn(
        utterance_id="u1", text="Which Pune venue accommodates a workshop",
        t0=0.0, t1=1.0, final=False,
    ))
    await asyncio.sleep(0.05)
    types = [event.event_type for event in engine.events]
    assert "PrefetchCompleted" in types
    assert "EvidenceValidated" in types
    assert "AnswerVersionCreated" not in types
    assert "EvidenceCommitted" not in types

    await engine.on_chunk(ChunkIn(
        utterance_id="u1", text="of 30 people?", t0=1.0, t1=2.0, final=True,
    ))
    types = [event.event_type for event in engine.events]
    assert "GroundingVerified" in types
    assert "AnswerVersionCreated" in types
    committed = [event for event in engine.events if event.event_type == "EvidenceCommitted"]
    if committed:
        grounding_ids = {event.event_id for event in engine.events if event.event_type == "GroundingVerified"}
        assert set(committed[-1].parent_event_ids) & grounding_ids
    completions = [event for event in engine.events if event.event_type == "RetrievalCompleted"]
    assert completions
    assert any(event.payload["cache_status"] in {"CAPER_VALIDATED", "MISS_STORED"} for event in completions)


@pytest.mark.asyncio
async def test_caper_invalidates_critical_city_revision(cfg):
    caper_cfg = cfg.with_overrides({
        "controller.mode": "eou",
        "controller.caper.enabled": True,
        "controller.caper.debounce_ms": 0,
        "controller.caper.finalization_wait_ms": 1000,
    })
    ctx = EngineContext(caper_cfg)
    engine = SessionEngine(ctx, "caper-revision")
    engine.start()
    await engine.on_chunk(ChunkIn(
        utterance_id="u1", text="Find a workshop venue in Pune",
        t0=0.0, t1=1.0, final=False,
    ))
    await asyncio.sleep(0.05)
    await engine.on_chunk(ChunkIn(
        utterance_id="u1", text="sorry, in Mumbai for 100 people",
        t0=1.0, t1=2.0, final=True,
    ))
    invalidations = [event for event in engine.events if event.event_type == "EvidenceInvalidated"]
    superseded = [event for event in engine.events if event.event_type == "PrefetchSuperseded"]
    assert invalidations or superseded
    for event in [e for e in engine.events if e.event_type == "EvidenceCommitted"]:
        assert all(ctx.corpus.chunk(chunk_id) is not None for chunk_id in event.chunk_ids)


def test_actual_search_accounting_distinguishes_cache_hits(cfg):
    ctx = EngineContext(cfg)
    query = RetrievalQuery.build("Pune workshop venue")
    result = ctx.index.search(query, execution_context={
        "retrieval_type": "PROVISIONAL", "session_id": "accounting", "intent_id": "K1", "query_version": 1,
    })
    assert result.search_id
    cache = EvidenceCache(max_entries=4, ttl_s=30)
    cache.put("f", result, "source", validated=True)
    assert cache.get("f", "source") is not None
    summary = ctx.index.search_cost_summary(session_id="accounting")
    assert summary["actual_hybrid_search_executions"] == 1
    assert summary["by_retrieval_type"] == {"PROVISIONAL": 1}


def test_cache_is_session_scoped_for_provisional_promotion(cfg):
    ctx = EngineContext(cfg)
    result = ctx.index.search(RetrievalQuery.build("Pune workshop venue"))
    cache = EvidenceCache(max_entries=4, ttl_s=30)
    hashes = {hit.chunk_id: ctx.corpus.chunk(hit.chunk_id).sha256 for hit in result.relevant_hits}
    cache.put_provisional(
        "f", result, "corpus", session_id="s1", intent_id="K1", query_version=1,
        constraint_fingerprint="c", source_hashes=hashes,
    )
    assert not cache.promote(
        "f", "corpus", session_id="s2", intent_id="K1", query_version=1,
        constraint_fingerprint="c", source_hashes=hashes,
    )
    assert cache.get("f", "corpus") is None


def test_zero_action_threshold_has_undefined_risk():
    rows = [{
        "stream_id": "s1", "candidate_id": "K1", "prefix_index": 1, "is_final": False,
        "question_family": "f1", "targets": {"proxy_strict_useful": 1},
    }]
    selected = _select_threshold(rows, [0.1], [1.0])
    zero = selected["selected"]
    assert zero["selected"] == 0
    assert zero["conditional_selected_action_risk"] is None
    assert zero["promotion_eligible"] is False


def test_terminal_retrieval_has_nonzero_continuation_value():
    base = {
        "stream_id": "s", "candidate_id": "K", "targets": {
            "gold_relevance": 1, "constraint_compatibility": 1, "evidence_survival": 1,
            "answerability": 1, "grounding_eligibility": 1, "time_saved_s": 0,
            "retrieval_cost": 1, "revision_cost": 0,
        },
    }
    rows = [
        {**base, "prefix_index": 1, "is_final": False},
        {**base, "prefix_index": 2, "is_final": True},
    ]
    labeled = label_optimal_stopping(rows)
    assert labeled[0]["q_wait"] != -0.015
    assert labeled[0]["terminal_fallback"]["source"] == "FINAL_PREFIX_RETRIEVAL_PROXY"


def test_default_deny_redaction_covers_nested_strings_and_arrays():
    secret = "sk-test-SYNTHETIC-DO-NOT-USE"
    payload = {
        "metadata": {"unexpected": secret},
        "values": [secret, {"title": "Person Name"}],
        "reason": "SAFE_ENUM_REASON",
    }
    redacted = _redact(payload)
    encoded = json.dumps(redacted)
    assert secret not in encoded
    assert "Person Name" not in encoded
    assert redacted["reason"] == "SAFE_ENUM_REASON"


def test_websocket_resume_uses_initial_message_and_rotates_token(cfg, tmp_path):
    local = cfg.with_overrides({
        "telemetry.trace_dir": str(tmp_path / "traces"),
        "telemetry.sqlite_path": str(tmp_path / "events.sqlite"),
    })
    client = TestClient(create_app(local))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        session_id, token = hello["session_id"], hello["resume_token"]

    with client.websocket_connect(f"/ws?session_id={session_id}") as resumed:
        resumed.send_json({
            "type": "resume", "session_id": session_id, "resume_token": token, "after_event_id": None,
        })
        second = resumed.receive_json()
        assert second["type"] == "hello" and second["resumed"] is True
        assert second["resume_token"] != token

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws?session_id={session_id}") as rejected:
            rejected.send_json({"type": "resume", "session_id": session_id, "resume_token": token})
            rejected.receive_json()


def test_authorized_persistent_trace_deletion(cfg, tmp_path):
    trace_dir = tmp_path / "traces"
    sqlite_path = tmp_path / "events.sqlite"
    local = cfg.with_overrides({
        "telemetry.trace_dir": str(trace_dir),
        "telemetry.sqlite_path": str(sqlite_path),
    })
    client = TestClient(create_app(local))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        session_id, token = hello["session_id"], hello["resume_token"]
        ws.send_json({"type": "chunk", "utterance_id": "private", "text": "synthetic private text", "final": True})
        while True:
            message = ws.receive_json()
            if message.get("type") == "event" and message["event"]["event_type"] == "TelemetryEmitted":
                break
    assert (trace_dir / f"{session_id}.jsonl").exists()
    response = client.delete(
        f"/api/sessions/{session_id}/data", headers={"X-Kairos-Session-Token": token}
    )
    assert response.status_code == 200
    assert response.json()["ended"] is True
    assert not (trace_dir / f"{session_id}.jsonl").exists()
    with sqlite3.connect(sqlite_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
