import asyncio
import re
from pathlib import Path

from kairos.engine import ChunkIn, SessionEngine
from kairos.replay import run_script_sync
from kairos.schemas.models import Claim
from kairos.telemetry.validator import validate


def _by(events, etype, utt=None):
    return [e for e in events if e.event_type == etype and (utt is None or e.utterance_id == utt)]


def test_pune_demo_end_to_end(ctx):
    events = run_script_sync(ctx, "pune-demo")
    decisions = [e.payload["decision"] for e in _by(events, "RetrievalDecision", "u1")]
    assert decisions[:2] == ["WAIT", "WAIT"]  # "I need to", "book a"
    # retrieval starts before the utterance ends
    end_t = _by(events, "TranscriptChunkReceived", "u1")[-1].timestamp
    assert min(e.timestamp for e in _by(events, "RetrievalStarted", "u1")) < end_t
    # three intents, retrieved (some in parallel)
    assert len({e.sub_query_id for e in _by(events, "RetrievalCompleted", "u1")}) == 3
    versions = _by(events, "AnswerVersionCreated")
    assert [v.payload["kind"] for v in versions] == ["SEMANTIC", "SEMANTIC", "PRESENTATION"]
    # late constraint: delta retrieval touches fewer sub-queries than a full restart would
    plan = _by(events, "DeltaPlanComputed")[0].payload
    assert plan["counterfactual"]["delta_retrievals"] < plan["counterfactual"]["full_restart_retrievals"]
    v2 = versions[1].payload
    assert v2["diff"]["kept"] and v2["diff"]["revised"]
    # unaffected claims carried verbatim (same ids, same text)
    v1_claims = {c["claim_id"]: c["text"] for c in versions[0].payload["claims"]}
    for cid in v2["diff"]["kept"]:
        assert next(c["text"] for c in v2["claims"] if c["claim_id"] == cid) == v1_claims[cid]
    # "make that shorter": suppressed, no retrieval at all in that utterance
    assert not _by(events, "RetrievalStarted", "u3")
    assert _by(events, "RetrievalSuppressed", "u3")
    v3_citations = {e for c in versions[2].payload["claims"] for e in c["evidence"]}
    v2_citations = {e for c in v2["claims"] for e in c["evidence"]}
    assert v3_citations <= v2_citations


def test_all_demo_traces_have_full_lineage(ctx):
    for script in ("pune-demo", "reimbursement-demo", "conflict-demo", "self-repair-demo"):
        assert validate(run_script_sync(ctx, script))["coverage"] == 1.0


def test_conflicting_sources_are_both_surfaced(ctx):
    events = run_script_sync(ctx, "conflict-demo")
    answer = _by(events, "AnswerVersionCreated")[0].payload
    docs = {c["evidence"][0].split(":")[0] for c in answer["claims"]}
    assert {"D03", "D08"} <= docs
    assert all(c["verdict"] == "CONFLICTING" for c in answer["claims"])
    assert answer["conflicts"][0]["more_recent_document"] == "D03"


def test_unanswerable_question_abstains(ctx):
    engine = SessionEngine(ctx, "t-abstain")
    engine.start()
    asyncio.run(engine.on_chunk(ChunkIn(utterance_id="u1", text="Is there a swimming pool at the Pune venue?",
                                        t0=0.0, t1=1.0, final=True)))
    answer = _by(engine.events, "AnswerVersionCreated")[0].payload
    assert answer["claims"] == []
    assert "Insufficient evidence" in answer["uncertainty"][0]["text"]


def test_firewall_blocks_fabricated_and_foreign_citations(ctx):
    fw = ctx.firewall
    fake = Claim("C99", "L99", "Cancellations are always free.", "I01", ["D99:C01"], "x", 1.0)
    assert fw.verify(fake, retrieved={"D03:C03"}, peer_chunk_ids=[]).verdict == "INVALID_CITATION"
    # a real chunk that was never retrieved in this session is also rejected
    unretrieved = Claim("C98", "L98", "Alcohol is not served at Company venues.", "I01", ["D04:C07"],
                        "Alcohol is not served at Company venues.", 1.0)
    assert fw.verify(unretrieved, retrieved=set(), peer_chunk_ids=[]).verdict == "INVALID_CITATION"
    wrong_number = Claim("C97", "L97", "Cancellations made 21 or more days before a standard event are free.",
                         "I01", ["D03:C03"], "", 1.0)
    assert fw.verify(wrong_number, retrieved={"D03:C03"}, peer_chunk_ids=[]).verdict != "SUPPORTED"


def test_sessions_are_isolated(ctx):
    async def run():
        a, b = SessionEngine(ctx, "iso-a"), SessionEngine(ctx, "iso-b")
        a.start(), b.start()
        await a.on_chunk(ChunkIn(utterance_id="u1", text="What is the cancellation policy?", t1=1.0, final=True))
        await b.on_chunk(ChunkIn(utterance_id="u1", text="Is alcohol served at company venues?", t1=1.0, final=True))
        return a, b
    a, b = asyncio.run(run())
    for engine in (a, b):
        cited = {e for v in engine.versions for c in v.claims.values() for e in c.evidence}
        assert cited <= engine.retrieved
        assert all(ev.session_id == engine.id for ev in engine.events)
    asyncio.run(a.end())
    assert not a.versions and not a.retrieved and a.ended


def test_engine_source_contains_no_benchmark_text():
    """Leakage guard: no dev-set utterance appears verbatim in engine code."""
    import json

    root = Path(__file__).resolve().parent.parent
    dev = json.loads((root / "benchmark/datasets/dev/sessions.json").read_text(encoding="utf-8"))
    phrases = {" ".join(u["chunks"]).lower() for s in dev["sessions"] for u in s["utterances"]}
    source = "\n".join(p.read_text(encoding="utf-8").lower() for p in (root / "kairos").rglob("*.py"))
    source = re.sub(r"\s+", " ", source)
    leaked = [p for p in phrases if len(p) > 20 and p in source]
    assert not leaked, leaked
