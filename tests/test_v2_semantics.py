import asyncio
import json
import json
from pathlib import Path

import pytest

from benchmark.runner import load_dataset, replay_session
from kairos.config import load_config, resolve_path
from kairos.corpus.adapters import get_adapter
from kairos.corpus.adapters.scifact_eval import build_labels
from kairos.corpus.ingest import load_corpus
from kairos.decomposer.efis import Decomposer
from kairos.engine import EngineContext, SessionEngine
from kairos.grounding.firewall import GroundingFirewall
from kairos.replay import load_script, run_script
from kairos.retrieval.index import RetrievalQuery
from kairos.schemas.models import Claim


def test_request_onsets_split_but_symptom_conjunction_does_not(ctx):
    frame = ctx.frame_builder.build(
        "find houses in Mumbai and explain what an AI agent does tell me about Goa travel", {}, 1.0, "c1"
    )
    assert len(frame.candidates) == 3
    symptoms = ctx.frame_builder.build("fever and cough and cold", {}, 1.0, "c1")
    assert len(symptoms.candidates) == 1


def test_stable_candidate_identity_survives_prefix_growth(ctx):
    first = ctx.frame_builder.build("find houses in Mumbai", {}, 1.0, "c1")
    grown = ctx.frame_builder.build("find houses in Mumbai under 20 lakh", first.slots, 2.0, "c2", first)
    assert grown.candidates[0].key == first.candidates[0].key
    assert grown.candidates[0].revision_history


def test_scoped_constraints_and_indian_number_normalization(ctx):
    frame = ctx.frame_builder.build(
        "Find 2BHK houses in Mumbai below 20 lakh and 3BHK houses in Delhi above 40 lakh", {}, 1.0, "c1"
    )
    assert len(frame.candidates) == 2
    a, b = frame.candidates
    assert {(c.kind, c.normalized_value, c.relation) for c in a.typed_constraints} >= {
        ("city", "Mumbai", "EQ"), ("bedrooms", 2, "EQ"), ("budget", 2_000_000, "MAX")
    }
    assert {(c.kind, c.normalized_value, c.relation) for c in b.typed_constraints} >= {
        ("city", "Delhi", "EQ"), ("bedrooms", 3, "EQ"), ("budget", 4_000_000, "MIN")
    }
    assert all(c.owner_key == a.key for c in a.typed_constraints)
    assert all(c.owner_key == b.key for c in b.typed_constraints)


def test_ambiguous_financial_amount_is_not_relabelled(ctx):
    frame = ctx.frame_builder.build("three lakh per month at 5 percent", {}, 1.0, "c1")
    # Word-number support is intentionally conservative; no EMI/income/loan interpretation is invented.
    assert "emi" not in frame.normalized().lower()
    numeric = ctx.frame_builder.build("3 lakh per month at 5 percent", {}, 1.0, "c1")
    amount = next(c for c in numeric.candidates[0].typed_constraints if c.kind == "amount")
    assert amount.normalized_value == 300_000 and amount.unit == "per_month" and amount.ambiguous


def test_irrelevant_footprints_do_not_merge_unrelated_intents(ctx):
    frame = ctx.frame_builder.build("Define an AI agent. Find houses in Mumbai.", {}, 1.0, "c1")
    dec = Decomposer(ctx.cfg["decomposer"], ctx.index)
    first = dec.assign(frame.candidates[0], frame, [], "I01", "u1", None).sub_query
    second = dec.assign(frame.candidates[1], frame, [first], "I02", "u1", first)
    assert second.created and second.sub_query.id == "I02"
    assert not second.footprint_valid


def test_cancellation_paraphrase_merges_with_reasoned_signals(ctx):
    frame = ctx.frame_builder.build(
        "What are the cancellation rules? Explain the cancellation policy.", {}, 1.0, "c1"
    )
    assert len(frame.candidates) == 2
    dec = Decomposer(ctx.cfg["decomposer"], ctx.index)
    first = dec.assign(frame.candidates[0], frame, [], "I01", "u1", None).sub_query
    second = dec.assign(frame.candidates[1], frame, [first], "I02", "u1", first)
    assert not second.created and second.merged_into == "I01"
    signals = second.decision_signals["I01"]
    assert signals["request_objective_compatibility"] > 0
    assert signals["constraint_compatibility"] is True
    assert signals["decision"] == "MERGE"


def test_v2_1_efis_pair_set_has_required_scale_and_categories():
    data = json.loads(resolve_path("benchmark/datasets/heldout/efis_pairs_v2_1.json").read_text(encoding="utf-8"))
    assert len(data["cases"]) >= 100
    assert {case["merge"] for case in data["cases"]} == {True, False}
    categories = {case["category"] for case in data["cases"]}
    assert {"semantic_paraphrase", "same_topic_different_objective", "constraint_conflict"} <= categories


def test_relevance_gate_rejects_sample_corpus_mismatch(ctx):
    for text in [
        "Find 2BHK houses in Mumbai below 20 lakh",
        "What is an AI agent?",
        "medicine for fever and cough",
        "nearest hospital from my location",
        "travel plans to Goa",
    ]:
        result = ctx.index.search(RetrievalQuery.build(text))
        assert result.evidence_status in {"OUT_OF_DOMAIN", "NO_RELEVANT_EVIDENCE"}
        assert result.relevant_hits == []


def test_probe_cache_avoids_duplicate_bm25_work(ctx):
    query = RetrievalQuery.build("cancellation policy")
    before = ctx.index.probe_cache_hits
    assert ctx.index.probe(query.terms) == ctx.index.probe(query.terms)
    assert ctx.index.probe_cache_hits >= before + 1


def test_grounding_requires_real_quote_number_and_checksum(ctx):
    hit = ctx.index.search(RetrievalQuery.build("Banyan Hall participants")).relevant_hits[0]
    chunk = ctx.corpus.chunk(hit.chunk_id)
    retrieved = {hit.chunk_id}
    false_quote = Claim("C1", "L1", "Banyan Hall seats 40 participants.", "I1", [hit.chunk_id],
                        "This quotation is not in the source.", 1.0,
                        evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"})
    assert ctx.firewall.verify(false_quote, retrieved, []).verdict == "UNSUPPORTED"
    wrong_number = Claim("C2", "L2", "Banyan Hall seats 400 participants.", "I1", [hit.chunk_id],
                         chunk.text.split(". ")[0] + ".", 1.0,
                         evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"})
    assert ctx.firewall.verify(wrong_number, retrieved, []).verdict != "SUPPORTED"
    wrong_checksum = Claim("C3", "L3", chunk.text, "I1", [hit.chunk_id], chunk.text, 1.0,
                           evidence_checksums={hit.chunk_id: "0" * 40})
    assert ctx.firewall.verify(wrong_checksum, retrieved, []).verdict == "INVALID_CITATION"
    missing_checksum = Claim("C4", "L4", chunk.text, "I1", [hit.chunk_id], chunk.text, 1.0)
    missing = ctx.firewall.verify(missing_checksum, retrieved, [])
    assert missing.verdict == "INVALID_CITATION"
    assert missing.missing_checksums == [hit.chunk_id]


def test_dataset_adapters_are_offline_deterministic_and_keep_labels_separate(tmp_path):
    fixtures = resolve_path("tests/fixtures/datasets")
    sci = get_adapter("scifact").run(fixtures / "scifact_rows.json", tmp_path / "science",
                                      limit=1, seed=42, revision="fixture")
    squad = get_adapter("squad").run(fixtures / "squad_rows.json", tmp_path / "general",
                                      limit=1, seed=42, revision="fixture")
    assert sci.manifest["exported_documents"] == 1
    assert squad.manifest["evaluation_labels"] == 1
    corpus = load_corpus([tmp_path / "science", tmp_path / "general"], profile="test")
    assert set(corpus.documents) == {"SCIFACT:118568", next(k for k in corpus.documents if k.startswith("SQUAD:"))}
    markdown = squad.documents[0].read_text(encoding="utf-8")
    labels = squad.labels.read_text(encoding="utf-8")
    assert "What is the oldest building" not in markdown
    assert "Old College" in labels
    assert all(doc["license"] == "cc-by-sa-4.0" for doc in corpus.documents.values())
    with pytest.raises(ValueError, match="checksum mismatch"):
        get_adapter("scifact").run(fixtures / "scifact_rows.json", tmp_path / "bad-checksum",
                                    limit=1, seed=42, revision="fixture", expected_checksum="0" * 64)


def test_scifact_qrels_join_is_namespaced_and_does_not_enter_corpus():
    labels = build_labels(
        [{"query-id": 7, "corpus-id": 118568, "score": 1},
         {"query-id": 8, "corpus-id": 999999, "score": 1}],
        [{"_id": "7", "text": "A scientific claim"}, {"_id": "8", "text": "Excluded"}],
        {"118568"},
    )
    assert labels == [{"query_id": "7", "query": "A scientific claim", "document_id": "SCIFACT:118568",
                       "relevance": 1, "split": "test"}]


def test_committed_external_profiles_index_and_query():
    science = EngineContext(load_config(overrides={"corpus.profile": "science"}))
    result = science.index.search(RetrievalQuery.build("What did rhAng-1 improve in murine sepsis?"))
    assert result.relevant_hits and result.relevant_hits[0].chunk_id.startswith("SCIFACT:")
    assert science.corpus.chunk(result.relevant_hits[0].chunk_id).source_dataset == "BeIR/scifact"
    general = EngineContext(load_config(overrides={"corpus.profile": "general"}))
    result = general.index.search(RetrievalQuery.build("What is the oldest building on the Notre Dame campus?"))
    assert result.relevant_hits and result.relevant_hits[0].chunk_id.startswith("SQUAD:")


def test_multi_domain_profile_routes_each_query_to_compatible_sources():
    ctx = EngineContext(load_config(overrides={"corpus.profile": "multi_domain_demo"}))
    cancellation = ctx.index.search(RetrievalQuery.build("international event cancellation policy"))
    assert cancellation.domain == "corporate_events"
    assert all(ctx.index.chunk_domains[ctx.index.pos[h.chunk_id]] == "corporate_events"
               for h in cancellation.relevant_hits)
    university = ctx.index.search(RetrievalQuery.build("oldest building on the Notre Dame campus"))
    assert all(ctx.corpus.chunk(h.chunk_id).source_dataset == "rajpurkar/squad"
               for h in university.relevant_hits)


def test_cddm_exact_affected_sets_for_documented_s10_s26(cfg):
    dataset = load_dataset("dev")

    async def predicted(session_id):
        session = next(item for item in dataset["sessions"] if item["id"] == session_id)
        events = await replay_session(EngineContext(cfg), session)
        event = next(e for e in events if e.event_type == "DeltaPlanComputed")
        return {row["sub_query_id"] for row in event.payload["affected_sub_queries"]}, events

    s10_ids, s10_events = asyncio.run(predicted("S10"))
    s26_ids, s26_events = asyncio.run(predicted("S26"))
    assert len(s10_ids) == 2
    assert len(s26_ids) == 1
    s26_plan = next(e for e in s26_events if e.event_type == "DeltaPlanComputed")
    assert any(row["sub_query_id"] not in s26_ids for row in s26_plan.payload["unaffected_sub_queries"])


def test_multi_domain_stress_abstains_without_supported_claims(ctx):
    async def execute():
        engine = SessionEngine(ctx, "stress-regression")
        await run_script(engine, load_script("multi-domain-stress"))
        return engine

    engine = asyncio.run(execute())
    answers = [e for e in engine.events if e.event_type == "AnswerVersionCreated"]
    assert answers
    assert answers[-1].payload["claims"] == []
    grounding = [e for e in engine.events if e.event_type == "GroundingVerified"][-1]
    assert grounding.payload["summary"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert len(engine.sub_queries) >= 5
    retrievals = [e for e in engine.events if e.event_type == "RetrievalCompleted"]
    assert retrievals and all(not e.payload["accepted_chunk_ids"] for e in retrievals)
    graph = engine.graph.snapshot()
    assert not [node for node in graph["nodes"] if node["kind"] == "claim"]
    assert [node for node in graph["nodes"] if node["kind"] == "chunk"]
