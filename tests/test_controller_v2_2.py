from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from benchmark.controller.dataset_v2_2 import load_split, validate_dataset, validate_streams
from kairos.config import load_config
from kairos.controller.cold_start import assess_cold_start
from kairos.controller.evidence_cache import CacheState, EvidenceCache, query_fingerprint
from kairos.controller.policy import StabilityModel
from kairos.engine import EngineContext
from kairos.retrieval.index import RetrievalQuery


def _fingerprint(ctx: EngineContext, query: RetrievalQuery, constraints=None, profile="sample") -> str:
    return query_fingerprint(
        query, corpus_profile=profile, corpus_version="corpus-v1",
        index_configuration={"mode": "hybrid", "top_k": 5},
        intent_objective=query.text, typed_constraints=constraints or [],
        retrieval_mode="hybrid", relevance_gate_configuration={"enabled": True},
    )


def test_v2_2_dataset_manifest_and_group_splits_validate():
    result = validate_dataset()
    assert result["status"] == "PASS"
    assert result["manifest_dataset_hash_matches"]
    assert result["streams"] >= 100


def test_v2_2_dataset_detects_group_leakage_and_missing_timestamps():
    rows = [dict(row) for row in load_split("calibration")[:2]]
    assert rows
    duplicate = dict(rows[0])
    duplicate["stream_id"] = "forced-other-split"
    duplicate["session_id"] = "forced-other-split"
    duplicate["split"] = "heldout"
    duplicate["content_hash"] = "unique-content-hash-for-leak-test"
    duplicate["prefix_timestamps"] = []
    result = validate_streams([*rows, duplicate])
    codes = {error["code"] for error in result["errors"]}
    assert "GROUP_SPLIT_LEAKAGE" in codes
    assert "MISSING_TIMESTAMP" in codes


def test_query_fingerprint_normalizes_wording_but_invalidates_constraints_and_corpus():
    ctx = EngineContext(load_config())
    direct = RetrievalQuery.build("What is the cancellation policy?")
    polite = RetrievalQuery.build("Could you please explain the cancellation policy?")
    assert _fingerprint(ctx, direct) == _fingerprint(ctx, polite)
    forward = RetrievalQuery.build("Does venue policy replace travel policy?")
    inverse = RetrievalQuery.build("Does travel policy replace venue policy?")
    assert _fingerprint(ctx, forward) != _fingerprint(ctx, inverse)
    standard = _fingerprint(ctx, direct, [{"kind": "qualifier", "normalized_value": "standard"}])
    international = _fingerprint(ctx, direct, [{"kind": "qualifier", "normalized_value": "international"}])
    assert standard != international
    assert _fingerprint(ctx, direct, profile="sample") != _fingerprint(ctx, direct, profile="science")


def test_evidence_cache_is_session_local_source_bound_and_thread_safe():
    ctx = EngineContext(load_config())
    query = RetrievalQuery.build("cancellation policy")
    result = ctx.index.search(query)
    fingerprint = _fingerprint(ctx, query)
    first = EvidenceCache(max_entries=4, ttl_s=60)
    second = EvidenceCache(max_entries=4, ttl_s=60)
    first.put(fingerprint, result, "source-a", validated=True)
    assert first.get(fingerprint, "source-a") is not None
    assert second.get(fingerprint, "source-a") is None
    assert first.get(fingerprint, "source-b") is None
    first.put(fingerprint, result, "source-a", validated=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hits = list(pool.map(lambda _: first.get(fingerprint, "source-a") is not None, range(8)))
    assert all(hits)


def test_evidence_cache_never_reuses_provisional_or_expired_entries():
    ctx = EngineContext(load_config())
    query = RetrievalQuery.build("cancellation policy")
    result = ctx.index.search(query)
    fingerprint = _fingerprint(ctx, query)
    cache = EvidenceCache(max_entries=2, ttl_s=60)
    cache.put(fingerprint, result, "source", validated=False)
    assert cache.get(fingerprint, "source") is None
    assert cache.status()[CacheState.PROVISIONAL.value] == 1
    cache.put(fingerprint, result, "source", validated=True)
    cache._entries[fingerprint].expires_monotonic = 0.0
    assert cache.get(fingerprint, "source") is None


def test_extended_stability_model_is_bounded_deterministic_and_handles_missing_rho():
    model = StabilityModel(
        {"bias": -0.2, "rho": 0.4, "first_observation": 0.8, "constraint_revision_risk": -1.2},
        "unit-test", features=["rho", "first_observation", "constraint_revision_risk"],
        calibration={"enabled": True, "method": "platt", "slope": 0.9, "intercept": -0.1},
    )
    features = {"rho": None, "first_observation": 1.0, "constraint_revision_risk": 0.2}
    assert 0.0 < model.predict(features) < 1.0
    assert model.predict(features) == model.predict(features)


def test_cold_start_separates_missing_history_from_constraint_risk():
    ctx = EngineContext(load_config())
    safe = ctx.frame_builder.build("What is the cancellation policy?", {}, 1.0, "safe").candidates[0]
    risky = ctx.frame_builder.build("Find a venue under", {}, 1.0, "risky").candidates[0]
    safe_q, risky_q = RetrievalQuery.build(safe.text), RetrievalQuery.build(risky.text)
    safe_result = assess_cold_start(ctx.index, safe_q, safe, ctx.index.probe(safe_q.terms), is_final=False)
    risky_result = assess_cold_start(ctx.index, risky_q, risky, ctx.index.probe(risky_q.terms), is_final=False)
    assert safe_result.constraint_risk < risky_result.constraint_risk
    assert safe_result.probability is None  # heuristic confidence is not mislabeled as calibrated
    assert risky_result.decision_reason == "MISSING_CRITICAL_CONSTRAINT"
