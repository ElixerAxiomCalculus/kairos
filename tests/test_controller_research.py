from __future__ import annotations

from kairos.config import load_config
from kairos.controller.asr_policy import committed_length
from kairos.controller.calibration import calibrate_probability
from kairos.controller.cold_start import assess_cold_start
from kairos.controller.intent_readiness import assess_intent_readiness
from kairos.controller.scheduling import priority
from kairos.controller.speculative import SpeculativeCache, SpeculativeEntry
from kairos.controller.utility import assess_utility
from kairos.engine import EngineContext
from kairos.retrieval.index import RetrievalQuery
from kairos.streaming.frame import CandidateNeed


def _candidate(ctx: EngineContext, text: str):
    return ctx.frame_builder.build(text, {}, 1.0, "c1").candidates[0]


def test_cold_start_distinguishes_informative_and_unknown_terms():
    ctx = EngineContext(load_config())
    informative = _candidate(ctx, "What is the cancellation policy")
    info_query = RetrievalQuery.build(informative.text)
    info = assess_cold_start(ctx.index, info_query, informative, ctx.index.probe(info_query.terms), is_final=False)
    unknown = _candidate(ctx, "What is the xylophonium policy")
    unknown_query = RetrievalQuery.build(unknown.text)
    weak = assess_cold_start(ctx.index, unknown_query, unknown, ctx.index.probe(unknown_query.terms), is_final=False)
    assert 0 <= info.confidence <= 1
    assert info.evidence_relevance > weak.evidence_relevance
    assert info.confidence > weak.confidence


def test_cold_start_rejects_missing_critical_constraint():
    ctx = EngineContext(load_config())
    candidate = _candidate(ctx, "Find a venue under")
    query = RetrievalQuery.build(candidate.text)
    assessment = assess_cold_start(ctx.index, query, candidate, ctx.index.probe(query.terms), is_final=False)
    assert assessment.constraint_readiness == 0.0


def test_intent_readiness_penalizes_anaphora():
    ctx = EngineContext(load_config())
    direct = _candidate(ctx, "What is the cancellation policy?")
    anaphoric = CandidateNeed("K-dependent", "Tell me more about it", [], False, True,
                              status="DEPENDENT_ELABORATION")
    signals = {"kappa": 0.9, "pi": 0.1}
    assert assess_intent_readiness(direct, signals, is_final=False).score > assess_intent_readiness(
        anaphoric, signals, is_final=False
    ).score


def test_calibration_is_deterministic_bounded_and_has_identity_fallback():
    assert calibrate_probability(0.37, {"enabled": False}) == 0.37
    first = calibrate_probability(0.37, {"enabled": True, "method": "platt", "slope": 1.2, "intercept": -0.1})
    second = calibrate_probability(0.37, {"enabled": True, "method": "platt", "slope": 1.2, "intercept": -0.1})
    assert first == second
    assert 0.0 < first < 1.0


def test_speculative_cache_promotes_once_and_rejects_stale_or_incompatible_queries():
    ctx = EngineContext(load_config())
    query = RetrievalQuery.build("cancellation policy venue bookings")
    result = ctx.index.search(query)
    cache = SpeculativeCache()
    entry = SpeculativeEntry("s", "u", "k", 1, query.dense_text.lower(), "sample", result, 1.0, 2.0)
    assert cache.put(entry)
    assert cache.promote("s", "u", "k", 2, query, "sample") is result
    assert cache.promote("s", "u", "k", 2, query, "sample") is None

    other = RetrievalQuery.build("nearest hospital medicine")
    cache.put(SpeculativeEntry("s", "u", "other", 1, query.dense_text.lower(), "sample", result, 1.0, 2.0))
    assert cache.promote("s", "u", "other", 2, other, "sample") is None


def test_utility_enforces_risk_and_budget_signal():
    safe = assess_utility(0.9, 1.2, {"max_predicted_risk": 0.35})
    risky = assess_utility(0.4, 1.2, {"max_predicted_risk": 0.35})
    assert safe.allowed
    assert not risky.allowed
    assert priority({"s_hat": 0.9, "kappa": 0.9, "nu": 1.0, "chi": 0.8},
                    already_retrieved=False, remaining_budget=0) == float("-inf")


def test_asr_strategies_preserve_correction_distinction():
    assert committed_length("book a", "book a hall", strategy="agreement") == 2
    assert committed_length("book a", "book a hall", strategy="fixed_tail", mutable_tail=2) == 0
    assert committed_length("find Pune", "find Mumbai", strategy="agreement") == 1
    assert committed_length("find Pune", "find Mumbai", strategy="provisional_probe") == 2
