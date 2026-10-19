"""Canonical online feature extraction for KAIROS V2.3.

This module is the single train/serve implementation for IC-COS.  Every value
is computable at the current prefix: it never performs a full retrieval and it
never inspects a later transcript chunk.  Historical V2.2 feature code remains
in :mod:`kairos.controller.policy` only to reproduce published baselines.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kairos.controller.cold_start import critical_constraint_risk
from kairos.controller.intent_readiness import assess_intent_readiness
from kairos.retrieval.index import HybridIndex, RetrievalQuery
from kairos.streaming.frame import CandidateNeed, Frame, completeness, continuation_hazard
from kairos.textutil import content_terms, rbo


FEATURE_SCHEMA_VERSION = "kairos-iccos-online-v1"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    definition: str
    value_range: str
    missing_value: str
    availability: str
    cost: str
    pre_retrieval: bool = True
    uses_history: bool = False
    safe_on_first_observation: bool = True


FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec("probe_section_rbo", "float", "RBO between current and previous cheap BM25 section rankings.", "[0,1]", "0 with ranking_history_available=0", "Current and prior candidate probe", "one cheap probe", uses_history=True),
    FeatureSpec("ranking_history_available", "float", "Whether a real prior ranking exists for this candidate.", "{0,1}", "0", "Candidate state", "constant"),
    FeatureSpec("probe_clarity", "float", "One minus normalized entropy of current probe scores.", "[0,1]", "0", "Current probe", "linear in probe depth"),
    FeatureSpec("syntactic_completeness", "float", "Deterministic current-clause completeness score.", "[0,1]", "0", "Current QueryFrame", "linear in prefix"),
    FeatureSpec("slot_stability", "float", "Fraction of current candidate slots unchanged from the prior observation.", "[0,1]", "0 with slot_history_available=0", "Current/prior candidate slot snapshots", "linear in candidate slots", uses_history=True),
    FeatureSpec("slot_history_available", "float", "Whether comparable prior slot state exists.", "{0,1}", "0", "Candidate state", "constant"),
    FeatureSpec("continuation_safety", "float", "One minus deterministic continuation hazard.", "[0,1]", "0", "Current QueryFrame", "linear in prefix"),
    FeatureSpec("query_informativeness", "float", "Mean corpus-normalized IDF plus bounded known-term coverage.", "[0,1]", "0", "Current query and BM25 vocabulary", "linear in query terms"),
    FeatureSpec("probe_evidence_relevance", "float", "Maximum cheap query-term coverage among chunks in the top probed section.", "[0,1]", "0", "Current probe and in-memory term sets", "linear in top-section chunks"),
    FeatureSpec("rank_separation", "float", "Bounded standardized gap between the first two probe scores.", "[0,1]", "0", "Current probe", "linear in probe depth"),
    FeatureSpec("objective_clarity", "float", "Current topic confidence with an explicit request-cue floor.", "[0,1]", "0", "Current QueryFrame", "constant"),
    FeatureSpec("constraint_readiness", "float", "Whether current typed constraints are non-dangling and unambiguous.", "[0,1]", "0", "Current QueryFrame", "linear in constraints"),
    FeatureSpec("constraint_revision_risk", "float", "Online risk from dangling, ambiguous, or already revised constraints.", "[0,1]", "1", "Current QueryFrame", "linear in constraints"),
    FeatureSpec("domain_availability", "float", "Compatibility transform: 0 unavailable, 0.65 unknown-compatible, otherwise 0.65+0.35 confidence.", "[0,1]", "0", "Current query and corpus domains", "linear in query terms"),
    FeatureSpec("candidate_age", "float", "Candidate-local prior observation count divided by five and clipped.", "[0,1]", "0", "Candidate state", "constant", uses_history=True),
    FeatureSpec("intent_readiness", "float", "Current intent-level completion/readiness blend.", "[0,1]", "0", "Current QueryFrame", "constant"),
    FeatureSpec("intent_independence", "float", "Penalty for anaphoric or dependent elaborations.", "[0,1]", "0", "Current QueryFrame", "constant"),
    FeatureSpec("previous_evidence_section_jaccard", "float", "Jaccard overlap of current probe sections and previously accepted evidence sections.", "[0,1]", "0 with previous_evidence_available=0", "Current probe and accepted candidate evidence", "linear in section count", uses_history=True),
    FeatureSpec("previous_evidence_available", "float", "Whether accepted evidence exists for this candidate.", "{0,1}", "0", "Candidate state", "constant", uses_history=True),
    FeatureSpec("previous_query_compatibility", "float", "Cosine similarity between current and prior candidate retrieval queries.", "[-1,1]", "0 with query_history_available=0", "Current/prior candidate queries", "two cached encodings", uses_history=True),
    FeatureSpec("query_history_available", "float", "Whether a real prior candidate query exists.", "{0,1}", "0", "Candidate state", "constant"),
    FeatureSpec("cache_reusable", "float", "Whether the exact provenance-aware retrieval fingerprint is reusable.", "{0,1}", "0", "Session cache metadata", "constant"),
    FeatureSpec("remaining_s", "float", "Expected remaining speech seconds from current end-of-utterance evidence.", "[0,+inf)", "0", "Current prefix and configured prior", "constant"),
    FeatureSpec("prefix_length", "float", "Current candidate content-term count divided by sixteen and clipped.", "[0,1]", "0", "Current QueryFrame", "linear in prefix"),
)

FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SCHEMA)


def feature_schema_hash() -> str:
    encoded = json.dumps([asdict(spec) for spec in FEATURE_SCHEMA], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def feature_schema_public() -> dict[str, Any]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "sha256": feature_schema_hash(),
        "features": [asdict(spec) for spec in FEATURE_SCHEMA],
    }


def probe_clarity(scores: list[float], temperature: float = 0.25) -> float:
    if not scores or scores[0] <= 0:
        return 0.0
    if len(scores) == 1:
        return 1.0
    values = np.asarray(scores, dtype=float) / (scores[0] * temperature)
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    return float(max(0.0, 1.0 - entropy / math.log(len(scores))))


def query_informativeness(index: HybridIndex, query: RetrievalQuery) -> float:
    terms = list(query.terms)
    if not terms:
        return 0.0
    known = [index.bm25.idf_of(term) for term in terms if term in index.bm25.idf]
    if not known:
        return 0.0
    corpus_max = max(index.bm25.idf.values(), default=1.0)
    return min(1.0, 0.8 * (sum(known) / len(known)) / max(corpus_max, 1e-9) + 0.2 * min(1.0, len(known) / 2.0))


def rank_separation(scores: list[float]) -> float:
    if not scores or scores[0] <= 0:
        return 0.0
    if len(scores) == 1:
        return 1.0
    values = np.asarray(scores, dtype=float)
    scale = float(values.std()) or float(values[0]) or 1.0
    return 1.0 - math.exp(-max(0.0, float(values[0] - values[1]) / scale))


def constraint_readiness(candidate: CandidateNeed) -> float:
    terms = content_terms(candidate.text)
    if not terms:
        return 0.0
    dangling = {"under", "below", "above", "over", "in", "for", "with", "without", "from", "to"}
    if candidate.text.lower().split()[-1].strip(".,?!") in dangling:
        return 0.0
    return 0.35 if any(item.ambiguous for item in candidate.typed_constraints) else 1.0


def _candidate_slots(candidate: CandidateNeed, frame: Frame) -> dict[str, str]:
    owned = {key: slot.value for key, slot in frame.slots.items() if slot.owner_key == candidate.key}
    if owned:
        return owned
    low = candidate.text.lower()
    return {key: slot.value for key, slot in frame.slots.items() if slot.value and slot.value.lower() in low}


def _domain_availability(index: HybridIndex, query: RetrievalQuery) -> float:
    domain, confidence, compatible = index.route_domain(query.dense_text)
    if not compatible:
        return 0.0
    return 0.65 if domain == "unknown" else 0.65 + 0.35 * confidence


def _top_probe_relevance(index: HybridIndex, query: RetrievalQuery, probe: list[tuple[str, float]]) -> float:
    if not probe:
        return 0.0
    section = probe[0][0]
    positions = (i for i, chunk in enumerate(index.chunks) if chunk.section_key == section)
    return max((index.coverage(query.terms, i) for i in positions), default=0.0)


def extract_online_features(
    *,
    index: HybridIndex,
    query: RetrievalQuery,
    candidate: CandidateNeed,
    frame: Frame,
    state: Any,
    probe: list[tuple[str, float]],
    is_final: bool,
    utterance_end_cue: bool,
    remaining_prior_s: float,
    cache_reusable: bool = False,
) -> dict[str, float]:
    """Extract the canonical online vector without a hidden full search."""
    keys = [key for key, _ in probe]
    ranking_available = bool(state.observations > 0 and state.last_probe)
    rho = rbo(keys, state.last_probe) if ranking_available and keys else 0.0

    current_slots = _candidate_slots(candidate, frame)
    prior_slots = getattr(state, "last_slots", {}) or {}
    comparable = set(current_slots) & set(prior_slots)
    slot_available = bool(comparable)
    sigma = sum(current_slots[key] == prior_slots[key] for key in comparable) / len(comparable) if comparable else 0.0

    kappa = completeness(candidate.text, candidate.closed, is_final)
    pi = continuation_hazard(candidate.text, candidate.closed, is_final)
    readiness = assess_intent_readiness(candidate, {"kappa": kappa, "pi": pi}, is_final=is_final)
    scores = [score for _, score in probe]
    prior_sections = set(getattr(state, "retrieved_sections", []) or [])
    evidence_available = bool(prior_sections)
    section_overlap = len(set(keys) & prior_sections) / len(set(keys) | prior_sections) if keys and prior_sections else 0.0
    prior_query = str(getattr(state, "last_query_text", "") or "")
    query_available = bool(prior_query)
    query_compatibility = index.cosine(query.dense_text, prior_query) if query_available and query.dense_text else 0.0
    p_end = 1.0 if is_final else (0.6 if utterance_end_cue else 0.0)

    values = {
        "probe_section_rbo": rho,
        "ranking_history_available": float(ranking_available),
        "probe_clarity": probe_clarity(scores),
        "syntactic_completeness": kappa,
        "slot_stability": sigma,
        "slot_history_available": float(slot_available),
        "continuation_safety": 1.0 - pi,
        "query_informativeness": query_informativeness(index, query),
        "probe_evidence_relevance": _top_probe_relevance(index, query, probe),
        "rank_separation": rank_separation(scores),
        "objective_clarity": max(float(candidate.confidence.get("topic", 0.0)), 0.75 if candidate.request_cue else 0.0),
        "constraint_readiness": constraint_readiness(candidate),
        "constraint_revision_risk": critical_constraint_risk(candidate),
        "domain_availability": _domain_availability(index, query),
        "candidate_age": min(float(state.observations) / 5.0, 1.0),
        "intent_readiness": readiness.score,
        "intent_independence": readiness.independent,
        "previous_evidence_section_jaccard": section_overlap,
        "previous_evidence_available": float(evidence_available),
        "previous_query_compatibility": query_compatibility,
        "query_history_available": float(query_available),
        "cache_reusable": float(cache_reusable),
        "remaining_s": (1.0 - p_end) * remaining_prior_s,
        "prefix_length": min(len(content_terms(candidate.text)) / 16.0, 1.0),
    }
    assert tuple(values) == FEATURE_NAMES
    return {name: round(float(value), 6) for name, value in values.items()}


def candidate_slot_snapshot(candidate: CandidateNeed, frame: Frame) -> dict[str, str]:
    """Public helper used by live and replay state transitions."""
    return _candidate_slots(candidate, frame)
