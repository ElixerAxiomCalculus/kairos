"""Interpretable confidence for the first meaningful PSRC observation.

The estimator intentionally uses only information available at the current
prefix.  It does not infer stability from a fabricated previous ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kairos.retrieval.index import HybridIndex, RetrievalQuery
from kairos.streaming.frame import CandidateNeed
from kairos.textutil import content_terms, lower_tokens
from kairos.controller.calibration import calibrate_probability


@dataclass(frozen=True)
class ColdStartAssessment:
    confidence: float
    probability: float | None
    informativeness: float
    rank_separation: float
    evidence_relevance: float
    domain_compatibility: float
    constraint_readiness: float
    objective_clarity: float
    asr_stability: float
    constraint_risk: float
    decision_uncertainty: float
    decision_reason: str

    def signals(self) -> dict[str, Any]:
        return {
            key: round(float(value), 4) if isinstance(value, (int, float)) else value
            for key, value in self.__dict__.items()
        }


def _informativeness(index: HybridIndex, query: RetrievalQuery) -> float:
    terms = list(query.terms)
    if not terms:
        return 0.0
    known = [index.bm25.idf_of(term) for term in terms if term in index.bm25.idf]
    if not known:
        return 0.0
    corpus_max = max(index.bm25.idf.values(), default=1.0)
    # Mean IDF avoids rewarding verbosity; the small coverage term rewards a
    # second independent discriminative word without making length dominant.
    mean_idf = sum(known) / len(known)
    coverage = min(1.0, len(known) / 2.0)
    return min(1.0, 0.8 * mean_idf / max(corpus_max, 1e-9) + 0.2 * coverage)


def _rank_separation(scores: list[float]) -> float:
    if not scores or scores[0] <= 0:
        return 0.0
    if len(scores) == 1:
        return 1.0
    values = np.asarray(scores, dtype=float)
    scale = float(values.std()) or float(values[0]) or 1.0
    z_margin = max(0.0, float(values[0] - values[1]) / scale)
    return 1.0 - math.exp(-z_margin)


def _constraint_readiness(candidate: CandidateNeed) -> float:
    low = lower_tokens(candidate.text)
    if not low:
        return 0.0
    dangling = {"under", "below", "above", "over", "in", "for", "with", "without", "from", "to"}
    if low[-1] in dangling:
        return 0.0
    if any(constraint.ambiguous for constraint in candidate.typed_constraints):
        return 0.35
    return 1.0


def critical_constraint_risk(candidate: CandidateNeed) -> float:
    """Online-only risk signal; unavailable future constraints are never inferred."""
    words = lower_tokens(candidate.text)
    if not words:
        return 1.0
    dangling = {"under", "below", "above", "over", "in", "for", "with", "without", "from", "to"}
    if words[-1] in dangling:
        return 1.0
    typed = candidate.typed_constraints
    if any(item.ambiguous for item in typed):
        return 0.85
    if any(getattr(item, "relation", "EQ") in {"MAX", "MIN"} and not item.normalized_value for item in typed):
        return 0.9
    if getattr(candidate, "revision_history", []):
        return 0.7
    # A recognized objective with no dangling/ambiguous constraint has low,
    # not zero, revision risk because future speech can still add a qualifier.
    return 0.18 if candidate.request_cue or len(words) >= 4 else 0.45


def assess_cold_start(
    index: HybridIndex,
    query: RetrievalQuery,
    candidate: CandidateNeed,
    probe: list[tuple[str, float]],
    *,
    is_final: bool,
    committed_prefix: bool = True,
    cfg: dict | None = None,
) -> ColdStartAssessment:
    scores = [score for _, score in probe]
    info = _informativeness(index, query)
    separation = _rank_separation(scores)
    domain, domain_confidence, compatible = index.route_domain(query.dense_text)
    domain_score = 0.0 if not compatible else (0.65 if domain == "unknown" else 0.65 + 0.35 * domain_confidence)

    relevance = 0.0
    if probe:
        section = probe[0][0]
        members = [i for i, chunk in enumerate(index.chunks) if chunk.section_key == section]
        relevance = max((index.coverage(query.terms, i) for i in members), default=0.0)

    constraint = _constraint_readiness(candidate)
    risk = critical_constraint_risk(candidate)
    objective = max(
        float(candidate.confidence.get("completeness", 0.0)),
        0.7 if candidate.request_cue else 0.0,
    )
    asr = 1.0 if is_final or candidate.closed else (0.8 if committed_prefix else 0.35)
    # Pre-registered, monotone, interpretable blend.  Coefficients are exposed
    # in decision traces and are not learned from the development labels.
    base_confidence = (
        0.20 * info
        + 0.15 * separation
        + 0.25 * relevance
        + 0.10 * domain_score
        + 0.15 * constraint
        + 0.10 * objective
        + 0.05 * asr
    )
    cold_cfg = cfg or {}
    variant = str(cold_cfg.get("variant", "CS0_original_heuristic"))
    confidence = base_confidence if variant.startswith("CS0") else base_confidence * (1.0 - 0.45 * risk)
    calibration_cfg = cold_cfg.get("calibration", {})
    probability = calibrate_probability(confidence, calibration_cfg) if calibration_cfg.get("enabled", False) else None
    decision_value = probability if probability is not None else confidence
    if relevance <= 0.0 or domain_score <= 0.0:
        reason = "INSUFFICIENT_CORPUS"
    elif risk >= 0.75:
        reason = "MISSING_CRITICAL_CONSTRAINT"
    elif decision_value >= float(cold_cfg.get("threshold", 0.68)):
        reason = "SAFE_COLD_START"
    else:
        reason = "COLD_START_RISK_OR_CONFIDENCE"
    return ColdStartAssessment(
        confidence=min(max(confidence, 0.0), 1.0),
        probability=min(max(probability, 0.0), 1.0) if probability is not None else None,
        informativeness=info,
        rank_separation=separation,
        evidence_relevance=relevance,
        domain_compatibility=domain_score,
        constraint_readiness=constraint,
        objective_clarity=objective,
        asr_stability=asr,
        constraint_risk=risk,
        decision_uncertainty=1.0 - abs(2.0 * decision_value - 1.0),
        decision_reason=reason,
    )
