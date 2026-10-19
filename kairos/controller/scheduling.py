"""Deterministic marginal-value scheduler for multi-intent experiments."""

from __future__ import annotations


def priority(signals: dict[str, float], *, already_retrieved: bool, remaining_budget: int) -> float:
    if remaining_budget <= 0:
        return float("-inf")
    readiness = float(signals.get("intent_readiness", signals.get("kappa", 0.0)))
    survival = float(signals.get("s_hat", 0.0))
    novelty = float(signals.get("nu", 1.0))
    relevance = float(signals.get("cold_evidence_relevance", signals.get("chi", 0.0)))
    refinement_penalty = 0.18 if already_retrieved else 0.0
    return 0.35 * readiness + 0.30 * survival + 0.20 * relevance + 0.15 * novelty - refinement_penalty
