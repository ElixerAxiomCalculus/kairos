"""Risk-aware retrieval utility without claims of formal optimality."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityAssessment:
    retrieve: float
    wait: float
    predicted_risk: float
    allowed: bool

    def signals(self) -> dict[str, float]:
        return {
            "utility_retrieve": round(self.retrieve, 4),
            "utility_wait": round(self.wait, 4),
            "predicted_risk": round(self.predicted_risk, 4),
            "utility_allowed": float(self.allowed),
        }


def assess_utility(probability: float, hidden_latency_s: float, cfg: dict) -> UtilityAssessment:
    latency_scale = max(float(cfg.get("latency_scale_s", 1.2)), 1e-6)
    latency_benefit = min(max(hidden_latency_s / latency_scale, 0.0), 1.0)
    refine_cost = float(cfg.get("refinement_cost", 0.45))
    extra_cost = float(cfg.get("extra_retrieval_cost", 0.12))
    retrieve = probability * latency_benefit - (1.0 - probability) * refine_cost - extra_cost
    wait = float(cfg.get("wait_utility", 0.0))
    risk = 1.0 - probability
    allowed = retrieve > wait and risk <= float(cfg.get("max_predicted_risk", 0.35))
    return UtilityAssessment(retrieve, wait, risk, allowed)
