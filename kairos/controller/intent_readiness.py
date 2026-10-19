"""Intent-level readiness distinct from whole-utterance completion."""

from __future__ import annotations

from dataclasses import dataclass

from kairos.streaming.frame import CandidateNeed


@dataclass(frozen=True)
class IntentReadiness:
    score: float
    completion: float
    objective_clarity: float
    constraint_readiness: float
    continuation_safety: float
    independent: float

    def signals(self) -> dict[str, float]:
        return {key: round(float(value), 4) for key, value in self.__dict__.items()}


def assess_intent_readiness(candidate: CandidateNeed, signals: dict[str, float], *, is_final: bool) -> IntentReadiness:
    completion = max(float(signals.get("kappa", 0.0)), 1.0 if is_final else (0.85 if candidate.closed else 0.0))
    objective = max(float(candidate.confidence.get("topic", 0.0)), 0.75 if candidate.request_cue else 0.0)
    ambiguous = any(constraint.ambiguous for constraint in candidate.typed_constraints)
    constraint = 0.35 if ambiguous else (1.0 if candidate.status == "VALID_STANDALONE" else 0.45)
    continuation = 1.0 - float(signals.get("pi", 1.0))
    independent = 0.25 if candidate.anaphoric or candidate.status == "DEPENDENT_ELABORATION" else 1.0
    score = 0.32 * completion + 0.23 * objective + 0.20 * constraint + 0.15 * continuation + 0.10 * independent
    return IntentReadiness(min(max(score, 0.0), 1.0), completion, objective, constraint, continuation, independent)
