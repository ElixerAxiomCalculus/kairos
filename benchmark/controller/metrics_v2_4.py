"""Metric primitives for CAPER real-engine evaluation.

Every public metric carries its population, eligibility rule, unit, timing
source and grounding status.  Prefixes are never treated as independent
samples; uncertainty is grouped by question family.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from benchmark.controller.metrics import ratio, wilson


def proportion(
    numerator: int,
    denominator: int,
    *,
    unit: str,
    eligibility: str,
    timing_source: str,
    grounding_status: str,
    split: str,
    measurement_source: str = "ACTUAL_SESSION_ENGINE_EVENTS",
) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": ratio(numerator, denominator),
        "wilson_95": wilson(numerator, denominator),
        "unit": unit,
        "eligibility": eligibility,
        "timing_source": timing_source,
        "grounding_status": grounding_status,
        "split": split,
        "measurement_source": measurement_source,
    }


def family_bootstrap(
    rows: list[dict[str, Any]],
    outcome: str,
    *,
    family_key: str = "question_family",
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster bootstrap of a binary stream outcome by question family."""
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get(outcome) is not None:
            grouped[str(row[family_key])].append(int(bool(row[outcome])))
    families = sorted(grouped)
    if not families:
        return {"estimate": None, "ci_95": [None, None], "families": 0, "samples": 0}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        picked = [rng.choice(families) for _ in families]
        values = [value for family in picked for value in grouped[family]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    observed = [value for family in families for value in grouped[family]]
    return {
        "estimate": round(sum(observed) / len(observed), 6),
        "ci_95": [round(estimates[int(samples * 0.025)], 6),
                  round(estimates[min(samples - 1, int(samples * 0.975))], 6)],
        "families": len(families),
        "samples": samples,
        "seed": seed,
        "resampling_unit": "question_family",
    }
