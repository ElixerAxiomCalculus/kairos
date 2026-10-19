from __future__ import annotations

import math
import statistics
import random


def ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    den = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def calibration_metrics(labels: list[int], probabilities: list[float], bins: int = 10) -> dict:
    if not labels:
        return {"examples": 0, "brier_score": None, "ece": None, "log_loss": None, "bins": []}
    clipped = [min(max(float(p), 1e-6), 1 - 1e-6) for p in probabilities]
    brier = statistics.mean((p - y) ** 2 for p, y in zip(clipped, labels))
    log_loss = -statistics.mean(y * math.log(p) + (1 - y) * math.log(1 - p)
                                for p, y in zip(clipped, labels))
    reliability = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [(p, y) for p, y in zip(clipped, labels)
                   if lower <= p < upper or (index == bins - 1 and p == 1.0)]
        if not members:
            continue
        predicted = statistics.mean(p for p, _ in members)
        observed = statistics.mean(y for _, y in members)
        ece += len(members) * abs(predicted - observed) / len(labels)
        reliability.append({"lower": lower, "upper": upper, "count": len(members),
                            "mean_score": round(predicted, 4), "observed": round(observed, 4)})
    return {"examples": len(labels), "brier_score": round(brier, 4), "ece": round(ece, 4),
            "log_loss": round(log_loss, 4), "bins": reliability}


def grouped_bootstrap_binary(groups: list[dict], *, seed: int = 42, samples: int = 2000) -> list[float] | None:
    """Percentile interval by resampling independent utterance/family groups."""
    if not groups:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        denominator = sum(int(item.get("eligible", 1)) for item in draw)
        numerator = sum(int(item.get("success", 0)) for item in draw)
        values.append(numerator / denominator if denominator else 0.0)
    values.sort()
    return [round(values[int(0.025 * (samples - 1))], 4), round(values[int(0.975 * (samples - 1))], 4)]
