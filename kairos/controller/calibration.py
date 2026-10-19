"""Small probability-calibration helpers used by PSRC experiments."""

from __future__ import annotations

import math


def logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def calibrate_probability(probability: float, cfg: dict) -> float:
    """Apply a frozen Platt transform; identity is the safe fallback."""
    if not cfg.get("enabled", False):
        return float(probability)
    method = cfg.get("method", "platt")
    if method == "platt":
        return sigmoid(float(cfg.get("slope", 1.0)) * logit(probability) + float(cfg.get("intercept", 0.0)))
    if method == "identity":
        return float(probability)
    raise ValueError(f"unsupported calibration method: {method}")
