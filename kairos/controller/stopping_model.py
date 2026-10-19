"""Strict, versioned IC-COS stopping-model loader and scorer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kairos.config import resolve_path
from kairos.controller.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, feature_schema_hash


class ModelCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class StoppingPrediction:
    delta_q: float
    beneficial_probability: float
    uncertainty: float


class StoppingModel:
    def __init__(self, artifact: dict[str, Any], source: str) -> None:
        self.artifact = artifact
        self.source = source
        self.features = tuple(artifact["required_features"])
        self.classifier = artifact["models"]["beneficial_stop_classifier"]
        self.regressor = artifact["models"]["delta_q_regressor"]
        self.calibration = artifact.get("calibration", {"method": "identity", "slope": 1.0, "intercept": 0.0})
        self.decision_threshold = float(artifact["decision_threshold"])
        self.risk = artifact.get("risk", {})

    @classmethod
    def load(cls, path_value: str | Path) -> "StoppingModel":
        path = resolve_path(path_value)
        if not path.exists():
            raise ModelCompatibilityError(f"MODEL_NOT_FOUND:{path}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelCompatibilityError(f"MODEL_UNREADABLE:{exc}") from exc
        required = {
            "model_schema_version", "feature_schema_version", "feature_schema_hash", "required_features",
            "dataset_hash", "calibration", "models", "decision_threshold", "objective", "validation",
        }
        missing = sorted(required - set(artifact))
        if missing:
            raise ModelCompatibilityError(f"MODEL_SCHEMA_MISSING:{','.join(missing)}")
        if artifact["model_schema_version"] != "kairos-iccos-model-v1":
            raise ModelCompatibilityError("MODEL_SCHEMA_VERSION_MISMATCH")
        if artifact["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ModelCompatibilityError("FEATURE_SCHEMA_VERSION_MISMATCH")
        if artifact["feature_schema_hash"] != feature_schema_hash():
            raise ModelCompatibilityError("FEATURE_SCHEMA_HASH_MISMATCH")
        unknown = sorted(set(artifact["required_features"]) - set(FEATURE_NAMES))
        if unknown:
            raise ModelCompatibilityError(f"UNKNOWN_REQUIRED_FEATURES:{','.join(unknown)}")
        for name in ("beneficial_stop_classifier", "delta_q_regressor"):
            spec = artifact["models"].get(name, {})
            if "weights" not in spec or "bias" not in spec:
                raise ModelCompatibilityError(f"INVALID_MODEL_WEIGHTS:{name}")
            if sorted(spec["weights"]) != sorted(artifact["required_features"]):
                raise ModelCompatibilityError(f"MODEL_FEATURE_WEIGHT_MISMATCH:{name}")
        return cls(artifact, str(path))

    def _vector(self, features: dict[str, Any]) -> dict[str, float]:
        missing = [name for name in self.features if name not in features or features[name] is None]
        if missing:
            raise ModelCompatibilityError(f"MISSING_LIVE_FEATURES:{','.join(missing)}")
        return {name: float(features[name]) for name in self.features}

    @staticmethod
    def _linear(spec: dict[str, Any], vector: dict[str, float]) -> float:
        return float(spec["bias"]) + sum(float(spec["weights"][name]) * value for name, value in vector.items())

    def predict(self, features: dict[str, Any]) -> StoppingPrediction:
        vector = self._vector(features)
        classifier_logit = self._linear(self.classifier, vector)
        raw_probability = 1.0 / (1.0 + math.exp(-max(min(classifier_logit, 40.0), -40.0)))
        method = self.calibration.get("method", "identity")
        if method == "platt":
            logit = math.log(min(max(raw_probability, 1e-6), 1 - 1e-6) / (1 - min(max(raw_probability, 1e-6), 1 - 1e-6)))
            z = float(self.calibration.get("slope", 1.0)) * logit + float(self.calibration.get("intercept", 0.0))
            probability = 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))
        elif method == "identity":
            probability = raw_probability
        else:
            raise ModelCompatibilityError(f"UNSUPPORTED_CALIBRATION:{method}")
        delta = self._linear(self.regressor, vector)
        return StoppingPrediction(delta, probability, 1.0 - abs(2.0 * probability - 1.0))
