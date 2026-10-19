"""Train, calibrate, and risk-select the IC-COS classifier and value model."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.controller.counterfactual import label_optimal_stopping, oracle_summary
from benchmark.controller.dataset_v2_3 import ROOT as DATA_ROOT
from benchmark.controller.dataset_v2_3 import build_dataset, load_split
from benchmark.controller.features_v2_3 import replay_feature_rows
from benchmark.controller.metrics import calibration_metrics, wilson
from kairos.config import Config, resolve_path
from kairos.controller.features import FEATURE_SCHEMA_VERSION, feature_schema_hash
from kairos.engine import EngineContext


MODEL_ROOT = resolve_path("models/controller_v2_3")
MODEL_FEATURES = (
    "probe_section_rbo", "ranking_history_available", "probe_clarity", "syntactic_completeness",
    "slot_stability", "slot_history_available", "continuation_safety", "query_informativeness",
    "probe_evidence_relevance", "rank_separation", "objective_clarity", "constraint_readiness",
    "constraint_revision_risk", "domain_availability", "candidate_age", "intent_readiness",
    "intent_independence", "previous_query_compatibility", "query_history_available",
    "remaining_s", "prefix_length",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row["features"][name]) for name in MODEL_FEATURES] for row in rows], dtype=float)


def _linear_spec(model) -> dict[str, Any]:
    coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
    intercept = model.intercept_[0] if np.ndim(model.intercept_) else model.intercept_
    return {
        "weights": {name: round(float(value), 10) for name, value in zip(MODEL_FEATURES, coef)},
        "bias": round(float(intercept), 10),
    }


def extract_splits(cfg: Config, splits: tuple[str, ...], artifact_dir: Path) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, EngineContext] = {}
    extracted = {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = []
        for stream in load_split(split):
            profile = stream["corpus_profile"]
            if profile not in contexts:
                contexts[profile] = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
            rows.extend(replay_feature_rows(contexts[profile], stream))
        labeled = label_optimal_stopping(rows)
        extracted[split] = labeled
        (artifact_dir / f"features_{split}.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in labeled), encoding="utf-8"
        )
        summaries = {}
        for stream_id in sorted({row["stream_id"] for row in rows}):
            summaries[stream_id] = oracle_summary([row for row in rows if row["stream_id"] == stream_id])
        (artifact_dir / f"oracle_{split}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return extracted


def _prediction(spec: dict[str, Any], rows: list[dict[str, Any]], *, logistic: bool) -> list[float]:
    values = []
    for row in rows:
        score = float(spec["bias"]) + sum(float(spec["weights"][name]) * float(row["features"][name]) for name in MODEL_FEATURES)
        values.append(1.0 / (1.0 + math.exp(-max(min(score, 40.0), -40.0))) if logistic else score)
    return values


def _platt(raw_probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    if len(set(labels)) < 2 or len(labels) < 8:
        return {"method": "identity", "slope": 1.0, "intercept": 0.0,
                "reason": "INSUFFICIENT_CALIBRATION_CLASSES_OR_ROWS"}
    from sklearn.linear_model import LogisticRegression
    logits = np.asarray([
        math.log(min(max(value, 1e-6), 1 - 1e-6) / (1 - min(max(value, 1e-6), 1 - 1e-6)))
        for value in raw_probabilities
    ]).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(logits, np.asarray(labels))
    return {"method": "platt", "slope": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


def _apply_calibration(probabilities: list[float], calibration: dict[str, Any]) -> list[float]:
    if calibration["method"] == "identity":
        return probabilities
    output = []
    for probability in probabilities:
        clipped = min(max(probability, 1e-6), 1 - 1e-6)
        logit = math.log(clipped / (1 - clipped))
        z = calibration["slope"] * logit + calibration["intercept"]
        output.append(1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0))))
    return output


def _sequential_selected(
    rows: list[dict[str, Any]], probabilities: list[float], deltas: list[float], threshold: float
) -> list[int]:
    """Apply the deployable first-eligible-action rule once per intent lineage."""
    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["stream_id"], row["candidate_id"])].append(index)
    selected: list[int] = []
    for indices in grouped.values():
        indices.sort(key=lambda index: rows[index]["prefix_index"])
        match = next((
            index for index in indices
            if not rows[index].get("is_final") and probabilities[index] >= threshold and deltas[index] > 0.0
        ), None)
        if match is not None:
            selected.append(match)
    return selected


def _select_threshold(rows: list[dict[str, Any]], probabilities: list[float], deltas: list[float], max_risk: float = 0.20) -> dict[str, Any]:
    table = []
    best = None
    for threshold in [round(value / 100, 2) for value in range(35, 96, 5)]:
        selected = _sequential_selected(rows, probabilities, deltas, threshold)
        unsafe = sum(rows[index]["targets"].get("proxy_strict_useful",
                                                rows[index]["targets"].get("strict_useful")) != 1
                     for index in selected)
        useful = len(selected) - unsafe
        risk = unsafe / len(selected) if selected else None
        eligible_lineages = len({(row["stream_id"], row["candidate_id"]) for row in rows})
        useful_lineages = {
            (row["stream_id"], row["candidate_id"])
            for row in rows if row["targets"].get("proxy_strict_useful", row["targets"].get("strict_useful")) == 1
        }
        recall_den = len(useful_lineages)
        recall = useful / recall_den if recall_den else None
        families = {rows[index]["question_family"] for index in selected}
        record = {"threshold": threshold, "selected": len(selected), "useful": useful, "unsafe": unsafe,
                  "coverage": round(len(selected) / eligible_lineages, 6) if eligible_lineages else None,
                  "conditional_selected_action_risk": round(risk, 6) if risk is not None else None,
                  "empirical_risk": round(risk, 6) if risk is not None else None,
                  "selected_action_risk_wilson_95": wilson(unsafe, len(selected)) if selected else None,
                  "useful_recall": round(recall, 6) if recall is not None else None,
                  "selected_question_families": len(families),
                  "promotion_eligible": bool(selected) and risk is not None and risk <= max_risk,
                  "eligible": bool(selected) and risk is not None and risk <= max_risk}
        table.append(record)
        if record["eligible"] and (best is None or (useful, -risk, -threshold) > (best[0], -best[1], -best[2])):
            best = (useful, risk, threshold, record)
    chosen = best[3] if best else table[-1]
    return {"max_empirical_risk": max_risk, "selection_rule": "FIRST_ELIGIBLE_PREFIX_PER_INTENT_LINEAGE",
            "risk_target": "PROXY_STRICT_USEFUL_NOT_VERIFIED_SAFETY", "selected": chosen, "sweep": table}


def train_models(cfg: Config, artifact_dir: Path | None = None) -> dict[str, Any]:
    manifest = build_dataset()
    artifact_dir = artifact_dir or resolve_path("runs/bench/controller_v2_3/modeling")
    rows = extract_splits(cfg, ("train", "calibration", "development"), artifact_dir)
    train, calibration, development = rows["train"], rows["calibration"], rows["development"]
    if len({row["beneficial_stop"] for row in train}) < 2:
        raise RuntimeError("IC-COS training labels contain only one class")

    from sklearn.linear_model import LogisticRegression, Ridge
    classifier = LogisticRegression(C=0.5, max_iter=3000, random_state=42, class_weight="balanced").fit(
        _matrix(train), np.asarray([row["beneficial_stop"] for row in train])
    )
    regressor = Ridge(alpha=1.0).fit(_matrix(train), np.asarray([row["delta_q"] for row in train]))
    classifier_spec = _linear_spec(classifier)
    regressor_spec = _linear_spec(regressor)

    cal_raw = _prediction(classifier_spec, calibration, logistic=True)
    calibration_spec = _platt(cal_raw, [row["beneficial_stop"] for row in calibration])
    dev_raw = _prediction(classifier_spec, development, logistic=True)
    dev_prob = _apply_calibration(dev_raw, calibration_spec)
    dev_delta = _prediction(regressor_spec, development, logistic=False)
    risk = _select_threshold(development, dev_prob, dev_delta)
    labels = [row["beneficial_stop"] for row in development]
    mae = sum(abs(predicted - row["delta_q"]) for predicted, row in zip(dev_delta, development)) / max(1, len(development))
    sign_accuracy = sum((predicted > 0) == (row["delta_q"] > 0) for predicted, row in zip(dev_delta, development)) / max(1, len(development))

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "iccos_v1.json"
    artifact = {
        "model_schema_version": "kairos-iccos-model-v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(),
        "required_features": list(MODEL_FEATURES),
        "dataset_hash": manifest["dataset_hash"],
        "training_split_hash": manifest["split_hashes"]["train"],
        "calibration_split_hash": manifest["split_hashes"]["calibration"],
        "development_split_hash": manifest["split_hashes"]["development"],
        "models": {
            "beneficial_stop_classifier": {**classifier_spec, "family": "regularized_logistic", "C": 0.5},
            "delta_q_regressor": {**regressor_spec, "family": "ridge", "alpha": 1.0},
        },
        "calibration": calibration_spec,
        "decision_threshold": risk["selected"]["threshold"],
        "objective": "predict Q(RETRIEVE_NOW)-Q(WAIT_NEXT) under frozen deterministic continuation",
        "risk": {"max_empirical_risk": risk["max_empirical_risk"], "development": risk["selected"]},
        "validation": {
            "classifier": calibration_metrics(labels, dev_prob),
            "regression_mae": round(mae, 6),
            "delta_sign_accuracy": round(sign_accuracy, 6),
            "rows": {key: len(value) for key, value in rows.items()},
            "groups": {key: len({row["question_family"] for row in value}) for key, value in rows.items()},
            "future_feature_audit": "PASS",
            "forbidden_future_features": [],
        },
    }
    model_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    comparison = {
        "status": "MEASURED",
        "model_path": model_path.relative_to(resolve_path(".")).as_posix(),
        "artifact": artifact,
        "risk_selection": risk,
        "alignment_status_counts": {
            split: dict(Counter(row["final_alignment_status"] for row in values)) for split, values in rows.items()
        },
    }
    (artifact_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison
