"""Frozen A0-A8 IC-COS ablations, profile checks, and promotion decision."""

from __future__ import annotations

import json
import math
import platform
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmark.controller.audit_v2_3 import audit_v22_empty_final_evidence
from benchmark.controller.counterfactual import label_optimal_stopping, validate_real_branches
from benchmark.controller.dataset_v2_3 import load_split
from benchmark.controller.features_v2_3 import replay_feature_rows
from benchmark.controller.metrics_v2_3 import evaluate_policy
from benchmark.controller.modeling_v2_3 import MODEL_FEATURES, extract_splits, train_models
from kairos.config import Config, resolve_path
from kairos.controller.stopping_model import StoppingModel
from kairos.engine import EngineContext
from kairos.engine import ChunkIn, SessionEngine


RUN_ROOT = resolve_path("runs/bench/controller_v2_3")


def _score(spec: dict[str, Any], row: dict[str, Any], logistic: bool) -> float:
    value = float(spec["bias"]) + sum(float(spec["weights"][name]) * float(row["features"][name])
                                      for name in spec["weights"])
    return 1.0 / (1.0 + math.exp(-max(min(value, 40.0), -40.0))) if logistic else value


def _calibrate(value: float, calibration: dict[str, Any]) -> float:
    if calibration.get("method") != "platt":
        return value
    clipped = min(max(value, 1e-6), 1 - 1e-6)
    logit = math.log(clipped / (1 - clipped))
    z = calibration["slope"] * logit + calibration["intercept"]
    return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))


def _hard_guards(row: dict[str, Any]) -> bool:
    features = row["features"]
    return (
        features["domain_availability"] > 0.0
        and features["probe_evidence_relevance"] >= 0.16
        and features["constraint_readiness"] >= 0.75
        and features["constraint_revision_risk"] <= 0.70
    )


def _fit_intent_psrc(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    names = ("probe_section_rbo", "probe_clarity", "syntactic_completeness", "slot_stability", "continuation_safety")
    x = np.asarray([[row["features"][name] for name in names] for row in train_rows])
    y = np.asarray([row["beneficial_stop"] for row in train_rows])
    model = LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight="balanced").fit(x, y)
    return {"features": names, "weights": dict(zip(names, model.coef_[0].tolist())), "bias": float(model.intercept_[0])}


def _intent_psrc_probability(spec: dict[str, Any], row: dict[str, Any]) -> float:
    z = spec["bias"] + sum(spec["weights"][name] * row["features"][name] for name in spec["features"])
    return 1.0 / (1.0 + math.exp(-max(min(z, 40), -40)))


def _historical() -> dict[str, Any]:
    return json.loads(resolve_path("runs/bench/controller_v2_2/latest.json").read_text(encoding="utf-8"))


def _prefix_deltas(prefixes: list[str]) -> list[str]:
    output, previous = [], ""
    for prefix in prefixes:
        output.append(prefix[len(previous):].strip() if previous and prefix.lower().startswith(previous.lower()) else prefix)
        previous = prefix
    return output


def _live_baseline_selected(cfg: Config, streams: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    async def execute() -> set[tuple[str, str, int]]:
        contexts: dict[str, EngineContext] = {}
        selected = set()
        for stream in streams:
            profile = stream["corpus_profile"]
            if profile not in contexts:
                contexts[profile] = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
            engine = SessionEngine(contexts[profile], f"final-baseline-{stream['stream_id']}")
            engine.start()
            start = 0.0
            for index, (text, end) in enumerate(zip(_prefix_deltas(stream["prefixes"]), stream["prefix_timestamps"]), start=1):
                before = len(engine.events)
                await engine.on_chunk(ChunkIn(
                    utterance_id=stream.get("utterance_id", "u1"), text=text,
                    t0=start, t1=float(end), final=index == len(stream["prefixes"]),
                ))
                for event in engine.events[before:]:
                    if event.event_type == "RetrievalDecision":
                        for candidate in event.payload.get("candidates", []):
                            if candidate.get("action") == "RETRIEVE" and candidate.get("trigger") == "EARLY":
                                selected.add((stream["stream_id"], candidate["key"], index))
                start = float(end)
        return selected
    return asyncio.run(execute())


def _profile_evaluation(cfg: Config, artifact_dir: Path, model: StoppingModel,
                        development_rows: list[dict[str, Any]]) -> dict[str, Any]:
    external = extract_splits(cfg, ("external",), artifact_dir / "profiles")["external"]
    by_profile = {}
    by_profile["sample"] = evaluate_policy(
        development_rows,
        lambda row: _hard_guards(row) and model.predict(row["features"]).delta_q > 0
                    and model.predict(row["features"]).beneficial_probability >= model.decision_threshold,
        probability=lambda row: model.predict(row["features"]).beneficial_probability,
    )
    for profile in sorted({row["corpus_profile"] for row in external}):
        values = [row for row in external if row["corpus_profile"] == profile]
        by_profile[profile] = evaluate_policy(
            values,
            lambda row: _hard_guards(row) and model.predict(row["features"]).delta_q > 0
                        and model.predict(row["features"]).beneficial_probability >= model.decision_threshold,
            probability=lambda row: model.predict(row["features"]).beneficial_probability,
        )
    # Cross-domain smoke evaluation uses sample questions against the combined
    # profile; it is not folded into selection.
    combined_streams = []
    for stream in load_split("development")[:5]:
        copied = dict(stream)
        copied["corpus_profile"] = "multi_domain_demo"
        combined_streams.append(copied)
    context = EngineContext(cfg.with_overrides({"corpus.profile": "multi_domain_demo"}))
    combined_rows = label_optimal_stopping([
        row for stream in combined_streams for row in replay_feature_rows(context, stream)
    ])
    by_profile["multi_domain_demo"] = evaluate_policy(
        combined_rows,
        lambda row: _hard_guards(row) and model.predict(row["features"]).delta_q > 0
                    and model.predict(row["features"]).beneficial_probability >= model.decision_threshold,
        probability=lambda row: model.predict(row["features"]).beneficial_probability,
    )
    return by_profile


def run_v2_3(cfg: Config) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    modeling = train_models(cfg, run_dir / "modeling")
    model = StoppingModel.load(modeling["model_path"])
    extracted = extract_splits(cfg, ("train", "development"), run_dir / "selection_features")
    train_rows, development = extracted["train"], extracted["development"]
    historical = _historical()
    h0 = historical["variants"]["P0_original_psrc"]["metrics"]
    h6 = historical["variants"]["P6_evidence_reuse"]["metrics"]
    intent_psrc = _fit_intent_psrc(train_rows)
    classifier = model.artifact["models"]["beneficial_stop_classifier"]
    regressor = model.artifact["models"]["delta_q_regressor"]
    calibration = model.artifact["calibration"]

    default_weights = cfg["controller"]["default_weights"]
    def a1(row):
        f = row["features"]
        z = default_weights["bias"] + default_weights["rho"] * f["probe_section_rbo"] + default_weights["chi"] * f["probe_clarity"]
        z += default_weights["kappa"] * f["syntactic_completeness"] + default_weights["sigma"] * f["slot_stability"]
        z += default_weights["cont"] * f["continuation_safety"]
        probability = 1.0 / (1.0 + math.exp(-z))
        hidden = min(float(cfg["controller"]["pipeline_latency_prior_s"]), f["remaining_s"])
        theta = 1.0 - cfg["controller"]["lambda_latency"] * hidden / (cfg["controller"]["cost_retrieval"] + cfg["controller"]["cost_noise"])
        theta = min(max(theta, cfg["controller"]["theta_min"]), cfg["controller"]["theta_max"])
        return probability >= theta and f["syntactic_completeness"] >= cfg["controller"]["kappa_min"] and (
            f["probe_section_rbo"] >= cfg["controller"]["rho_min"] or f["remaining_s"] <= cfg["controller"]["pipeline_latency_prior_s"]
        )

    raw_probability = lambda row: _score(classifier, row, True)
    delta = lambda row: _score(regressor, row, False)
    calibrated = lambda row: _calibrate(raw_probability(row), calibration)
    policies: dict[str, tuple[Callable, Callable | None, str]] = {
        "A1_psrc_shared_features": (a1, None, "historical PSRC weights over corrected shared online features"),
        "A2_psrc_intent_labels": (lambda row: _hard_guards(row) and _intent_psrc_probability(intent_psrc, row) >= 0.5,
                                  lambda row: _intent_psrc_probability(intent_psrc, row), "five-feature PSRC retrained on intent-conditioned action labels"),
        "A3_iccos_classifier": (lambda row: _hard_guards(row) and raw_probability(row) >= 0.5,
                                raw_probability, "uncalibrated beneficial-stop classifier"),
        "A4_iccos_value_regression": (lambda row: _hard_guards(row) and delta(row) > 0.0,
                                      None, "direct positive DeltaQ regression"),
        "A5_iccos_calibrated": (lambda row: _hard_guards(row) and delta(row) > 0.0 and calibrated(row) >= 0.5,
                                calibrated, "calibrated classifier plus positive DeltaQ"),
        "A6_iccos_risk_selected": (lambda row: _hard_guards(row) and delta(row) > 0.0 and calibrated(row) >= model.decision_threshold,
                                   calibrated, "development-selected empirical-risk threshold"),
        "A7_iccos_cache": (lambda row: _hard_guards(row) and delta(row) > 0.0 and calibrated(row) >= model.decision_threshold,
                           calibrated, "A6 policy with cache treated as an independent execution optimization"),
        "A8_full_integration": (lambda row: _hard_guards(row) and delta(row) > 0.0 and calibrated(row) >= model.decision_threshold,
                                calibrated, "historically named A8; Stage-A offline guarded policy predicate, not full-engine integration"),
        "fixed_4": (lambda row: row["features"]["prefix_length"] * 16 >= 4, None, "retrieve at four content terms"),
        "fixed_8": (lambda row: row["features"]["prefix_length"] * 16 >= 8, None, "retrieve at eight content terms"),
        "end_of_utterance": (lambda row: False, None, "no non-final retrieval"),
    }
    variants = {
        name: {"description": description, "evaluation_stage": "A_OFFLINE_POLICY_SIMULATION",
               "metrics": evaluate_policy(development, decision, probability=probability)}
        for name, (decision, probability, description) in policies.items()
    }
    variants["A0_historical_psrc"] = {
        "description": "unaltered V2.2 historical metric record; denominator and aliases are not rewritten",
        "historical_run_id": historical["run_id"],
        "metrics": h0,
        "metric_status": "HISTORICAL_DEFINITION",
    }
    variants["v2_2_cold_start"] = {
        "description": "unaltered V2.2 P3 cold-start comparison",
        "historical_run_id": historical["run_id"],
        "metrics": historical["variants"]["P3_constraint_cold_start"]["metrics"],
        "metric_status": "HISTORICAL_DEFINITION",
    }

    cache_validation = {
        "source_run_id": historical["run_id"],
        "policy_metrics_equal": all(h0[key] == h6[key] for key in (
            "early_retrieval_rate", "useful_early_retrieval_rate", "premature_retrieval_rate",
            "evidence_survival_rate", "grounded_claim_rate", "citation_precision", "citation_recall",
        )),
        "baseline_full_searches": h0["full_searches"],
        "cache_full_searches": h6["full_searches"],
        "searches_avoided": h0["full_searches"] - h6["full_searches"],
        "promotion_independent_of_policy_gate": True,
        "selected": h6["full_searches"] < h0["full_searches"],
    }
    a1_metrics = variants["A1_psrc_shared_features"]["metrics"]
    a8_metrics = variants["A8_full_integration"]["metrics"]
    promotes = bool(
        a8_metrics["strict_safe_early"]["numerator"] > a1_metrics["strict_safe_early"]["numerator"]
        and (a8_metrics["premature"]["rate"] or 0.0) <= 0.20
        and a8_metrics["strict_safe_early"]["numerator"] > 0
    )
    selection = {
        "frozen_before_final_evaluation": True,
        "selected_policy": "A8_full_integration" if promotes else "A0_historical_psrc",
        "runtime_default": "iccos enabled" if promotes else "baseline PSRC; IC-COS disabled",
        "policy_promoted": promotes,
        "reasons": [
            "A8 improves strict-safe useful early retrieval under the corrected denominator" if promotes else
            "A8 did not demonstrate a non-zero strict-safe improvement over the corrected PSRC comparison",
            "The new final set lacks independent authorship and cannot alone justify promotion",
        ],
        "cache_selection": cache_validation,
    }
    (run_dir / "selection_freeze.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # Only after writing the frozen selection record do we read/evaluate final.
    final_rows = extract_splits(cfg, ("final",), run_dir / "post_selection")["final"]
    historical_rows = extract_splits(cfg, ("historical_frozen",), run_dir / "post_selection")["historical_frozen"]
    if promotes:
        chosen = policies["A8_full_integration"]
        final_decision, final_probability = chosen[0], chosen[1]
        historical_decision, historical_probability = chosen[0], chosen[1]
        evaluated_policy = "A8_full_integration"
    else:
        final_streams = load_split("final")
        historical_streams = load_split("historical_frozen")
        final_keys = _live_baseline_selected(cfg, final_streams)
        historical_keys = _live_baseline_selected(cfg, historical_streams)
        final_decision = lambda row: (row["stream_id"], row["candidate_id"], row["prefix_index"]) in final_keys
        historical_decision = lambda row: (row["stream_id"], row["candidate_id"], row["prefix_index"]) in historical_keys
        final_probability = historical_probability = None
        evaluated_policy = "A0_historical_psrc_live_session_engine"
    post_selection = {
        "evaluated_policy": evaluated_policy,
        "final": evaluate_policy(final_rows, final_decision, probability=final_probability),
        "historical_frozen_comparison": evaluate_policy(historical_rows, historical_decision, probability=historical_probability),
        "final_limitations": [
            "project-authored rather than independently authored",
            "same sample corpus as internal development",
            "synthetic written-question timing",
        ],
    }
    profile_evaluation = _profile_evaluation(cfg, run_dir, model, development)
    branch_candidates = [row for row in development if row["targets"].get("strict_useful") == 1][:3]
    stage_b = [validate_real_branches(cfg, next(stream for stream in load_split("development")
                                                 if stream["stream_id"] == row["stream_id"]), row)
               for row in branch_candidates]
    anomaly = audit_v22_empty_final_evidence(cfg, run_dir / "v2_2_empty_final_evidence_audit.json")

    report = {
        "status": "MEASURED",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "random_seed": 42},
        "modeling": modeling,
        "variants": variants,
        "selection": selection,
        "post_selection": post_selection,
        "profile_evaluation": profile_evaluation,
        "counterfactual_stage_b": stage_b,
        "experiment_stages": {
            "A": "offline evidence and sequential policy simulation",
            "B": f"real SessionEngine paired branches ({len(stage_b)} development cases)",
            "C": "not measured in V2.3",
            "correction": "The historical A8 name did not denote an end-to-end SessionEngine evaluation.",
        },
        "v2_2_anomaly_audit": {key: value for key, value in anomaly.items() if key != "rows"},
        "metric_reconciliation": {
            "historical_G2": {"value": h0["early_retrieval_rate"], "numerator": 11, "denominator": 16,
                              "status": "PRESERVED_HISTORICAL_UTTERANCE_LEVEL"},
            "oracle": "separate gold/constraint/useful/strict-safe earliest-prefix fields; final excluded",
            "intent_metric_alias": "V2.2 intent_level_useful_early_retrieval_rate was an alias; V2.3 uses actual candidate-lineage denominators",
            "latency_alias": "V2.2 first_useful_evidence_latency_ms was retrieval execution duration; V2.3 reports execution, elapsed-to-useful, and lead-before-end separately",
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
