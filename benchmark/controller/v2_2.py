"""End-to-end PSRC V2.2 research runner with frozen selection and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.controller.dataset_v2_2 import ROOT as DATA_ROOT
from benchmark.controller.dataset_v2_2 import build_dataset, validate_dataset
from benchmark.controller.experiments import _run_variant, _write_svg
from benchmark.controller.modeling_v2_2 import MODEL_ROOT, extract_split, train_models
from benchmark.controller.oracle import oracle_rows, write_oracle
from benchmark.runner import load_dataset
from kairos.config import Config, resolve_path
from kairos.engine import EngineContext

RUN_ROOT = resolve_path("runs/bench/controller_v2_2")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=resolve_path("."), check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _preserve_baseline(out_dir: Path, cfg: Config) -> dict:
    target = out_dir / "baseline"
    target.mkdir(parents=True, exist_ok=False)
    copies = {
        "historical_stability.json": resolve_path("models/stability.json"),
        "prior_controller_latest.json": resolve_path("runs/bench/controller/latest.json"),
        "default_configuration.yaml": resolve_path("configs/default.yaml"),
    }
    hashes = {}
    for name, source in copies.items():
        if source.exists():
            shutil.copy2(source, target / name)
            hashes[name] = _sha_file(target / name)
    status = _git("status", "--short")
    (target / "working_tree_status.txt").write_text(status + "\n", encoding="utf-8")
    environment = {
        "commit_sha": _git("rev-parse", "HEAD"), "git_status": status.splitlines(),
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor() or "not-reported",
        "model_and_artifact_hashes": hashes,
        "note": "Immutable run-local copies; prior runs outside this directory were not modified.",
    }
    (target / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return environment


def _variants(modeling: dict) -> dict[str, dict[str, object]]:
    m1 = modeling["models"]["M1_retrained_historical"]["path"]
    m4 = modeling["models"]["M4_calibrated_logistic"]["path"]
    cold = modeling["cold_start_calibration"]["configuration"]
    calibrated_cold = {
        "controller.cold_start.calibration.enabled": cold["enabled"],
        "controller.cold_start.calibration.method": cold["method"],
        "controller.cold_start.calibration.slope": cold["slope"],
        "controller.cold_start.calibration.intercept": cold["intercept"],
    }
    return {
        "P0_original_psrc": {"controller.policy_variant": "P0_original_psrc"},
        "P1_retrained_stability": {"controller.policy_variant": "P1_retrained_stability", "controller.stability_model": m1},
        "P2_calibrated_stability": {"controller.policy_variant": "P2_calibrated_stability", "controller.stability_model": m4},
        "P3_constraint_cold_start": {"controller.policy_variant": "P3_constraint_cold_start",
            "controller.cold_start.enabled": True, "controller.cold_start.variant": "CS1_constraint_aware"},
        "P4_calibrated_cold_start": {"controller.policy_variant": "P4_calibrated_cold_start",
            "controller.cold_start.enabled": True, "controller.cold_start.variant": "CS2_calibrated", **calibrated_cold},
        "P5_guarded_intent_bypass": {"controller.policy_variant": "P5_guarded_intent_bypass",
            "controller.intent_completion.enabled": True,
            "controller.intent_completion.guarded_first_observation_bypass": True},
        "P6_evidence_reuse": {"controller.policy_variant": "P6_evidence_reuse", "controller.evidence_reuse.enabled": True},
        "P7_calibrated_reuse": {"controller.policy_variant": "P7_calibrated_reuse",
            "controller.stability_model": m4, "controller.evidence_reuse.enabled": True},
        "P8_calibrated_cold_reuse": {"controller.policy_variant": "P8_calibrated_cold_reuse",
            "controller.stability_model": m4, "controller.cold_start.enabled": True,
            "controller.cold_start.variant": "CS4_calibrated_reuse", "controller.evidence_reuse.enabled": True,
            **calibrated_cold},
        "P9_calibrated_intent": {"controller.policy_variant": "P9_calibrated_intent",
            "controller.stability_model": m4, "controller.intent_completion.enabled": True,
            "controller.intent_completion.guarded_first_observation_bypass": True},
        "P10_cold_intent_reuse": {"controller.policy_variant": "P10_cold_intent_reuse",
            "controller.stability_model": m4, "controller.cold_start.enabled": True,
            "controller.intent_completion.enabled": True,
            "controller.intent_completion.guarded_first_observation_bypass": True,
            "controller.evidence_reuse.enabled": True, **calibrated_cold},
        "P11_risk_utility": {"controller.policy_variant": "P11_risk_utility",
            "controller.stability_model": m4, "controller.evidence_reuse.enabled": True,
            "controller.utility.enabled": True, "controller.utility.max_predicted_risk": 0.45,
            "controller.utility.extra_retrieval_cost": 0.03},
        "P12_measured_latency": {"controller.policy_variant": "P12_measured_latency",
            "controller.stability_model": m4, "controller.evidence_reuse.enabled": True,
            "controller.adapt_latency": True, "controller.pipeline_latency_prior_s": 0.02},
    }


def _cold_start_variants(modeling: dict) -> dict[str, dict[str, object]]:
    cold = modeling["cold_start_calibration"]["configuration"]
    calibrated = {
        "controller.cold_start.calibration.enabled": cold["enabled"],
        "controller.cold_start.calibration.method": cold["method"],
        "controller.cold_start.calibration.slope": cold["slope"],
        "controller.cold_start.calibration.intercept": cold["intercept"],
    }
    return {
        "CS0_original_heuristic": {"controller.policy_variant": "CS0_original_heuristic",
                                   "controller.cold_start.enabled": True,
                                   "controller.cold_start.variant": "CS0_original_heuristic"},
        "CS1_constraint_aware": {"controller.policy_variant": "CS1_constraint_aware",
                                 "controller.cold_start.enabled": True,
                                 "controller.cold_start.variant": "CS1_constraint_aware"},
        "CS2_calibrated": {"controller.policy_variant": "CS2_calibrated",
                           "controller.cold_start.enabled": True,
                           "controller.cold_start.variant": "CS2_calibrated", **calibrated},
        "CS3_constraint_reuse": {"controller.policy_variant": "CS3_constraint_reuse",
                                 "controller.cold_start.enabled": True,
                                 "controller.cold_start.variant": "CS3_constraint_reuse",
                                 "controller.evidence_reuse.enabled": True},
        "CS4_calibrated_reuse": {"controller.policy_variant": "CS4_calibrated_reuse",
                                 "controller.cold_start.enabled": True,
                                 "controller.cold_start.variant": "CS4_calibrated_reuse",
                                 "controller.evidence_reuse.enabled": True, **calibrated},
    }


def _pareto(variants: dict[str, dict]) -> list[str]:
    frontier = []
    for name, record in variants.items():
        metric = record["metrics"]
        point = (metric.get("useful_early_retrieval_rate") or 0.0,
                 -(metric.get("premature_retrieval_rate") or 0.0),
                 metric.get("evidence_survival_rate") or 0.0,
                 -(metric.get("full_searches") or math.inf))
        dominated = False
        for other_name, other_record in variants.items():
            if other_name == name:
                continue
            other = other_record["metrics"]
            comparison = (other.get("useful_early_retrieval_rate") or 0.0,
                          -(other.get("premature_retrieval_rate") or 0.0),
                          other.get("evidence_survival_rate") or 0.0,
                          -(other.get("full_searches") or math.inf))
            if all(left >= right for left, right in zip(comparison, point)) and any(
                left > right for left, right in zip(comparison, point)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


def _select(variants: dict[str, dict]) -> dict:
    baseline = variants["P0_original_psrc"]["metrics"]
    feasible, rejected = [], []
    for name, record in variants.items():
        metrics = record["metrics"]
        failures = []
        if metrics.get("false_retrieval_rate") not in (0, 0.0):
            failures.append("FALSE_PRESENTATION_RETRIEVAL")
        if metrics.get("telemetry_coverage") != 1.0:
            failures.append("TELEMETRY_REGRESSION")
        if (metrics.get("grounded_claim_rate") or 0.0) < (baseline.get("grounded_claim_rate") or 0.0):
            failures.append("GROUNDING_REGRESSION")
        if (metrics.get("premature_retrieval_rate") or 0.0) > (baseline.get("premature_retrieval_rate") or 0.0):
            failures.append("PREMATURE_ABOVE_BASELINE")
        if (metrics.get("evidence_survival_rate") or 0.0) < (baseline.get("evidence_survival_rate") or 0.0):
            failures.append("SURVIVAL_BELOW_BASELINE")
        if (metrics.get("full_searches") or math.inf) > (baseline.get("full_searches") or math.inf):
            failures.append("FULL_SEARCH_COST_ABOVE_BASELINE")
        if name != "P0_original_psrc" and (metrics.get("early_retrieval_rate") or 0.0) < 0.80:
            failures.append("G2_BELOW_0_80")
        item = {"variant": name, "failures": failures}
        (rejected if failures else feasible).append(item)
    feasible_names = {item["variant"] for item in feasible}
    candidates = sorted(
        (name for name in feasible_names),
        key=lambda name: (
            variants[name]["metrics"].get("useful_early_retrieval_rate") or 0.0,
            -(variants[name]["metrics"].get("full_searches") or math.inf),
            -len(variants[name]["overrides"]),
        ), reverse=True,
    )
    candidate = candidates[0] if candidates else "P0_original_psrc"
    improved = (variants[candidate]["metrics"].get("useful_early_retrieval_rate") or 0.0) > (
        baseline.get("useful_early_retrieval_rate") or 0.0
    )
    selected = candidate if candidate != "P0_original_psrc" and improved else "P0_original_psrc"
    return {
        "selected": selected, "promoted": selected != "P0_original_psrc", "feasible": feasible,
        "rejected": rejected, "pareto_frontier": _pareto(variants),
        "rules": ["correctness", "G2 >= 0.80 for promotion", "premature <= baseline",
                  "survival >= baseline", "full searches <= baseline", "maximize gold-useful early", "simplicity"],
        "held_out_status": "PENDING_FROZEN_EVALUATION",
        "note": "The untouched held-out split is evaluated only after this development selection is frozen.",
    }


def _plots(out_dir: Path, variants: dict, modeling: dict, baseline_rows: list[dict], heldout: dict) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pairs = [(name, item["metrics"]) for name, item in variants.items()]
    def points(x: str, y: str):
        return [(float(metric.get(x) or 0.0), float(metric.get(y) or 0.0), name) for name, metric in pairs]
    m4 = modeling["models"]["M4_calibrated_logistic"]["metrics"]
    _write_svg(plots / "01_reliability.svg", "V2.2 survival reliability (development)",
               [(float(item["mean_score"]), float(item["observed"]), f"n={item['count']}") for item in m4.get("bins", [])],
               "predicted probability", "observed survival")
    model_points = [(index + 1.0, float(item.get("metrics", {}).get("brier_score") or 0.0), name)
                    for index, (name, item) in enumerate(modeling["models"].items()) if item.get("metrics")]
    _write_svg(plots / "02_brier_comparison.svg", "Brier comparison (development)", model_points,
               "model index", "Brier score")
    _write_svg(plots / "03_early_vs_premature.svg", "Early versus premature", points("premature_retrieval_rate", "early_retrieval_rate"),
               "premature rate", "G2 early rate")
    _write_svg(plots / "04_useful_vs_cost.svg", "Useful early versus full-search cost", points("full_searches", "useful_early_retrieval_rate"),
               "actual full searches", "gold-useful early rate")
    _write_svg(plots / "05_survival_vs_coverage.svg", "Evidence survival versus useful coverage", points("useful_early_retrieval_rate", "evidence_survival_rate"),
               "gold-useful early rate", "evidence survival")
    cold = [(float(row["features"].get("cold_start_confidence", 0.0)), float(row["survival_label"]), row["stream_id"])
            for row in baseline_rows if row["features"].get("first_observation") == 1.0]
    _write_svg(plots / "06_cold_start_distribution.svg", "Cold-start confidence and outcome", cold[:80],
               "heuristic confidence", "survival label")
    trajectories = [(float(row["prefix_index"]), float(row["survival_score"]), row["stream_id"])
                    for row in baseline_rows[:80]]
    _write_svg(plots / "07_survival_trajectories.svg", "Per-prefix measured survival", trajectories,
               "prefix index", "measured survival")
    _write_svg(plots / "08_intent_vs_decisions.svg", "Intent-ready arms", [
        (float(metric.get("useful_early_retrieval_rate") or 0.0), float(metric.get("early_retrieval_rate") or 0.0), name)
        for name, metric in pairs if "intent" in name.lower()], "useful early", "all early")
    _write_svg(plots / "09_search_cost_breakdown.svg", "Shadow and total full searches", [
        (float(metric.get("shadow_retrievals") or 0.0), float(metric.get("full_searches") or 0.0), name)
        for name, metric in pairs], "shadow searches", "actual full searches")
    _write_svg(plots / "10_cache_outcomes.svg", "Validated cache outcomes", [
        (float(metric.get("reused_full_searches") or 0.0), float(metric.get("duplicate_retrievals_avoided") or 0.0), name)
        for name, metric in pairs], "reused full results", "duplicate searches avoided")
    _write_svg(plots / "11_pareto_frontier.svg", "Policy Pareto view", points("premature_retrieval_rate", "useful_early_retrieval_rate"),
               "premature rate", "gold-useful early rate")
    selected_metrics = heldout["metrics"]
    _write_svg(plots / "12_heldout_comparison.svg", "Frozen held-out selected policy", [
        (float(selected_metrics.get("full_searches") or 0.0), float(selected_metrics.get("useful_early_retrieval_rate") or 0.0), heldout["selected_variant"])
    ], "actual full searches", "gold-useful early rate")


def run_v2_2(cfg: Config) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUN_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    baseline_environment = _preserve_baseline(out_dir, cfg)
    dataset_manifest = build_dataset()
    dataset_validation = validate_dataset()
    modeling = train_models(cfg, artifact_dir=out_dir / "modeling")
    arms = _variants(modeling)
    dataset = load_dataset("dev")
    variants: dict[str, dict] = {}
    traces: list[dict] = []
    per_utterance: dict[str, object] = {}
    for name, overrides in arms.items():
        metrics, details, events = _run_variant(cfg, name, overrides, dataset)
        variants[name] = {"variant": name, "overrides": overrides, "metrics": metrics,
                          "validation": {key: value for key, value in details["validation"].items() if key != "failures"}}
        per_utterance[name] = details["per_session"]
        traces.extend({"variant": name, **row} for row in details["diagnostics"]["rows"])
    cold_start_ablation = {}
    for name, overrides in _cold_start_variants(modeling).items():
        metrics, details, _ = _run_variant(cfg, name, overrides, dataset)
        cold_start_ablation[name] = {
            "overrides": overrides, "metrics": metrics,
            "calibration_supported": bool(modeling["cold_start_calibration"]["configuration"]["enabled"])
                              if "calibrated" in name else None,
        }
    selection = _select(variants)
    frozen_name = selection["selected"]
    heldout_dataset = load_dataset("heldout_streaming")
    heldout_metrics, heldout_details, _ = _run_variant(cfg, f"heldout:{frozen_name}", arms[frozen_name], heldout_dataset)
    heldout = {
        "status": "MEASURED_FROZEN_NO_POST_SELECTION_TUNING", "selected_variant": frozen_name,
        "dataset": heldout_dataset["name"], "metrics": heldout_metrics,
        "validation": {key: value for key, value in heldout_details["validation"].items() if key != "failures"},
        "limitations": heldout_dataset.get("note"),
    }
    selection["held_out_status"] = heldout["status"]
    selection["held_out_metrics"] = heldout_metrics
    dev_features = extract_split(cfg, "development", out_dir / "modeling")
    baseline_rows = [row for row in traces if row["variant"] == "P0_original_psrc"]
    oracle = oracle_rows(dataset, baseline_rows, EngineContext(cfg).index)
    write_oracle(oracle, out_dir / "oracle.json", out_dir / "oracle.csv")

    ctx = EngineContext(cfg)
    corpus_hash = hashlib.sha256("\n".join(f"{c.chunk_id}:{c.sha256}" for c in ctx.corpus.chunks).encode()).hexdigest()
    model_hashes = {path.name: _sha_file(path) for path in MODEL_ROOT.glob("*.json")}
    cost_breakdown = {name: {
        "actual_full_searches": item["metrics"].get("full_searches"),
        "shadow_searches": item["metrics"].get("shadow_retrievals"),
        "reused_full_searches": item["metrics"].get("reused_full_searches"),
        "probe_computations": item["metrics"].get("probe_computations"),
        "retrievals_per_need": item["metrics"].get("retrievals_per_information_need"),
    } for name, item in variants.items()}
    report = {
        "status": "MEASURED", "run_id": run_id, "commit_sha": _git("rev-parse", "HEAD"),
        "corpus_profile": cfg["corpus"]["profile"], "corpus_hash": corpus_hash,
        "dataset_hash": dataset_manifest["dataset_hash"], "split_hashes": dataset_manifest["split_hashes"],
        "model_hashes": model_hashes, "random_seed": 42,
        "environment": baseline_environment, "configuration": dict(cfg),
        "dataset_validation": dataset_validation, "modeling": modeling,
        "variants": variants, "cold_start_ablation": cold_start_ablation,
        "cost_breakdown": cost_breakdown, "selection": selection, "held_out": heldout,
        "claims_boundary": "Engineering evidence on small, mixed-source streams; not production-generalization evidence.",
        "hardware_microphone_validation": "PENDING",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "configuration.json").write_text(json.dumps(dict(cfg), indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(baseline_environment, indent=2), encoding="utf-8")
    (out_dir / "per_utterance_results.json").write_text(json.dumps(per_utterance, indent=2), encoding="utf-8")
    (out_dir / "per_intent_results.json").write_text(json.dumps(baseline_rows, indent=2), encoding="utf-8")
    (out_dir / "per_prefix_results.json").write_text(json.dumps(dev_features, indent=2), encoding="utf-8")
    (out_dir / "decision_traces.jsonl").write_text("\n".join(json.dumps(row) for row in traces), encoding="utf-8")
    (out_dir / "cost_breakdown.json").write_text(json.dumps(cost_breakdown, indent=2), encoding="utf-8")
    (out_dir / "calibration_results.json").write_text(json.dumps(modeling, indent=2), encoding="utf-8")
    (out_dir / "selection_result.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (out_dir / "heldout_results.json").write_text(json.dumps(heldout, indent=2), encoding="utf-8")
    _plots(out_dir, variants, modeling, dev_features, heldout)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_latest() -> dict:
    return json.loads((RUN_ROOT / "latest.json").read_text(encoding="utf-8"))
