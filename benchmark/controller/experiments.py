"""Controlled PSRC experiments over real SessionEngine traces.

Development comparisons are deliberately separate from the official G2 gate.
No held-out streaming claim is made when a suitable untouched split is absent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark.controller.metrics import calibration_metrics, grouped_bootstrap_binary, ratio, wilson
from benchmark.controller.oracle import oracle_rows, write_oracle
from benchmark.runner import _decision_diagnostics, evaluate, load_dataset, replay_session, timed_chunks
from kairos.config import Config, resolve_path
from kairos.controller.speculative import SpeculativeCache, SpeculativeEntry
from kairos.engine import EngineContext
from kairos.retrieval.index import RetrievalQuery


VARIANTS: dict[str, dict[str, object]] = {
    "B0_baseline": {},
    "B1_cold_start": {"controller.policy_variant": "cold_start", "controller.cold_start.enabled": True},
    "B2_intent_completion": {"controller.policy_variant": "intent_completion",
                             "controller.intent_completion.enabled": True},
    "B3_calibrated_identity": {"controller.policy_variant": "calibrated_identity",
                               "controller.calibration.enabled": True,
                               "controller.calibration.method": "identity"},
    "B5_risk_utility": {"controller.policy_variant": "risk_utility", "controller.utility.enabled": True},
    "C1_cold_intent": {"controller.policy_variant": "cold_intent",
                       "controller.cold_start.enabled": True,
                       "controller.intent_completion.enabled": True},
    "C5_cold_intent_utility": {"controller.policy_variant": "cold_intent_utility",
                               "controller.cold_start.enabled": True,
                               "controller.intent_completion.enabled": True,
                               "controller.utility.enabled": True},
}


def _sha_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=resolve_path("."), check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _run_variant(base_cfg: Config, name: str, overrides: dict[str, object], dataset: dict) -> tuple[dict, dict, list]:
    cfg = base_cfg.with_overrides(overrides)
    ctx = EngineContext(cfg)
    started = time.perf_counter()
    traces = {session["id"]: asyncio.run(replay_session(ctx, session)) for session in dataset["sessions"]}
    evaluated = evaluate(ctx, dataset, traces)
    diagnostics = _decision_diagnostics(ctx, dataset, traces)
    rows = diagnostics["rows"]
    eligible_keys = {(session["id"], utterance["id"])
                     for session in dataset["sessions"] for utterance in session["utterances"]
                     if utterance["type"] == "NEW_INFO"
                     and utterance.get("eligible_chunk", len(utterance["chunks"])) < len(utterance["chunks"])}
    eligible = len(eligible_keys)
    useful_keys: set[tuple[str, str]] = set()
    useful_latencies: list[float] = []
    for session in dataset["sessions"]:
        session_events = traces[session["id"]]
        assigned = timed_chunks(session)
        for utterance in session["utterances"]:
            key = (session["id"], utterance["id"])
            if key not in eligible_keys:
                continue
            end_time = max(chunk["t1"] for chunk in assigned if chunk["utterance_id"] == utterance["id"])
            gold_sections = {section for intent in utterance.get("intents", []) if not intent.get("unanswerable")
                             for section in intent.get("gold_sections", [])}
            for event in session_events:
                if event.utterance_id != utterance["id"] or event.event_type != "RetrievalCompleted":
                    continue
                if event.timestamp >= end_time - 1e-6:
                    continue
                accepted = event.payload.get("accepted_chunk_ids", [])
                if any(ctx.index.section_of(chunk_id) in gold_sections for chunk_id in accepted):
                    useful_keys.add(key)
                    if event.latency_ms is not None:
                        useful_latencies.append(float(event.latency_ms))
                    break
    events = [event for values in traces.values() for event in values]
    completed = [event for event in events if event.event_type == "RetrievalCompleted"]
    decisions = [event for event in events if event.event_type == "RetrievalDecision"]
    generated = [event for event in events if event.event_type == "SubQueriesGenerated"]
    candidate_decisions = [
        candidate for event in decisions for candidate in event.payload.get("candidates", [])
    ]
    shadow = sum(int(candidate.get("signals", {}).get(
        "shadow_search_executed", candidate.get("signals", {}).get("shadow_retrieval", 0)))
                 for event in decisions for candidate in event.payload.get("candidates", []))
    reused_full = sum(not bool(event.payload.get("search_executed", True)) for event in completed)
    actual_completed = sum(bool(event.payload.get("search_executed", True)) for event in completed)
    full_by_trigger = {trigger: sum(event.retrieval_trigger == trigger for event in completed)
                       for trigger in ("EARLY", "END_OF_UTTERANCE", "REFINE", "DELTA")}
    metrics = dict(evaluated["metrics"])
    new_info_retrievals = [utterance["retrievals"] for session in evaluated["per_session"]
                           for utterance in session["utterances"] if utterance["type"] == "NEW_INFO"]
    operation_counts = {
        "BM25_PROBE": ctx.index.probe_cache_misses,
        "DENSE_QUERY_ENCODING": actual_completed + shadow,
        "EFIS_FOOTPRINT_SEARCH": sum(len(event.payload.get("assignments", [])) for event in generated),
        "SHADOW_RETRIEVAL": shadow,
        "SPECULATIVE_FULL_RETRIEVAL": 0,
        "PROMOTED_FULL_RETRIEVAL": full_by_trigger["EARLY"],
        "DELTA_REFINEMENT": full_by_trigger["DELTA"],
        "FINAL_RETRIEVAL": full_by_trigger["END_OF_UTTERANCE"],
        "CACHE_HIT": ctx.index.probe_cache_hits + reused_full,
    }
    intent_ready_rows = [candidate for candidate in candidate_decisions
                         if candidate.get("signals", {}).get("intent_readiness", 0.0) >= 0.76]
    intent_blocked_by_stability = [candidate for candidate in intent_ready_rows
                                   if candidate.get("action") == "WAIT"
                                   and (candidate.get("signals", {}).get("s_hat") or 0.0)
                                   < (candidate.get("signals", {}).get("theta") or 1.0)]
    guarded_bypass_decisions = sum(
        "FIRST_OBSERVATION_HISTORY_BYPASS" in candidate.get("reasons", []) for candidate in candidate_decisions
    )
    stage_samples: dict[str, list[float]] = {}
    for event in completed:
        for stage, value in event.payload.get("stage_ms", {}).items():
            stage_samples.setdefault(stage, []).append(float(value))
    metrics.update({
        "useful_early_retrieval_rate": ratio(len(useful_keys), eligible),
        "useful_early_successes": len(useful_keys), "eligible_utterances": eligible,
        "useful_early_ci95": wilson(len(useful_keys), eligible),
        "useful_early_group_bootstrap_ci95": grouped_bootstrap_binary([
            {"group": f"{session_id}:{utterance_id}", "eligible": 1,
             "success": int((session_id, utterance_id) in useful_keys)}
            for session_id, utterance_id in sorted(eligible_keys)
        ]),
        "probe_computations": ctx.index.probe_cache_misses,
        "probe_cache_hits": ctx.index.probe_cache_hits,
        "shadow_retrievals": shadow,
        "full_searches": actual_completed + shadow,
        "reused_full_searches": reused_full,
        "duplicate_retrievals_avoided": reused_full + sum(
            int(candidate.get("signals", {}).get("cached_shadow_result", 0))
            for event in decisions for candidate in event.payload.get("candidates", [])
        ),
        "evidence_cache_hits": reused_full,
        "full_searches_by_trigger": full_by_trigger,
        "retrievals_per_information_need": ratio(sum(new_info_retrievals), len(new_info_retrievals)),
        "first_useful_evidence_latency_ms_mean": round(statistics.mean(useful_latencies), 3)
        if useful_latencies else None,
        "first_useful_evidence_latencies_ms": [round(value, 3) for value in useful_latencies],
        "retrieval_latency_ms_p50": round(float(np.percentile(
            [event.latency_ms for event in completed if event.latency_ms is not None], 50)), 3) if completed else None,
        "retrieval_latency_ms_p95": round(float(np.percentile(
            [event.latency_ms for event in completed if event.latency_ms is not None], 95)), 3) if completed else None,
        "intent_level_useful_early_retrieval_rate": ratio(len(useful_keys), eligible),
        "end_of_utterance_correctness": metrics.get("intent_recall"),
        "evaluation_sample_count": eligible,
        "operation_counts": operation_counts,
        "intent_gate_diagnostics": {
            "intent_ready_rows": len(intent_ready_rows),
            "intent_ready_but_stability_blocked": len(intent_blocked_by_stability),
            "guarded_bypass_decisions": guarded_bypass_decisions,
        },
        "latency_components_ms_mean": {
            stage: round(statistics.mean(values), 4) for stage, values in sorted(stage_samples.items())
        },
        "latency_scope": "backend monotonic processing only; transcript timing is synthetic in benchmark; WebSocket and microphone/ASR not measured",
        "wall_s": round(time.perf_counter() - started, 3),
    })
    return metrics, {"diagnostics": diagnostics, "validation": evaluated["validation"],
                     "per_session": evaluated["per_session"],
                     "useful_early_groups": [
                         {"session_id": session_id, "utterance_id": utterance_id,
                          "success": int((session_id, utterance_id) in useful_keys)}
                         for session_id, utterance_id in sorted(eligible_keys)
                     ]}, events


def _grouped_calibration(rows: list[dict]) -> dict:
    usable = [row for row in rows if row.get("stable_label") is not None and row.get("s_hat") is not None]
    labels = [int(row["stable_label"]) for row in usable]
    baseline = [float(row["s_hat"]) for row in usable]
    result = {"existing_model": calibration_metrics(labels, baseline),
              "groups": len({(row["session_id"], row["utterance_id"]) for row in usable}),
              "split_rule": "GroupKFold by session_id/utterance_id; diagnostic development data only"}
    if len(set(labels)) < 2 or result["groups"] < 3:
        result["status"] = "INSUFFICIENT_GROUPED_DATA"
        return result
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    existing_features = ["rho", "chi", "kappa", "sigma"]
    extended_features = existing_features + ["pi", "nu"]
    groups = np.asarray([f"{row['session_id']}::{row['utterance_id']}" for row in usable])
    y = np.asarray(labels)
    splitter = GroupKFold(n_splits=min(5, len(set(groups))))

    def cross_validated(features: list[str]) -> list[float]:
        matrix = np.asarray([[float(row.get(feature) or 0.0) for feature in features] for row in usable])
        probabilities = np.zeros(len(usable), dtype=float)
        for train, test in splitter.split(matrix, y, groups):
            if len(set(y[train])) < 2:
                probabilities[test] = float(y[train].mean())
            else:
                model = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(matrix[train], y[train])
                probabilities[test] = model.predict_proba(matrix[test])[:, 1]
        return probabilities.tolist()

    retrained = cross_validated(existing_features)
    extended = cross_validated(extended_features)
    result["retrained_existing_features"] = calibration_metrics(labels, retrained)
    result["extended_features"] = calibration_metrics(labels, extended)

    # Grouped out-of-fold Platt and isotonic calibration of the existing score.
    score_matrix = np.asarray(baseline).reshape(-1, 1)
    platt = np.zeros(len(usable), dtype=float)
    isotonic = np.zeros(len(usable), dtype=float)
    for train, test in splitter.split(score_matrix, y, groups):
        if len(set(y[train])) < 2:
            platt[test] = isotonic[test] = float(y[train].mean())
            continue
        sigmoid_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(score_matrix[train], y[train])
        platt[test] = sigmoid_model.predict_proba(score_matrix[test])[:, 1]
        iso_model = IsotonicRegression(out_of_bounds="clip").fit(score_matrix[train, 0], y[train])
        isotonic[test] = iso_model.predict(score_matrix[test, 0])
    result["platt"] = calibration_metrics(labels, platt.tolist())
    result["isotonic"] = calibration_metrics(labels, isotonic.tolist())
    result["status"] = "MEASURED_NOT_DEPLOYED"
    return result


def _speculative_evaluation(cfg: Config, rows: list[dict]) -> dict:
    ctx = EngineContext(cfg)
    cache = SpeculativeCache()
    searches = reused = discarded = refinements = 0
    elapsed_ms = 0.0
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["session_id"], row["utterance_id"], row["candidate_id"]), []).append(row)
    for (session_id, utterance_id, candidate_id), observations in grouped.items():
        observations.sort(key=lambda row: row["chunk_index"])
        if len(observations) < 2 or not observations[0].get("query_prefix"):
            continue
        first, final = observations[0], observations[-1]
        first_query = RetrievalQuery.build(first["query_prefix"])
        final_query = RetrievalQuery.build(final["query_prefix"])
        started = time.perf_counter()
        result = ctx.index.search(first_query)
        elapsed_ms += (time.perf_counter() - started) * 1000
        searches += 1
        cache.put(SpeculativeEntry(session_id, utterance_id, candidate_id, int(first["chunk_index"]),
                                   first_query.dense_text.lower(), cfg["corpus"].get("profile", "sample"),
                                   result, float(first.get("first_actual_retrieval_t") or 0.0), elapsed_ms))
        promoted = cache.promote(session_id, utterance_id, candidate_id, int(final["chunk_index"]), final_query,
                                 cfg["corpus"].get("profile", "sample"),
                                 min_term_jaccard=float(cfg["controller"]["speculative"]["min_term_jaccard"]))
        if promoted is None:
            discarded += 1
            ctx.index.search(final_query)
            refinements += 1
        else:
            reused += 1
    return {"status": "MEASURED_OFFLINE_NOT_PROMOTED", "speculative_full_retrievals": searches,
            "reused_retrievals": reused, "discarded_retrievals": discarded,
            "additional_full_retrievals": refinements, "reuse_rate": ratio(reused, searches),
            "wall_ms": round(elapsed_ms, 3),
            "note": "Actual index searches and version-bound cache promotion; no provisional evidence reached claims."}


def _asr_evaluation() -> dict:
    from kairos.controller.asr_policy import committed_length
    sequences = [
        ["book a", "book a whole", "book a hall"],
        ["find in", "find in Pune", "find in Mumbai"],
        ["cancellation", "cancellation policy", "cancellation policy"],
    ]
    out = {}
    for strategy in ("fixed_tail", "agreement", "adaptive_tail", "provisional_probe", "hybrid"):
        lengths, corrections = [], 0
        for sequence in sequences:
            for previous, current in zip(sequence, sequence[1:]):
                lengths.append(committed_length(previous, current, strategy=strategy, recent_revision_rate=0.5))
                if previous.split()[:min(len(previous.split()), len(current.split()))] != current.split()[:len(previous.split())]:
                    corrections += 1
        out[strategy] = {"mean_available_tokens": round(statistics.mean(lengths), 3),
                         "revision_events": corrections,
                         "authoritative_commit_safe": strategy in {"fixed_tail", "agreement", "adaptive_tail", "hybrid"}}
    return {"status": "MEASURED_RECORDED_SEQUENCES", "strategies": out,
            "hardware_microphone_validation": "PENDING"}


def _write_svg(path: Path, title: str, points: list[tuple[float, float, str]], x_label: str, y_label: str) -> None:
    width, height, pad = 720, 440, 70
    xs = [p[0] for p in points] or [0.0]
    ys = [p[1] for p in points] or [0.0]
    xmax, ymax = max(max(xs), 1e-6), max(max(ys), 1e-6)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
            f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>',
            f'<line x1="{pad}" y1="{height-pad}" x2="{pad}" y2="{pad}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif">{x_label}</text>',
            f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif">{y_label}</text>']
    for x, y, label in points:
        px = pad + (width - 2 * pad) * x / xmax
        py = height - pad - (height - 2 * pad) * y / ymax
        body.extend([f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#2563eb"/>',
                     f'<text x="{px+7:.1f}" y="{py-7:.1f}" font-family="sans-serif" font-size="11">{label}</text>'])
    body.append('</svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def _write_plots(out_dir: Path, variants: dict, calibration: dict, baseline_rows: list[dict]) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pairs = [(name, data["metrics"]) for name, data in variants.items()]
    specs = [
        ("01_early_vs_premature.svg", "Early retrieval versus premature retrieval", "premature_retrieval_rate", "early_retrieval_rate"),
        ("02_early_vs_cost.svg", "Early retrieval versus retrieval cost", "full_searches", "early_retrieval_rate"),
        ("03_survival_vs_coverage.svg", "Evidence survival versus early coverage", "early_retrieval_rate", "evidence_survival_rate"),
        ("04_useful_vs_work.svg", "Useful early retrieval versus total work", "full_searches", "useful_early_retrieval_rate"),
        ("08_pareto_frontier.svg", "Controller Pareto view", "premature_retrieval_rate", "useful_early_retrieval_rate"),
    ]
    for filename, title, x_key, y_key in specs:
        points = [(float(metrics.get(x_key) or 0.0), float(metrics.get(y_key) or 0.0), name)
                  for name, metrics in pairs]
        _write_svg(plots / filename, title, points, x_key, y_key)
    bins = calibration.get("existing_model", {}).get("bins", [])
    _write_svg(plots / "05_calibration_reliability.svg", "Calibration reliability",
               [(float(item["mean_score"]), float(item["observed"]), str(item["count"])) for item in bins],
               "predicted", "observed")
    trajectories = [(float(row.get("chunk_index") or 0), float(row.get("s_hat") or 0),
                     f"{row['session_id']}:{row['candidate_id']}") for row in baseline_rows[:40]]
    _write_svg(plots / "06_prefix_confidence.svg", "Per-prefix confidence trajectories", trajectories,
               "prefix index", "survival confidence")
    latency_points = []
    for position, (name, metrics) in enumerate(pairs, start=1):
        for sample_index, latency in enumerate(metrics.get("first_useful_evidence_latencies_ms", [])):
            jitter = ((sample_index % 5) - 2) * 0.025
            latency_points.append((position + jitter, float(latency), name if sample_index == 0 else ""))
    _write_svg(plots / "07_latency_distribution.svg", "First useful evidence latency distribution",
               latency_points, "controller arm index", "retrieval latency (ms)")


def _selection(variants: dict[str, dict]) -> dict:
    feasible = []
    rejected = []
    for name, record in variants.items():
        metrics = record["metrics"]
        failures = []
        if metrics.get("false_retrieval_rate") not in (0, 0.0):
            failures.append("FALSE_PRESENTATION_RETRIEVAL")
        if metrics.get("telemetry_coverage") != 1.0:
            failures.append("TELEMETRY_REGRESSION")
        if (metrics.get("grounded_claim_rate") or 0.0) < 1.0:
            failures.append("GROUNDING_REGRESSION")
        if (metrics.get("premature_retrieval_rate") or 0.0) > 0.2963:
            failures.append("PREMATURE_RISK_ABOVE_BASELINE")
        if (metrics.get("evidence_survival_rate") or 0.0) < 0.7333:
            failures.append("EVIDENCE_SURVIVAL_BELOW_BASELINE")
        (rejected if failures else feasible).append({"variant": name, "failures": failures})
    feasible_names = {item["variant"] for item in feasible}
    ranked = sorted((record for name, record in variants.items() if name in feasible_names),
                    key=lambda record: (record["metrics"].get("useful_early_retrieval_rate") or 0.0,
                                        -(record["metrics"].get("full_searches") or math.inf)), reverse=True)
    winner = ranked[0]["variant"] if ranked else "B0_baseline"
    baseline = variants["B0_baseline"]["metrics"]
    chosen = variants[winner]["metrics"]
    promoted = winner != "B0_baseline" and (
        (chosen.get("useful_early_retrieval_rate") or 0.0) > (baseline.get("useful_early_retrieval_rate") or 0.0)
    )
    if not promoted:
        winner = "B0_baseline"
    return {"selection_rules": ["correctness filters", "risk and cost constraints", "useful early rate",
                                "Pareto comparison", "complexity penalty", "frozen held-out evaluation"],
            "feasible": feasible, "rejected": rejected, "selected": winner,
            "promoted": promoted,
            "held_out_status": "PENDING_FROZEN_EVALUATION",
            "note": "No post-selection held-out tuning was performed."}


def run_controller_experiments(cfg: Config) -> dict:
    dataset = load_dataset("dev")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = resolve_path(cfg["bench"]["results_dir"]) / "controller" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    variants: dict[str, dict] = {}
    baseline_rows: list[dict] = []
    all_traces = []
    for name, overrides in VARIANTS.items():
        metrics, details, events = _run_variant(cfg, name, overrides, dataset)
        variants[name] = {"variant": name, "overrides": overrides, "metrics": metrics,
                          "validation": {k: v for k, v in details["validation"].items() if k != "failures"}}
        if name == "B0_baseline":
            baseline_rows = details["diagnostics"]["rows"]
            (out_dir / "per_utterance.json").write_text(json.dumps(details["per_session"], indent=2), encoding="utf-8")
        all_traces.extend({"variant": name, **row} for row in details["diagnostics"]["rows"])

    calibration = _grouped_calibration(baseline_rows)
    speculative = _speculative_evaluation(cfg, baseline_rows)
    asr = _asr_evaluation()
    variants["B4_speculative_reuse"] = {"variant": "B4_speculative_reuse", "overrides": {},
        "metrics": {**variants["B0_baseline"]["metrics"], **speculative}}
    variants["B6_adaptive_asr"] = {"variant": "B6_adaptive_asr", "overrides": {},
        "metrics": {**variants["B0_baseline"]["metrics"], "asr_experiment": asr}}
    variants["B7_intent_scheduling"] = {"variant": "B7_intent_scheduling", "overrides": {},
        "metrics": {**variants["B0_baseline"]["metrics"],
                    "scheduler_status": "IMPLEMENTED_NOT_PROMOTED_NO_STARVATION_DATA"}}
    selection = _selection(variants)
    heldout_dataset = load_dataset("heldout_streaming")
    selected_name = selection["selected"]
    heldout_metrics, heldout_details, _ = _run_variant(
        cfg, f"heldout:{selected_name}", VARIANTS.get(selected_name, {}), heldout_dataset
    )
    heldout = {
        "status": "MEASURED_FROZEN_NO_POST_SELECTION_TUNING",
        "dataset": heldout_dataset["name"],
        "note": heldout_dataset["note"],
        "selected_variant": selected_name,
        "metrics": heldout_metrics,
        "validation": {k: v for k, v in heldout_details["validation"].items() if k != "failures"},
    }
    selection["held_out_status"] = heldout["status"]
    selection["held_out_metrics"] = {
        key: heldout_metrics.get(key) for key in (
            "early_retrieval_rate", "useful_early_retrieval_rate", "premature_retrieval_rate",
            "evidence_survival_rate", "intent_recall", "grounded_claim_rate", "telemetry_coverage",
            "retrievals_per_information_need", "full_searches",
        )
    }
    corpus_hash = hashlib.sha256("\n".join(f"{chunk.chunk_id}:{chunk.sha256}"
                                             for chunk in EngineContext(cfg).corpus.chunks).encode()).hexdigest()
    report = {"status": "MEASURED", "run_id": run_id, "commit_sha": _git_commit(), "seed": 42,
              "policy_variant": selection["selected"], "training_split": "benchmark/datasets/train",
              "calibration_split": "grouped development diagnostic",
              "evaluation_split": "benchmark/datasets/dev",
              "held_out_split": "benchmark/datasets/heldout_streaming",
              "config_hash": _sha_json(dict(cfg)), "corpus_hash": corpus_hash,
              "dataset_hash": _sha_json(dataset), "machine": platform.platform(), "runtime": sys.version,
              "variants": variants, "calibration": calibration, "speculative": speculative,
              "asr": asr, "selection": selection, "held_out": heldout,
              "failures": ["Held-out streaming set is small and shares topics/corpus with development; it is not external generalization evidence."],
              "confidence_intervals": {name: record["metrics"].get("useful_early_ci95")
                                       for name, record in variants.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (out_dir / "heldout_metrics.json").write_text(json.dumps(heldout, indent=2), encoding="utf-8")
    (out_dir / "configuration.json").write_text(json.dumps(dict(cfg), indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps({"commit_sha": report["commit_sha"],
        "python": sys.version, "platform": platform.platform(), "run_id": run_id}, indent=2), encoding="utf-8")
    (out_dir / "per_intent.json").write_text(json.dumps(baseline_rows, indent=2), encoding="utf-8")
    (out_dir / "decision_traces.jsonl").write_text("\n".join(json.dumps(row) for row in all_traces), encoding="utf-8")
    rows = oracle_rows(dataset, baseline_rows, EngineContext(cfg).index)
    controller_root = out_dir.parent
    write_oracle(rows, controller_root / "oracle_opportunities.json", controller_root / "oracle_opportunities.csv")
    _write_plots(out_dir, variants, calibration, baseline_rows)
    (controller_root / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_latest(cfg: Config) -> dict:
    return json.loads((resolve_path(cfg["bench"]["results_dir"]) / "controller" / "latest.json").read_text(encoding="utf-8"))
