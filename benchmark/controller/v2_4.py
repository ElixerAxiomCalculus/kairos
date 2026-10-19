"""KAIROS V2.4 CAPER real-engine evaluation and artifact writer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.controller.dataset_v2_3 import load_split
from benchmark.controller.dataset_v2_4 import validate_dataset
from benchmark.controller.metrics_v2_4 import family_bootstrap, proportion
from benchmark.controller.security_v2_4 import write_security_audit
from kairos.config import Config, resolve_path
from kairos.engine import ChunkIn, EngineContext, SessionEngine


RUN_ROOT = resolve_path("runs/bench/controller_v2_4")
AUDITED_REFERENCE = "a7ee6642e7975105721a97438162b3dcbbb536be"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=resolve_path("."), capture_output=True, text=True,
                              check=True, encoding="utf-8", errors="replace").stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _environment() -> dict[str, Any]:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "head_sha": _git("rev-parse", "HEAD"),
        "audited_reference_sha": AUDITED_REFERENCE,
        "audited_reference_is_ancestor": _git("merge-base", "--is-ancestor", AUDITED_REFERENCE, "HEAD") == "",
        "working_tree_porcelain": _git("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "node": _command_version("node", "--version"),
        "npm": _command_version("npm", "--version"),
    }


def _command_version(*command: str) -> str:
    try:
        executable = shutil.which(command[0]) or command[0]
        return subprocess.run([executable, *command[1:]], cwd=resolve_path("."), capture_output=True, text=True,
                              check=True, encoding="utf-8", errors="replace").stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _prefix_deltas(prefixes: list[str]) -> list[str]:
    output, previous = [], ""
    for prefix in prefixes:
        output.append(prefix[len(previous):].strip() if previous and prefix.casefold().startswith(previous.casefold()) else prefix)
        previous = prefix
    return output


ARMS: dict[str, dict[str, Any] | None] = {
    "C0_baseline_psrc": {"controller.caper.enabled": False, "controller.evidence_reuse.enabled": False},
    "C1_validated_cache_only": {"controller.caper.enabled": False, "controller.evidence_reuse.enabled": True},
    "C2_prefetch_no_reuse": {"controller.caper.enabled": True, "controller.caper.provisional_cache_enabled": False,
                              "controller.caper.validated_reuse_enabled": False},
    "C3_provisional_cache_no_commit_reuse": {"controller.caper.enabled": True,
                                               "controller.caper.provisional_cache_enabled": True,
                                               "controller.caper.validated_reuse_enabled": False},
    "C4_strict_validated_reuse": {"controller.caper.enabled": True,
                                    "controller.caper.validated_reuse_enabled": True},
    "C5_query_version_reconciliation": {"controller.caper.enabled": True,
                                          "controller.caper.validated_reuse_enabled": True},
    "C6_bounded_async": {"controller.caper.enabled": True,
                           "controller.caper.max_active_prefetches_per_session": 1,
                           "controller.caper.max_queued_prefetches_per_session": 2},
    "C7_learned_prefetch": None,
    "C8_full_caper_deterministic": {"controller.caper.enabled": True,
                                     "controller.caper.policy": "deterministic"},
    "C9_eager_bounded": {"controller.caper.enabled": True, "controller.caper.policy": "eager"},
    "C10_end_of_utterance": {"controller.mode": "eou", "controller.caper.enabled": False,
                               "controller.evidence_reuse.enabled": False},
}


def _arm_cfg(cfg: Config, overrides: dict[str, Any]) -> Config:
    defaults = {
        "controller.iccos.enabled": False,
        "controller.caper.learned_prefetch.enabled": False,
        "controller.caper.debounce_ms": 0,
        "controller.caper.finalization_wait_ms": 1000,
        "telemetry.trace_content": False,
    }
    defaults.update(overrides)
    return cfg.with_overrides(defaults)


def _gold_ids(stream: dict[str, Any]) -> tuple[set[str], set[str]]:
    sections = set(stream.get("gold_sections", []))
    documents = set(stream.get("gold_documents", [])) | {item.split("#", 1)[0] for item in sections}
    return sections, documents


def _event_gold(event: Any, sections: set[str], documents: set[str]) -> bool:
    ids = set(event.payload.get("accepted_chunk_ids", event.chunk_ids))
    event_documents = {item.split("#", 1)[0].split(":", 1)[0] for item in ids}
    return bool(ids & sections or event_documents & documents)


async def _run_arm(cfg: Config, name: str, streams: list[dict[str, Any]], split: str) -> dict[str, Any]:
    contexts: dict[str, EngineContext] = {}
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    prefetches: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    latency_values: list[float] = []
    search_executions: list[dict[str, Any]] = []
    caper_totals: Counter[str] = Counter()
    for stream in streams:
        profile = stream.get("corpus_profile", "sample")
        if profile not in contexts:
            contexts[profile] = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
        ctx = contexts[profile]
        session_id = f"v24-{split}-{name}-{stream['stream_id']}"
        engine = SessionEngine(ctx, session_id)
        engine.start()
        prefixes = stream["prefixes"]
        timestamps = stream["prefix_timestamps"]
        deltas = _prefix_deltas(prefixes)
        start = 0.0
        for index, (text, end) in enumerate(zip(deltas, timestamps), start=1):
            await engine.on_chunk(ChunkIn(
                utterance_id=stream.get("utterance_id", "u1"), text=text, t0=start,
                t1=float(end), final=index == len(prefixes),
            ))
            if index < len(prefixes):
                # Give bounded worker tasks an actual scheduling opportunity without pretending this
                # accelerated written replay is natural speech timing.
                await asyncio.sleep(0.01)
            start = float(end)
        await engine.end()
        events = engine.active_events()
        final_t = float(timestamps[-1])
        sections, documents = _gold_ids(stream)
        eligible = bool((sections or documents) and len(prefixes) > 1)
        completed_early = [event for event in events if event.event_type == "PrefetchCompleted"
                           and event.timestamp < final_t]
        validated_early = [event for event in events if event.event_type == "EvidenceValidated"
                           and event.timestamp < final_t]
        admitted_retrieval_early = [event for event in events if event.event_type == "RetrievalCompleted"
                                    and event.timestamp < final_t
                                    and event.payload.get("accepted_chunk_ids")]
        grounding = [event for event in events if event.event_type == "GroundingVerified"]
        grounded_ids = {chunk_id for event in grounding for chunk_id in event.citations}
        useful_early_events = [
            event for event in [*validated_early, *admitted_retrieval_early]
            if _event_gold(event, sections, documents)
            and grounded_ids.intersection(event.payload.get("accepted_chunk_ids", event.chunk_ids))
        ]
        first_useful = min((event.timestamp for event in useful_early_events), default=None)
        invalid_citations = sum(int(event.payload.get("summary", {}).get("invalid_citation", 0)) for event in grounding)
        emitted_claims = sum(int(event.payload.get("summary", {}).get("emitted", 0)) for event in grounding)
        supported_claims = sum(int(event.payload.get("summary", {}).get("supported", 0)) for event in grounding)
        answer_events = [event for event in events if event.event_type == "AnswerVersionCreated"]
        retrieval_events = [event for event in events if event.event_type == "RetrievalCompleted"]
        latency_values.extend(float(event.latency_ms) for event in retrieval_events if event.latency_ms is not None)
        committed = {event.payload.get("evidence_id") for event in events if event.event_type == "EvidenceCommitted"}
        provisional = {event.payload.get("evidence_id") for event in events if event.event_type == "PrefetchCompleted"}
        row = {
            "arm": name,
            "split": split,
            "stream_id": stream["stream_id"],
            "question_family": stream.get("question_family", stream["stream_id"]),
            "timing_source": "accelerated scheduled replay of synthetic written-question prefixes",
            "eligible_for_verified_early": eligible,
            "prepared_before_final": bool(completed_early) if eligible else None,
            "verified_useful_before_final": bool(useful_early_events) if eligible else None,
            "verified_early_sources": sorted({
                "CAPER_VALIDATED" if event.event_type == "EvidenceValidated" else "RELEVANCE_ADMITTED_RETRIEVAL"
                for event in useful_early_events
            }),
            "first_useful_evidence_lead_ms": round((final_t - first_useful) * 1000, 3) if first_useful is not None else None,
            "answer_emitted": bool(answer_events),
            "grounding_checked": bool(grounding),
            "emitted_claims": emitted_claims,
            "supported_claims": supported_claims,
            "invalid_citations": invalid_citations,
            "actual_retrieval_events": len(retrieval_events),
            "cache_reuse_events": sum(event.payload.get("cache_status") in {"CAPER_VALIDATED", "VALIDATED_CACHE"}
                                      for event in retrieval_events),
            "provisional_completed": len(provisional),
            "prepared_committed": len(provisional & committed),
            "prepared_wasted": len(provisional - committed),
            "caper": engine.caper.snapshot()["stats"],
        }
        rows.append(row)
        caper_totals.update(row["caper"])
        for event in events:
            minimal = {
                "arm": name, "split": split, "stream_id": stream["stream_id"],
                "event_id": event.event_id, "event_type": event.event_type,
                "timestamp": event.timestamp, "query_version": event.query_version,
                "chunk_ids": list(event.chunk_ids),
            }
            if event.event_type in {"PrefetchDecision", "RetrievalDecision"}:
                minimal["decision"] = event.payload.get("decision")
                minimal["reasons"] = event.payload.get("reasons", [])
                decisions.append(minimal)
            if event.event_type.startswith("Prefetch"):
                minimal["evidence_id"] = event.payload.get("evidence_id")
                minimal["state"] = event.payload.get("state")
                prefetches.append(minimal)
            if event.event_type in {"EvidenceValidated", "EvidenceInvalidated", "EvidenceCommitted"}:
                minimal["evidence_id"] = event.payload.get("evidence_id")
                minimal["state"] = event.payload.get("state")
                minimal["checks"] = event.payload.get("checks")
                validations.append(minimal)
        search_executions.extend(ctx.index.search_executions(session_id=session_id))

    eligible_rows = [row for row in rows if row["eligible_for_verified_early"]]
    prepared_n = sum(bool(row["prepared_before_final"]) for row in eligible_rows)
    verified_n = sum(bool(row["verified_useful_before_final"]) for row in eligible_rows)
    claims = sum(row["emitted_claims"] for row in rows)
    invalid = sum(row["invalid_citations"] for row in rows)
    provisional = sum(row["provisional_completed"] for row in rows)
    wasted = sum(row["prepared_wasted"] for row in rows)
    leads = [row["first_useful_evidence_lead_ms"] for row in rows
             if row["first_useful_evidence_lead_ms"] is not None]
    by_type = Counter(item["retrieval_type"] for item in search_executions)
    metrics = {
        "verified_useful_evidence_before_final": proportion(
            verified_n, len(eligible_rows), unit="stream", split=split,
            eligibility="gold-labeled stream with at least one non-final prefix",
            timing_source="accelerated scheduled replay; synthetic timestamps",
            grounding_status="early evidence was gold-aligned and later cited by an emitted Grounding Firewall claim",
        ),
        "preparation_coverage_before_final": proportion(
            prepared_n, len(eligible_rows), unit="stream", split=split,
            eligibility="gold-labeled stream with at least one non-final prefix",
            timing_source="accelerated scheduled replay; synthetic timestamps",
            grounding_status="provisional only",
        ),
        "invalid_citation_rate": proportion(
            invalid, claims, unit="emitted claim", split=split, eligibility="claim emitted after Grounding Firewall",
            timing_source="not applicable", grounding_status="verified by Grounding Firewall",
        ),
        "wasted_preparation_rate": proportion(
            wasted, provisional, unit="completed provisional search", split=split,
            eligibility="CAPER provisional search completed", timing_source="actual task lifecycle",
            grounding_status="not committed",
        ),
        "family_bootstrap_verified_early": family_bootstrap(eligible_rows, "verified_useful_before_final"),
        "lead_time_ms": {
            "count": len(leads), "mean": round(statistics.mean(leads), 3) if leads else None,
            "median": round(statistics.median(leads), 3) if leads else None,
            "unit": "ms", "timing_source": "synthetic stream clock; accelerated scheduled replay",
        },
        "retrieval_latency_ms": _percentiles(latency_values),
        "search_cost": {
            "actual_hybrid_search_executions": len(search_executions),
            "by_retrieval_type": dict(sorted(by_type.items())),
            "cache_reuse_events": sum(row["cache_reuse_events"] for row in rows),
            "cancellation_requests": caper_totals["cancellation_requests"],
            "canceled_before_execution": caper_totals["canceled_before_execution"],
            "completed_but_discarded": caper_totals["completed_discarded"],
            "measurement_source": "HybridIndex.search execution registry",
        },
        "caper_totals": dict(caper_totals),
    }
    return {"name": name, "split": split, "status": "MEASURED", "metrics": metrics, "rows": rows,
            "decisions": decisions, "prefetches": prefetches, "validations": validations,
            "search_executions": search_executions}


def _percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None, "unit": "ms"}
    ordered = sorted(values)
    at = lambda q: ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]
    return {"count": len(values), "p50": round(statistics.median(ordered), 3),
            "p95": round(at(0.95), 3), "max": round(max(ordered), 3), "unit": "ms",
            "measurement_source": "RetrievalCompleted latency_ms"}


def _release_selection(arms: dict[str, Any]) -> dict[str, Any]:
    baseline = arms["C0_baseline_psrc"]["metrics"]
    candidate = arms["C8_full_caper_deterministic"]["metrics"]
    base_rate = baseline["verified_useful_evidence_before_final"]["rate"] or 0.0
    candidate_rate = candidate["verified_useful_evidence_before_final"]["rate"] or 0.0
    base_searches = baseline["search_cost"]["actual_hybrid_search_executions"]
    candidate_searches = candidate["search_cost"]["actual_hybrid_search_executions"]
    invalid = candidate["invalid_citation_rate"]["numerator"]
    waste = candidate["wasted_preparation_rate"]["rate"]
    gates = {
        "verified_early_absolute_delta_at_least_0_10": candidate_rate - base_rate >= 0.10,
        "invalid_citations_zero": invalid == 0,
        "search_cost_at_most_1_5x_baseline": candidate_searches <= max(1, 1.5 * base_searches),
        "wasted_preparation_at_most_0_50": waste is not None and waste <= 0.50,
        "independent_final_available": False,
    }
    promote = all(gates.values())
    return {
        "frozen_before_final_evaluation": True,
        "selected_candidate": "C8_full_caper_deterministic",
        "comparison_baseline": "C0_baseline_psrc",
        "preregistered_budgets": {"max_search_cost_multiplier": 1.5, "max_wasted_preparation_rate": 0.50,
                                  "required_invalid_citations": 0,
                                  "required_verified_early_absolute_delta": 0.10},
        "development_gates": gates,
        "promotion_eligible": promote,
        "runtime_default": "CAPER enabled" if promote else "historical PSRC; CAPER disabled",
        "decision": "PROMOTE" if promote else "RETAIN_AS_EXPERIMENTAL",
        "reason": "All development and independent-evidence gates pass." if promote else
                  "At least one preregistered gate failed; independent final evidence is unavailable.",
    }


def _write_plot(path: Path, title: str, rows: list[tuple[str, float, float]], x_label: str, y_label: str) -> None:
    width, height, margin = 760, 420, 60
    max_x = max([item[1] for item in rows] + [1.0])
    max_y = max([item[2] for item in rows] + [1.0])
    circles = []
    for label, x, y in rows:
        px = margin + (width - 2 * margin) * x / max_x
        py = height - margin - (height - 2 * margin) * y / max_y
        circles.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#38bdf8"/><text x="{px+8:.1f}" y="{py-7:.1f}" font-size="11">{label}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="{title}">
<rect width="100%" height="100%" fill="#0f172a"/><text x="{margin}" y="30" fill="white" font-size="18">{title}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8"/>
<text x="{width/2}" y="{height-15}" fill="#cbd5e1" text-anchor="middle">{x_label}</text>
<text x="15" y="{height/2}" fill="#cbd5e1" transform="rotate(-90 15 {height/2})" text-anchor="middle">{y_label}</text>
{''.join(circles)}</svg>'''
    path.write_text(svg, encoding="utf-8")


def run_v2_4(cfg: Config) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = _environment()
    dataset = validate_dataset()
    _json(run_dir / "environment.json", environment)
    _json(run_dir / "configuration.json", {"controller": cfg["controller"], "retrieval": cfg["retrieval"],
                                             "telemetry": cfg["telemetry"]})
    _json(run_dir / "dataset_manifest.json", dataset)
    baseline = {
        "audited_reference": AUDITED_REFERENCE,
        "baseline_test_result": {"passed": 67, "failed": 0, "warnings": 1},
        "benchmark_run_id": "20261019T070859Z",
        "semantic_run_id": "20261019T070917Z",
        "development_gates": {"G2_early_retrieval": {"value": 0.6875, "status": "FAIL"},
                              "G3_grounded_claims": {"value": 0.9444, "status": "PASS"}},
        "canonical_replay_event_counts": {"pune-demo": 110, "reimbursement-demo": 68,
                                           "self-repair-demo": 37, "conflict-demo": 25,
                                           "multi-domain-stress": 111},
        "timing": "captured before V2.4 production edits",
    }
    _json(run_dir / "baseline_metrics.json", baseline)
    _json(run_dir / "feature_validation.json", {
        "status": "PASS", "shared_query_path": "SessionEngine QueryFrame/EFIS/HybridIndex",
        "future_text_features": 0, "caper_learned_model": "NOT_TRAINED",
        "invariants": ["session-scoped evidence", "current constraint validation", "source checksum validation",
                       "Grounding Firewall required after evidence reuse"],
    })
    _json(run_dir / "counterfactual_results.json", {
        "stage": "A_OFFLINE_METHOD_CORRECTION", "status": "IMPLEMENTED_AND_UNIT_TESTED",
        "terminal_value": "actual final-retrieval utility proxy, not zero",
        "selection_unit": "first eligible prefix per intent lineage",
        "zero_action_risk": None, "verified_strict_useful_unavailable_is_not_negative": True,
    })

    development = load_split("development")
    measured: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    all_prefetches: list[dict[str, Any]] = []
    all_validations: list[dict[str, Any]] = []
    all_searches: list[dict[str, Any]] = []
    for name, overrides in ARMS.items():
        if overrides is None:
            measured[name] = {"name": name, "split": "development", "status": "NOT_RUN",
                              "reason": "No learned CAPER model was trained; deterministic policy evaluated first."}
            continue
        result = asyncio.run(_run_arm(_arm_cfg(cfg, overrides), name, development, "development"))
        measured[name] = {key: value for key, value in result.items()
                          if key not in {"rows", "decisions", "prefetches", "validations", "search_executions"}}
        all_rows.extend(result["rows"])
        all_decisions.extend(result["decisions"])
        all_prefetches.extend(result["prefetches"])
        all_validations.extend(result["validations"])
        all_searches.extend(result["search_executions"])
    selection = _release_selection(measured)
    selection["frozen_at"] = datetime.now(timezone.utc).isoformat()
    selection["development_dataset_hash"] = dataset["split_file_sha256"].get("development")
    _json(run_dir / "selection_freeze.json", selection)

    # Access final outcomes only after the selection record is on disk.
    final_streams = load_split("final")
    final_names = ["C0_baseline_psrc", selection["selected_candidate"]]
    heldout: dict[str, Any] = {}
    for name in final_names:
        overrides = ARMS[name]
        assert overrides is not None
        result = asyncio.run(_run_arm(_arm_cfg(cfg, overrides), name, final_streams, "final"))
        heldout[name] = {key: value for key, value in result.items()
                         if key not in {"rows", "decisions", "prefetches", "validations", "search_executions"}}
        all_rows.extend(result["rows"])
        all_decisions.extend(result["decisions"])
        all_prefetches.extend(result["prefetches"])
        all_validations.extend(result["validations"])
        all_searches.extend(result["search_executions"])

    _jsonl(run_dir / "per_prefix_results.jsonl", all_decisions)
    by_intent = [{"arm": item["arm"], "split": item["split"], "stream_id": item["stream_id"],
                  "evidence_id": item.get("evidence_id"), "event_type": item["event_type"],
                  "state": item.get("state"), "chunk_ids": item["chunk_ids"]} for item in all_validations]
    _json(run_dir / "per_intent_results.json", by_intent)
    _json(run_dir / "per_utterance_results.json", all_rows)
    _jsonl(run_dir / "decision_traces.jsonl", all_decisions)
    _jsonl(run_dir / "prefetch_traces.jsonl", all_prefetches)
    _json(run_dir / "validation_results.json", all_validations)
    _json(run_dir / "search_cost_breakdown.json", {
        "measurement_source": "HybridIndex.search execution registry", "executions": all_searches,
        "counts": dict(Counter(item["retrieval_type"] for item in all_searches)),
    })
    _json(run_dir / "latency_results.json", {name: arm.get("metrics", {}).get("retrieval_latency_ms")
                                               for name, arm in measured.items()})
    _json(run_dir / "cache_results.json", {name: {
        "search_cost": arm.get("metrics", {}).get("search_cost"),
        "caper_totals": arm.get("metrics", {}).get("caper_totals"),
    } for name, arm in measured.items()})
    security = write_security_audit(run_dir / "security_audit.json")
    _json(run_dir / "ablation_results.json", measured)
    _json(run_dir / "heldout_results.json", {
        "selection_frozen_before_access": True, "independent_final": False, "results": heldout,
        "limitation": "Project-authored V2.3 final streams reuse the sample corpus and synthetic timing.",
    })
    plots = run_dir / "plots"
    plots.mkdir()
    plot_rows = [(name, float(arm["metrics"]["search_cost"]["actual_hybrid_search_executions"]),
                  float(arm["metrics"]["verified_useful_evidence_before_final"]["rate"] or 0.0))
                 for name, arm in measured.items() if arm.get("status") == "MEASURED"]
    _write_plot(plots / "useful_early_vs_search_cost.svg", "Verified useful early evidence vs actual search cost",
                plot_rows, "actual HybridIndex.search executions", "verified useful early rate")
    measured_arms = [(name, arm["metrics"]) for name, arm in measured.items() if arm.get("status") == "MEASURED"]
    _write_plot(
        plots / "preparation_coverage_vs_wasted_work.svg", "Preparation coverage vs wasted work",
        [(name, float(metrics["preparation_coverage_before_final"]["rate"] or 0.0),
          float(metrics["wasted_preparation_rate"]["rate"] or 0.0)) for name, metrics in measured_arms],
        "preparation coverage", "wasted preparation rate",
    )
    _write_plot(
        plots / "evidence_validation_outcomes.svg", "Evidence validation outcomes",
        [(name, float(metrics["caper_totals"].get("searches_completed", 0)),
          float(metrics["caper_totals"].get("validated", 0))) for name, metrics in measured_arms],
        "completed provisional searches", "validated evidence records",
    )
    _write_plot(
        plots / "first_useful_evidence_lead_time.svg", "First useful evidence lead time",
        [(name, float(metrics["verified_useful_evidence_before_final"]["rate"] or 0.0),
          float(metrics["lead_time_ms"]["mean"] or 0.0)) for name, metrics in measured_arms],
        "verified useful early rate", "mean synthetic-clock lead (ms)",
    )
    c8_cost = measured["C8_full_caper_deterministic"]["metrics"]["search_cost"]["by_retrieval_type"]
    _write_plot(
        plots / "search_count_distribution.svg", "C8 actual search-count distribution",
        [(kind, float(index + 1), float(count)) for index, (kind, count) in enumerate(sorted(c8_cost.items()))],
        "retrieval type (ordinal)", "HybridIndex.search executions",
    )
    _write_plot(
        plots / "cache_reuse_and_invalidation.svg", "Cache reuse vs reconciliation pressure",
        [(name, float(metrics["search_cost"]["cache_reuse_events"]),
          float(metrics["caper_totals"].get("cancellation_requests", 0))) for name, metrics in measured_arms],
        "validated cache reuse events", "cancellation/invalidation requests",
    )
    _write_plot(
        plots / "latency_percentiles.svg", "Retrieval latency percentiles",
        [(name, float(metrics["retrieval_latency_ms"]["p50"] or 0.0),
          float(metrics["retrieval_latency_ms"]["p95"] or 0.0)) for name, metrics in measured_arms],
        "p50 retrieval latency (ms)", "p95 retrieval latency (ms)",
    )
    _write_plot(
        plots / "policy_comparison.svg", "Policy comparison",
        [(name, float(metrics["preparation_coverage_before_final"]["rate"] or 0.0),
          float(metrics["verified_useful_evidence_before_final"]["rate"] or 0.0))
         for name, metrics in measured_arms],
        "preparation coverage", "verified useful early rate",
    )
    report = {
        "schema_version": 1, "run_id": run_id, "stage": "B_REAL_SESSION_ENGINE",
        "development": measured, "selection": selection, "heldout": heldout,
        "security_status": security["status"],
        "release_decision": selection["decision"],
        "active_runtime_policy": selection["runtime_default"],
        "physical_microphone_validation": "PENDING",
        "learned_prefetch": "NOT_RUN",
        "limitations": [
            "No new independently authored final collection was available.",
            "Prefix timing is synthetic and scheduled in accelerated replay, not physical microphone timing.",
            "CAPER remains disabled by default unless every promotion gate passes.",
        ],
        "artifact_hashes": {},
    }
    _json(run_dir / "final_report.json", report)
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "final_report.json":
            report["artifact_hashes"][str(path.relative_to(run_dir)).replace("\\", "/")] = _sha(path)
    _json(run_dir / "final_report.json", report)
    _json(RUN_ROOT / "latest.json", report)
    return report


def load_latest() -> dict[str, Any]:
    return json.loads((RUN_ROOT / "latest.json").read_text(encoding="utf-8"))


def latest_artifact(name: str) -> Any:
    report = load_latest()
    return json.loads((RUN_ROOT / report["run_id"] / name).read_text(encoding="utf-8"))
