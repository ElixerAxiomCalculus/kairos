"""Corrected intent-level metrics for V2.3 stopping experiments."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Callable

from benchmark.controller.metrics import ratio, wilson


def evaluate_policy(
    rows: list[dict[str, Any]],
    decide: Callable[[dict[str, Any]], bool],
    *,
    probability: Callable[[dict[str, Any]], float | None] | None = None,
) -> dict[str, Any]:
    trajectories: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectories[(row["stream_id"], row["candidate_id"])].append(row)
    selections = []
    for key, values in trajectories.items():
        values.sort(key=lambda item: item["prefix_index"])
        selected = next((row for row in values if decide(row)), None)
        selections.append((key, values, selected))

    denominator = len(selections)
    selected_rows = [selected for _, _, selected in selections if selected is not None]
    useful_rows = [row for row in selected_rows if row["targets"].get("gold_relevance") == 1]
    strict_rows = [row for row in selected_rows if row["targets"].get("strict_useful") == 1]
    safe_rows = [row for row in strict_rows if row.get("beneficial_stop") == 1]
    unsafe = len(selected_rows) - len(strict_rows)
    first_useful_elapsed = [row["prefix_timestamp_s"] * 1000 for row in strict_rows]
    lead = [row["targets"]["time_saved_s"] * 1000 for row in strict_rows]
    chunks = [row["accepted_chunk_rbo"] for row in selected_rows if row.get("accepted_chunk_rbo") is not None]
    sections = [row["accepted_section_jaccard"] for row in selected_rows if row.get("accepted_section_jaccard") is not None]
    probabilities = [probability(row) for row in selected_rows] if probability else []
    probabilities = [value for value in probabilities if value is not None]
    # Every lineage needs one final fallback if it does not stop early. This is
    # an explicit search count; shadow/speculative searches are zero here.
    full_searches = denominator
    return {
        "intent_denominator": denominator,
        "raw_early": {"numerator": len(selected_rows), "denominator": denominator,
                      "rate": ratio(len(selected_rows), denominator), "wilson_95": wilson(len(selected_rows), denominator)},
        "gold_useful_early": {"numerator": len(useful_rows), "denominator": denominator,
                             "rate": ratio(len(useful_rows), denominator), "wilson_95": wilson(len(useful_rows), denominator)},
        "strict_useful_early": {"numerator": len(strict_rows), "denominator": denominator,
                               "rate": ratio(len(strict_rows), denominator), "wilson_95": wilson(len(strict_rows), denominator)},
        "strict_safe_early": {"numerator": len(safe_rows), "denominator": denominator,
                             "rate": ratio(len(safe_rows), denominator), "wilson_95": wilson(len(safe_rows), denominator)},
        "premature": {"numerator": unsafe, "denominator": len(selected_rows),
                      "rate": ratio(unsafe, len(selected_rows)), "wilson_95": wilson(unsafe, len(selected_rows))},
        "accepted_chunk_survival_mean": round(statistics.mean(chunks), 4) if chunks else None,
        "accepted_section_survival_mean": round(statistics.mean(sections), 4) if sections else None,
        "latency": {
            "retrieval_execution_latency_ms": None,
            "retrieval_execution_latency_status": "NOT_MEASURED_BY_OFFLINE_POLICY_REPLAY",
            "first_useful_evidence_elapsed_ms_mean": round(statistics.mean(first_useful_elapsed), 3) if first_useful_elapsed else None,
            "lead_before_end_ms_mean": round(statistics.mean(lead), 3) if lead else None,
        },
        "costs": {
            "full_searches": full_searches,
            "primary_searches": full_searches,
            "shadow_searches": 0,
            "speculative_searches": 0,
            "refinement_searches": 0,
            "final_fallback_searches": denominator - len(selected_rows),
        },
        "uncertainty": {
            "selected_with_probability": len(probabilities),
            "mean_selected_probability": round(statistics.mean(probabilities), 4) if probabilities else None,
        },
        "selected_rows": [{"stream_id": row["stream_id"], "candidate_id": row["candidate_id"],
                           "prefix_index": row["prefix_index"]} for row in selected_rows],
        "definitions": {
            "intent_denominator": "candidate lineages with at least one non-final opportunity",
            "raw_early": "lineage has a selected non-final retrieval",
            "gold_useful_early": "selected prefix retrieves its aligned gold evidence",
            "strict_useful_early": "gold relevance + constraints + survival + answerability + grounding eligibility",
            "strict_safe_early": "strict useful and counterfactual action advantage is positive",
        },
    }
