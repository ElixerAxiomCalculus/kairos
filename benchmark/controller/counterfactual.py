"""Counterfactual action labeling and finite-horizon stopping for IC-COS."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from benchmark.controller.intent_alignment import gold_for_stream
from kairos.engine import ChunkIn, EngineContext, SessionEngine


@dataclass(frozen=True)
class UtilityWeights:
    gold_relevance: float = 0.30
    constraint_compatibility: float = 0.15
    evidence_survival: float = 0.22
    answerability: float = 0.10
    grounding_eligibility: float = 0.13
    latency_per_second: float = 0.08
    retrieval_cost: float = 0.12
    revision_cost: float = 0.20
    wait_step_cost: float = 0.015


def retrieve_utility(row: dict[str, Any], weights: UtilityWeights = UtilityWeights()) -> tuple[float, dict[str, float]]:
    target = row["targets"]
    components = {
        "gold_relevance": weights.gold_relevance * float(target.get("gold_relevance") or 0.0),
        "constraint_compatibility": weights.constraint_compatibility * float(target.get("constraint_compatibility") or 0.0),
        "evidence_survival": weights.evidence_survival * float(target.get("evidence_survival") or 0.0),
        "answerability": weights.answerability * float(target.get("answerability") or 0.0),
        "grounding_eligibility": weights.grounding_eligibility * float(target.get("grounding_eligibility") or 0.0),
        "latency_value": weights.latency_per_second * min(float(target.get("time_saved_s") or 0.0), 3.0) / 3.0,
        "retrieval_cost": -weights.retrieval_cost * float(target.get("retrieval_cost") or 0.0),
        "revision_cost": -weights.revision_cost * float(target.get("revision_cost") or 0.0),
    }
    return sum(components.values()), components


def label_optimal_stopping(rows: list[dict[str, Any]], weights: UtilityWeights = UtilityWeights()) -> list[dict[str, Any]]:
    """Backward induction over each candidate lineage; final observations are excluded."""
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stream_id"], row["candidate_id"])].append(row)
    labeled = []
    for trajectory in grouped.values():
        trajectory.sort(key=lambda item: item["prefix_index"])
        future_value = 0.0
        future_action = "TERMINAL"
        reverse_rows = []
        for row in reversed(trajectory):
            if row["is_final"]:
                # End-of-utterance retrieval is the terminal fallback, not an
                # early-stopping opportunity or oracle candidate.
                future_value = 0.0
                future_action = "END_OF_UTTERANCE"
                continue
            q_retrieve, components = retrieve_utility(row, weights)
            q_wait = future_value - weights.wait_step_cost
            delta = q_retrieve - q_wait
            optimal = "RETRIEVE" if delta > 0.0 else "WAIT"
            value = max(q_retrieve, q_wait)
            enriched = dict(row)
            enriched.update({
                "q_retrieve": round(q_retrieve, 8),
                "q_wait": round(q_wait, 8),
                "delta_q": round(delta, 8),
                "beneficial_stop": int(delta > 0.0),
                "optimal_action": optimal,
                "continuation_action": future_action,
                "utility_components": {key: round(value, 8) for key, value in components.items()},
                "counterfactual_stage": "A_DETERMINISTIC_REPLAY",
            })
            reverse_rows.append(enriched)
            future_value, future_action = value, optimal
        labeled.extend(reversed(reverse_rows))
    return sorted(labeled, key=lambda item: (item["stream_id"], item["prefix_index"], item["candidate_id"]))


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute distinct earliest opportunities. Final prefixes are never eligible."""
    nonfinal = [row for row in rows if not row.get("is_final")]
    def earliest(predicate):
        matched = [row["prefix_index"] for row in nonfinal if predicate(row)]
        return min(matched) if matched else None
    return {
        "earliest_gold_relevant_prefix": earliest(lambda row: row["targets"].get("gold_relevance") == 1),
        "earliest_constraint_ready_prefix": earliest(lambda row: row["targets"].get("constraint_compatibility") == 1),
        "earliest_useful_prefix": earliest(lambda row: row["targets"].get("strict_useful") == 1),
        "earliest_strict_safe_prefix": earliest(lambda row: row["targets"].get("strict_useful") == 1 and row.get("beneficial_stop") == 1),
        "eligible_nonfinal_rows": len(nonfinal),
        "terminal_rows_excluded": sum(bool(row.get("is_final")) for row in rows),
    }


def _deltas(prefixes: list[str]) -> list[str]:
    output, previous = [], ""
    for prefix in prefixes:
        if previous and prefix.lower().startswith(previous.lower()):
            output.append(prefix[len(previous):].strip())
        else:
            output.append(prefix)
        previous = prefix
    return output


async def _branch(cfg, stream: dict[str, Any], candidate_id: str, observation: int, action: str) -> dict[str, Any]:
    branch_cfg = cfg.with_overrides({"corpus.profile": stream["corpus_profile"]})
    ctx = EngineContext(branch_cfg)
    engine = SessionEngine(ctx, f"cf-{stream['stream_id']}-{action.lower()}")
    engine.controller.set_counterfactual_action(candidate_id, observation, action)
    engine.start()
    timestamps = stream["prefix_timestamps"]
    start = 0.0
    for index, (text, end) in enumerate(zip(_deltas(stream["prefixes"]), timestamps), start=1):
        await engine.on_chunk(ChunkIn(
            utterance_id=stream.get("utterance_id", "u1"), text=text,
            t0=start, t1=float(end), final=index == len(timestamps),
        ))
        start = float(end)
    completions = [event for event in engine.events if event.event_type == "RetrievalCompleted"]
    accepted = sorted({chunk for event in completions for chunk in event.payload.get("accepted_chunk_ids", [])})
    sections = sorted({ctx.index.section_of(chunk) for chunk in accepted})
    docs = sorted({ctx.corpus.chunk(chunk).doc_id for chunk in accepted if ctx.corpus.chunk(chunk)})
    gold = gold_for_stream(stream)
    useful = any(set(sections) & set(item.get("gold_sections", [])) or set(docs) & set(item.get("gold_documents", [])) for item in gold)
    full_searches = sum(bool(event.payload.get("search_executed", True)) for event in completions)
    full_searches += sum(
        int(candidate.get("signals", {}).get("shadow_search_executed", 0))
        for event in engine.events if event.event_type == "RetrievalDecision"
        for candidate in event.payload.get("candidates", [])
    )
    first_early = next((event.timestamp for event in engine.events if event.event_type == "RetrievalStarted" and event.retrieval_trigger == "EARLY"), None)
    return {
        "action": action,
        "accepted_chunks": accepted,
        "accepted_sections": sections,
        "accepted_documents": docs,
        "gold_relevant": useful,
        "full_searches": full_searches,
        "first_early_retrieval_elapsed_s": first_early,
        "events": len(engine.events),
    }


def validate_real_branches(cfg, stream: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Stage B: reconstruct identical state and branch one live SessionEngine action."""
    async def execute():
        return await asyncio.gather(
            _branch(cfg, stream, row["candidate_id"], row["candidate_observation"], "RETRIEVE"),
            _branch(cfg, stream, row["candidate_id"], row["candidate_observation"], "WAIT"),
        )
    retrieve, wait = asyncio.run(execute())
    return {
        "stage": "B_REAL_SESSION_ENGINE",
        "stream_id": stream["stream_id"],
        "candidate_id": row["candidate_id"],
        "prefix_index": row["prefix_index"],
        "same_continuation": True,
        "same_corpus_and_config": True,
        "retrieve_now": retrieve,
        "wait_next": wait,
    }
