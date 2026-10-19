"""Policy-prefix feature replay and intent-conditioned V2.3 target extraction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from benchmark.controller.intent_alignment import (
    AlignmentStatus,
    classify_final_alignment,
    evidence_identity,
    gold_for_stream,
    matching_gold_intents,
    survival,
)
from kairos.controller.features import FEATURE_NAMES, candidate_slot_snapshot, extract_online_features
from kairos.controller.policy import CandidateState
from kairos.engine import SessionEngine


def replay_feature_rows(ctx, stream: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay cumulative prefixes through the production QueryFrame and feature code.

    Full retrieval is executed only after the online vector is complete and is
    stored exclusively as an offline target.  It cannot influence a feature.
    """
    # SessionEngine owns the production controller/cache configuration.  Using
    # its controller here prevents a separate training-only construction path.
    controller = SessionEngine(ctx, f"feature-replay-{stream['stream_id']}").controller
    states: dict[str, CandidateState] = {}
    previous_slots = {}
    previous_frame = None
    provisional: list[dict[str, Any]] = []
    timestamps = stream.get("prefix_timestamps") or list(range(1, len(stream["prefixes"]) + 1))
    for prefix_index, (prefix, timestamp) in enumerate(zip(stream["prefixes"], timestamps), start=1):
        is_final = prefix_index == len(stream["prefixes"])
        frame = ctx.frame_builder.build(
            prefix, previous_slots, float(timestamp), f"{stream['stream_id']}:{prefix_index}", previous_frame
        )
        for candidate in frame.candidates:
            state = states.setdefault(candidate.key, CandidateState(candidate.key))
            query = controller.query_for(candidate, frame, 0.4)
            probe = ctx.index.probe(query.terms)
            features = extract_online_features(
                index=ctx.index,
                query=query,
                candidate=candidate,
                frame=frame,
                state=state,
                probe=probe,
                is_final=is_final,
                utterance_end_cue=prefix.rstrip().endswith((".", "?", "!")),
                remaining_prior_s=float(controller.cfg["remaining_prior_s"]),
                cache_reusable=False,
            )
            # Target-only search: deliberately occurs after feature extraction.
            result = ctx.index.search(query)
            identity = evidence_identity(ctx, result)
            provisional.append({
                "stream_id": stream["stream_id"],
                "session_id": stream.get("session_id", stream["stream_id"]),
                "utterance_id": stream.get("utterance_id", "u1"),
                "question_family": stream.get("question_family", stream["stream_id"]),
                "source_dataset": stream.get("source_dataset", "unknown"),
                "source_question_id": stream.get("source_question_id", stream["stream_id"]),
                "corpus_profile": stream["corpus_profile"],
                "split": stream.get("split", "unknown"),
                "prefix_index": prefix_index,
                "candidate_observation": state.observations,
                "prefix_timestamp_s": float(timestamp),
                "prefix_text": prefix,
                "is_final": is_final,
                "candidate_id": candidate.key,
                "candidate_topic_id": candidate.topic_id,
                "candidate_text": candidate.text,
                "candidate_status": candidate.status,
                "lineage_source": "QUERYFRAME_STABLE_KEY",
                "features": features,
                "feature_names": list(FEATURE_NAMES),
                "prefix_evidence": identity.public(),
                "prefix_result_status": result.evidence_status,
                "query_text": query.dense_text,
            })
            state.observations += 1
            state.last_text = candidate.text
            state.last_probe = [key for key, _ in probe]
            state.last_query_text = query.dense_text
            state.last_slots = candidate_slot_snapshot(candidate, frame)
        previous_slots = frame.slots
        previous_frame = frame

    final_rows = {row["candidate_id"]: row for row in provisional if row["is_final"]}
    final_topics: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows.values():
        if row["candidate_topic_id"]:
            final_topics[row["candidate_topic_id"]].append(row)
    gold = gold_for_stream(stream)
    end_time = float(timestamps[-1]) if timestamps else 0.0
    for row in provisional:
        final = final_rows.get(row["candidate_id"])
        lineage_source = "QUERYFRAME_STABLE_KEY"
        if final is None and row["candidate_topic_id"] and len(final_topics[row["candidate_topic_id"]]) == 1:
            final = final_topics[row["candidate_topic_id"]][0]
            lineage_source = "UNIQUE_TOPIC_LINEAGE"
        final_identity = None
        if final is not None:
            public = final["prefix_evidence"]
            from benchmark.controller.intent_alignment import EvidenceIdentity
            final_identity = EvidenceIdentity(tuple(public["chunks"]), tuple(public["sections"]), tuple(public["documents"]))
        public = row["prefix_evidence"]
        from benchmark.controller.intent_alignment import EvidenceIdentity
        prefix_identity = EvidenceIdentity(tuple(public["chunks"]), tuple(public["sections"]), tuple(public["documents"]))
        alignment = classify_final_alignment(
            identity=final_identity,
            gold_intents=gold,
            domain_available=row["features"]["domain_availability"] > 0.0,
            lineage_defined=final is not None,
        )
        gold_matches = matching_gold_intents(prefix_identity, gold)
        survival_values = survival(prefix_identity, final_identity)
        constraint_ok = row["features"]["constraint_readiness"] >= 0.75 and row["features"]["constraint_revision_risk"] <= 0.45
        answerable = bool(prefix_identity.chunks) and row["prefix_result_status"] == "RELEVANT_SUFFICIENT"
        grounding_eligible = answerable and bool(gold_matches or not gold)
        useful = bool(gold_matches and constraint_ok and survival_values["survival_label"] == 1 and grounding_eligible)
        row.update({
            "lineage_source": lineage_source,
            "final_candidate_id": final["candidate_id"] if final else None,
            "final_evidence": final_identity.public() if final_identity else None,
            "final_alignment_status": alignment.value,
            "matched_gold_intents": gold_matches,
            "targets": {
                "gold_relevance": int(bool(gold_matches)) if gold else None,
                "constraint_compatibility": int(constraint_ok),
                "evidence_survival": survival_values["survival_label"],
                "answerability": int(answerable),
                "retrieval_cost": 1.0,
                "time_saved_s": round(max(0.0, end_time - row["prefix_timestamp_s"]), 6),
                "revision_cost": round(float(row["features"]["constraint_revision_risk"]), 6),
                "grounding_eligibility": int(grounding_eligible),
                "proxy_strict_useful": int(useful),
                # Prefix feature replay does not execute generated claims
                # through GroundingFirewall.  Verification is therefore
                # unavailable, not a negative outcome.
                "verified_strict_useful": None,
                "verified_strict_useful_status": "UNAVAILABLE_NO_CLAIM_EXECUTION",
                # Historical alias retained solely for V2.3 reproducibility.
                "strict_useful": int(useful),
            },
            **survival_values,
        })
    return provisional


def feature_parity_differences(training_rows: list[dict], decision_events: list[Any]) -> list[dict[str, Any]]:
    """Compare offline replay vectors with live RetrievalDecision candidate signals."""
    live: dict[tuple[int, str], dict[str, float]] = {}
    chunk_index = 0
    for event in decision_events:
        kind = event.event_type if hasattr(event, "event_type") else event["event_type"]
        payload = event.payload if hasattr(event, "payload") else event["payload"]
        if kind == "TranscriptChunkReceived":
            chunk_index += 1
        elif kind == "RetrievalDecision":
            for candidate in payload.get("candidates", []):
                live[(chunk_index, candidate["key"])] = candidate.get("signals", {})
    differences = []
    for row in training_rows:
        key = (row["prefix_index"], row["candidate_id"])
        signals = live.get(key)
        if signals is None:
            differences.append({"key": key, "reason": "MISSING_LIVE_CANDIDATE"})
            continue
        for name, expected in row["features"].items():
            actual = signals.get(name)
            if actual is None or abs(float(actual) - float(expected)) > 1e-4:
                differences.append({"key": key, "feature": name, "training": expected, "live": actual})
    return differences
