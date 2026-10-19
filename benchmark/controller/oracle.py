from __future__ import annotations

import csv
import json
from pathlib import Path

from kairos.retrieval.index import RetrievalQuery
from kairos.textutil import lower_tokens


def oracle_rows(dataset: dict, decision_rows: list[dict], index) -> list[dict]:
    lookup = {(session["id"], utterance["id"]): utterance for session in dataset["sessions"]
              for utterance in session["utterances"]}
    end_times: dict[tuple[str, str], float] = {}
    for session in dataset["sessions"]:
        timestamp = 1.0
        for utterance in session["utterances"]:
            for chunk in utterance["chunks"]:
                timestamp += 0.3 * len(lower_tokens(chunk)) + 0.1
            end_times[(session["id"], utterance["id"])] = round(timestamp, 3)
            timestamp += 2.5
    out = []
    for row in decision_rows:
        utterance = lookup[(row["session_id"], row["utterance_id"])]
        prefix_index = max(1, int(row.get("chunk_index") or 1))
        survival = row.get("final_evidence_survival")
        stable = survival is not None and survival >= 0.7
        gold_sections = {section for intent in utterance.get("intents", []) if not intent.get("unanswerable")
                         for section in intent.get("gold_sections", [])}
        prefix_query = row.get("query_prefix", "")
        prefix_result = index.search(RetrievalQuery.build(prefix_query)) if prefix_query else None
        prefix_sections = {index.section_of(hit.chunk_id) for hit in (prefix_result.relevant_hits if prefix_result else [])}
        relevance = bool(prefix_sections & gold_sections)
        constraints_ready = (row.get("kappa") or 0.0) >= 0.30
        oracle_eligible = bool(stable and relevance and constraints_ready)
        gold_eligible = prefix_index >= int(utterance.get("eligible_chunk", len(utterance["chunks"])))
        reasons = []
        if survival is None:
            reasons.append("UNDEFINED_FINAL_ALIGNMENT")
        elif not stable:
            reasons.append("EVIDENCE_DOES_NOT_SURVIVE")
        if not relevance:
            reasons.append("NO_GOLD_RELEVANT_ACCEPTED_EVIDENCE")
        if not constraints_ready:
            reasons.append("CRITICAL_CONSTRAINTS_INCOMPLETE")
        if oracle_eligible and row.get("decision") not in {"RETRIEVE", "REFINE"}:
            reasons.append("MISSED_ORACLE_OPPORTUNITY")
        if not gold_sections:
            attribution = "INSUFFICIENT_CORPUS"
        elif not relevance:
            attribution = "LOW_EVIDENCE_RELEVANCE"
        elif not constraints_ready:
            attribution = "INCOMPLETE_CONSTRAINTS"
        elif stable and row.get("decision") not in {"RETRIEVE", "REFINE"}:
            attribution = "UNNECESSARY_HARD_GATE"
        elif not stable and row.get("s_hat") is not None and row.get("theta") is not None:
            attribution = "LOW_STABILITY_ESTIMATE"
        else:
            attribution = "NO_SAFE_EARLY_OPPORTUNITY"
        out.append({
            "session_id": row["session_id"], "utterance_id": row["utterance_id"],
            "candidate_id": row["candidate_id"], "prefix_index": prefix_index,
            "prefix_text": row.get("query_prefix", ""), "oracle_eligible": oracle_eligible,
            "gold_eligible": gold_eligible, "retrieval_decision": row.get("decision"),
            "controller_signals": {key: row.get(key) for key in
                                   ("rho", "chi", "kappa", "sigma", "pi", "nu", "s_hat")},
            "threshold": row.get("theta"), "retrieval_time": row.get("first_actual_retrieval_t"),
            "utterance_end_time": end_times[(row["session_id"], row["utterance_id"])],
            "evidence_survival": survival, "gold_sections": sorted(gold_sections),
            "prefix_evidence_sections": sorted(prefix_sections),
            "failure_reason": reasons or ["NONE"],
            "failure_attribution": attribution,
        })
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in out:
        grouped.setdefault((row["session_id"], row["utterance_id"]), []).append(row)
    for key, values in grouped.items():
        values.sort(key=lambda item: item["prefix_index"])
        utterance = lookup[key]
        summary = {
            "earliest_relevant_prefix": next((item["prefix_index"] for item in values
                                               if item["prefix_evidence_sections"]), None),
            "earliest_constraint_ready_prefix": next((item["prefix_index"] for item in values
                                                       if (item["controller_signals"].get("kappa") or 0.0) >= 0.30), None),
            "earliest_gold_useful_prefix": next((item["prefix_index"] for item in values if item["oracle_eligible"]), None),
            "first_actual_retrieval_prefix": next((item["prefix_index"] for item in values
                                                    if item["retrieval_decision"] in {"RETRIEVE", "REFINE"}), None),
            "first_useful_retrieval_prefix": next((item["prefix_index"] for item in values
                                                    if item["retrieval_decision"] in {"RETRIEVE", "REFINE"}
                                                    and item["oracle_eligible"]), None),
            "finalized_query": " ".join(utterance["chunks"]),
            "retrieval_cost": sum(item["retrieval_decision"] in {"RETRIEVE", "REFINE"} for item in values),
        }
        for item in values:
            item.update(summary)
    return out


def write_oracle(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"status": "MEASURED", "rows": rows}, indent=2), encoding="utf-8")
    fields = ["session_id", "utterance_id", "candidate_id", "prefix_index", "prefix_text",
              "oracle_eligible", "gold_eligible", "retrieval_decision", "threshold", "retrieval_time",
              "utterance_end_time", "evidence_survival", "gold_sections", "prefix_evidence_sections",
              "failure_reason", "failure_attribution", "earliest_relevant_prefix",
              "earliest_constraint_ready_prefix", "earliest_gold_useful_prefix",
              "first_actual_retrieval_prefix", "first_useful_retrieval_prefix",
              "finalized_query", "retrieval_cost", "controller_signals"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "gold_sections": "|".join(row["gold_sections"]),
                             "prefix_evidence_sections": "|".join(row["prefix_evidence_sections"]),
                             "failure_reason": "|".join(row["failure_reason"]),
                             "controller_signals": json.dumps(row["controller_signals"], sort_keys=True)})
