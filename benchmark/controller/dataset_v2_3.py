"""Build the append-only KAIROS V2.3 IC-COS dataset and freeze its manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.controller.dataset_v2_2 import ROOT as V22_ROOT
from benchmark.controller.dataset_v2_2 import build_dataset as build_v22
from benchmark.controller.dataset_v2_2 import load_split as load_v22_split
from kairos.config import resolve_path


ROOT = resolve_path("benchmark/datasets/controller_v2_3")
SPLITS = ("train", "calibration", "development", "historical_frozen", "external", "final")
FREEZE_ID = "KAIROS-ICCOS-FINAL-2026-10-19"
FROZEN_AT = "2026-10-19T00:00:00+00:00"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _segment(question: str) -> list[str]:
    words = question.split()
    cuts = sorted({max(2, round(len(words) * fraction)) for fraction in (0.28, 0.55, 0.78)} | {len(words)})
    return [" ".join(words[:cut]) for cut in cuts]


def _timestamps(prefixes: list[str]) -> list[float]:
    elapsed, previous = 0.0, 0
    values = []
    for prefix in prefixes:
        count = len(prefix.split())
        elapsed += 0.3 * max(1, count - previous) + 0.1
        values.append(round(elapsed, 3))
        previous = count
    return values


FINAL_SPEC = (
    ("F01", "What deadline applies if we cancel a standard venue booking?", ["D03#3.1"], []),
    ("F02", "For an overseas company event, when must the guest list reach security?", ["D06#2.2"], []),
    ("F03", "Which Pune room can hold twenty people, and what standard AV equipment is included?", ["D01#2", "D01#3", "D07#1"], []),
    ("F04", "Can catering accommodate dietary needs, and what is the minimum order size?", ["D04#2", "D04#3"], []),
    ("F05", "How long does expense reimbursement take after a complete claim is filed?", ["D05#5"], []),
    ("F06", "Can I revise attendee numbers after reserving a venue?", ["D03#4", "D08#2"], []),
    ("F07", "Is alcohol available for an internal corporate function?", ["D04#5"], []),
    ("F08", "What parking and accessibility information is provided for the Pune venues?", ["D01#5"], []),
    ("F09", "For a hybrid meeting, what extra technical support can be requested?", ["D07#2", "D07#3"], []),
    ("F10", "What paperwork is needed for small travel expenses and when are receipts mandatory?", ["D05#2"], []),
    ("F11", "Find a beachfront venue on Mars for a thousand guests.", [], []),
    ("F12", "How do I register visitors and keep them out of restricted areas?", ["D06#1", "D06#3"], []),
)


def _final_streams() -> list[dict[str, Any]]:
    rows = []
    for item_id, question, sections, documents in FINAL_SPEC:
        prefixes = _segment(question)
        rows.append({
            "stream_id": f"iccos-final-{item_id}",
            "session_id": f"iccos-final-{item_id}",
            "utterance_id": "u1",
            "source_dataset": "kairos-iccos-project-authored-final-v1",
            "source_question_id": item_id,
            "question_family": f"iccos-final:{item_id}",
            "corpus_profile": "sample",
            "full_text": question,
            "prefixes": prefixes,
            "prefix_order": list(range(1, len(prefixes) + 1)),
            "prefix_timestamps": _timestamps(prefixes),
            "timing_source": "synthetic written-question segmentation; not recorded speech",
            "annotation_source": "project-authored V2.3 gold section annotation",
            "split": "final",
            "source_revision": FREEZE_ID,
            "gold_sections": sections,
            "gold_documents": documents,
            "gold_intents": [{
                "intent_id": f"{item_id}:A",
                "gold_sections": sections,
                "gold_documents": documents,
                "unanswerable": not bool(sections or documents),
            }],
            "category": "FROZEN_FINAL_WRITTEN_QUESTION",
            "answerable": bool(sections or documents),
            "content_hash": hashlib.sha256(question.lower().encode()).hexdigest(),
            "freeze_id": FREEZE_ID,
        })
    return rows


def _source_intents() -> dict[tuple[str, str], list[dict[str, Any]]]:
    mapping: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset_path in (
        resolve_path("benchmark/datasets/dev/sessions.json"),
        resolve_path("benchmark/datasets/heldout_streaming/sessions.json"),
    ):
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        for session in data["sessions"]:
            for utterance in session["utterances"]:
                mapping[(session["id"], utterance["id"])] = [
                    {
                        "intent_id": f"{session['id']}:{utterance['id']}:{item.get('id', index)}",
                        "gold_sections": item.get("gold_sections", []),
                        "gold_documents": item.get("gold_documents", []),
                        "unanswerable": bool(item.get("unanswerable", False)),
                    }
                    for index, item in enumerate(utterance.get("intents", []), start=1)
                ]
    return mapping


def _enrich(rows: list[dict[str, Any]], split: str, source: dict) -> list[dict[str, Any]]:
    enriched = []
    for original in rows:
        row = dict(original)
        row["split"] = split
        parts = str(row.get("source_question_id", "")).split(":", 1)
        if len(parts) == 2 and (parts[0], parts[1]) in source:
            row["gold_intents"] = source[(parts[0], parts[1])]
            row["intent_annotation_scope"] = "source_utterance_intents"
        elif row.get("gold_sections") or row.get("gold_documents"):
            row["gold_intents"] = [{
                "intent_id": f"{row['stream_id']}:aggregate",
                "gold_sections": row.get("gold_sections", []),
                "gold_documents": row.get("gold_documents", []),
                "unanswerable": False,
            }]
            row["intent_annotation_scope"] = "aggregate_external_label"
        else:
            row["gold_intents"] = []
            row["intent_annotation_scope"] = "missing_gold_annotation"
        enriched.append(row)
    return enriched


def build_dataset() -> dict[str, Any]:
    build_v22()
    source = _source_intents()
    split_map = {
        "train": "train",
        "calibration": "calibration",
        "development": "development",
        "historical_frozen": "heldout",
        "external": "external",
    }
    all_rows = []
    split_hashes = {}
    for target, origin in split_map.items():
        rows = _enrich(load_v22_split(origin), target, source)
        all_rows.extend(rows)
        folder = ROOT / target
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 2, "split": target, "streams": rows}
        (folder / "streams.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        split_hashes[target] = _sha(rows)
    final_rows = _final_streams()
    all_rows.extend(final_rows)
    (ROOT / "final").mkdir(parents=True, exist_ok=True)
    (ROOT / "final" / "streams.json").write_text(
        json.dumps({"schema_version": 2, "split": "final", "streams": final_rows}, indent=2), encoding="utf-8"
    )
    split_hashes["final"] = _sha(final_rows)

    validation = validate_dataset(all_rows)
    manifest = {
        "schema_version": 2,
        "name": "kairos-controller-v2.3-iccos",
        "dataset_hash": _sha(all_rows),
        "split_hashes": split_hashes,
        "feature_schema": "kairos-iccos-online-v1",
        "parent_dataset": "kairos-controller-v2.2",
        "parent_dataset_hash": json.loads((V22_ROOT / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))["dataset_hash"],
        "freeze": {
            "id": FREEZE_ID,
            "recorded_at": FROZEN_AT,
            "selection_access": "final split excluded until model, calibration, and risk threshold are frozen",
            "independent_authorship": False,
        },
        "counts": dict(Counter(row["split"] for row in all_rows)),
        "status": "PASS" if not validation["errors"] else "FAIL",
        "validation": validation,
        "limitations": [
            "The new final questions were authored by the project, not by independent annotators.",
            "They are genuinely excluded from V2.3 training, calibration, development selection, and threshold selection, but reuse the sample corpus.",
            "Written questions use deterministic synthetic segmentation and are not microphone recordings.",
            "The historical V2.2 held-out set is retained as historical_frozen and is not described as untouched.",
        ],
    }
    (ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "manifests" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "annotations").mkdir(parents=True, exist_ok=True)
    (ROOT / "annotations" / "alignment_review.json").write_text(json.dumps({
        "status": "NOT_INDEPENDENTLY_HUMAN_REVIEWED",
        "review_protocol": "Review QueryFrame stable-key lineage, final candidate assignment, and each gold-intent evidence match.",
        "required_fields": ["reviewer", "reviewed_at", "stream_id", "candidate_id", "decision", "notes"],
        "rows": [],
    }, indent=2), encoding="utf-8")
    return manifest


def validate_dataset(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        rows = []
        for split in SPLITS:
            path = ROOT / split / "streams.json"
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["streams"])
    errors = []
    ids = set()
    families: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["stream_id"] in ids:
            errors.append({"code": "DUPLICATE_STREAM_ID", "stream_id": row["stream_id"]})
        ids.add(row["stream_id"])
        families[row["question_family"]].add(row["split"])
        if row["prefix_order"] != list(range(1, len(row["prefixes"]) + 1)):
            errors.append({"code": "INVALID_PREFIX_ORDER", "stream_id": row["stream_id"]})
        if len(row["prefixes"]) != len(row["prefix_timestamps"]):
            errors.append({"code": "MISSING_TIMESTAMPS", "stream_id": row["stream_id"]})
    for family, assigned in families.items():
        if len(assigned) > 1:
            errors.append({"code": "GROUP_SPLIT_LEAKAGE", "question_family": family, "splits": sorted(assigned)})
    return {"streams": len(rows), "families": len(families), "errors": errors}


def load_split(split: str) -> list[dict[str, Any]]:
    return json.loads((ROOT / split / "streams.json").read_text(encoding="utf-8"))["streams"]
