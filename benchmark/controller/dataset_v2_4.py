"""Append-only V2.4 evaluation manifest over the frozen V2.3 collection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.controller.dataset_v2_3 import ROOT as V23_ROOT, SPLITS, load_split
from kairos.config import resolve_path


ROOT = resolve_path("benchmark/datasets/controller_v2_4")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset() -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    ids: set[str] = set()
    family_splits: defaultdict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    split_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = V23_ROOT / split / "streams.json"
        if not path.exists():
            errors.append({"code": "MISSING_SPLIT", "split": split})
            continue
        split_hashes[split] = _sha(path)
        for row in load_split(split):
            stream_id = str(row.get("stream_id", ""))
            if not stream_id or stream_id in ids:
                errors.append({"code": "MISSING_OR_DUPLICATE_STREAM_ID", "stream_id": stream_id})
            ids.add(stream_id)
            counts[split] += 1
            family_splits[str(row.get("question_family", stream_id))].add(split)
            if len(row.get("prefixes", [])) != len(row.get("prefix_timestamps", [])):
                errors.append({"code": "PREFIX_TIMESTAMP_LENGTH_MISMATCH", "stream_id": stream_id})
            if row.get("timing_source") is None:
                errors.append({"code": "MISSING_TIMING_SOURCE", "stream_id": stream_id})
    for family, family_split_set in family_splits.items():
        selectable = family_split_set & {"train", "calibration", "development", "final"}
        if len(selectable) > 1:
            errors.append({"code": "QUESTION_FAMILY_SPLIT_LEAKAGE", "question_family": family,
                           "splits": sorted(selectable)})
    warnings.append({
        "code": "NO_NEW_INDEPENDENT_FINAL_COLLECTION",
        "detail": "V2.4 reuses the frozen, project-authored V2.3 final set; it is not independent evidence for generalization.",
    })
    manifest = {
        "schema_version": 1,
        "name": "kairos-controller-v2.4-caper-evaluation",
        "parent": "kairos-controller-v2.3-iccos",
        "parent_manifest_sha256": _sha(V23_ROOT / "manifests" / "dataset_manifest.json"),
        "split_file_sha256": split_hashes,
        "counts": dict(counts),
        "validation": {"status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": warnings},
        "independent_final_available": False,
        "timing_scope": "synthetic written-question prefix timestamps; not microphone timing",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
