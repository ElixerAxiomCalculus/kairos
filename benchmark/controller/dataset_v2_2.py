"""Build and validate the provenance-controlled PSRC V2.2 stream dataset.

Written questions are deterministically segmented and explicitly labelled as
synthetic written-question timing.  They are never described as recorded speech.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from kairos.config import resolve_path
from kairos.textutil import lower_tokens

ROOT = resolve_path("benchmark/datasets/controller_v2_2")
SPLITS = ("train", "calibration", "development", "heldout", "external")


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dataset_hash(streams: Iterable[dict]) -> str:
    return _sha(sorted(streams, key=lambda row: row["stream_id"]))


def _prefixes_from_chunks(chunks: list[str]) -> list[str]:
    assembled: list[str] = []
    for chunk in chunks:
        text = chunk.strip()
        if text:
            assembled.append(text)
    return [" ".join(assembled[: index + 1]) for index in range(len(assembled))]


def _segment_written_question(question: str) -> list[str]:
    tokens = question.strip().split()
    if len(tokens) <= 4:
        cuts = [max(1, len(tokens) - 1), len(tokens)]
    else:
        cuts = sorted({max(2, round(len(tokens) * fraction)) for fraction in (0.28, 0.55, 0.78)} | {len(tokens)})
    return [" ".join(tokens[:cut]) for cut in cuts if cut <= len(tokens)]


def _timestamps(prefixes: list[str]) -> list[float]:
    elapsed = 0.0
    previous = 0
    out = []
    for prefix in prefixes:
        words = len(lower_tokens(prefix))
        elapsed += 0.3 * max(1, words - previous) + 0.1
        out.append(round(elapsed, 3))
        previous = words
    return out


def _record(
    *, stream_id: str, source_dataset: str, source_question_id: str, question_family: str,
    corpus_profile: str, full_text: str, prefixes: list[str], split: str,
    source_revision: str, annotation_source: str, timing_source: str,
    gold_sections: list[str] | None = None, gold_documents: list[str] | None = None,
    category: str = "NATURAL_QUESTION", answerable: bool | None = None,
    utterance_id: str = "u1",
) -> dict:
    return {
        "stream_id": stream_id,
        "session_id": stream_id,
        "utterance_id": utterance_id,
        "source_dataset": source_dataset,
        "source_question_id": source_question_id,
        "question_family": question_family,
        "corpus_profile": corpus_profile,
        "full_text": full_text.strip(),
        "prefixes": prefixes,
        "prefix_order": list(range(1, len(prefixes) + 1)),
        "prefix_timestamps": _timestamps(prefixes),
        "timing_source": timing_source,
        "annotation_source": annotation_source,
        "split": split,
        "source_revision": source_revision,
        "gold_sections": sorted(set(gold_sections or [])),
        "gold_documents": sorted(set(gold_documents or [])),
        "category": category,
        "answerable": answerable,
        "content_hash": hashlib.sha256(full_text.strip().lower().encode()).hexdigest(),
    }


def _internal_streams(source_split: str, output_split: str) -> list[dict]:
    path = resolve_path(f"benchmark/datasets/{source_split}/sessions.json")
    dataset = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for session in dataset["sessions"]:
        for utterance in session["utterances"]:
            prefixes = _prefixes_from_chunks(utterance["chunks"])
            gold = [section for intent in utterance.get("intents", []) for section in intent.get("gold_sections", [])]
            records.append(_record(
                stream_id=f"kairos-{source_split}-{session['id']}-{utterance['id']}",
                source_dataset=dataset["name"], source_question_id=f"{session['id']}:{utterance['id']}",
                question_family=f"kairos:{session['id']}:{utterance['id']}", corpus_profile="sample",
                full_text=prefixes[-1], prefixes=prefixes, split=output_split,
                source_revision=_sha(dataset), annotation_source="KAIROS authored benchmark annotation",
                timing_source="benchmark synthetic chunk timing (0.3 seconds/word + 0.1 seconds/chunk)",
                gold_sections=gold, category=utterance.get("type", "NATURAL_QUESTION"),
                answerable=bool(gold),
                utterance_id=utterance["id"],
            ))
    return records


def _training_streams() -> list[dict]:
    path = resolve_path("benchmark/datasets/train/utterances.txt")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        chunks = [part.strip() for part in line.split("|") if part.strip()]
        prefixes = _prefixes_from_chunks(chunks)
        records.append(_record(
            stream_id=f"kairos-train-{number:03d}", source_dataset="kairos-authored-training-v1",
            source_question_id=f"train-line-{number}", question_family=f"train-line-{number}",
            corpus_profile="sample", full_text=prefixes[-1], prefixes=prefixes, split="train",
            source_revision=hashlib.sha256(path.read_bytes()).hexdigest(),
            annotation_source="policy-independent retrieval labels generated by V2.2 extractor",
            timing_source="benchmark synthetic chunk timing (0.3 seconds/word + 0.1 seconds/chunk)",
            category="AUTHORED_STREAM", answerable=None,
        ))
    return records


def _document_split(document_id: str) -> str:
    bucket = int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 65:
        return "train"
    if bucket < 82:
        return "calibration"
    return "external"


def _squad_streams() -> list[dict]:
    labels_path = resolve_path("corpus/imported/squad_v2_1/evaluation-labels.jsonl")
    manifest = json.loads(resolve_path("corpus/imported/squad_v2_1/manifest.json").read_text(encoding="utf-8"))
    items = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    # Assign whole source documents, not questions/prefixes, to one split.  The
    # deterministic rank gives the small local subset a usable calibration set.
    documents = sorted({item["document_id"] for item in items},
                       key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    train_cut = round(len(documents) * 0.62)
    calibration_cut = round(len(documents) * 0.81)
    document_splits = {
        document: ("train" if index < train_cut else "calibration" if index < calibration_cut else "external")
        for index, document in enumerate(documents)
    }
    records = []
    for item in items:
        split = document_splits[item["document_id"]]
        question = item["question"].strip()
        records.append(_record(
            stream_id=f"squad-{item['id']}", source_dataset=manifest["dataset"],
            source_question_id=item["id"], question_family=f"squad-doc:{item['document_id']}",
            corpus_profile="general_expanded", full_text=question,
            prefixes=_segment_written_question(question), split=split,
            source_revision=manifest["revision"], annotation_source="SQuAD answer/document label",
            timing_source="synthetic written-question segmentation; not recorded speech",
            gold_documents=[item["document_id"]], category="EXTERNAL_FACTUAL_QA", answerable=True,
        ))
    return records


def _scifact_streams() -> list[dict]:
    path = resolve_path("corpus/imported/scifact/scifact-qrels-test.jsonl")
    manifest = json.loads(resolve_path("corpus/imported/scifact/manifest.json").read_text(encoding="utf-8"))
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        query = item["query"].strip()
        records.append(_record(
            stream_id=f"scifact-{item['query_id']}", source_dataset=manifest["dataset"],
            source_question_id=item["query_id"], question_family=f"scifact:{item['query_id']}",
            corpus_profile="science", full_text=query, prefixes=_segment_written_question(query),
            split="external", source_revision=manifest["revision"],
            annotation_source="SciFact/BEIR qrel", timing_source="synthetic written-question segmentation; not recorded speech",
            gold_documents=[item["document_id"]], category="EXTERNAL_SCIENTIFIC_CLAIM", answerable=True,
        ))
    return records


def all_source_streams() -> list[dict]:
    return [
        *_training_streams(), *_internal_streams("dev", "development"),
        *_internal_streams("heldout_streaming", "heldout"), *_squad_streams(), *_scifact_streams(),
    ]


def validate_streams(streams: Iterable[dict]) -> dict:
    rows = list(streams)
    errors: list[dict] = []
    valid_profiles = {"sample", "science", "general", "general_expanded", "multi_domain_demo"}
    valid_splits = set(SPLITS)
    required = {
        "stream_id", "session_id", "utterance_id", "source_dataset", "source_question_id",
        "question_family", "corpus_profile", "full_text", "prefixes", "prefix_order",
        "prefix_timestamps", "timing_source", "annotation_source", "split", "source_revision", "content_hash",
    }
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    family_splits: defaultdict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            errors.append({"stream_id": row.get("stream_id"), "code": "MISSING_METADATA", "values": missing})
            continue
        stream_id = row["stream_id"]
        if stream_id in ids:
            errors.append({"stream_id": stream_id, "code": "DUPLICATE_STREAM_ID"})
        ids.add(stream_id)
        if not row["full_text"].strip() or not row["prefixes"]:
            errors.append({"stream_id": stream_id, "code": "EMPTY_QUERY"})
        if row.get("split") not in valid_splits:
            errors.append({"stream_id": stream_id, "code": "INVALID_SPLIT", "value": row.get("split")})
        if row.get("corpus_profile") not in valid_profiles:
            errors.append({"stream_id": stream_id, "code": "INVALID_CORPUS_REFERENCE", "value": row.get("corpus_profile")})
        if not isinstance(row.get("answerable"), (bool, type(None))):
            errors.append({"stream_id": stream_id, "code": "INVALID_ANSWERABLE_LABEL"})
        if not str(row.get("source_question_id", "")).strip():
            errors.append({"stream_id": stream_id, "code": "INVALID_SOURCE_QUESTION_ID"})
        internal_sources = {"kairos-dev-v1", "kairos-controller-heldout-v1"}
        source_utterance_id = str(row.get("source_question_id", "")).rsplit(":", 1)[-1]
        if row.get("source_dataset") in internal_sources and source_utterance_id != row["utterance_id"]:
            errors.append({"stream_id": stream_id, "code": "INCONSISTENT_UTTERANCE_ID"})
        if not all(isinstance(value, str) and value.strip() for value in row.get("gold_sections", [])):
            errors.append({"stream_id": stream_id, "code": "INVALID_GOLD_SECTION_REFERENCE"})
        if not all(isinstance(value, str) and value.strip() for value in row.get("gold_documents", [])):
            errors.append({"stream_id": stream_id, "code": "INVALID_GOLD_DOCUMENT_REFERENCE"})
        if row["prefix_order"] != list(range(1, len(row["prefixes"]) + 1)):
            errors.append({"stream_id": stream_id, "code": "INVALID_PREFIX_ORDER"})
        if len(row["prefixes"]) != len(row["prefix_timestamps"]):
            errors.append({"stream_id": stream_id, "code": "MISSING_TIMESTAMP"})
        if row["prefix_timestamps"] != sorted(row["prefix_timestamps"]):
            errors.append({"stream_id": stream_id, "code": "NON_MONOTONIC_TIMESTAMP"})
        if any(not row["prefixes"][index + 1].lower().startswith(row["prefixes"][index].lower())
               for index in range(len(row["prefixes"]) - 1)):
            errors.append({"stream_id": stream_id, "code": "NON_CUMULATIVE_PREFIX"})
        if row["content_hash"] in hashes:
            errors.append({"stream_id": stream_id, "code": "DUPLICATE_UTTERANCE", "other": hashes[row["content_hash"]]})
        hashes[row["content_hash"]] = stream_id
        family_splits[row["question_family"]].add(row["split"])
        split_counts[row["split"]] += 1
    for family, splits in family_splits.items():
        if len(splits) > 1:
            errors.append({"question_family": family, "code": "GROUP_SPLIT_LEAKAGE", "splits": sorted(splits)})

    # Near duplicates are checked across splits at the utterance level, never across correlated prefixes.
    token_sets = [(row["stream_id"], row["split"], set(lower_tokens(row["full_text"]))) for row in rows]
    near_duplicates = []
    for index, (left_id, left_split, left) in enumerate(token_sets):
        for right_id, right_split, right in token_sets[index + 1:]:
            if left_split == right_split or not left or not right:
                continue
            similarity = len(left & right) / len(left | right)
            if similarity >= 0.9:
                near_duplicates.append({"left": left_id, "right": right_id, "jaccard": round(similarity, 4)})
    if near_duplicates:
        errors.append({"code": "NEAR_DUPLICATE_SPLIT_LEAKAGE", "pairs": near_duplicates})
    return {
        "status": "PASS" if not errors else "FAIL", "streams": len(rows),
        "prefixes": sum(len(row.get("prefixes", [])) for row in rows),
        "split_counts": dict(sorted(split_counts.items())), "families": len(family_splits),
        "errors": errors, "validation_hash": _dataset_hash(rows),
    }


def build_dataset(root: Path = ROOT) -> dict:
    streams = all_source_streams()
    validation = validate_streams(streams)
    if validation["status"] != "PASS":
        raise ValueError(f"controller V2.2 dataset validation failed: {validation['errors'][:3]}")
    for split in SPLITS:
        target = root / split / "streams.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        selected = [row for row in streams if row["split"] == split]
        target.write_text(json.dumps({"schema_version": 1, "split": split, "streams": selected}, indent=2), encoding="utf-8")
    provenance = {
        "kairos_train": "benchmark/datasets/train/utterances.txt (project-authored)",
        "kairos_development": "benchmark/datasets/dev/sessions.json (project-authored)",
        "kairos_heldout": "benchmark/datasets/heldout_streaming/sessions.json (separately authored, small)",
        "squad": "rajpurkar/squad, revision and CC BY-SA 4.0 metadata in corpus/imported/squad_v2_1/manifest.json",
        "scifact": "BeIR/scifact, revision and CC BY-SA 4.0 metadata in corpus/imported/scifact/manifest.json",
        "timing_warning": "External written questions use synthetic segmentation and are not recorded speech.",
    }
    (root / "sources").mkdir(parents=True, exist_ok=True)
    (root / "sources" / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "label_definition.json").write_text(json.dumps({
        "survival_threshold": 0.7,
        "target": "accepted evidence at a prefix remains useful for the finalized question",
        "components": ["rank-biased overlap", "accepted section overlap", "independent gold-document overlap when available"],
        "policy_independent": True,
    }, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1, "name": "kairos-controller-v2.2", "dataset_hash": _dataset_hash(streams),
        "split_hashes": {split: _sha([row for row in streams if row["split"] == split]) for split in SPLITS},
        "source_revision": "local working tree plus immutable imported manifests", **validation,
        "limitations": [
            "Initial engineering dataset contains fewer than several hundred independent utterances.",
            "Most examples are written questions with synthetic segmentation, not microphone recordings.",
            "Project-authored development and held-out streams share the sample corpus and are not external generalization evidence.",
        ],
    }
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(split: str, root: Path = ROOT) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown controller split: {split}")
    return json.loads((root / split / "streams.json").read_text(encoding="utf-8"))["streams"]


def validate_dataset(root: Path = ROOT) -> dict:
    streams = [row for split in SPLITS for row in load_split(split, root)]
    result = validate_streams(streams)
    manifest = json.loads((root / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    result["manifest_dataset_hash_matches"] = _dataset_hash(streams) == manifest["dataset_hash"]
    if not result["manifest_dataset_hash_matches"]:
        result["status"] = "FAIL"
        result["errors"].append({"code": "MANIFEST_HASH_MISMATCH"})
    return result
