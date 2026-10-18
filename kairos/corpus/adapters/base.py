"""Dataset adapter primitives and registry.

Runtime indexing has no Hugging Face dependency.  Preparation uses the public Dataset Viewer API
through the Python standard library, or accepts a previously downloaded JSON fixture for offline CI.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DATASETS = {
    "scifact": {
        "repo": "BeIR/scifact", "config": "corpus", "split": "corpus",
        "license": "cc-by-sa-4.0", "adapter": "beir",
        "source_url": "https://huggingface.co/datasets/BeIR/scifact",
    },
    "squad": {
        "repo": "rajpurkar/squad", "config": "plain_text", "split": "train",
        "license": "cc-by-sa-4.0", "adapter": "squad",
        "source_url": "https://huggingface.co/datasets/rajpurkar/squad",
    },
}


@dataclass
class ImportResult:
    manifest: dict[str, Any]
    documents: list[Path]
    labels: Path | None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:80] or "document"


def _read_json(source: str | Path) -> tuple[dict, str]:
    if str(source).startswith(("http://", "https://")):
        request = urllib.request.Request(str(source), headers={"User-Agent": "kairos-corpus-import/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
    else:
        raw = Path(source).read_bytes()
    return json.loads(raw), _sha256(raw)


def dataset_viewer_url(repo: str, config: str, split: str, length: int) -> str:
    query = urllib.parse.urlencode({
        "dataset": repo, "config": config, "split": split, "offset": 0,
        "length": min(max(length, 1), 100),
    })
    return f"https://datasets-server.huggingface.co/rows?{query}"


class CorpusAdapter:
    namespace = "DATASET"

    def __init__(self, spec: dict[str, str]) -> None:
        self.spec = spec

    def records_and_checksum(self, source: str | Path | None, fetch_count: int) -> tuple[list[dict], str, str]:
        source_ref = str(source) if source else dataset_viewer_url(
            self.spec["repo"], self.spec["config"], self.spec["split"], fetch_count
        )
        payload, checksum = _read_json(source_ref)
        rows = payload.get("rows", payload if isinstance(payload, list) else [])
        records = [item.get("row", item) for item in rows]
        if not isinstance(records, list):
            raise ValueError("dataset source must contain a list or Dataset Viewer rows")
        return records, checksum, source_ref

    def validate(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        raise NotImplementedError

    def transform(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        raise NotImplementedError

    def export(self, records: list[dict], destination: Path, *, revision: str, source_checksum: str,
               source_ref: str, limit: int, seed: int, dry_run: bool = False) -> ImportResult:
        valid, rejected = self.validate(records)
        rng = random.Random(seed)
        order = list(range(len(valid)))
        rng.shuffle(order)
        selected = [valid[i] for i in order[:limit]]
        documents, labels = self.transform(selected)
        documents.sort(key=lambda d: d["doc_id"])
        document_entries = [{
            "doc_id": document["doc_id"],
            "source_record_id": document["source_record_id"],
            "content_checksum_sha256": _sha256(document["body"].encode("utf-8")),
        } for document in documents]
        manifest = {
            "schema_version": 1,
            "dataset": self.spec["repo"],
            "adapter": self.spec["adapter"],
            "namespace": self.namespace,
            "config": self.spec["config"],
            "split": self.spec["split"],
            "revision": revision,
            "license": self.spec["license"],
            "source_url": self.spec["source_url"],
            "source_ref": source_ref,
            "source_checksum_sha256": source_checksum,
            "seed": seed,
            "requested_limit": limit,
            "input_records": len(records),
            "valid_records": len(valid),
            "rejected_records": len(rejected),
            "exported_documents": len(documents),
            "evaluation_labels": len(labels),
            "transformation": "source passage to Markdown; questions/answers/qrels remain separate labels",
            "rejected": rejected[:20],
            "documents": document_entries,
        }
        paths: list[Path] = []
        labels_path = destination / "evaluation-labels.jsonl" if labels else None
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            for doc in documents:
                body = doc["body"]
                meta = {
                    **{key: value for key, value in doc.items() if key != "body"},
                    "source_dataset": self.spec["repo"],
                    "source_url": self.spec["source_url"],
                    "license": self.spec["license"],
                    "dataset_revision": revision,
                    "content_checksum": _sha256(body.encode("utf-8")),
                    "transformation": "verbatim source passage wrapped in Markdown headings",
                }
                path = destination / f"{_safe(meta['doc_id'])}.md"
                front = yaml.safe_dump(meta, sort_keys=True, allow_unicode=True).strip()
                path.write_text(f"---\n{front}\n---\n\n# {meta['title']}\n\n## Source passage\n\n{body.strip()}\n", encoding="utf-8")
                paths.append(path)
            if labels_path:
                labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels),
                                       encoding="utf-8")
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
        return ImportResult(manifest, paths, labels_path)

    def run(self, source: str | Path | None, destination: Path, *, limit: int, seed: int,
            revision: str, dry_run: bool = False, validate_only: bool = False,
            expected_checksum: str | None = None) -> ImportResult:
        records, checksum, source_ref = self.records_and_checksum(source, max(limit * 4, limit))
        if expected_checksum and checksum.lower() != expected_checksum.lower():
            raise ValueError(f"source checksum mismatch: expected {expected_checksum}, got {checksum}")
        if validate_only:
            valid, rejected = self.validate(records)
            return ImportResult({"valid_records": len(valid), "rejected_records": len(rejected),
                                 "rejected": rejected}, [], None)
        return self.export(records, destination, revision=revision, source_checksum=checksum,
                           source_ref=source_ref, limit=limit, seed=seed, dry_run=dry_run)


def get_adapter(name: str) -> CorpusAdapter:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"unsupported dataset {name!r}; choose from {', '.join(sorted(DATASETS))}")
    if DATASETS[key]["adapter"] == "squad":
        from kairos.corpus.adapters.squad import SquadAdapter
        return SquadAdapter(DATASETS[key])
    from kairos.corpus.adapters.beir import BeirCorpusAdapter
    return BeirCorpusAdapter(DATASETS[key])
