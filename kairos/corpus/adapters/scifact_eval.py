"""Official BEIR SciFact query/qrel import kept strictly outside retrievable documents."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def _viewer_rows(repo: str, config: str, split: str) -> tuple[list[dict], str, list[str]]:
    rows: list[dict] = []
    checksums: list[str] = []
    urls: list[str] = []
    offset = 0
    total = None
    while total is None or offset < total:
        query = urllib.parse.urlencode({
            "dataset": repo, "config": config, "split": split, "offset": offset, "length": 100,
        })
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "kairos-scifact-eval/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        payload = json.loads(raw)
        page = [item.get("row", item) for item in payload.get("rows", [])]
        rows.extend(page)
        checksums.append(hashlib.sha256(raw).hexdigest())
        urls.append(url)
        total = int(payload.get("num_rows_total", len(rows)))
        if not page:
            break
        offset += len(page)
    aggregate = hashlib.sha256("\n".join(checksums).encode("ascii")).hexdigest()
    return rows, aggregate, urls


def build_labels(qrels: list[dict], queries: list[dict], corpus_source_ids: set[str]) -> list[dict]:
    query_text = {
        str(row.get("_id") or row.get("id")): str(row.get("text") or "").strip()
        for row in queries
    }
    labels = []
    for row in qrels:
        query_id = str(row.get("query-id") or row.get("query_id") or "").strip()
        corpus_id = str(row.get("corpus-id") or row.get("corpus_id") or "").strip()
        score = int(row.get("score", 0))
        if corpus_id not in corpus_source_ids or score <= 0 or not query_text.get(query_id):
            continue
        labels.append({
            "query_id": query_id,
            "query": query_text[query_id],
            "document_id": f"SCIFACT:{corpus_id}",
            "relevance": score,
            "split": "test",
        })
    return sorted(labels, key=lambda row: (int(row["query_id"]), row["document_id"]))


def import_scifact_evaluation(destination: Path, *, corpus_revision: str, qrels_revision: str) -> dict:
    corpus_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    source_ids = {str(row["source_record_id"]) for row in corpus_manifest["documents"]}
    qrels, qrels_checksum, qrels_urls = _viewer_rows("BeIR/scifact-qrels", "default", "test")
    queries, query_checksum, query_urls = _viewer_rows("BeIR/scifact", "queries", "queries")
    labels = build_labels(qrels, queries, source_ids)
    labels_path = destination / "scifact-qrels-test.jsonl"
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": "BeIR/scifact",
        "qrels_dataset": "BeIR/scifact-qrels",
        "license": "cc-by-sa-4.0",
        "corpus_revision": corpus_revision,
        "qrels_revision": qrels_revision,
        "split": "test",
        "source": "Hugging Face Dataset Viewer API (paginated, read-only)",
        "query_response_pages_sha256": query_checksum,
        "qrels_response_pages_sha256": qrels_checksum,
        "query_pages": len(query_urls),
        "qrels_pages": len(qrels_urls),
        "indexed_subset_documents": len(source_ids),
        "matched_qrels": len(labels),
        "label_file_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "leakage_guard": "queries and qrels are evaluation labels and are not indexed",
    }
    (destination / "evaluation-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
