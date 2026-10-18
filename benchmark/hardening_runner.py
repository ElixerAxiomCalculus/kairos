"""V2.1 scale, concurrency and soak measurements over official SciFact corpus rows."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import tracemalloc
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from benchmark.runner import report_provenance
from kairos.config import Config, resolve_path
from kairos.corpus.ingest import Chunk, Corpus
from kairos.engine import EngineContext, SessionEngine
from kairos.replay import load_script, run_script
from kairos.retrieval.index import HybridIndex, RetrievalQuery


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return None
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError, AttributeError):
        return None


def _fetch_scifact(limit: int) -> tuple[list[dict], str]:
    """Read the immutable official Parquet artifact; use the row API only as a fallback."""
    parquet_cache = resolve_path("runs/tmp/scifact-corpus-b3b5335.parquet")
    parquet_url = (
        "https://huggingface.co/datasets/BeIR/scifact/resolve/"
        "b3b5335604bf5ee3c4447671af975ea25143d4f5/corpus/corpus-00000-of-00001.parquet"
    )
    if not parquet_cache.exists():
        parquet_cache.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(parquet_url, headers={"User-Agent": "kairos-scale/1.0"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    parquet_cache.write_bytes(response.read())
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    break
                time.sleep(2 ** attempt)
    if parquet_cache.exists():
        import pandas as pd

        raw = parquet_cache.read_bytes()
        frame = pd.read_parquet(parquet_cache)
        return frame.head(limit).to_dict(orient="records"), hashlib.sha256(raw).hexdigest()

    cache = resolve_path("runs/tmp/scifact-scale-rows.json")
    rows: list[dict] = []
    page_hashes: list[str] = []
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if len(data.get("rows", [])) >= limit:
            return data["rows"][:limit], data["response_pages_sha256"]
        rows = data.get("rows", [])
        page_hashes = data.get("page_sha256", [])
    while len(rows) < limit:
        query = urllib.parse.urlencode({
            "dataset": "BeIR/scifact", "config": "corpus", "split": "corpus",
            "revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
            "offset": len(rows), "length": min(100, limit - len(rows)),
        })
        request = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{query}",
                                         headers={"User-Agent": "kairos-scale/1.0"})
        raw = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        assert raw is not None
        page_hashes.append(hashlib.sha256(raw).hexdigest())
        payload = json.loads(raw)
        page = [item.get("row", item) for item in payload.get("rows", [])]
        if not page:
            break
        rows.extend(page)
        aggregate = hashlib.sha256("\n".join(page_hashes).encode("ascii")).hexdigest()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"rows": rows, "page_sha256": page_hashes,
                                     "response_pages_sha256": aggregate}), encoding="utf-8")
    aggregate = hashlib.sha256("\n".join(page_hashes).encode("ascii")).hexdigest()
    return rows[:limit], aggregate


def _corpus(records: list[dict], profile: str) -> Corpus:
    chunks: list[Chunk] = []
    documents = {}
    for record in records:
        source_id = str(record.get("_id") or record.get("id"))
        doc_id = f"SCIFACT:{source_id}"
        title = str(record.get("title") or f"SciFact article {source_id}").strip()
        words = str(record.get("text") or "").split()
        parts = [" ".join(words[i:i + 110]) for i in range(0, len(words), 110)] or [""]
        documents[doc_id] = {"doc_id": doc_id, "title": title, "chunks": len(parts),
                             "source_dataset": "BeIR/scifact", "dataset_revision":
                             "b3b5335604bf5ee3c4447671af975ea25143d4f5"}
        for ordinal, text in enumerate(parts, 1):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:C{ordinal:02d}", doc_id=doc_id, doc_title=title,
                section_key=f"{doc_id}#abstract", section_number="abstract", section_title="Abstract",
                section_path="Abstract", parent_key=None, text=text,
                sha1=hashlib.sha1(text.encode()).hexdigest(), sha256=hashlib.sha256(text.encode()).hexdigest(),
                facets={}, effective_date=None, doc_type="scientific_abstract", ordinal=ordinal,
                source_dataset="BeIR/scifact", source_record_id=source_id,
                source_url="https://huggingface.co/datasets/BeIR/scifact", license="cc-by-sa-4.0",
                dataset_revision="b3b5335604bf5ee3c4447671af975ea25143d4f5", corpus_profile=profile,
            ))
    return Corpus(chunks, documents, {}, {}, set(), set(), {}, set())


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(percentile * len(ordered)))], 3)


def _scale(cfg: Config, records: list[dict], sizes: list[int]) -> list[dict]:
    rows = []
    queries = ["podocytes injury motility", "murine sepsis vascular leakage", "protein gene expression",
               "clinical trial patient outcome", "cell signaling inflammation"]
    for size in sizes:
        gc.collect()
        rss_before = _rss_bytes()
        corpus = _corpus(records[:size], f"scifact-scale-{size}")
        tracemalloc.start()
        start = time.perf_counter()
        index = HybridIndex(corpus, cfg["retrieval"])
        build_ms = (time.perf_counter() - start) * 1000
        _current, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _rss_bytes()
        latencies = []
        stages: dict[str, list[float]] = {}
        for query in queries * 4:
            started = time.perf_counter()
            result = index.search(RetrievalQuery.build(query), top_k=10)
            latencies.append((time.perf_counter() - started) * 1000)
            for key, value in result.stage_ms.items():
                stages.setdefault(key, []).append(value)
        avg_words = statistics.mean(len(chunk.text.split()) for chunk in corpus.chunks)
        rows.append({
            "documents": len(corpus.documents), "chunks": len(corpus.chunks),
            "average_chunk_words": round(avg_words, 2), "vocabulary_size": len(index.bm25.idf),
            "index_build_ms": round(build_ms, 3), "process_rss_before_bytes": rss_before,
            "process_rss_after_build_bytes": rss_after,
            "process_rss_delta_bytes": None if rss_before is None or rss_after is None else rss_after - rss_before,
            "python_traced_peak_bytes": python_peak, "index_disk_size_bytes": None,
            "index_persistence_status": "NOT_IMPLEMENTED",
            "bm25_ms_mean": round(statistics.mean(stages.get("sparse_ms", [0])), 3),
            "dense_ms_mean": round(statistics.mean(stages.get("dense_ms", [0])), 3),
            "fusion_ms_mean": round(statistics.mean(stages.get("fusion_ms", [0])), 3),
            "rerank_ms_mean": round(statistics.mean(stages.get("rerank_ms", [0])), 3),
            "dedup_ms_mean": round(statistics.mean(stages.get("dedup_ms", [0])), 3),
            "search_p50_ms": _percentile(latencies, 0.50), "search_p95_ms": _percentile(latencies, 0.95),
            "search_p99_ms": _percentile(latencies, 0.99),
            "single_session_qps": round(1000 / statistics.mean(latencies), 2),
        })
        del index, corpus
    return rows


async def _concurrent_arm(ctx: EngineContext, sessions: int) -> dict:
    rss_before = _rss_bytes()
    engines = [SessionEngine(ctx, f"load-{sessions}-{index}") for index in range(sessions)]
    start = time.perf_counter()
    results = await asyncio.gather(*(run_script(engine, load_script("pune-demo")) for engine in engines))
    elapsed = time.perf_counter() - start
    rss_after = _rss_bytes()
    ids = [{event.session_id for event in events} for events in results]
    isolated = all(value == {engines[index].id} for index, value in enumerate(ids))
    return {
        "sessions": sessions, "wall_ms": round(elapsed * 1000, 3),
        "sessions_per_second": round(sessions / elapsed, 3), "session_isolation": isolated,
        "events": sum(len(events) for events in results), "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_growth_bytes": None if rss_before is None or rss_after is None else rss_after - rss_before,
    }


def run_hardening_benchmark(cfg: Config, *, max_documents: int = 5000, write: bool = True) -> dict:
    sizes = [size for size in (100, 500, 1000, 5000) if size <= max_documents]
    records, response_checksum = _fetch_scifact(max(sizes))
    scale = _scale(cfg, records, sizes)
    sample = EngineContext(cfg)
    concurrency = [asyncio.run(_concurrent_arm(sample, sessions)) for sessions in (1, 2, 4, 8)]
    soak_start = _rss_bytes()
    soak_started = time.perf_counter()
    for _ in range(10):
        asyncio.run(_concurrent_arm(sample, 4))
    soak_end = _rss_bytes()
    soak_duration = time.perf_counter() - soak_started
    report = {
        "status": "MEASURED", "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "provenance": report_provenance(cfg, sample, seed=42, repeats=1),
        "dataset": {"name": "BeIR/scifact", "revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
                    "license": "cc-by-sa-4.0", "source_artifact_sha256": response_checksum,
                    "source_artifact": "corpus/corpus-00000-of-00001.parquet",
                    "selection": "first N official corpus rows; no duplicated or synthetic documents"},
        "scale": scale,
        "concurrency": concurrency,
        "soak": {"rounds": 10, "sessions_per_round": 4,
                 "duration_s": round(soak_duration, 3),
                 "rss_start_bytes": soak_start, "rss_end_bytes": soak_end,
                 "rss_growth_bytes": None if soak_start is None or soak_end is None else soak_end - soak_start,
                 "qualification": "short automated soak, not a long-duration production leak test"},
    }
    if write:
        out = resolve_path(cfg["bench"]["results_dir"]) / "hardening"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{report['run_id']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
