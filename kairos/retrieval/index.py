"""Hybrid index: BM25 + dense (exact inner product) → RRF → rerank → facet filter → dedup.

Every stage exists for a measured reason (see ablation B):
* BM25 catches exact policy vocabulary ("deposit", "per diem", section names, numbers).
* Dense catches paraphrase ("call off" ~ "cancellation").
* RRF fuses without score calibration.
* The reranker produces an *absolute* relevance in [0, 1] that the synthesizer uses as a
  sufficiency signal (answer vs. "insufficient evidence").
* Probes are BM25-only and section-level: they are cheap enough to run on every chunk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from kairos.corpus.ingest import Chunk, Corpus
from kairos.retrieval.bm25 import BM25
from kairos.retrieval.dense import build_encoder
from kairos.textutil import content_terms


@dataclass
class RetrievalQuery:
    text: str                                   # human-readable standalone query
    terms: dict[str, float]                     # BM25 term weights
    dense_text: str
    facet_constraints: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def build(need: str, anchor: str = "", constraints: list[str] | None = None,
              anchor_weight: float = 0.4, facet_constraints: dict[str, str] | None = None,
              display: str | None = None) -> "RetrievalQuery":
        terms: dict[str, float] = {}
        for t in content_terms(need):
            terms[t] = 1.0
        for c in constraints or []:
            for t in content_terms(c):
                terms[t] = max(terms.get(t, 0.0), 1.0)
        for t in content_terms(anchor):
            terms.setdefault(t, anchor_weight)
        dense_text = " ".join(x for x in [need, " ".join(constraints or []), anchor] if x)
        return RetrievalQuery(display or dense_text, terms, dense_text, dict(facet_constraints or {}))


@dataclass
class Hit:
    chunk_id: str
    rank: int
    sparse_score: float
    sparse_rank: int | None
    dense_score: float
    dense_rank: int | None
    fused_score: float
    rerank_score: float
    facet_match: bool | None
    duplicate_of: str | None = None

    def to_public(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "sparse_score": round(self.sparse_score, 4),
            "sparse_rank": self.sparse_rank,
            "dense_score": round(self.dense_score, 4),
            "dense_rank": self.dense_rank,
            "fused_score": round(self.fused_score, 5),
            "rerank_score": round(self.rerank_score, 4),
            "facet_match": self.facet_match,
        }


@dataclass
class SearchResult:
    hits: list[Hit]
    duplicates: list[tuple[str, str]]           # (duplicate, canonical)
    filtered: list[str]                         # chunk ids removed by facet constraints
    stage_ms: dict[str, float]
    mode: str


class HybridIndex:
    def __init__(self, corpus: Corpus, cfg: dict) -> None:
        self.corpus = corpus
        self.cfg = cfg
        self.chunks: list[Chunk] = corpus.chunks
        self.pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        boost = int(cfg.get("heading_boost", 2))
        self.doc_terms: list[list[str]] = []
        for c in self.chunks:
            heading = content_terms(f"{c.doc_title} {c.section_path}")
            self.doc_terms.append(heading * boost + content_terms(c.text))
        self.term_sets = [set(t) for t in self.doc_terms]
        self.bm25 = BM25(self.doc_terms, k1=cfg["bm25_k1"], b=cfg["bm25_b"])
        texts = [c.index_text for c in self.chunks]
        t0 = time.perf_counter()
        self.encoder = build_encoder(cfg, texts)
        self.vectors = self.encoder.encode(texts)
        self.build_ms = (time.perf_counter() - t0) * 1000
        self._dedup_vectors = self._tfidf_vectors([c.text for c in self.chunks])
        self.cross_encoder = None
        if cfg.get("reranker") == "cross-encoder":
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(cfg["cross_encoder_model"])
        self._encode_cache: dict[str, np.ndarray] = {}
        self.chunk_places = [self.term_sets[i] & corpus.places for i in range(len(self.chunks))]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tfidf_vectors(texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        mat = TfidfVectorizer(sublinear_tf=True).fit_transform(texts).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def encode(self, text: str) -> np.ndarray:
        vec = self._encode_cache.get(text)
        if vec is None:
            vec = self.encoder.encode([text])[0]
            if len(self._encode_cache) > 4096:
                self._encode_cache.clear()
            self._encode_cache[text] = vec
        return vec

    def cosine(self, a: str, b: str) -> float:
        return float(np.dot(self.encode(a), self.encode(b)))

    def coverage(self, terms: dict[str, float], idx: int) -> float:
        """IDF-weighted fraction of the query's term mass present in the chunk."""
        total = sum(self.bm25.idf_of(t) * w for t, w in terms.items())
        if total <= 0:
            return 0.0
        hit = sum(self.bm25.idf_of(t) * w for t, w in terms.items() if t in self.term_sets[idx])
        return hit / total

    def section_of(self, chunk_id: str) -> str:
        return self.chunks[self.pos[chunk_id]].section_key

    # ------------------------------------------------------------------ probe
    def probe(self, terms: dict[str, float], k: int | None = None) -> list[tuple[str, float]]:
        """Cheap BM25-only, section-level ranking used by the controller and decomposer."""
        k = k or self.cfg["probe_k"]
        if not terms:
            return []
        scores = self.bm25.scores(terms)
        order = np.argsort(-scores, kind="stable")
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for i in order:
            if scores[i] <= 0:
                break
            key = self.chunks[i].section_key
            if key in seen:
                continue
            seen.add(key)
            out.append((key, float(scores[i])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ search
    def search(self, query: RetrievalQuery, top_k: int | None = None, mode: str | None = None) -> SearchResult:
        cfg = self.cfg
        top_k = top_k or cfg["top_k"]
        mode = mode or cfg["mode"]
        depth = cfg["candidates"]
        stage: dict[str, float] = {}

        t = time.perf_counter()
        sparse = self.bm25.scores(query.terms) if mode in ("hybrid", "sparse") else np.zeros(len(self.chunks))
        sparse_order = [i for i in np.argsort(-sparse, kind="stable")[:depth] if sparse[i] > 0]
        stage["sparse_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense = self.vectors @ self.encode(query.dense_text)
        dense_order = list(np.argsort(-dense, kind="stable")[:depth]) if mode in ("hybrid", "dense") else []
        stage["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        k = cfg["rrf_k"]
        fused: dict[int, float] = {}
        sparse_rank = {i: r + 1 for r, i in enumerate(sparse_order)}
        dense_rank = {i: r + 1 for r, i in enumerate(dense_order)}
        for ranks in (sparse_rank, dense_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
        stage["fusion_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        candidates = sorted(fused, key=lambda i: (-fused[i], i))
        query_places = set(query.terms) & self.corpus.places
        filtered: list[str] = []
        scored: list[tuple[float, int, bool | None]] = []
        ce_scores = None
        if self.cross_encoder is not None and candidates:
            pairs = [(query.text, self.chunks[i].index_text) for i in candidates]
            raw = np.asarray(self.cross_encoder.predict(pairs))
            ce_scores = 1 / (1 + np.exp(-raw))
        for n, i in enumerate(candidates):
            chunk = self.chunks[i]
            match: bool | None = None
            reject = False
            for group, value in query.facet_constraints.items():
                have = chunk.facets.get(group)
                if have is None:
                    continue
                if have != value:
                    reject = True
                else:
                    match = True
            if reject:
                filtered.append(chunk.chunk_id)
                continue
            if cfg["reranker"] == "none":
                rel = fused[i] * k / 2  # ≈ 1.0 when ranked first by both retrievers
            elif ce_scores is not None:
                rel = float(ce_scores[n])
            else:
                rel = 0.6 * self.coverage(query.terms, i) + 0.4 * max(float(dense[i]), 0.0)
            if match:
                rel += cfg["facet_bonus"]
            if query_places and self.chunk_places[i] and not (query_places & self.chunk_places[i]):
                rel *= cfg.get("place_mismatch_penalty", 0.5)  # names only a different place than asked
            scored.append((rel, i, match))
        scored.sort(key=lambda x: (-x[0], x[1]))
        stage["rerank_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits: list[Hit] = []
        duplicates: list[tuple[str, str]] = []
        for rel, i, match in scored:
            dup_of = None
            for h in hits:
                j = self.pos[h.chunk_id]
                if float(self._dedup_vectors[i] @ self._dedup_vectors[j]) >= cfg["dedup_threshold"]:
                    dup_of = h.chunk_id
                    break
            if dup_of:
                duplicates.append((self.chunks[i].chunk_id, dup_of))
                continue
            hits.append(Hit(
                chunk_id=self.chunks[i].chunk_id,
                rank=len(hits) + 1,
                sparse_score=float(sparse[i]),
                sparse_rank=sparse_rank.get(i),
                dense_score=float(dense[i]),
                dense_rank=dense_rank.get(i),
                fused_score=fused[i],
                rerank_score=min(rel, 1.0),
                facet_match=match,
            ))
            if len(hits) >= top_k:
                break
        stage["dedup_ms"] = (time.perf_counter() - t) * 1000
        return SearchResult(hits, duplicates, filtered, stage, mode)

    def status(self) -> dict:
        return {
            "documents": len(self.corpus.documents),
            "chunks": len(self.chunks),
            "dense_backend": self.encoder.name,
            "embedding_model": self.encoder.model_id,
            "sparse_index": "bm25 (k1={}, b={})".format(self.cfg["bm25_k1"], self.cfg["bm25_b"]),
            "reranker": self.cfg["reranker"] if self.cross_encoder is None else self.cfg["cross_encoder_model"],
            "facet_dimensions": {g: sorted(m) for g, m in self.corpus.contrast_groups.items()},
            "build_ms": round(self.build_ms, 1),
        }
