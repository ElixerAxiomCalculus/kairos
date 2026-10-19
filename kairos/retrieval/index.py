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
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kairos.corpus.ingest import Chunk, Corpus
from kairos.retrieval.bm25 import BM25
from kairos.retrieval.dense import build_encoder
from kairos.textutil import content_terms


DOMAIN_TERMS = {
    "corporate_events": {"venue", "workshop", "catering", "cancellation", "event", "booking", "reimbursement", "per diem"},
    "housing": {"house", "housing", "apartment", "property", "bhk", "listing", "mortgage"},
    "finance": {"finance", "financial", "interest", "loan", "compound", "investment", "bank"},
    "biomedical_research": {"fever", "cough", "medicine", "medical", "disease", "clinical", "patient", "hospital"},
    "travel": {"travel", "tourism", "flight", "hotel", "destination", "itinerary"},
    "scientific_literature": {"study", "research", "protein", "gene", "cell", "scientific", "experiment"},
    "general_knowledge": {"definition", "history", "agent", "university", "encyclopedia"},
}

# Terms that identify a broad request context but not its information objective.  Admission uses the
# remaining high-weight terms to prevent an "international event" document from satisfying a
# cancellation question merely because organization/domain vocabulary overlaps.
OBJECTIVE_STOP = {
    "policy", "rule", "option", "information", "detail", "event", "international", "standard",
    "general", "current", "today", "document", "company", "organization", "available",
}


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
    accepted: bool = True
    relevance_status: str = "RELEVANT_SUFFICIENT"
    query_coverage: float = 0.0
    objective_coverage: float = 0.0
    relevance_grade: int = 0
    relevance_reasons: list[str] = field(default_factory=list)

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
            "accepted": self.accepted,
            "relevance_status": self.relevance_status,
            "query_coverage": round(self.query_coverage, 4),
            "objective_coverage": round(self.objective_coverage, 4),
            "relevance_grade": self.relevance_grade,
            "relevance_reasons": self.relevance_reasons,
        }


@dataclass
class SearchResult:
    hits: list[Hit]
    duplicates: list[tuple[str, str]]           # (duplicate, canonical)
    filtered: list[str]                         # chunk ids removed by facet constraints
    stage_ms: dict[str, float]
    mode: str
    evidence_status: str = "NO_RELEVANT_EVIDENCE"
    domain: str = "unknown"
    domain_confidence: float = 0.0
    domain_fallback_used: bool = False
    search_id: str | None = None
    retrieval_type: str = "PRIMARY"

    @property
    def relevant_hits(self) -> list[Hit]:
        return [hit for hit in self.hits if hit.accepted]


@dataclass
class SearchExecution:
    search_id: str
    retrieval_type: str
    session_id: str | None
    intent_id: str | None
    query_version: int | None
    evidence_id: str | None
    execution_started: float
    execution_completed: float | None = None
    execution_canceled_before_start: bool = False
    result_discarded: bool = False
    cache_reused: bool = False
    failed: bool = False
    error_type: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "retrieval_type": self.retrieval_type,
            "session_id": self.session_id,
            "intent_id": self.intent_id,
            "query_version": self.query_version,
            "evidence_id": self.evidence_id,
            "execution_started": self.execution_started,
            "execution_completed": self.execution_completed,
            "execution_canceled_before_start": self.execution_canceled_before_start,
            "result_discarded": self.result_discarded,
            "cache_reused": self.cache_reused,
            "failed": self.failed,
            "error_type": self.error_type,
        }


class HybridIndex:
    def __init__(self, corpus: Corpus, cfg: dict) -> None:
        self.corpus = corpus
        self.cfg = cfg
        self.chunks: list[Chunk] = corpus.chunks
        self.pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        boost = int(cfg.get("heading_boost", 2))
        self.doc_terms: list[list[str]] = []
        self.heading_term_sets: list[set[str]] = []
        for c in self.chunks:
            heading = content_terms(f"{c.doc_title} {c.section_path}")
            self.heading_term_sets.append(set(heading))
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
        self._probe_cache: dict[tuple, list[tuple[str, float]]] = {}
        self.probe_cache_hits = 0
        self.probe_cache_misses = 0
        self.chunk_places = [self.term_sets[i] & corpus.places for i in range(len(self.chunks))]
        self.chunk_domains = [self._infer_chunk_domain(c, self.term_sets[i]) for i, c in enumerate(self.chunks)]
        self.available_domains = {d for d in self.chunk_domains if d != "unknown"}
        self._search_lock = threading.RLock()
        self._search_executions: dict[str, SearchExecution] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tfidf_vectors(texts: list[str]):
        """Keep TF-IDF sparse; dense materialisation dominated memory on imported corpora."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        return normalize(TfidfVectorizer(sublinear_tf=True).fit_transform(texts), norm="l2", copy=False).tocsr()

    @staticmethod
    def _infer_chunk_domain(chunk: Chunk, terms: set[str]) -> str:
        source = (chunk.source_dataset or "").lower()
        doc_type = (chunk.doc_type or "").lower()
        title = chunk.doc_title.lower()
        if "scifact" in source or "scientific" in doc_type:
            return "scientific_literature"
        if "squad" in source or "encyclopedic" in doc_type:
            return "general_knowledge"
        if not source and any(marker in title for marker in (
            "venue", "catering", "travel and expense", "event security", "av and it support"
        )):
            return "corporate_events"
        scores = {domain: len(words & terms) for domain, words in DOMAIN_TERMS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] else "unknown"

    def route_domain(self, text: str) -> tuple[str, float, bool]:
        terms = set(content_terms(text))
        scores = {domain: len(words & terms) for domain, words in DOMAIN_TERMS.items()}
        best = max(scores, key=scores.get)
        matched = scores[best]
        if not matched:
            return "unknown", 0.0, True
        confidence = min(1.0, matched / max(2, len(terms)))
        compatible = best in self.available_domains
        # Travel reimbursement belongs to the corporate-policy corpus, while live travel planning does not.
        if best == "travel" and {"reimbursement", "policy"} & terms and "corporate_events" in self.available_domains:
            return "corporate_events", confidence, True
        return best, confidence, compatible

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

    def objective_terms(self, query: RetrievalQuery) -> dict[str, float]:
        terms = {
            term: weight for term, weight in query.terms.items()
            if weight >= 0.95 and term not in OBJECTIVE_STOP and term not in self.corpus.places
        }
        if not terms:
            terms = {term: weight for term, weight in query.terms.items() if weight >= 0.95}
        return terms

    def section_of(self, chunk_id: str) -> str:
        return self.chunks[self.pos[chunk_id]].section_key

    # ------------------------------------------------------------------ probe
    def probe(self, terms: dict[str, float], k: int | None = None) -> list[tuple[str, float]]:
        """Cheap BM25-only, section-level ranking used by the controller and decomposer."""
        k = k or self.cfg["probe_k"]
        if not terms:
            return []
        cache_key = (tuple(sorted((term, round(weight, 4)) for term, weight in terms.items())), k)
        if self.cfg.get("probe_cache_enabled", True):
            cached = self._probe_cache.get(cache_key)
            if cached is not None:
                self.probe_cache_hits += 1
                return list(cached)
        self.probe_cache_misses += 1
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
        if self.cfg.get("probe_cache_enabled", True):
            if len(self._probe_cache) >= int(self.cfg.get("probe_cache_size", 4096)):
                self._probe_cache.pop(next(iter(self._probe_cache)))
            self._probe_cache[cache_key] = list(out)
        return out

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
        mode: str | None = None,
        execution_context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute one authoritative full search and record its actual work.

        Cache hits never call this method.  A coroutine cancellation after this
        method starts therefore cannot erase the recorded underlying work.
        """
        context = execution_context or {}
        search_id = str(context.get("search_id") or f"search-{uuid.uuid4().hex[:16]}")
        record = SearchExecution(
            search_id=search_id,
            retrieval_type=str(context.get("retrieval_type", "PRIMARY")),
            session_id=context.get("session_id"),
            intent_id=context.get("intent_id"),
            query_version=context.get("query_version"),
            evidence_id=context.get("evidence_id"),
            execution_started=time.perf_counter(),
        )
        with self._search_lock:
            self._search_executions[search_id] = record
        try:
            result = self._search_impl(query, top_k=top_k, mode=mode)
        except Exception as exc:
            with self._search_lock:
                record.execution_completed = time.perf_counter()
                record.failed = True
                record.error_type = type(exc).__name__
            raise
        with self._search_lock:
            record.execution_completed = time.perf_counter()
        result.search_id = search_id
        result.retrieval_type = record.retrieval_type
        return result

    def mark_search_discarded(self, search_id: str | None) -> None:
        if not search_id:
            return
        with self._search_lock:
            if search_id in self._search_executions:
                self._search_executions[search_id].result_discarded = True

    def search_executions(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._search_lock:
            records = list(self._search_executions.values())
            if session_id is not None:
                records = [record for record in records if record.session_id == session_id]
            return [record.public() for record in records]

    def search_cost_summary(self, *, session_id: str | None = None) -> dict[str, Any]:
        records = self.search_executions(session_id=session_id)
        types: dict[str, int] = {}
        for record in records:
            key = str(record["retrieval_type"])
            types[key] = types.get(key, 0) + 1
        durations = [
            (float(record["execution_completed"]) - float(record["execution_started"])) * 1000
            for record in records if record["execution_completed"] is not None
        ]
        return {
            "actual_hybrid_search_executions": len(records),
            "by_retrieval_type": types,
            "completed": sum(record["execution_completed"] is not None for record in records),
            "failed": sum(bool(record["failed"]) for record in records),
            "completed_but_discarded": sum(bool(record["result_discarded"]) for record in records),
            "execution_wall_ms_total": round(sum(durations), 3),
            "executions": records,
        }

    def _search_impl(self, query: RetrievalQuery, top_k: int | None = None, mode: str | None = None) -> SearchResult:
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
        # A place inherited through low-weight anchor context must not become a hard entity filter
        # for a later, independently worded request (for example cancellation policy after Pune venue).
        query_places = {term for term, weight in query.terms.items() if weight >= 0.9} & self.corpus.places
        filtered: list[str] = []
        scored: list[tuple[float, int, bool | None, float]] = []
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
            if any(weight >= 0.95 and term in self.heading_term_sets[i] for term, weight in query.terms.items()):
                rel += float(cfg.get("heading_focus_bonus", 0.12))
            if query_places and self.chunk_places[i] and not (query_places & self.chunk_places[i]):
                rel *= cfg.get("place_mismatch_penalty", 0.5)  # names only a different place than asked
            scored.append((rel, i, match, self.coverage(query.terms, i)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        stage["rerank_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits: list[Hit] = []
        duplicates: list[tuple[str, str]] = []
        domain, domain_confidence, domain_available = self.route_domain(query.text)
        query_negative = bool(re.search(
            r"\b(?:not|non[- ]?reimbursable|excluded|ineligible|without)\b", query.text, re.I
        ))
        gate = bool(cfg.get("relevance_gate", True))
        objective_terms = self.objective_terms(query)
        enforce_domain = cfg.get("domain_routing", True) and domain_confidence >= float(
            cfg.get("domain_routing_min_confidence", 0.25)
        )
        for rel, i, match, coverage in scored:
            dup_of = None
            for h in hits:
                j = self.pos[h.chunk_id]
                similarity = float(self._dedup_vectors[i].multiply(self._dedup_vectors[j]).sum())
                if similarity >= cfg["dedup_threshold"]:
                    dup_of = h.chunk_id
                    break
            if dup_of:
                duplicates.append((self.chunks[i].chunk_id, dup_of))
                continue
            focus_present = sum(1 for term, weight in query.terms.items()
                                if weight >= 0.95 and term in self.term_sets[i])
            objective_coverage = self.coverage(objective_terms, i)
            reasons: list[str] = []
            if not domain_available and domain != "unknown":
                reasons.append("DOMAIN_NOT_AVAILABLE")
            elif enforce_domain and domain != "unknown" and self.chunk_domains[i] not in {domain, "unknown"}:
                reasons.append("DOMAIN_MISMATCH")
            if coverage < float(cfg.get("relevance_min_coverage", 0.16)):
                reasons.append("LOW_IDF_COVERAGE")
            if rel < float(cfg.get("relevance_min_rerank", 0.28)):
                reasons.append("LOW_RERANK_SCORE")
            if float(dense[i]) < float(cfg.get("relevance_min_dense", 0.08)) and sparse[i] <= 0:
                reasons.append("NO_SPARSE_OR_DENSE_SUPPORT")
            if focus_present < int(cfg.get("relevance_min_focus_terms", 1)):
                reasons.append("NO_FOCUS_TERM")
            if objective_terms and objective_coverage < float(cfg.get("relevance_min_objective_coverage", 0.14)):
                reasons.append("LOW_OBJECTIVE_COVERAGE")
            if query_places and not (query_places & self.chunk_places[i]):
                reasons.append("QUERY_ENTITY_NOT_IN_EVIDENCE")
            if query_negative and not re.search(
                r"\b(?:not|non[- ]?reimbursable|excluded|ineligible|without)\b", self.chunks[i].index_text,
                re.I
            ):
                reasons.append("NEGATION_SCOPE_MISMATCH")
            accepted = not gate or not reasons
            grade = 3 if accepted and coverage >= 0.35 and objective_coverage >= 0.35 else (
                2 if accepted and objective_coverage >= float(cfg.get("relevance_min_objective_coverage", 0.14))
                else 1 if coverage >= float(cfg.get("relevance_min_coverage", 0.16)) else 0
            )
            relevance_status = "RELEVANT_SUFFICIENT" if grade == 3 else (
                "RELEVANT_PARTIAL" if accepted else "OUT_OF_DOMAIN" if "DOMAIN_NOT_AVAILABLE" in reasons
                else "NO_RELEVANT_EVIDENCE"
            )
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
                accepted=accepted,
                relevance_status=relevance_status,
                query_coverage=coverage,
                objective_coverage=objective_coverage,
                relevance_grade=grade,
                relevance_reasons=reasons,
            ))
            if len(hits) >= top_k:
                break
        stage["dedup_ms"] = (time.perf_counter() - t) * 1000
        relevant = [hit for hit in hits if hit.accepted]
        if relevant:
            evidence_status = "RELEVANT_SUFFICIENT" if relevant[0].relevance_status == "RELEVANT_SUFFICIENT" else "RELEVANT_PARTIAL"
        elif domain != "unknown" and not domain_available:
            evidence_status = "OUT_OF_DOMAIN"
        else:
            evidence_status = "NO_RELEVANT_EVIDENCE"
        fallback_used = bool(cfg.get("domain_routing", True) and domain != "unknown" and not enforce_domain)
        return SearchResult(hits, duplicates, filtered, stage, mode, evidence_status, domain, domain_confidence,
                            fallback_used)

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
            "corpus_profiles": sorted({c.corpus_profile for c in self.chunks if c.corpus_profile}),
            "available_domains": sorted(self.available_domains),
            "probe_cache": {"hits": self.probe_cache_hits, "misses": self.probe_cache_misses,
                            "entries": len(self._probe_cache)},
        }
