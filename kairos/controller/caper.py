"""CAPER: bounded anticipatory retrieval with verified evidence commitment.

CAPER deliberately separates *preparing* evidence from *committing* it.  This
module owns only the preparation lifecycle.  A prepared result can be consumed
by :class:`SessionEngine` only after the current intent, constraints, corpus,
source checksums, and expiry have been revalidated.  Claims still pass through
the ordinary Grounding Firewall before a record can become ``COMMITTED``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from kairos.controller.evidence_cache import EvidenceCache, canonical_objective
from kairos.retrieval.index import HybridIndex, RetrievalQuery, SearchResult
from kairos.textutil import content_terms


class EvidenceState(StrEnum):
    ABSENT = "ABSENT"
    PROBE_READY = "PROBE_READY"
    PREFETCH_QUEUED = "PREFETCH_QUEUED"
    PREFETCH_RUNNING = "PREFETCH_RUNNING"
    PROVISIONAL = "PROVISIONAL"
    VALIDATION_PENDING = "VALIDATION_PENDING"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    STALE = "STALE"
    INVALIDATED = "INVALIDATED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


_TRANSITIONS: dict[EvidenceState, set[EvidenceState]] = {
    EvidenceState.ABSENT: {EvidenceState.PROBE_READY},
    EvidenceState.PROBE_READY: {EvidenceState.PREFETCH_QUEUED, EvidenceState.INVALIDATED},
    EvidenceState.PREFETCH_QUEUED: {
        EvidenceState.PREFETCH_RUNNING, EvidenceState.INVALIDATED, EvidenceState.FAILED,
    },
    EvidenceState.PREFETCH_RUNNING: {
        EvidenceState.PROVISIONAL, EvidenceState.INVALIDATED, EvidenceState.FAILED,
    },
    EvidenceState.PROVISIONAL: {
        EvidenceState.VALIDATION_PENDING, EvidenceState.STALE, EvidenceState.INVALIDATED,
        EvidenceState.EXPIRED,
    },
    EvidenceState.VALIDATION_PENDING: {
        EvidenceState.VALIDATED, EvidenceState.INVALIDATED, EvidenceState.STALE,
        EvidenceState.EXPIRED,
    },
    EvidenceState.VALIDATED: {
        EvidenceState.COMMITTED, EvidenceState.STALE, EvidenceState.INVALIDATED,
        EvidenceState.EXPIRED,
    },
    EvidenceState.COMMITTED: {EvidenceState.STALE, EvidenceState.INVALIDATED},
    EvidenceState.STALE: {EvidenceState.VALIDATION_PENDING, EvidenceState.INVALIDATED},
    EvidenceState.INVALIDATED: set(),
    EvidenceState.FAILED: set(),
    EvidenceState.EXPIRED: set(),
}


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def constraint_fingerprint(typed_constraints: list[dict[str, Any]]) -> str:
    return _hash(sorted(typed_constraints, key=lambda item: json.dumps(item, sort_keys=True)))


def semantic_query_fingerprint(
    query: RetrievalQuery,
    typed_constraints: list[dict[str, Any]],
    corpus_version: str,
    retrieval_config_hash: str,
) -> str:
    return _hash({
        "objective": canonical_objective(query.text),
        "terms": sorted((key, round(float(value), 6)) for key, value in query.terms.items()),
        "facets": dict(sorted(query.facet_constraints.items())),
        "constraints": sorted(typed_constraints, key=lambda item: json.dumps(item, sort_keys=True)),
        "corpus_version": corpus_version,
        "retrieval_config_hash": retrieval_config_hash,
    })


@dataclass
class PreparedEvidence:
    evidence_id: str
    session_id: str
    utterance_id: str
    intent_id: str
    intent_version: int
    query_version: int
    query_fingerprint: str
    semantic_fingerprint: str
    corpus_version: str
    retrieval_config_hash: str
    constraint_fingerprint: str
    state: EvidenceState
    created_at: float
    expires_at: float
    parent_event_id: str | None = None
    task_id: str | None = None
    source_chunk_ids: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    completed_at: float | None = None
    validated_at: float | None = None
    committed_at: float | None = None
    validation_record: dict[str, Any] = field(default_factory=dict)
    query_terms: dict[str, float] = field(default_factory=dict)
    typed_constraints: list[dict[str, Any]] = field(default_factory=list)
    result: SearchResult | None = field(default=None, repr=False)
    superseded: bool = False
    search_started: bool = False
    consumed: bool = False

    def transition(self, target: EvidenceState) -> None:
        if target == self.state:
            return
        if target not in _TRANSITIONS[self.state]:
            raise ValueError(f"invalid CAPER evidence transition {self.state.value}->{target.value}")
        self.state = target

    def public(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "session_id": self.session_id,
            "utterance_id": self.utterance_id,
            "intent_id": self.intent_id,
            "intent_version": self.intent_version,
            "query_version": self.query_version,
            "query_fingerprint": self.query_fingerprint,
            "semantic_fingerprint": self.semantic_fingerprint,
            "corpus_version": self.corpus_version,
            "retrieval_config_hash": self.retrieval_config_hash,
            "constraint_fingerprint": self.constraint_fingerprint,
            "state": self.state.value,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_hashes": dict(self.source_hashes),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "validated_at": self.validated_at,
            "committed_at": self.committed_at,
            "expires_at": self.expires_at,
            "validation_record": dict(self.validation_record),
            "search_started": self.search_started,
            "consumed": self.consumed,
        }


@dataclass(frozen=True)
class PrefetchDecision:
    action: str  # PREFETCH | WAIT | REUSE | SUPERSEDE
    reasons: tuple[str, ...]
    semantic_fingerprint: str
    constraint_fingerprint: str
    expected_usefulness: float


class CaperManager:
    """Session-local CAPER scheduler and lifecycle registry.

    Blocking hybrid search runs in a bounded worker semaphore.  Running worker
    calls are never reported as canceled: superseded work is allowed to finish
    and is then counted as completed-but-discarded.
    """

    def __init__(
        self,
        *,
        session_id: str,
        index: HybridIndex,
        cache: EvidenceCache,
        corpus_version: str,
        cfg: dict[str, Any],
        emit: Callable[..., Any],
    ) -> None:
        self.session_id = session_id
        self.index = index
        self.cache = cache
        self.corpus_version = corpus_version
        self.cfg = cfg
        self.emit = emit
        self.enabled = bool(cfg.get("enabled", False))
        self.policy = str(cfg.get("policy", "deterministic")).casefold()
        self.provisional_cache_enabled = bool(cfg.get("provisional_cache_enabled", True))
        self.validated_reuse_enabled = bool(cfg.get("validated_reuse_enabled", True))
        self.max_active_session = max(1, int(cfg.get("max_active_prefetches_per_session", 2)))
        self.max_active_intent = max(1, int(cfg.get("max_active_prefetches_per_intent", 1)))
        self.max_queued = max(0, int(cfg.get("max_queued_prefetches_per_session", 4)))
        self.ttl_s = max(0.01, float(cfg.get("provisional_ttl_s", 30.0)))
        self.debounce_s = max(0.0, float(cfg.get("debounce_ms", 80.0)) / 1000.0)
        self.finalization_wait_s = max(0.0, float(cfg.get("finalization_wait_ms", 250.0)) / 1000.0)
        self._retrieval_config_hash = _hash({
            key: index.cfg.get(key) for key in (
                "mode", "dense_backend", "dense_model", "lsa_dims", "bm25_k1", "bm25_b",
                "rrf_k", "candidates", "top_k", "reranker", "dedup_threshold",
                "relevance_gate", "relevance_min_coverage", "relevance_min_rerank",
            )
        })
        self._records: dict[str, PreparedEvidence] = {}
        self._latest_by_intent: dict[str, str] = {}
        self._intent_versions: dict[str, int] = {}
        self._current_semantic: dict[str, str] = {}
        self._current_constraints: dict[str, str] = {}
        self._current_queries: dict[str, RetrievalQuery] = {}
        self._current_typed_constraints: dict[str, list[dict[str, Any]]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._queued_order: list[str] = []
        self._lock = threading.RLock()
        self._semaphore: asyncio.Semaphore | None = None
        self._closed = False
        self.stats: dict[str, int] = {
            "decisions": 0,
            "queued": 0,
            "searches_started": 0,
            "searches_completed": 0,
            "cancellation_requests": 0,
            "canceled_before_execution": 0,
            "completed_discarded": 0,
            "validated": 0,
            "committed": 0,
            "reused": 0,
            "failed": 0,
            "expired": 0,
            "backpressure_rejections": 0,
        }

    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_active_session)
        return self._semaphore

    def _active_for(self, intent_id: str) -> list[PreparedEvidence]:
        active = {
            EvidenceState.PROBE_READY, EvidenceState.PREFETCH_QUEUED, EvidenceState.PREFETCH_RUNNING,
            EvidenceState.PROVISIONAL, EvidenceState.VALIDATION_PENDING, EvidenceState.VALIDATED,
            EvidenceState.STALE,
        }
        return [record for record in self._records.values() if record.intent_id == intent_id and record.state in active]

    @staticmethod
    def _constraint_map(items: list[dict[str, Any]]) -> dict[str, str]:
        output: dict[str, str] = {}
        for item in items:
            key = str(item.get("kind") or item.get("key") or item.get("type") or item.get("scope") or "unknown")
            normalized = item.get("normalized_value", item.get("normalized", item.get("value", item.get("text"))))
            value = json.dumps({
                "value": normalized, "unit": item.get("unit"), "relation": item.get("relation", "EQ")
            }, sort_keys=True, default=str)
            output[key] = value.casefold()
        return output

    def _request_compatible(
        self, record: PreparedEvidence, query: RetrievalQuery, typed_constraints: list[dict[str, Any]]
    ) -> bool:
        old_terms = set(record.query_terms)
        new_terms = set(query.terms)
        if not old_terms or not new_terms:
            return False
        overlap = len(old_terms & new_terms) / max(1, len(old_terms | new_terms))
        if overlap < float(self.cfg.get("min_revision_term_overlap", 0.45)):
            return False
        old_constraints = self._constraint_map(record.typed_constraints)
        new_constraints = self._constraint_map(typed_constraints)
        return not any(key in new_constraints and new_constraints[key] != value
                       for key, value in old_constraints.items())

    def decide(
        self,
        *,
        intent_id: str,
        query: RetrievalQuery,
        typed_constraints: list[dict[str, Any]],
        probe: list[str],
        signals: dict[str, Any],
        is_final: bool,
    ) -> PrefetchDecision:
        semantic = semantic_query_fingerprint(
            query, typed_constraints, self.corpus_version, self._retrieval_config_hash
        )
        constraints = constraint_fingerprint(typed_constraints)
        if not self.enabled:
            return PrefetchDecision("WAIT", ("CAPER_DISABLED",), semantic, constraints, 0.0)
        active = self._active_for(intent_id)
        if is_final:
            if any(record.semantic_fingerprint == semantic and not record.superseded for record in active):
                return PrefetchDecision("REUSE", ("IDENTICAL_FINAL_REQUEST",), semantic, constraints, 0.0)
            if any(self._request_compatible(record, query, typed_constraints) and not record.superseded
                   for record in active):
                return PrefetchDecision("REUSE", ("COMPATIBLE_FINAL_REFINEMENT",), semantic, constraints, 0.0)
            if active:
                return PrefetchDecision("INVALIDATE", ("FINAL_REQUEST_CHANGED",), semantic, constraints, 0.0)
            return PrefetchDecision("WAIT", ("FINALIZATION_USES_COMMIT_LANE",), semantic, constraints, 0.0)
        terms = content_terms(query.text)
        if self.policy != "eager" and len(terms) < int(self.cfg.get("min_content_terms", 2)):
            return PrefetchDecision("WAIT", ("INSUFFICIENT_OBJECTIVE_CONTENT",), semantic, constraints, 0.0)
        if not probe:
            return PrefetchDecision("WAIT", ("NO_PLAUSIBLE_EVIDENCE",), semantic, constraints, 0.0)
        if float(signals.get("domain_availability", 1.0)) <= 0.0:
            return PrefetchDecision("WAIT", ("UNSUPPORTED_DOMAIN",), semantic, constraints, 0.0)
        relevance = float(signals.get("probe_evidence_relevance", signals.get("chi", 0.0)) or 0.0)
        if self.policy != "eager" and relevance < float(self.cfg.get("min_probe_relevance", 0.10)):
            return PrefetchDecision("WAIT", ("LOW_PROBE_RELEVANCE",), semantic, constraints, relevance)
        if any(record.semantic_fingerprint == semantic and not record.superseded for record in active):
            return PrefetchDecision("REUSE", ("IDENTICAL_ACTIVE_OR_PREPARED",), semantic, constraints, relevance)
        if any(self._request_compatible(record, query, typed_constraints) and not record.superseded
               for record in active):
            return PrefetchDecision("REUSE", ("COMPATIBLE_QUERY_REFINEMENT",), semantic, constraints, relevance)
        if len(self._queued_order) >= self.max_queued:
            return PrefetchDecision("WAIT", ("SESSION_QUEUE_SATURATED",), semantic, constraints, relevance)
        action = "SUPERSEDE" if active else "PREFETCH"
        return PrefetchDecision(action, ("MEANINGFUL_QUERY_CHANGE",) if active else ("BOUNDED_EARLY_PREPARATION",),
                                semantic, constraints, relevance)

    def observe(
        self,
        *,
        utterance_id: str,
        intent_id: str,
        query_version: int,
        query: RetrievalQuery,
        typed_constraints: list[dict[str, Any]],
        probe: list[str],
        signals: dict[str, Any],
        is_final: bool,
        parent_event_id: str | None,
        timestamp: float,
        allow_prefetch: bool = True,
    ) -> PrefetchDecision:
        decision = self.decide(
            intent_id=intent_id, query=query, typed_constraints=typed_constraints,
            probe=probe, signals=signals, is_final=is_final,
        )
        previous_semantic = self._current_semantic.get(intent_id)
        semantic_changed = previous_semantic is not None and previous_semantic != decision.semantic_fingerprint
        original_action = decision.action
        if semantic_changed and original_action in {"SUPERSEDE", "INVALIDATE"}:
            self._supersede_intent(intent_id, decision.semantic_fingerprint, parent_event_id)
        if not allow_prefetch and decision.action in {"PREFETCH", "SUPERSEDE"}:
            decision = PrefetchDecision(
                "WAIT", ("PRIMARY_SEARCH_ALREADY_SELECTED",), decision.semantic_fingerprint,
                decision.constraint_fingerprint, decision.expected_usefulness,
            )
        self.stats["decisions"] += 1
        if previous_semantic != decision.semantic_fingerprint:
            self._intent_versions[intent_id] = self._intent_versions.get(intent_id, 0) + 1
        self._current_semantic[intent_id] = decision.semantic_fingerprint
        self._current_constraints[intent_id] = decision.constraint_fingerprint
        self._current_queries[intent_id] = copy.deepcopy(query)
        self._current_typed_constraints[intent_id] = copy.deepcopy(typed_constraints)
        if semantic_changed and original_action == "REUSE":
            for record in self._active_for(intent_id):
                if record.state == EvidenceState.VALIDATED:
                    record.transition(EvidenceState.STALE)
        self.emit(
            "PrefetchDecision", [parent_event_id], utterance_id=utterance_id, query_version=query_version,
            payload={
                "intent_id": intent_id, "intent_version": self._intent_versions.get(intent_id, 1),
                "decision": decision.action, "reasons": list(decision.reasons),
                "semantic_fingerprint": decision.semantic_fingerprint,
                "constraint_fingerprint": decision.constraint_fingerprint,
                "expected_usefulness": round(decision.expected_usefulness, 4),
            },
        )
        if decision.action == "SUPERSEDE" and not semantic_changed:
            self._supersede_intent(intent_id, decision.semantic_fingerprint, parent_event_id)
        if decision.action not in {"PREFETCH", "SUPERSEDE"}:
            if decision.action == "REUSE":
                self.stats["reused"] += 1
            elif "SESSION_QUEUE_SATURATED" in decision.reasons:
                self.stats["backpressure_rejections"] += 1
            return decision
        if self._closed:
            return PrefetchDecision("WAIT", ("SESSION_CLOSING",), decision.semantic_fingerprint,
                                    decision.constraint_fingerprint, decision.expected_usefulness)
        intent_version = self._intent_versions.get(intent_id, 1)
        evidence_id = f"pe-{uuid.uuid4().hex[:16]}"
        record = PreparedEvidence(
            evidence_id=evidence_id,
            session_id=self.session_id,
            utterance_id=utterance_id,
            intent_id=intent_id,
            intent_version=intent_version,
            query_version=query_version,
            query_fingerprint=_hash({"semantic": decision.semantic_fingerprint, "query_version": query_version}),
            semantic_fingerprint=decision.semantic_fingerprint,
            corpus_version=self.corpus_version,
            retrieval_config_hash=self._retrieval_config_hash,
            constraint_fingerprint=decision.constraint_fingerprint,
            state=EvidenceState.ABSENT,
            created_at=timestamp,
            expires_at=time.monotonic() + self.ttl_s,
            parent_event_id=parent_event_id,
            query_terms=dict(query.terms),
            typed_constraints=copy.deepcopy(typed_constraints),
        )
        record.transition(EvidenceState.PROBE_READY)
        record.transition(EvidenceState.PREFETCH_QUEUED)
        task_id = f"prefetch-{uuid.uuid4().hex[:12]}"
        record.task_id = task_id
        with self._lock:
            self._records[evidence_id] = record
            self._latest_by_intent[intent_id] = evidence_id
            self._queued_order.append(evidence_id)
        self.stats["queued"] += 1
        queued = self.emit(
            "PrefetchQueued", [parent_event_id], utterance_id=utterance_id, query_version=query_version,
            payload={
                "evidence_id": evidence_id, "task_id": task_id, "intent_id": intent_id,
                "intent_version": intent_version, "state": record.state.value,
                "queue_depth": len(self._queued_order),
            },
        )
        record.parent_event_id = queued.event_id
        task = asyncio.create_task(self._run(record, copy.deepcopy(query)), name=task_id)
        self._tasks[evidence_id] = task
        return decision

    def _supersede_intent(self, intent_id: str, current_semantic: str, parent_event_id: str | None) -> None:
        for record in self._active_for(intent_id):
            if record.semantic_fingerprint == current_semantic:
                continue
            record.superseded = True
            self.stats["cancellation_requests"] += 1
            task = self._tasks.get(record.evidence_id)
            canceled_before = bool(task and not record.search_started and task.cancel())
            if canceled_before:
                self.stats["canceled_before_execution"] += 1
            if record.state in {EvidenceState.PROBE_READY, EvidenceState.PREFETCH_QUEUED,
                                EvidenceState.PREFETCH_RUNNING, EvidenceState.PROVISIONAL,
                                EvidenceState.VALIDATION_PENDING, EvidenceState.VALIDATED}:
                record.transition(EvidenceState.INVALIDATED)
            self.cache.invalidate(record.query_fingerprint)
            self.emit(
                "PrefetchSuperseded", [parent_event_id, record.parent_event_id],
                utterance_id=record.utterance_id, query_version=record.query_version,
                payload={
                    "evidence_id": record.evidence_id, "task_id": record.task_id,
                    "intent_id": intent_id, "state": record.state.value,
                    "cancellation_requested": True,
                    "canceled_before_execution": canceled_before,
                    "underlying_search_may_continue": record.search_started,
                },
            )

    async def _run(self, record: PreparedEvidence, query: RetrievalQuery) -> None:
        try:
            if self.debounce_s:
                await asyncio.sleep(self.debounce_s)
            async with self._sem():
                if record.superseded or record.state == EvidenceState.INVALIDATED:
                    return
                with self._lock:
                    if record.evidence_id in self._queued_order:
                        self._queued_order.remove(record.evidence_id)
                record.transition(EvidenceState.PREFETCH_RUNNING)
                record.search_started = True
                self.stats["searches_started"] += 1
                started = self.emit(
                    "PrefetchStarted", [record.parent_event_id], utterance_id=record.utterance_id,
                    query_version=record.query_version, payload={
                        "evidence_id": record.evidence_id, "task_id": record.task_id,
                        "intent_id": record.intent_id, "intent_version": record.intent_version,
                        "state": record.state.value,
                    },
                )
                record.parent_event_id = started.event_id
                result = await asyncio.to_thread(
                    self.index.search,
                    query,
                    None,
                    None,
                    {
                        "retrieval_type": "PROVISIONAL",
                        "session_id": self.session_id,
                        "intent_id": record.intent_id,
                        "query_version": record.query_version,
                        "evidence_id": record.evidence_id,
                    },
                )
                self.stats["searches_completed"] += 1
                record.completed_at = time.monotonic()
                record.result = result
                record.source_chunk_ids = [hit.chunk_id for hit in result.relevant_hits]
                record.source_hashes = {
                    chunk_id: self.index.corpus.chunk(chunk_id).sha256
                    for chunk_id in record.source_chunk_ids if self.index.corpus.chunk(chunk_id)
                }
                if record.superseded or record.state == EvidenceState.INVALIDATED:
                    self.stats["completed_discarded"] += 1
                    self.index.mark_search_discarded(result.search_id)
                    self.emit(
                        "PrefetchDiscarded", [record.parent_event_id], utterance_id=record.utterance_id,
                        query_version=record.query_version, chunk_ids=list(record.source_chunk_ids),
                        payload={
                            "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                            "reason": "SUPERSEDED_DURING_BLOCKING_SEARCH", "state": record.state.value,
                            "search_id": result.search_id,
                        },
                    )
                    return
                record.transition(EvidenceState.PROVISIONAL)
                if self.provisional_cache_enabled:
                    self.cache.put_provisional(
                        record.query_fingerprint, result, self.corpus_version,
                        session_id=self.session_id, intent_id=record.intent_id,
                        query_version=record.query_version,
                        constraint_fingerprint=record.constraint_fingerprint,
                        source_hashes=record.source_hashes,
                    )
                completed = self.emit(
                    "PrefetchCompleted", [record.parent_event_id], utterance_id=record.utterance_id,
                    query_version=record.query_version, chunk_ids=list(record.source_chunk_ids),
                    latency_ms=round(sum(result.stage_ms.values()), 3), payload={
                        "evidence_id": record.evidence_id, "task_id": record.task_id,
                        "intent_id": record.intent_id, "intent_version": record.intent_version,
                        "state": record.state.value, "search_id": result.search_id,
                        "evidence_status": result.evidence_status,
                        "accepted_chunk_ids": list(record.source_chunk_ids),
                    },
                )
                record.parent_event_id = completed.event_id
                self._validate(record)
        except asyncio.CancelledError:
            if not record.search_started and record.state == EvidenceState.PREFETCH_QUEUED:
                record.transition(EvidenceState.INVALIDATED)
            raise
        except Exception as exc:
            self.stats["failed"] += 1
            if record.state in {EvidenceState.PREFETCH_QUEUED, EvidenceState.PREFETCH_RUNNING}:
                record.transition(EvidenceState.FAILED)
            self.emit(
                "PrefetchDiscarded", [record.parent_event_id], utterance_id=record.utterance_id,
                query_version=record.query_version, payload={
                    "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                    "reason": "SEARCH_FAILED", "error_type": type(exc).__name__,
                    "state": record.state.value,
                },
            )
        finally:
            with self._lock:
                if record.evidence_id in self._queued_order:
                    self._queued_order.remove(record.evidence_id)

    def _validate(self, record: PreparedEvidence) -> bool:
        if record.state not in {EvidenceState.PROVISIONAL, EvidenceState.STALE}:
            return False
        record.transition(EvidenceState.VALIDATION_PENDING)
        started = self.emit(
            "EvidenceValidationStarted", [record.parent_event_id], utterance_id=record.utterance_id,
            query_version=record.query_version, payload={
                "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                "state": record.state.value,
            },
        )
        current_query = self._current_queries.get(record.intent_id)
        current_typed = self._current_typed_constraints.get(record.intent_id, [])
        request_compatible = bool(current_query and self._request_compatible(record, current_query, current_typed))
        if current_query and record.semantic_fingerprint == self._current_semantic.get(record.intent_id):
            request_compatible = True
        source_ok = all(
            self.index.corpus.chunk(chunk_id) is not None
            and self.index.corpus.chunk(chunk_id).sha256 == checksum
            for chunk_id, checksum in record.source_hashes.items()
        )
        critical_values = [
            str(item.get("normalized_value") or item.get("value") or item.get("surface") or "").casefold()
            for item in current_typed if str(item.get("kind", "")).casefold() in {"city", "document_scope"}
        ]
        constraint_evidence_ok = all(
            any(value in self.index.corpus.chunk(chunk_id).index_text.casefold()
                for chunk_id in record.source_chunk_ids if self.index.corpus.chunk(chunk_id))
            for value in critical_values if value
        ) if critical_values else True
        current_relevance = bool(record.result and record.result.relevant_hits)
        if current_query and record.source_chunk_ids:
            threshold = float(self.index.cfg.get("relevance_min_coverage", 0.16))
            current_relevance = any(
                self.index.coverage(current_query.terms, self.index.pos[chunk_id]) >= threshold
                for chunk_id in record.source_chunk_ids if chunk_id in self.index.pos
            )
        checks = {
            "session_scope": record.session_id == self.session_id,
            "intent_current": self._latest_by_intent.get(record.intent_id) == record.evidence_id,
            "semantic_query_compatible": request_compatible,
            "constraints_compatible": constraint_evidence_ok,
            "corpus_version_current": record.corpus_version == self.corpus_version,
            "source_checksums_current": source_ok,
            "not_expired": record.expires_at > time.monotonic(),
            "relevance_admitted": current_relevance,
        }
        record.validation_record = checks
        if all(checks.values()):
            record.transition(EvidenceState.VALIDATED)
            record.validated_at = time.monotonic()
            self.stats["validated"] += 1
            if self.provisional_cache_enabled:
                self.cache.promote(
                    record.query_fingerprint, self.corpus_version,
                    session_id=self.session_id, intent_id=record.intent_id,
                    query_version=record.query_version,
                    constraint_fingerprint=record.constraint_fingerprint,
                    source_hashes=record.source_hashes,
                )
            self.emit(
                "EvidenceValidated", [started.event_id], utterance_id=record.utterance_id,
                query_version=record.query_version, chunk_ids=list(record.source_chunk_ids), payload={
                    "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                    "state": record.state.value, "checks": checks,
                    "claim_grounding_required": True,
                },
            )
            return True
        if not checks["not_expired"]:
            record.transition(EvidenceState.EXPIRED)
            self.stats["expired"] += 1
            reason = "EXPIRED"
        else:
            record.transition(EvidenceState.INVALIDATED)
            reason = "VALIDATION_FAILED"
        self.cache.invalidate(record.query_fingerprint)
        self.emit(
            "EvidenceInvalidated", [started.event_id], utterance_id=record.utterance_id,
            query_version=record.query_version, chunk_ids=list(record.source_chunk_ids), payload={
                "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                "state": record.state.value, "reason": reason, "checks": checks,
            },
        )
        return False

    async def validated_for(
        self,
        intent_ids: list[str],
        *,
        wait_s: float = 0.0,
    ) -> PreparedEvidence | None:
        if not self.validated_reuse_enabled:
            return None
        candidates = [self._latest_by_intent.get(intent_id) for intent_id in intent_ids]
        tasks = [self._tasks[evidence_id] for evidence_id in candidates
                 if evidence_id and evidence_id in self._tasks and not self._tasks[evidence_id].done()]
        if tasks and wait_s > 0:
            done, _ = await asyncio.wait(tasks, timeout=wait_s)
            for task in done:
                try:
                    task.result()
                except (asyncio.CancelledError, Exception):
                    pass
        now = time.monotonic()
        for intent_id in intent_ids:
            evidence_id = self._latest_by_intent.get(intent_id)
            record = self._records.get(evidence_id or "")
            if record is None or record.consumed:
                continue
            if record.expires_at <= now and record.state in {EvidenceState.PROVISIONAL, EvidenceState.VALIDATED}:
                record.transition(EvidenceState.EXPIRED)
                self.cache.invalidate(record.query_fingerprint)
                self.stats["expired"] += 1
                continue
            if record.state == EvidenceState.STALE:
                self._validate(record)
            if record.state == EvidenceState.VALIDATED and record.result is not None:
                return record
        return None

    def mark_consumed(self, record: PreparedEvidence) -> None:
        if record.state != EvidenceState.VALIDATED:
            raise ValueError("only validated CAPER evidence may enter the retrieval/grounding lane")
        record.consumed = True
        self.stats["reused"] += 1

    def invalidate_utterance(self, utterance_id: str, parent_event_id: str | None, reason: str) -> None:
        """Revoke every nonterminal preparation owned by a revised utterance."""
        for record in self._records.values():
            if record.utterance_id != utterance_id or record.state in {
                EvidenceState.INVALIDATED, EvidenceState.FAILED, EvidenceState.EXPIRED,
            }:
                continue
            record.superseded = True
            task = self._tasks.get(record.evidence_id)
            canceled_before = bool(task and not record.search_started and task.cancel())
            if canceled_before:
                self.stats["canceled_before_execution"] += 1
            self.stats["cancellation_requests"] += 1
            if record.state != EvidenceState.COMMITTED:
                record.transition(EvidenceState.INVALIDATED)
                self.cache.invalidate(record.query_fingerprint)
            else:
                record.transition(EvidenceState.STALE)
            self.emit(
                "EvidenceInvalidated", [parent_event_id, record.parent_event_id],
                utterance_id=utterance_id, query_version=record.query_version,
                chunk_ids=list(record.source_chunk_ids), payload={
                    "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                    "state": record.state.value, "reason": reason,
                    "canceled_before_execution": canceled_before,
                },
            )

    def commit_grounded(self, chunk_ids: set[str], parent_event_id: str | None, query_version: int) -> None:
        if not chunk_ids:
            return
        for record in self._records.values():
            if record.state != EvidenceState.VALIDATED or not record.consumed:
                continue
            supported = sorted(chunk_ids & set(record.source_chunk_ids))
            if not supported:
                continue
            record.transition(EvidenceState.COMMITTED)
            record.committed_at = time.monotonic()
            self.stats["committed"] += 1
            self.emit(
                "EvidenceCommitted", [parent_event_id], utterance_id=record.utterance_id,
                query_version=query_version, chunk_ids=supported, payload={
                    "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                    "state": record.state.value, "grounding_verified": True,
                },
            )

    async def close(self) -> None:
        self._closed = True
        for record in self._records.values():
            if record.state not in {EvidenceState.PREFETCH_QUEUED, EvidenceState.PREFETCH_RUNNING}:
                continue
            task = self._tasks.get(record.evidence_id)
            canceled_before = bool(record.state == EvidenceState.PREFETCH_QUEUED and task and task.cancel())
            if canceled_before:
                self.stats["canceled_before_execution"] += 1
            self.stats["cancellation_requests"] += 1
            record.superseded = True
            record.transition(EvidenceState.INVALIDATED)
            self.cache.invalidate(record.query_fingerprint)
            self.emit(
                "EvidenceInvalidated", [record.parent_event_id], utterance_id=record.utterance_id,
                query_version=record.query_version, chunk_ids=list(record.source_chunk_ids), payload={
                    "evidence_id": record.evidence_id, "intent_id": record.intent_id,
                    "state": record.state.value, "reason": "SESSION_CLOSED",
                    "canceled_before_execution": canceled_before,
                    "underlying_search_may_continue": record.search_started,
                },
            )
        running = [task for task in self._tasks.values() if not task.done()]
        if running:
            done, _ = await asyncio.wait(running, timeout=self.finalization_wait_s)
            for task in done:
                try:
                    task.result()
                except (asyncio.CancelledError, Exception):
                    pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "records": [record.public() for record in self._records.values()],
            "stats": dict(self.stats),
            "limits": {
                "max_active_prefetches_per_session": self.max_active_session,
                "max_active_prefetches_per_intent": self.max_active_intent,
                "max_queued_prefetches_per_session": self.max_queued,
            },
        }
