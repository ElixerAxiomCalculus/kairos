"""Session-local, provenance-aware cache for validated retrieval results.

Cache reuse is an execution optimization only.  A hit still travels through the
normal evidence graph, admission, grounding, and claim verification paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kairos.retrieval.index import RetrievalQuery, SearchResult
from kairos.textutil import content_terms


class CacheState(StrEnum):
    PROVISIONAL = "PROVISIONAL"
    VALIDATED = "VALIDATED"
    REUSABLE = "REUSABLE"
    STALE = "STALE"
    INVALIDATED = "INVALIDATED"


_POLITENESS = re.compile(r"\b(?:please|could you|can you|would you|tell me|explain|about)\b", re.I)


def canonical_objective(text: str) -> str:
    """Conservative wording normalization; constraints remain in the key."""
    reduced = _POLITENESS.sub(" ", text)
    return " ".join(content_terms(reduced))


def canonical_weighted_terms(query: RetrievalQuery) -> list[tuple[str, float]]:
    """Retain meaningful query weights while discarding normalized request boilerplate."""
    meaningful = set(content_terms(_POLITENESS.sub(" ", query.dense_text)))
    return sorted((term, round(float(weight), 8)) for term, weight in query.terms.items() if term in meaningful)


def query_fingerprint(
    query: RetrievalQuery,
    *,
    corpus_profile: str,
    corpus_version: str,
    index_configuration: dict[str, Any],
    intent_objective: str | None = None,
    typed_constraints: list[dict[str, Any]] | None = None,
    retrieval_mode: str | None = None,
    relevance_gate_configuration: dict[str, Any] | None = None,
) -> str:
    payload = {
        "corpus_profile": corpus_profile,
        "corpus_version": corpus_version,
        "index_configuration": index_configuration,
        "normalized_query": canonical_objective(query.dense_text),
        "weighted_terms": canonical_weighted_terms(query),
        "intent_objective": canonical_objective(intent_objective or query.text),
        "typed_constraints": sorted(typed_constraints or [], key=lambda item: json.dumps(item, sort_keys=True)),
        "facet_constraints": dict(sorted(query.facet_constraints.items())),
        "retrieval_mode": retrieval_mode,
        "relevance_gate_configuration": relevance_gate_configuration or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class EvidenceCacheEntry:
    fingerprint: str
    result: SearchResult
    state: CacheState
    created_monotonic: float
    expires_monotonic: float
    source_hash: str
    hits: int = 0


class EvidenceCache:
    """Bounded, thread-safe cache.  Instantiate once per SessionEngine."""

    def __init__(self, *, max_entries: int = 64, ttl_s: float = 300.0) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_s = max(0.001, float(ttl_s))
        self._entries: OrderedDict[str, EvidenceCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        self.evictions = 0

    def put(self, fingerprint: str, result: SearchResult, source_hash: str, *, validated: bool) -> None:
        now = time.monotonic()
        state = CacheState.VALIDATED if validated else CacheState.PROVISIONAL
        with self._lock:
            entry = EvidenceCacheEntry(
                fingerprint, copy.deepcopy(result), state, now, now + self.ttl_s, source_hash
            )
            if validated:
                entry.state = CacheState.REUSABLE
            self._entries[fingerprint] = entry
            self._entries.move_to_end(fingerprint)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1

    def get(self, fingerprint: str, source_hash: str) -> SearchResult | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(fingerprint)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_monotonic <= now:
                entry.state = CacheState.STALE
            if entry.source_hash != source_hash:
                entry.state = CacheState.INVALIDATED
                self.invalidations += 1
            if entry.state != CacheState.REUSABLE:
                self.misses += 1
                if entry.state in {CacheState.STALE, CacheState.INVALIDATED}:
                    self._entries.pop(fingerprint, None)
                return None
            entry.hits += 1
            self.hits += 1
            self._entries.move_to_end(fingerprint)
            return copy.deepcopy(entry.result)

    def is_reusable(self, fingerprint: str, source_hash: str) -> bool:
        """Read cache eligibility without changing counters, order, or entry state.

        Feature extraction may observe execution metadata, but must not consume a
        cache hit or alter subsequent retrieval behaviour.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(fingerprint)
            return bool(
                entry is not None
                and entry.state == CacheState.REUSABLE
                and entry.expires_monotonic > now
                and entry.source_hash == source_hash
            )

    def invalidate(self, fingerprint: str) -> None:
        with self._lock:
            entry = self._entries.get(fingerprint)
            if entry is not None:
                entry.state = CacheState.INVALIDATED
                self.invalidations += 1

    def status(self) -> dict[str, int]:
        with self._lock:
            states = {state.value: 0 for state in CacheState}
            for entry in self._entries.values():
                states[entry.state.value] += 1
            return {
                "entries": len(self._entries), "hits": self.hits, "misses": self.misses,
                "invalidations": self.invalidations, "evictions": self.evictions, **states,
            }
