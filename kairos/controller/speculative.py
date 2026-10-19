"""Version-bound provisional retrieval cache for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass

from kairos.retrieval.index import RetrievalQuery, SearchResult
from kairos.textutil import jaccard


@dataclass
class SpeculativeEntry:
    session_id: str
    utterance_id: str
    candidate_id: str
    query_version: int
    normalized_query: str
    corpus_profile: str
    result: SearchResult
    retrieval_timestamp: float
    retrieval_cost_ms: float
    provisional: bool = True
    promoted: bool = False
    stale: bool = False


class SpeculativeCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], SpeculativeEntry] = {}

    def put(self, entry: SpeculativeEntry) -> bool:
        key = (entry.session_id, entry.utterance_id, entry.candidate_id)
        current = self._entries.get(key)
        if current and current.query_version >= entry.query_version:
            return False
        if current:
            current.stale = True
        self._entries[key] = entry
        return True

    def promote(
        self,
        session_id: str,
        utterance_id: str,
        candidate_id: str,
        query_version: int,
        query: RetrievalQuery,
        corpus_profile: str,
        *,
        min_term_jaccard: float = 0.75,
    ) -> SearchResult | None:
        entry = self._entries.get((session_id, utterance_id, candidate_id))
        if entry is None or entry.stale or entry.promoted or entry.query_version > query_version:
            return None
        compatible = (
            entry.corpus_profile == corpus_profile
            and jaccard(entry.normalized_query.split(), query.dense_text.lower().split()) >= min_term_jaccard
        )
        if not compatible:
            entry.stale = True
            return None
        entry.promoted = True
        entry.provisional = False
        return entry.result

    def invalidate(self, session_id: str, utterance_id: str, candidate_id: str) -> None:
        entry = self._entries.get((session_id, utterance_id, candidate_id))
        if entry:
            entry.stale = True
