"""Shared synthesis types and evidence-sentence scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from kairos.retrieval.index import Hit, HybridIndex, RetrievalQuery
from kairos.textutil import content_set, content_terms, estimate_tokens, lower_tokens, norm, sentences

PRONOUN_START = re.compile(r"^(It|This|They|These|That|Its|There)\b")


@dataclass
class ClaimDraft:
    text: str
    quote: str
    evidence: list[str]
    score: float
    sub_query_id: str


@dataclass
class SynthesisResult:
    drafts: list[ClaimDraft]
    gap: str | None
    tokens: dict = field(default_factory=dict)
    context_tokens_est: int = 0
    backend: str = "extractive"


@dataclass
class EvidenceSentence:
    chunk_id: str
    text: str
    relevance: float
    section_title: str


def _sentence_coverage(index: HybridIndex, terms: dict[str, float], sentence: str, context: set[str]) -> float:
    """IDF-weighted query coverage of a sentence; terms found only in the chunk's heading/document
    context ('Mumbai Venue Catalogue > Marine Hall') earn half credit."""
    total = sum(index.bm25.idf_of(t) * w for t, w in terms.items())
    if total <= 0:
        return 0.0
    present = content_set(sentence)
    got = 0.0
    for t, w in terms.items():
        if t in present:
            got += index.bm25.idf_of(t) * w
        elif t in context:
            got += 0.5 * index.bm25.idf_of(t) * w
    return got / total


HEADCOUNT_RE = re.compile(r"(\d+)\s+([A-Za-z]+)")


def _people_counts(text: str) -> list[int]:
    return [int(m.group(1)) for m in HEADCOUNT_RE.finditer(text) if norm(m.group(2)) == "people"]


def _headcount_fit(wanted: list[int], sentence: str) -> float:
    """Numeric slot compatibility: '30 people' prefers 'up to 40 participants' over 'up to 24'."""
    if not wanted:
        return 0.0
    have = _people_counts(sentence)
    if not have:
        return 0.0
    need = max(wanted)
    return 0.1 if any(h >= need for h in have) else -0.15


def score_sentences(index: HybridIndex, query: RetrievalQuery, hits: list[Hit],
                    avoid_facets: dict[str, str] | None = None,
                    prefer_terms: set[str] | None = None) -> list[EvidenceSentence]:
    """Rank evidence sentences for a sub-query. Sentences mentioning a *conflicting* facet value
    (e.g. 'standard event' under an 'international' constraint) are excluded."""
    avoid_terms: set[str] = set()
    corpus = index.corpus
    for group, value in (avoid_facets or {}).items():
        avoid_terms |= {m for m in corpus.contrast_groups.get(group, set()) if m != value}
    qvec = index.encode(query.dense_text)
    want_people = _people_counts(query.dense_text)
    out: list[EvidenceSentence] = []
    for hit in hits:
        chunk = corpus.chunk(hit.chunk_id)
        if chunk is None:
            continue
        sents = sentences(chunk.text)
        if not sents:
            continue
        vecs = index.encoder.encode(sents)
        context = content_set(f"{chunk.doc_title} {chunk.section_path}")
        for sent, vec in zip(sents, vecs):
            terms = set(content_terms(sent))
            if avoid_terms & terms:
                continue
            rel = 0.5 * _sentence_coverage(index, query.terms, sent, context) + 0.3 * max(float(np.dot(vec, qvec)), 0.0) \
                + 0.2 * hit.rerank_score
            if prefer_terms and prefer_terms & terms:
                rel += 0.15
            rel += _headcount_fit(want_people, sent)
            out.append(EvidenceSentence(hit.chunk_id, sent, round(min(rel, 1.0), 4), chunk.section_title))
    out.sort(key=lambda s: (-s.relevance, s.chunk_id, s.text))
    return out


def claim_text_for(sentence: EvidenceSentence) -> str:
    """Make an extracted sentence standalone: 'It is suited to…' -> 'Banyan Hall: It is suited to…'."""
    if PRONOUN_START.match(sentence.text):
        return f"{sentence.section_title}: {sentence.text}"
    return sentence.text


def context_tokens(index: HybridIndex, hits: list[Hit]) -> int:
    return sum(estimate_tokens(index.corpus.chunk(h.chunk_id).text) for h in hits if index.corpus.chunk(h.chunk_id))


def focus_terms(index: HybridIndex, sq_need_texts: list[str], k: int = 2) -> list[str]:
    """The most specific terms of a need: highest-IDF content terms that are not places or numbers."""
    corpus = index.corpus
    terms = []
    for t in dict.fromkeys(content_terms(" ".join(sq_need_texts))):
        if t in corpus.places or any(ch.isdigit() for ch in t) or len(t) < 3:
            continue
        terms.append(t)
    terms.sort(key=lambda t: -index.bm25.idf_of(t))
    return terms[:k]


def evidence_covers_focus(index: HybridIndex, need_texts: list[str], hits: list[Hit]) -> tuple[bool, list[str]]:
    """Sufficiency: at least one focus term must occur in the retrieved evidence. A question about a gym at
    a venue retrieves that venue's chunks, but if none mentions a gym the answer abstains."""
    focus = focus_terms(index, need_texts)
    if not focus:
        return True, focus
    present: set[str] = set()
    for h in hits:
        present |= index.term_sets[index.pos[h.chunk_id]]
    surface = {norm(w): w for w in lower_tokens(" ".join(need_texts))}
    return any(t in present for t in focus), [surface.get(t, t) for t in focus]
