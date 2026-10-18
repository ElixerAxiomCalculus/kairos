"""Incremental QueryFrame construction: utterance prefix -> clauses, candidate needs, slots, repairs.

A *candidate need* is a clause that asks for information. Constraint clauses
("for 30 people", "in Pune") attach to the need they modify. Candidates are keyed by their
ordinal within the utterance, which is stable while the utterance grows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kairos.corpus.ingest import Corpus
from kairos.textutil import (
    CLOSED_CLASS,
    PREPOSITIONS,
    REQUEST_CUES,
    STOPWORDS,
    content_terms,
    is_number,
    lower_tokens,
    norm,
    tokens,
)

# Boundaries: punctuation, and discourse markers that introduce a new request.
REQUEST_ONSET = (
    r"tell me|let me know|what|whether|how|is there|are there|can|could|do|does|give me|show me|"
    r"check|find|list|explain|which|when|where|who|also"
)
HARD_BOUNDARY_RE = re.compile(rf"[,;.?!]+\s*|\s+(?:and|also|plus|then)\s+(?=(?:{REQUEST_ONSET})\b)|\s+(?=(?:tell me|let me know)\b)", re.I)
SOFT_BOUNDARY_RE = re.compile(r"\s+(?:and|or)\s+", re.I)
REPAIR_RE = re.compile(r"(?:[,—-]\s*)?\b(?:sorry|i mean|no wait|rather|correction)\b[,]?\s+", re.I)
UNIT_WORDS = {
    "people", "person", "persons", "participants", "attendees", "guests", "pax", "day", "days",
    "night", "nights", "hour", "hours", "week", "weeks", "rupees", "inr", "usd",
}
ANAPHORA = {"that", "it", "its", "there", "this", "these", "those", "same"}


@dataclass
class Slot:
    key: str            # stable identity, e.g. "place", "domestic|international|standard", "number:people"
    kind: str           # place | proper | qualifier | number | term
    value: str
    first_seen_t: float
    last_changed_t: float
    source_chunk_id: str
    revised_from: str | None = None

    def to_public(self) -> dict:
        return {
            "key": self.key, "kind": self.kind, "value": self.value,
            "first_seen_t": round(self.first_seen_t, 3), "last_changed_t": round(self.last_changed_t, 3),
            "source_chunk_id": self.source_chunk_id, "revised_from": self.revised_from,
        }


@dataclass
class Span:
    text: str
    start: int
    kind: str            # NEED | CONSTRAINT | FILLER
    closed: bool         # followed by a boundary (speaker moved on)
    soft: bool = False   # created by a soft (and/or) split


@dataclass
class CandidateNeed:
    key: str
    need_text: str
    constraints: list[str]
    closed: bool
    anaphoric: bool

    @property
    def text(self) -> str:
        return " ".join([self.need_text, *self.constraints]).strip()


@dataclass
class Frame:
    raw_text: str
    analysis_text: str
    spans: list[Span]
    candidates: list[CandidateNeed]
    slots: dict[str, Slot]
    repairs: list[dict] = field(default_factory=list)
    anchor: str = ""

    def normalized(self) -> str:
        return " · ".join(c.text for c in self.candidates)


def _has_request_cue(text: str) -> bool:
    low = f" {text.lower()} "
    return any(f" {cue} " in low or low.strip().startswith(cue) for cue in REQUEST_CUES)


def _slot_like(term: str, corpus: Corpus) -> bool:
    return (
        is_number(term)
        or term in UNIT_WORDS
        or term in corpus.places
        or term in corpus.qualifier_group
        or term in {"event", "events"}
    )


def classify_span(text: str, corpus: Corpus) -> str:
    terms = [t for t in lower_tokens(text) if t not in STOPWORDS]
    if not terms:
        return "FILLER"
    if _has_request_cue(text):
        return "NEED"
    first = lower_tokens(text)[0]
    normed = [norm(t) for t in terms]
    if all(_slot_like(t, corpus) or _slot_like(n, corpus) for t, n in zip(terms, normed)):
        return "CONSTRAINT"
    if first in PREPOSITIONS and len(terms) <= 3 and any(
        _slot_like(t, corpus) or t in corpus.proper_nouns for t in terms
    ):
        return "CONSTRAINT"
    return "NEED"


def apply_repairs(text: str, corpus: Corpus) -> tuple[str, list[dict]]:
    """'…in Pune, sorry, Mumbai for 30…' -> '…in Mumbai for 30…' (same-class replacement)."""
    repairs = []
    while True:
        m = REPAIR_RE.search(text)
        if not m:
            break
        before, after = text[: m.start()], text[m.end():]
        after_toks = tokens(after)
        if not after_toks:
            break  # repair cue at the end of a partial utterance: wait for more speech
        new = after_toks[0]
        new_low = new.lower()
        cls = "place" if new_low in corpus.places else "qualifier" if new_low in corpus.qualifier_group else (
            "number" if is_number(new_low) else None)
        replaced = False
        if cls:
            before_toks = list(re.finditer(r"[₹$]?\d[\d,]*|[A-Za-z]+", before))
            for tm in reversed(before_toks):
                w = tm.group(0).lower()
                same = (
                    (cls == "place" and w in corpus.places)
                    or (cls == "qualifier" and corpus.qualifier_group.get(w) == corpus.qualifier_group.get(new_low))
                    or (cls == "number" and is_number(w))
                )
                if same:
                    text = before[: tm.start()] + after
                    repairs.append({"kind": cls, "from": tm.group(0), "to": new})
                    replaced = True
                    break
        if not replaced:
            text = before + " " + after  # drop the cue, keep content
    return re.sub(r"\s+", " ", text).strip(), repairs


def segment(text: str, corpus: Corpus) -> list[Span]:
    spans: list[Span] = []
    pos = 0
    pieces: list[tuple[int, int, bool]] = []
    for m in HARD_BOUNDARY_RE.finditer(text):
        if m.start() > pos:
            pieces.append((pos, m.start(), True))
        pos = m.end()
    if pos < len(text):
        pieces.append((pos, len(text), False))
    for start, end, closed in pieces:
        chunk = text[start:end]
        sub_pos = 0
        subs = list(SOFT_BOUNDARY_RE.finditer(chunk))
        parts = []
        for sm in subs:
            parts.append((sub_pos, sm.start(), True))
            sub_pos = sm.end()
        parts.append((sub_pos, len(chunk), closed))
        for i, (s, e, c) in enumerate(parts):
            piece = chunk[s:e].strip()
            if not piece:
                continue
            spans.append(Span(piece, start + s, classify_span(piece, corpus), c, soft=i > 0))
    return spans


def build_candidates(spans: list[Span]) -> list[CandidateNeed]:
    cands: list[CandidateNeed] = []
    pending: list[str] = []
    for span in spans:
        if span.kind == "FILLER":
            continue
        if span.kind == "CONSTRAINT":
            if cands:
                cands[-1].constraints.append(span.text)
                cands[-1].closed = span.closed
            else:
                pending.append(span.text)
            continue
        if span.soft and cands and not _has_request_cue(span.text):
            # "the venue and a projector": a conjunct that inherits the previous request.
            pass
        anaphoric = any(t in ANAPHORA for t in lower_tokens(span.text))
        cands.append(CandidateNeed(f"k{len(cands)}", span.text, list(pending), span.closed, anaphoric))
        pending = []
    if pending and not cands:
        cands.append(CandidateNeed("k0", " ".join(pending), [], spans[-1].closed if spans else False, False))
    return cands


def extract_slots(text: str, corpus: Corpus) -> dict[str, tuple[str, str]]:
    """key -> (kind, value)."""
    out: dict[str, tuple[str, str]] = {}
    toks = tokens(text)
    for i, tok in enumerate(toks):
        low = tok.lower()
        if is_number(low):
            unit = toks[i + 1].lower() if i + 1 < len(toks) else ""
            unit_n = norm(unit) if unit in UNIT_WORDS else "value"
            out[f"number:{unit_n}"] = ("number", tok)
        elif low in corpus.places:
            out["place"] = ("place", tok)
        elif low in corpus.qualifier_group:
            out[corpus.qualifier_group[low]] = ("qualifier", low)
        elif low in corpus.proper_nouns and tok[:1].isupper() and i > 0:
            out[f"proper:{low}"] = ("proper", tok)
    return out


def anchor_of(candidates: list[CandidateNeed]) -> str:
    """Context shared by later needs: content terms of the first need (numbers excluded)."""
    if not candidates:
        return ""
    first = candidates[0].text
    terms = [t for t in content_terms(first) if not is_number(t) and t not in UNIT_WORDS and t not in {"book"}]
    return " ".join(terms[:5])


def completeness(span_text: str, closed: bool, is_final: bool) -> float:
    """κ: is this clause syntactically and semantically complete enough to act on?"""
    low = lower_tokens(span_text)
    if not low:
        return 0.0
    content = [t for t in low if t not in STOPWORDS]
    last = low[-1]
    score = 0.35 * min(len(content) / 3.0, 1.0)
    score += 0.25 if last not in CLOSED_CLASS else 0.0
    score += 0.2 if _has_request_cue(span_text) or len(content) >= 3 else 0.0
    score += 0.2 if (closed or is_final) else 0.0
    if last in CLOSED_CLASS and not (closed or is_final):
        score *= 0.5
    return round(min(score, 1.0), 4)


def continuation_hazard(span_text: str, closed: bool, is_final: bool) -> float:
    """π: probability that more speech will materially change this clause."""
    if is_final:
        return 0.0
    low = lower_tokens(span_text)
    if not low or low[-1] in CLOSED_CLASS:
        return 0.85
    return 0.1 if closed else 0.5


class FrameBuilder:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def build(self, raw_text: str, prev_slots: dict[str, Slot], t: float, chunk_id: str) -> Frame:
        analysis, repairs = apply_repairs(raw_text, self.corpus)
        spans = segment(analysis, self.corpus)
        candidates = build_candidates(spans)
        slots: dict[str, Slot] = {}
        for key, (kind, value) in extract_slots(analysis, self.corpus).items():
            prev = prev_slots.get(key)
            if prev is None:
                slots[key] = Slot(key, kind, value, t, t, chunk_id)
            elif prev.value.lower() != value.lower():
                slots[key] = Slot(key, kind, value, prev.first_seen_t, t, chunk_id, revised_from=prev.value)
            else:
                slots[key] = prev
        return Frame(raw_text, analysis, spans, candidates, slots, repairs, anchor_of(candidates))


# Preserve the original implementation for explicit benchmark ablation only.
LegacyCandidateNeed = CandidateNeed
LegacyFrame = Frame
LegacySlot = Slot
legacy_apply_repairs = apply_repairs
legacy_segment = segment
legacy_build_candidates = build_candidates
legacy_extract_slots = extract_slots
legacy_anchor_of = anchor_of


class LegacyFrameBuilder:
    """Frozen V1 frame builder used only by the original-vs-improved benchmark arm."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def build(self, raw_text: str, prev_slots: dict[str, LegacySlot], t: float, chunk_id: str) -> LegacyFrame:
        analysis, repairs = legacy_apply_repairs(raw_text, self.corpus)
        spans = legacy_segment(analysis, self.corpus)
        candidates = legacy_build_candidates(spans)
        slots: dict[str, LegacySlot] = {}
        for key, (kind, value) in legacy_extract_slots(analysis, self.corpus).items():
            prev = prev_slots.get(key)
            if prev is None:
                slots[key] = LegacySlot(key, kind, value, t, t, chunk_id)
            elif prev.value.lower() != value.lower():
                slots[key] = LegacySlot(key, kind, value, prev.first_seen_t, t, chunk_id, revised_from=prev.value)
            else:
                slots[key] = prev
        return LegacyFrame(raw_text, analysis, spans, candidates, slots, repairs, legacy_anchor_of(candidates))

# V2 implementation.  These explicit re-exports keep the original public import path stable for
# the engine, tests and downstream integrations while the legacy definitions above remain useful
# to the original-vs-improved ablation.
from kairos.streaming.frame_v2 import (  # noqa: E402,F401
    DEFAULT_PLACES,
    UNIT_WORDS,
    CandidateNeed,
    Frame,
    FrameBuilder,
    ScopedConstraint,
    Slot,
    Span,
    anchor_of,
    apply_repairs,
    build_candidates,
    classify_span,
    completeness,
    continuation_hazard,
    extract_slots,
    extract_typed_constraints,
    segment,
)
