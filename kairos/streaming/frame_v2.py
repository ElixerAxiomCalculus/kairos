"""Structured incremental QueryFrame with stable identities and scoped constraints.

This module stays deterministic and lightweight: it uses linguistic onset rules, local text
statistics and a small configurable gazetteer rather than an LLM.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from kairos.corpus.ingest import Corpus
from kairos.textutil import (
    CLOSED_CLASS,
    FILLERS,
    PREPOSITIONS,
    REQUEST_CUES,
    STOPWORDS,
    content_terms,
    is_number,
    jaccard,
    lower_tokens,
    norm,
    tokens,
)

REQUEST_ONSET = (
    r"tell me|let me know|what(?:\s+(?:is|are|does|do|was|were))?|whether|how|is there|are there|"
    r"can|could|do|does|give me|show me|check|find(?:\s+me)?|list|explain|define|suggest|"
    r"which|when|where|who"
)
STRONG_ONSET_RE = re.compile(rf"\b(?:{REQUEST_ONSET})\b", re.I)
BOUNDARY_RE = re.compile(
    rf"[;.?!]+\s*|\s+(?:and|also|plus|then)\s+(?=(?:{REQUEST_ONSET})\b)|"
    rf"\s+(?=(?:what\s+(?:is|are|does|do)|how\s+|where\s+|find(?:\s+me)?\s+|"
    rf"suggest\s+|explain\s+|define\s+|tell\s+me\s+)\b)",
    re.I,
)
PARALLEL_REQUEST_RE = re.compile(
    r"\s+and\s+(?=(?:\d+\s*bhk|houses?\b|apartments?\b).{0,60}\bin\s+[A-Z])", re.I
)
GENERIC_ONSET_RE = re.compile(r"\s+(?=(?:what|when|where|who|how|find|tell\s+me|explain|suggest|define)\b)", re.I)
NOMINAL_TOPIC_RE = re.compile(
    r"\s+(?=(?:[A-Za-z]+\s+(?:plans?|prices?|options?|rules?)|polic(?:y|ies)|definition|meaning)\s+"
    r"(?:to|in|for|about|of)\b)", re.I
)
REPAIR_RE = re.compile(
    r"(?:[,—-]\s*)?\b(?:sorry|i mean|no(?:,)?\s+|no wait|rather|correction(?:\s*:)?)\b[,]?\s+",
    re.I,
)
FILLER_PHRASES = {"wait", "you know", "okay", "ok", "well", "so", "um", "uh", "erm", "hmm"}
UNIT_WORDS = {
    "people", "person", "persons", "participants", "attendees", "guests", "pax", "day", "days",
    "night", "nights", "hour", "hours", "week", "weeks", "month", "months", "year", "years",
    "rupees", "inr", "usd", "lakh", "lakhs", "crore", "crores", "percent", "percentage",
}
ANAPHORA = {"that", "it", "its", "there", "this", "these", "those", "same"}
DEFAULT_PLACES = {
    "ahmedabad", "bengaluru", "bangalore", "chennai", "delhi", "goa", "gurugram", "hyderabad",
    "jaipur", "kolkata", "mumbai", "new delhi", "noida", "pune",
}
RELATION_CUES = {
    "MAX": re.compile(r"\b(?:under|below|less than|at most|up to|maximum|max)\b", re.I),
    "MIN": re.compile(r"\b(?:over|above|more than|at least|starting from|minimum|min)\b", re.I),
}
AMOUNT_RE = re.compile(
    r"(?P<surface>(?:₹|rs\.?|inr\s*)?\s*\d[\d,]*(?:\.\d+)?\s*(?:lakh(?:s)?|crore(?:s)?)?)",
    re.I,
)
BHK_RE = re.compile(r"\b(?P<n>\d+)\s*[- ]?bhk\b", re.I)
PERCENT_RE = re.compile(r"\b(?P<n>\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)\b", re.I)
DURATION_RE = re.compile(r"\b(?P<n>\d+(?:\.\d+)?)\s+(?P<unit>day|week|month|year|hour)s?\b", re.I)


@dataclass
class ScopedConstraint:
    kind: str
    surface: str
    normalized_value: str | int | float
    owner_key: str
    relation: str = "EQ"
    unit: str | None = None
    currency: str | None = None
    ambiguous: bool = False
    start: int = 0
    end: int = 0

    def to_public(self) -> dict:
        return {
            "kind": self.kind, "surface": self.surface, "normalized_value": self.normalized_value,
            "owner_key": self.owner_key, "relation": self.relation, "unit": self.unit,
            "currency": self.currency, "ambiguous": self.ambiguous, "span": [self.start, self.end],
        }


@dataclass
class Slot:
    key: str
    kind: str
    value: str
    first_seen_t: float
    last_changed_t: float
    source_chunk_id: str
    revised_from: str | None = None
    owner_key: str | None = None
    normalized_value: str | int | float | None = None
    unit: str | None = None
    relation: str = "EQ"
    ambiguous: bool = False
    surface: str | None = None

    def to_public(self) -> dict:
        return {
            "key": self.key, "kind": self.kind, "value": self.value,
            "first_seen_t": round(self.first_seen_t, 3), "last_changed_t": round(self.last_changed_t, 3),
            "source_chunk_id": self.source_chunk_id, "revised_from": self.revised_from,
            "owner_key": self.owner_key, "normalized_value": self.normalized_value, "unit": self.unit,
            "relation": self.relation, "ambiguous": self.ambiguous, "surface": self.surface or self.value,
        }


@dataclass
class Span:
    text: str
    start: int
    kind: str
    closed: bool
    soft: bool = False
    end: int = 0
    request_cue: str | None = None


@dataclass
class CandidateNeed:
    key: str
    need_text: str
    constraints: list[str]
    closed: bool
    anaphoric: bool
    normalized_text: str = ""
    source_span: tuple[int, int] = (0, 0)
    source_chunk_range: tuple[str, str] | None = None
    request_cue: str | None = None
    topic_id: str = ""
    status: str = "VALID_STANDALONE"
    revision_history: list[dict] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    typed_constraints: list[ScopedConstraint] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join([self.need_text, *self.constraints]).strip()

    def to_public(self) -> dict:
        return {
            "key": self.key, "text": self.text, "need": self.need_text, "constraints": self.constraints,
            "typed_constraints": [c.to_public() for c in self.typed_constraints],
            "normalized_text": self.normalized_text, "source_span": list(self.source_span),
            "source_chunk_range": list(self.source_chunk_range) if self.source_chunk_range else None,
            "request_cue": self.request_cue, "topic_id": self.topic_id, "completion_status": self.status,
            "closed": self.closed, "anaphoric": self.anaphoric, "revision_history": self.revision_history,
            "confidence": self.confidence,
        }


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
    return bool(STRONG_ONSET_RE.search(text)) or any(
        f" {cue} " in low or low.strip().startswith(cue) for cue in REQUEST_CUES
    )


def _request_cue(text: str) -> str | None:
    match = STRONG_ONSET_RE.search(text)
    return match.group(0).lower() if match else None


def _slot_like(term: str, corpus: Corpus, places: set[str]) -> bool:
    return (is_number(term) or term in UNIT_WORDS or term in places or term in corpus.qualifier_group
            or term in {"event", "events", "bhk"})


def classify_span(text: str, corpus: Corpus, places: set[str] | None = None) -> str:
    places = places or (set(corpus.places) | DEFAULT_PLACES)
    low_text = re.sub(r"[^a-z ]", "", text.lower()).strip()
    terms = [t for t in lower_tokens(text) if t not in STOPWORDS]
    if not terms or low_text in FILLER_PHRASES or all(t in FILLERS | {"wait", "actually"} for t in lower_tokens(text)):
        return "FILLER"
    if _has_request_cue(text):
        return "NEED"
    first = lower_tokens(text)[0]
    normalized = [norm(t) for t in terms]
    if all(_slot_like(t, corpus, places) or _slot_like(n, corpus, places) for t, n in zip(terms, normalized)):
        return "CONSTRAINT"
    if first in PREPOSITIONS and len(terms) <= 5 and any(
        _slot_like(t, corpus, places) or t in corpus.proper_nouns for t in terms
    ):
        return "CONSTRAINT"
    return "NEED"


def apply_repairs(text: str, corpus: Corpus, places: set[str] | None = None) -> tuple[str, list[dict]]:
    """Apply only explicit, same-class self-repairs and retain their audit history."""
    places = {p.lower() for p in (places or set())} | set(corpus.places) | DEFAULT_PLACES
    repairs: list[dict] = []
    while True:
        match = REPAIR_RE.search(text)
        if not match:
            break
        before, after = text[:match.start()], text[match.end():]
        after_tokens = tokens(after)
        if not after_tokens:
            break
        new = after_tokens[0]
        new_low = new.lower()
        cls = ("place" if new_low in places else
               "qualifier" if new_low in corpus.qualifier_group else
               "number" if is_number(new_low) else None)
        replaced = False
        if cls:
            for token_match in reversed(list(re.finditer(r"[₹$]?\d[\d,]*|[A-Za-z]+", before))):
                old = token_match.group(0)
                old_low = old.lower()
                same_class = (
                    (cls == "place" and old_low in places)
                    or (cls == "qualifier" and corpus.qualifier_group.get(old_low) == corpus.qualifier_group.get(new_low))
                    or (cls == "number" and is_number(old_low))
                )
                if same_class:
                    text = before[:token_match.start()] + after
                    repairs.append({"kind": cls, "from": old, "to": new})
                    replaced = True
                    break
        if not replaced:
            text = before + " " + after
    return re.sub(r"\s+", " ", text).strip(), repairs


def segment(text: str, corpus: Corpus, places: set[str] | None = None) -> list[Span]:
    """Split on punctuation or strong new-request onsets, never on a bare conjunction alone."""
    places = places or (set(corpus.places) | DEFAULT_PLACES)
    boundaries = list(BOUNDARY_RE.finditer(text)) + list(PARALLEL_REQUEST_RE.finditer(text))
    for match in GENERIC_ONSET_RE.finditer(text):
        prefix_tokens = lower_tokens(text[:match.start()])
        onset = text[match.end():].split(maxsplit=1)[0].lower() if text[match.end():].strip() else ""
        # "explain what..." and "tell me what..." are one request, not nested requests.
        if prefix_tokens and prefix_tokens[-1] in {"explain", "define"}:
            continue
        if onset == "what" and len(prefix_tokens) >= 2 and prefix_tokens[-2:] == ["tell", "me"]:
            continue
        boundaries.append(match)
    for match in NOMINAL_TOPIC_RE.finditer(text):
        prefix_tokens = lower_tokens(text[:match.start()])
        if prefix_tokens and prefix_tokens[-1] in {
            "suggest", "find", "explain", "define", "show", "list", "is", "are", "the", "a", "an"
        }:
            continue
        if len(prefix_tokens) >= 2 and prefix_tokens[-2:] == ["tell", "me"]:
            continue
        boundaries.append(match)
    boundaries.sort(key=lambda m: (m.start(), m.end()))
    spans: list[Span] = []
    pos = 0
    for match in boundaries:
        if match.start() < pos:
            continue
        if match.start() > pos:
            raw = text[pos:match.start()]
            piece = raw.strip(" ,")
            if piece:
                offset = pos + raw.find(piece)
                spans.append(Span(piece, offset, classify_span(piece, corpus, places), True,
                                  end=offset + len(piece), request_cue=_request_cue(piece)))
        pos = match.end()
    if pos < len(text):
        raw = text[pos:]
        piece = raw.strip(" ,")
        if piece:
            offset = pos + raw.find(piece)
            spans.append(Span(piece, offset, classify_span(piece, corpus, places), False,
                              end=offset + len(piece), request_cue=_request_cue(piece)))
    return spans


def _topic_terms(text: str, places: set[str]) -> list[str]:
    ignored = {norm(p) for p in places} | UNIT_WORDS | {
        "house", "houses", "apartment", "apartments", "bhk", "price", "cost", "policy", "policies",
    }
    terms = [t for t in content_terms(text) if t not in ignored and not is_number(t)]
    if not terms:
        terms = [t for t in content_terms(text) if t not in UNIT_WORDS and not is_number(t)]
    return list(dict.fromkeys(terms))


def _topic_id(text: str, places: set[str]) -> str:
    seed = "|".join(_topic_terms(text, places)[:3]) or "provisional"
    return "T-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


def _candidate_key(text: str, start: int, places: set[str]) -> str:
    terms = _topic_terms(text, places)[:3]
    seed = "|".join(terms) if terms else f"pending@{start}"
    return "K-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def _relation(text: str, start: int) -> str:
    prefix = text[max(0, start - 28):start]
    for relation, pattern in RELATION_CUES.items():
        if pattern.search(prefix):
            return relation
    return "EQ"


def _amount_value(surface: str) -> tuple[float, str | None]:
    clean = re.sub(r"(?:₹|rs\.?|inr)", "", surface, flags=re.I).strip().lower().replace(",", "")
    multiplier = 10_000_000 if "crore" in clean else 100_000 if "lakh" in clean else 1
    number_match = re.search(r"\d+(?:\.\d+)?", clean)
    value = float(number_match.group(0)) * multiplier if number_match else 0.0
    currency = "INR" if multiplier > 1 or re.search(r"₹|rs\.?|inr", surface, re.I) else None
    return value, currency


def extract_typed_constraints(text: str, owner_key: str, places: set[str]) -> list[ScopedConstraint]:
    out: list[ScopedConstraint] = []
    occupied: list[tuple[int, int]] = []
    for match in BHK_RE.finditer(text):
        out.append(ScopedConstraint("bedrooms", match.group(0), int(match.group("n")), owner_key,
                                    unit="BHK", start=match.start(), end=match.end()))
        occupied.append(match.span())
    for match in PERCENT_RE.finditer(text):
        out.append(ScopedConstraint("percentage", match.group(0), float(match.group("n")), owner_key,
                                    unit="percent", ambiguous="rate" not in text.lower(),
                                    start=match.start(), end=match.end()))
        occupied.append(match.span())
    for match in DURATION_RE.finditer(text):
        out.append(ScopedConstraint("duration", match.group(0), float(match.group("n")), owner_key,
                                    unit=match.group("unit").lower(), relation=_relation(text, match.start()),
                                    start=match.start(), end=match.end()))
        occupied.append(match.span())
    for match in AMOUNT_RE.finditer(text):
        if any(a <= match.start() < b for a, b in occupied):
            continue
        surface = match.group("surface").strip()
        if not re.search(r"₹|rs\.?|inr|lakh|crore", surface, re.I):
            continue
        value, currency = _amount_value(surface)
        relation = _relation(text, match.start())
        nearby = text[max(0, match.start() - 35):min(len(text), match.end() + 20)].lower()
        per_month = bool(re.search(r"\bper\s+month\b", nearby))
        kind = "budget" if relation in {"MIN", "MAX"} or "budget" in nearby or "price" in nearby else "amount"
        out.append(ScopedConstraint(kind, surface, int(value) if value.is_integer() else value, owner_key,
                                    relation=relation, unit="per_month" if per_month else None,
                                    currency=currency or "INR", ambiguous=kind == "amount",
                                    start=match.start(), end=match.end()))
    low = text.lower()
    for place in sorted(places, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(place)}\b", low):
            if any(c.kind == "city" and c.start == match.start() for c in out):
                continue
            out.append(ScopedConstraint("city", text[match.start():match.end()], place.title(), owner_key,
                                        start=match.start(), end=match.end()))
    out.sort(key=lambda c: (c.start, c.kind))
    return out


def build_candidates(spans: list[Span], previous: list[CandidateNeed] | None = None,
                     places: set[str] | None = None, chunk_id: str | None = None) -> list[CandidateNeed]:
    places = places or DEFAULT_PLACES
    previous = previous or []
    candidates: list[CandidateNeed] = []
    pending: list[str] = []
    used_previous: set[str] = set()
    for span in spans:
        if span.kind == "FILLER":
            continue
        if span.kind == "CONSTRAINT":
            if candidates:
                candidates[-1].constraints.append(span.text)
                candidates[-1].closed = span.closed
            else:
                pending.append(span.text)
            continue
        text = span.text
        key = _candidate_key(text, span.start, places)
        best = None
        best_score = 0.0
        new_terms = set(content_terms(text))
        for old in previous:
            if old.key in used_previous:
                continue
            overlap = jaccard(new_terms, content_terms(old.need_text))
            same_origin = abs(old.source_span[0] - span.start) <= 8
            score = overlap + (0.55 if same_origin else 0.0)
            if score > best_score and (overlap >= 0.2 or same_origin):
                best, best_score = old, score
        history: list[dict] = []
        if best is not None:
            key = best.key
            used_previous.add(best.key)
            history = list(best.revision_history)
            if best.need_text != text:
                history.append({"from": best.need_text, "to": text})
        combined = " ".join([text, *pending]).strip()
        topic_terms = _topic_terms(combined, places)
        score = completeness(combined, span.closed, False)
        status = "VALID_STANDALONE"
        if score < 0.45 or not topic_terms:
            status = "INCOMPLETE"
        elif any(t in ANAPHORA for t in lower_tokens(text)) and not span.request_cue:
            status = "DEPENDENT_ELABORATION"
        candidate = CandidateNeed(
            key, text, list(pending), span.closed, any(t in ANAPHORA for t in lower_tokens(text)),
            normalized_text=" ".join(content_terms(combined)), source_span=(span.start, span.end),
            source_chunk_range=(chunk_id, chunk_id) if chunk_id else None,
            request_cue=span.request_cue or _request_cue(text), topic_id=_topic_id(combined, places),
            status=status, revision_history=history,
            confidence={"completeness": score, "topic": round(min(len(topic_terms) / 2, 1.0), 3)},
        )
        candidate.typed_constraints = extract_typed_constraints(candidate.text, key, places)
        candidates.append(candidate)
        pending = []
    if pending and not candidates:
        text = " ".join(pending)
        key = _candidate_key(text, spans[0].start if spans else 0, places)
        candidate = CandidateNeed(key, text, [], spans[-1].closed if spans else False, False,
                                  normalized_text=" ".join(content_terms(text)), status="CONSTRAINT_ONLY")
        candidate.typed_constraints = extract_typed_constraints(text, key, places)
        candidates.append(candidate)
    return candidates


def extract_slots(text: str, corpus: Corpus, candidates: list[CandidateNeed] | None = None,
                  places: set[str] | None = None) -> dict[str, tuple]:
    """Return slot tuples; owner-qualified keys prevent constraints leaking across intents."""
    places = places or (set(corpus.places) | DEFAULT_PLACES)
    candidates = candidates or [CandidateNeed("K-global", text, [], False, False)]
    out: dict[str, tuple] = {}
    multi = len(candidates) > 1
    for candidate in candidates:
        typed = candidate.typed_constraints or extract_typed_constraints(candidate.text, candidate.key, places)
        for constraint in typed:
            kind = "place" if constraint.kind == "city" else "number"
            base = "place" if kind == "place" else f"{constraint.kind}:{constraint.unit or 'value'}"
            key = f"{candidate.key}:{base}" if multi else base
            out[key] = (kind, constraint.surface, candidate.key, constraint.normalized_value,
                        constraint.unit, constraint.relation, constraint.ambiguous)
        for token in tokens(candidate.text):
            low = token.lower()
            if low in corpus.qualifier_group:
                base = corpus.qualifier_group[low]
                key = f"{candidate.key}:{base}" if multi else base
                out[key] = ("qualifier", low, candidate.key, low, None, "EQ", False)
    return out


def anchor_of(candidates: list[CandidateNeed]) -> str:
    if not candidates:
        return ""
    terms = [t for t in content_terms(candidates[0].text)
             if not is_number(t) and t not in UNIT_WORDS and t not in {"book"}]
    return " ".join(terms[:5])


def completeness(span_text: str, closed: bool, is_final: bool) -> float:
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
    if re.search(r"\b(?:find me|tell me|what is the|starting from|under)\s*$", span_text, re.I):
        score = min(score, 0.25)
    return round(min(score, 1.0), 4)


def continuation_hazard(span_text: str, closed: bool, is_final: bool) -> float:
    if is_final:
        return 0.0
    low = lower_tokens(span_text)
    if not low or low[-1] in CLOSED_CLASS or completeness(span_text, closed, False) < 0.45:
        return 0.85
    return 0.1 if closed else 0.5


class FrameBuilder:
    def __init__(self, corpus: Corpus, places: set[str] | None = None) -> None:
        self.corpus = corpus
        self.places = {p.lower() for p in (places or set())} | set(corpus.places) | DEFAULT_PLACES

    def build(self, raw_text: str, prev_slots: dict[str, Slot], t: float, chunk_id: str,
              previous_frame: Frame | None = None) -> Frame:
        analysis, repairs = apply_repairs(raw_text, self.corpus, self.places)
        spans = segment(analysis, self.corpus, self.places)
        candidates = build_candidates(spans, previous_frame.candidates if previous_frame else None,
                                      self.places, chunk_id)
        extracted = extract_slots(analysis, self.corpus, candidates, self.places)
        slots: dict[str, Slot] = {}
        for key, (kind, value, owner, normalized, unit, relation, ambiguous) in extracted.items():
            prev = prev_slots.get(key)
            if prev is None:
                slots[key] = Slot(key, kind, value, t, t, chunk_id, owner_key=owner,
                                  normalized_value=normalized, unit=unit, relation=relation,
                                  ambiguous=ambiguous, surface=value)
            elif prev.value.lower() != value.lower():
                slots[key] = Slot(key, kind, value, prev.first_seen_t, t, chunk_id,
                                  revised_from=prev.value, owner_key=owner, normalized_value=normalized,
                                  unit=unit, relation=relation, ambiguous=ambiguous, surface=value)
            else:
                slots[key] = prev
        return Frame(raw_text, analysis, spans, candidates, slots, repairs, anchor_of(candidates))
