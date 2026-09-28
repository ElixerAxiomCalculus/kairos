"""Lightweight, deterministic text processing (no model downloads).

Closed-class word lists stand in for a POS tagger: the controller only needs to know
whether an utterance prefix ends on a function word ("I need to book a…"), which these
lists answer exactly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

TOKEN_RE = re.compile(r"[₹$]\s?\d[\d,]*(?:\.\d+)?|\d+(?:[.,]\d+)*%?|[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹$(])|\n+")
NUMBER_RE = re.compile(r"^[₹$]?\s?\d[\d,]*(?:\.\d+)?%?$")

DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "our", "your", "their",
    "his", "her", "its", "some", "any", "each", "every", "no", "another",
}
PREPOSITIONS = {
    "in", "on", "at", "for", "to", "of", "with", "from", "by", "about", "into", "over",
    "under", "between", "during", "before", "after", "within", "without", "per", "near",
    "around", "across", "via", "than", "like",
}
CONJUNCTIONS = {"and", "or", "but", "nor", "so", "yet", "whether", "if", "because", "while", "plus"}
AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have",
    "has", "had", "can", "could", "will", "would", "should", "may", "might", "must", "shall",
}
PARTICLES = {"to", "not", "up", "off"}
PRONOUNS = {
    "i", "me", "we", "us", "you", "he", "she", "it", "they", "them", "what", "which", "who",
    "whom", "whose", "there", "here", "one", "ones",
}
FILLERS = {"um", "uh", "erm", "hmm", "uhh", "umm", "okay", "ok", "so", "well", "like", "please"}
CLOSED_CLASS = DETERMINERS | PREPOSITIONS | CONJUNCTIONS | AUXILIARIES | PARTICLES

STOPWORDS = CLOSED_CLASS | PRONOUNS | FILLERS | {
    "i'm", "it's", "that's", "there's", "what's", "let", "let's", "just", "also", "really", "very",
    "tell", "know", "need", "want", "get", "give", "show", "find", "like", "would", "any",
    "all", "how", "when", "where", "why", "much", "many", "more", "most", "other", "such",
    "only", "own", "same", "too", "s", "t", "don't", "able", "then", "than", "as", "okay",
    "actually", "please", "thanks", "thank", "yes", "no", "hi", "hello", "these", "those",
    "am", "me", "my", "mine", "being", "into", "out", "again", "further", "once",
}

REQUEST_CUES = (
    "tell me", "let me know", "what", "whether", "how", "is there", "are there", "can i",
    "can we", "could you", "do you", "does", "do we", "do i", "i need", "we need", "find",
    "book", "summarize", "summarise", "explain", "list", "show me", "which", "when", "where",
    "who", "is it", "are we", "am i", "give me", "check", "looking for", "i want", "we want",
)


def tokens(text: str) -> list[str]:
    """Surface tokens preserving case."""
    return TOKEN_RE.findall(text)


def lower_tokens(text: str) -> list[str]:
    return [t.lower() for t in tokens(text)]


# Headcount nouns are interchangeable in queries ("30 people" ~ "30 participants"). Linguistic
# normalisation only — no factual knowledge is introduced.
CANONICAL = {"participant": "people", "attendee": "people", "person": "people", "pax": "people",
             "persons": "people", "participants": "people", "attendees": "people"}


def norm(token: str) -> str:
    """Light, deterministic suffix stripping so 'bookings'~'booking', 'caterers'~'caterer'."""
    t = _norm(token)
    return CANONICAL.get(t, t)


def _norm(token: str) -> str:
    t = token.lower().strip("'")
    if t.endswith("'s"):
        t = t[:-2]
    if NUMBER_RE.match(t):
        return t.replace(",", "").replace(" ", "")
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("sses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    if len(t) > 6 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 5 and t.endswith("ed") and not t.endswith("eed"):
        return t[:-2]
    return t


def content_terms(text: str) -> list[str]:
    """Normalised content terms (stopwords removed), order preserved, duplicates kept."""
    out = []
    for tok in lower_tokens(text):
        if tok in STOPWORDS or len(tok) < 2 and not tok.isdigit():
            continue
        out.append(norm(tok))
    return out


def content_set(text: str) -> set[str]:
    return set(content_terms(text))


def is_number(tok: str) -> bool:
    return bool(NUMBER_RE.match(tok.strip()))


def sentences(text: str) -> list[str]:
    out = []
    for part in SENT_SPLIT_RE.split(text):
        s = part.strip().lstrip("-*• ").strip()
        if len(s) >= 12 and any(c.isalpha() for c in s):
            out.append(s)
    return out


def anchors(text: str) -> set[str]:
    """Verifiable anchors: numbers/amounts/percentages and mid-sentence proper nouns."""
    found: set[str] = set()
    toks = tokens(text)
    for i, tok in enumerate(toks):
        if any(ch.isdigit() for ch in tok):
            found.add(tok.replace(",", "").replace(" ", "").lower())
        elif tok[:1].isupper() and i > 0 and tok.lower() not in STOPWORDS and len(tok) > 2:
            prev = toks[i - 1]
            if not prev.endswith((".", "!", "?")):
                found.add(tok.lower())
    return found


def estimate_tokens(text: str) -> int:
    """Approximate LLM token count (≈1.3 tokens per word-ish unit). Labelled '_est' wherever reported."""
    return int(math.ceil(len(re.findall(r"\w+|[^\w\s]", text)) * 1.1))


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rbo(a: Sequence, b: Sequence, p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap (Webber et al., 2010) for two finite rankings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    k = max(len(a), len(b))
    score = 0.0
    agreement = 0.0
    for d in range(1, k + 1):
        agreement = len(set(a[:d]) & set(b[:d])) / d
        score += agreement * (p ** (d - 1))
    return float((1 - p) * score + agreement * (p ** k))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
