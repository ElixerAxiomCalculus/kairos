"""Turn classifier: NEW_INFO | CONSTRAINT | TRANSFORM | ELABORATE.

Feature-based and deterministic. The decisive feature is *information novelty*: content terms
that are neither presentation vocabulary nor already present in the session's evidence/query
vocabulary. A presentation request ("make that shorter", "two bullets", "in Hindi") carries none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kairos.corpus.ingest import Corpus
from kairos.streaming.frame import UNIT_WORDS
from kairos.textutil import content_terms, is_number, lower_tokens, norm

TRANSFORM_PATTERNS = [
    r"\bshort(?:er|en)?\b", r"\bbrief(?:er|ly)?\b", r"\bconcise\b", r"\btl;?dr\b",
    r"\bsummar(?:y|ise|ize)\s+(?:that|it|this|the\s+(?:answer|above|previous))\b",
    r"\b(?:\w+|\d+)\s+bullets?\b", r"\bbullet(?:\s+points?)?\b", r"\bbullets\b",
    r"\btranslate\b", r"\bin\s+(?:hindi|marathi|tamil|telugu|kannada|bengali|french|german|spanish|japanese)\b",
    r"\bsimpl(?:er|y|ify)\b", r"\bplain(?:er)?\s+(?:language|english|words)\b", r"\brephrase\b", r"\breword\b",
    r"\bas\s+a\s+(?:table|list)\b", r"\bin\s+(?:a\s+)?(?:table|list)\b", r"\bone\s+(?:line|sentence|paragraph)\b",
    r"\bformat\b", r"\bnumbered\b",
]
ELABORATE_PATTERNS = [
    r"\bmore\s+detail(?:s|ed)?\b", r"\btell\s+me\s+more\b", r"\belaborate\b", r"\bexpand\s+on\b",
    r"\bexplain\s+(?:more|further|that|the)\b", r"\bgo\s+deeper\b", r"\bin\s+detail\b",
]
CONSTRAINT_CUES = [
    r"^\s*actually\b", r"\bonly\b", r"\binstead\b", r"\b(?:this|it)\s+(?:is|'s|will\s+be)\s+for\b", r"\bit's\s+for\b",
    r"\brather\b", r"\bwhat\s+if\b", r"\bswitch\s+to\b", r"\bassume\b", r"^\s*no[,\s]", r"\bsorry\b",
    r"\bi\s+mean\b", r"\bchange\s+(?:it|that)\s+to\b", r"\bmake\s+it\s+(?:for|an?)\b",
]
PRESENTATION_VOCAB = {
    "make", "shorter", "short", "shorten", "brief", "briefer", "briefly", "concise", "summary", "summarize",
    "summarise", "bullet", "bullets", "point", "points", "translate", "hindi", "marathi", "tamil", "telugu",
    "kannada", "bengali", "french", "german", "spanish", "japanese", "simpler", "simple", "simply",
    "simplify", "plain", "language", "english", "word", "words", "rephrase", "reword", "table", "list",
    "line", "sentence", "paragraph", "format", "numbered", "answer", "above", "previous", "version",
    "explain", "put", "write", "rewrite", "turn", "into", "two", "three", "four", "five", "one", "tl", "dr",
    "detail", "details", "detailed", "elaborate", "expand", "deeper", "go", "more", "less", "long", "longer",
    "keep", "same", "could", "can", "pleas", "please", "okay", "now",
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


@dataclass
class TurnResult:
    turn_type: str
    confidence: float
    features: dict = field(default_factory=dict)
    transform: dict = field(default_factory=dict)

    def to_public(self) -> dict:
        return {"turn_type": self.turn_type, "confidence": round(self.confidence, 3),
                "features": self.features, "transform": self.transform}


def parse_transform(text: str) -> dict:
    low = text.lower()
    spec: dict = {"ops": []}
    m = re.search(r"\b(\d+|one|two|three|four|five|six)\s+bullet", low)
    if m:
        n = m.group(1)
        spec["bullets"] = int(n) if n.isdigit() else NUMBER_WORDS[n]
        spec["ops"].append("bullets")
    elif re.search(r"\bbullet", low) or re.search(r"\bas\s+a\s+list\b|\bin\s+(?:a\s+)?list\b|\bnumbered\b", low):
        spec["ops"].append("bullets")
    if re.search(r"\bshort(?:er|en)?\b|\bbrief|\bconcise\b|\btl;?dr\b|\bsummar|\bone\s+(?:line|sentence|paragraph)\b", low):
        spec["ops"].append("condense")
    lang = re.search(r"\b(?:into|in|to)\s+(hindi|marathi|tamil|telugu|kannada|bengali|french|german|spanish|japanese)\b", low)
    if lang or "translate" in low:
        spec["ops"].append("translate")
        spec["language"] = lang.group(1).title() if lang else None
    if re.search(r"\bsimpl|\bplain", low):
        spec["ops"].append("simplify")
    if not spec["ops"]:
        spec["ops"].append("rephrase")
    return spec


class TurnClassifier:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def _slot_like(self, term: str) -> bool:
        return (
            is_number(term) or term in UNIT_WORDS or term in self.corpus.places
            or term in self.corpus.qualifier_group or term in {"event", "events"}
        )

    def classify(self, text: str, has_answer: bool, session_vocab: set[str], is_final: bool) -> TurnResult:
        low = text.lower()
        terms = content_terms(text)
        new_terms = [t for t in terms if t not in PRESENTATION_VOCAB and t not in session_vocab]
        slot_terms = [t for t in lower_tokens(text) if self._slot_like(t) or self._slot_like(norm(t))]
        non_slot_new = [t for t in new_terms if not self._slot_like(t)]
        features = {
            "has_answer": has_answer,
            "transform_cue": _any(TRANSFORM_PATTERNS, low),
            "elaborate_cue": _any(ELABORATE_PATTERNS, low),
            "constraint_cue": _any(CONSTRAINT_CUES, low),
            "content_terms": terms,
            "novel_terms": new_terms,
            "slot_terms": slot_terms,
        }
        if not has_answer:
            return TurnResult("NEW_INFO", 1.0, features)
        if features["transform_cue"] and not new_terms:
            return TurnResult("TRANSFORM", 0.95 if is_final else 0.8, features, parse_transform(text))
        if features["elaborate_cue"] and not non_slot_new:
            return TurnResult("ELABORATE", 0.85, features)
        if features["constraint_cue"] and not non_slot_new:
            conf = 0.9 if slot_terms or new_terms else 0.55
            return TurnResult("CONSTRAINT", conf, features)
        if slot_terms and not non_slot_new and len(terms) <= 4:
            return TurnResult("CONSTRAINT", 0.7, features)
        qualifier_terms = [t for t in slot_terms if t in self.corpus.qualifier_group or t in self.corpus.places]
        if features["constraint_cue"] and qualifier_terms and len(non_slot_new) <= 1 and len(terms) <= 5:
            # "Actually the trip is international": one new noun alongside a known facet value.
            return TurnResult("CONSTRAINT", 0.75, features)
        if features["transform_cue"] and len(non_slot_new) <= 1 and len(terms) <= 4:
            return TurnResult("TRANSFORM", 0.6, features, parse_transform(text))
        if not terms and not is_final:
            return TurnResult("NEW_INFO", 0.3, features)
        return TurnResult("NEW_INFO", 0.85 if non_slot_new else 0.6, features)
