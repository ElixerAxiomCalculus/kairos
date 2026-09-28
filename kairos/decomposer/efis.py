"""Evidence-Footprint Intent Segmentation (EFIS).

Operational definition: two clauses are separate retrieval intents iff they need different
evidence AND each is answerable on its own.

* footprint(c)  = parent-section keys of the top-5 hybrid hits for clause c's OWN terms (no anchor
                  context, which would make every later clause look like the first)
* merge(i, j)   ⟺ RBO_0.8(F_i, F_j) ≥ θ_merge  ∨  cos(e_i, e_j) ≥ θ_sem
  Parent level: 'Cancellation > Standard' and 'Cancellation > International' are one topic.
* standalone(c) ⟺ clarity(c) ≥ χ_min

Constraint clauses never become intents; they are attached to the need they modify (frame.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from kairos.retrieval.index import HybridIndex, RetrievalQuery
from kairos.schemas.models import SubQuery
from kairos.streaming.frame import UNIT_WORDS, CandidateNeed, Frame
from kairos.textutil import STOPWORDS, content_terms, is_number, lower_tokens, norm, rbo

LABEL_STOP = {"book", "need", "tell", "whether", "available", "availability", "know", "find", "check", "detail",
              "summarize", "summarise", "explain", "list", "policy", "rule", "option", "info", "information"}


@dataclass
class Assignment:
    candidate_key: str
    sub_query: SubQuery
    created: bool
    merged_into: str | None
    footprint_overlap: dict[str, float]


def _surface_terms(text: str) -> list[str]:
    """Content words as spoken (lower-cased), filtered on their normalised form."""
    out = []
    for tok in lower_tokens(text):
        n = norm(tok)
        if tok in STOPWORDS or n in LABEL_STOP or n in UNIT_WORDS or tok in UNIT_WORDS or is_number(tok) or len(tok) < 2:
            continue
        out.append(tok)
    return out


def make_label(cand: CandidateNeed) -> str:
    """Short human label: salient need words plus attached constraint words."""
    terms = _surface_terms(cand.need_text)
    if "policy" in lower_tokens(cand.need_text):
        terms.append("policy")
    base = " ".join(dict.fromkeys(terms))[:40] or cand.need_text[:40]
    extras = [" ".join(w for w in lower_tokens(c) if w not in STOPWORDS) for c in cand.constraints]
    return " · ".join([base, *[e for e in extras if e]])


class Decomposer:
    def __init__(self, cfg: dict, index: HybridIndex) -> None:
        self.cfg = cfg
        self.index = index

    def query_for(self, need_texts: list[str], constraints: list[str], anchor: str,
                  facet_constraints: dict[str, str] | None = None, display: str | None = None) -> RetrievalQuery:
        return RetrievalQuery.build(" ; ".join(need_texts), anchor, constraints, self.cfg["anchor_weight"],
                                    facet_constraints, display)

    def footprint(self, q: RetrievalQuery) -> list[str]:
        res = self.index.search(q, top_k=5)
        keys: list[str] = []
        for h in res.hits:
            chunk = self.index.corpus.chunk(h.chunk_id)
            key = chunk.parent_key or chunk.section_key
            if key not in keys:
                keys.append(key)
        return keys

    def own_footprint(self, need_texts: list[str], constraints: list[str]) -> list[str]:
        return self.footprint(RetrievalQuery.build(" ; ".join(need_texts), "", constraints))

    def assign(self, cand: CandidateNeed, frame: Frame, existing: list[SubQuery], next_id: str,
               utterance_id: str, first_sq: SubQuery | None) -> Assignment:
        anchor = frame.anchor if cand.key != "k0" else ""
        q = self.query_for([cand.need_text], cand.constraints, anchor)
        fp = self.own_footprint([cand.need_text], cand.constraints)
        overlaps: dict[str, float] = {}
        best: SubQuery | None = None
        best_val = 0.0
        cvec = self.index.encode(" ".join([cand.need_text, *cand.constraints]))
        for sq in existing:
            overlap = rbo(fp, sq.footprint, 0.8)
            cos = float(cvec @ self.index.encode(" ".join(sq.need_texts)))
            overlaps[sq.id] = round(overlap, 3)
            if (overlap >= self.cfg["theta_merge"] or cos >= self.cfg["theta_sem"]) and overlap + cos > best_val:
                best, best_val = sq, overlap + cos
        if best is not None:
            best.need_texts.append(cand.need_text)
            best.constraints.extend(c for c in cand.constraints if c not in best.constraints)
            best.candidate_keys.append(cand.key)
            best.merged_from.append(cand.key)
            best.query = self.query_for(best.need_texts, best.constraints, best.anchor, best.facet_constraints,
                                        best.text)
            best.footprint = self.own_footprint(best.need_texts, best.constraints)
            return Assignment(cand.key, best, False, best.id, overlaps)

        sq = SubQuery(
            id=next_id,
            utterance_id=utterance_id,
            label=make_label(cand),
            need_texts=[cand.need_text],
            constraints=list(cand.constraints),
            anchor=anchor,
            candidate_keys=[cand.key],
            footprint=fp,
        )
        if first_sq is not None and cand.key != "k0" and (cand.anaphoric or anchor):
            sq.depends_on = [first_sq.id]
            sq.dependency_type = "soft"
        sq.query = self.query_for(sq.need_texts, sq.constraints, sq.anchor, None, sq.text)
        return Assignment(cand.key, sq, True, None, overlaps)

    def refresh(self, sq: SubQuery, cand: CandidateNeed) -> None:
        """Candidate text grew after retrieval (REFINE): rebuild the sub-query from its latest text."""
        idx = sq.candidate_keys.index(cand.key) if cand.key in sq.candidate_keys else 0
        if idx < len(sq.need_texts):
            sq.need_texts[idx] = cand.need_text
        for c in cand.constraints:
            if c not in sq.constraints:
                sq.constraints.append(c)
        sq.label = make_label(cand) if idx == 0 else sq.label
        sq.query = self.query_for(sq.need_texts, sq.constraints, sq.anchor, sq.facet_constraints, sq.text)
        sq.footprint = self.own_footprint(sq.need_texts, sq.constraints)
