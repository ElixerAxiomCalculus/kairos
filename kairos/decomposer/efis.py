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
from kairos.textutil import STOPWORDS, content_terms, is_number, jaccard, lower_tokens, norm, rbo

LABEL_STOP = {"book", "need", "tell", "whether", "available", "availability", "know", "find", "check", "detail",
              "summarize", "summarise", "explain", "list", "policy", "rule", "option", "info", "information"}

# Small, auditable equivalence classes for request-objective vocabulary.  These are linguistic
# normalisations, not domain facts: they let "cancellation rules" and "cancellation policy" agree
# without making the much broader topic word "event" sufficient for a merge.
OBJECTIVE_ALIASES = {
    "rule": "policy", "regulation": "policy", "guideline": "policy", "term": "policy",
    "define": "definition", "meaning": "definition", "describe": "explanation",
    "explain": "explanation", "cost": "price", "fee": "price", "charge": "price",
    "deadline": "timing", "period": "timing", "notice": "timing", "when": "timing",
    "available": "availability", "availability": "availability", "offer": "availability",
    "capacity": "capacity", "accommodate": "capacity", "seat": "capacity",
}
OBJECTIVE_MARKERS = {"policy", "definition", "explanation", "price", "timing", "availability", "capacity",
                     "penalty", "refund", "eligibility", "location", "process", "comparison"}


@dataclass
class Assignment:
    candidate_key: str
    sub_query: SubQuery
    created: bool
    merged_into: str | None
    footprint_overlap: dict[str, float]
    footprint_valid: bool = False
    standalone_status: str = "VALID_STANDALONE"
    merge_reason: str = "NEW_INTENT"
    decision_signals: dict[str, dict[str, float | bool | str]] | None = None


def _semantic_signature(text: str) -> tuple[set[str], set[str]]:
    """Return (subject, objective) terms used only as a conservative merge signal."""
    canonical = {OBJECTIVE_ALIASES.get(term, term) for term in content_terms(text)}
    objectives = canonical & OBJECTIVE_MARKERS
    subjects = canonical - objectives - {"event", "request", "detail", "information", "option"}
    return subjects, objectives


def _objective_compatibility(left: str, right: str) -> tuple[float, float]:
    left_subject, left_objective = _semantic_signature(left)
    right_subject, right_objective = _semantic_signature(right)
    subject_similarity = jaccard(left_subject, right_subject)
    if left_objective or right_objective:
        objective_similarity = jaccard(left_objective, right_objective)
    else:
        objective_similarity = subject_similarity
    return subject_similarity, objective_similarity


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

    def footprint_with_quality(self, q: RetrievalQuery) -> tuple[list[str], float]:
        res = self.index.search(q, top_k=5)
        keys: list[str] = []
        relevant = res.relevant_hits
        for h in relevant:
            chunk = self.index.corpus.chunk(h.chunk_id)
            key = chunk.parent_key or chunk.section_key
            if key not in keys:
                keys.append(key)
        quality = relevant[0].rerank_score if relevant else 0.0
        return keys, quality

    def footprint(self, q: RetrievalQuery) -> list[str]:
        return self.footprint_with_quality(q)[0]

    def own_footprint(self, need_texts: list[str], constraints: list[str]) -> list[str]:
        return self.footprint(RetrievalQuery.build(" ; ".join(need_texts), "", constraints))

    def assign(self, cand: CandidateNeed, frame: Frame, existing: list[SubQuery], next_id: str,
               utterance_id: str, first_sq: SubQuery | None) -> Assignment:
        first_key = frame.candidates[0].key if frame.candidates else cand.key
        anchor = frame.anchor if cand.key != first_key else ""
        q = self.query_for([cand.need_text], cand.constraints, anchor)
        own_query = RetrievalQuery.build(cand.need_text, "", cand.constraints)
        fp, fp_quality = self.footprint_with_quality(own_query)
        footprint_valid = (not self.cfg.get("footprint_validity_gate", True) and bool(fp)) or (
            fp_quality >= float(self.cfg.get("footprint_min_relevance", 0.35))
        )
        overlaps: dict[str, float] = {}
        decision_signals: dict[str, dict[str, float | bool | str]] = {}
        best: SubQuery | None = None
        best_val = -1.0
        best_reason = "NEW_INTENT"
        cvec = self.index.encode(" ".join([cand.need_text, *cand.constraints]))
        cand_terms = set(content_terms(cand.need_text))
        for sq in existing:
            overlap = rbo(fp, sq.footprint, 0.8)
            cos = float(cvec @ self.index.encode(" ".join(sq.need_texts)))
            overlaps[sq.id] = round(overlap, 3)
            lexical = jaccard(cand_terms, content_terms(" ".join(sq.need_texts)))
            subject_similarity, objective_compatibility = _objective_compatibility(
                cand.need_text, " ".join(sq.need_texts)
            )
            constraints_compatible = not any(
                a["kind"] == b["kind"] and a["normalized_value"] != b["normalized_value"]
                for a in [c.to_public() for c in cand.typed_constraints]
                for b in sq.typed_constraints
            )
            explicit_topic_shift = bool(
                cand.request_cue
                and subject_similarity < float(self.cfg.get("semantic_subject_min", 0.5))
                and lexical < float(self.cfg.get("lexical_merge_min", 0.28))
            )
            existing_valid = (not self.cfg.get("footprint_validity_gate", True) and bool(sq.footprint)) or (
                sq.footprint_quality >= float(self.cfg.get("footprint_min_relevance", 0.35))
            )
            footprint_merge = (footprint_valid and existing_valid
                               and overlap >= self.cfg["theta_merge"])
            semantic_merge = cos >= self.cfg["theta_sem"] and lexical >= float(self.cfg.get("lexical_merge_min", 0.28))
            objective_merge = (
                self.cfg.get("semantic_objective_merge", True)
                and
                subject_similarity >= float(self.cfg.get("semantic_subject_min", 0.5))
                and objective_compatibility >= float(self.cfg.get("objective_compatibility_min", 0.75))
            )
            can_merge = constraints_compatible and objective_compatibility > 0 and (
                footprint_merge or semantic_merge or objective_merge
            )
            if self.cfg.get("topic_shift_guard", True) and explicit_topic_shift:
                can_merge = False
            score = (
                0.25 * overlap + 0.25 * cos + 0.15 * lexical
                + 0.20 * subject_similarity + 0.15 * objective_compatibility
            )
            reason = (
                "SEMANTIC_OBJECTIVE_EQUIVALENCE" if objective_merge
                else "VALIDATED_EVIDENCE_FOOTPRINT" if footprint_merge
                else "DENSE_LEXICAL_EQUIVALENCE" if semantic_merge
                else "TOPIC_TRANSITION" if explicit_topic_shift
                else "CONSTRAINT_CONFLICT" if not constraints_compatible
                else "INSUFFICIENT_EQUIVALENCE"
            )
            decision_signals[sq.id] = {
                "semantic_equivalence": round(max(cos, subject_similarity), 4),
                "evidence_footprint_similarity": round(overlap, 4),
                "evidence_footprint_validity": bool(footprint_valid and existing_valid),
                "constraint_compatibility": constraints_compatible,
                "topic_transition": explicit_topic_shift,
                "request_objective_compatibility": round(objective_compatibility, 4),
                "merge_score": round(score, 4),
                "decision": "MERGE" if can_merge else "SPLIT",
                "reason": reason,
            }
            if can_merge and score > best_val:
                best, best_val, best_reason = sq, score, reason
        if best is not None:
            best.need_texts.append(cand.need_text)
            best.constraints.extend(c for c in cand.constraints if c not in best.constraints)
            best.candidate_keys.append(cand.key)
            best.merged_from.append(cand.key)
            best.query = self.query_for(best.need_texts, best.constraints, best.anchor, best.facet_constraints,
                                        best.text)
            best.footprint = self.own_footprint(best.need_texts, best.constraints)
            best.footprint_quality = max(best.footprint_quality, fp_quality)
            return Assignment(cand.key, best, False, best.id, overlaps, footprint_valid, cand.status,
                              best_reason, decision_signals)

        sq = SubQuery(
            id=next_id,
            utterance_id=utterance_id,
            label=make_label(cand),
            need_texts=[cand.need_text],
            constraints=list(cand.constraints),
            anchor=anchor,
            candidate_keys=[cand.key],
            footprint=fp,
            typed_constraints=[c.to_public() for c in cand.typed_constraints],
            footprint_quality=fp_quality,
        )
        if cand.status == "VALID_STANDALONE" and cand.confidence.get("topic", 1.0) < self.cfg.get("chi_min", 0.12):
            cand.status = "REQUIRES_CLARIFICATION"
        if first_sq is not None and cand.key != first_key and cand.anaphoric:
            sq.depends_on = [first_sq.id]
            sq.dependency_type = "soft"
        sq.query = self.query_for(sq.need_texts, sq.constraints, sq.anchor, None, sq.text)
        return Assignment(cand.key, sq, True, None, overlaps, footprint_valid, cand.status,
                          "NEW_INTENT", decision_signals)

    def refresh(self, sq: SubQuery, cand: CandidateNeed) -> None:
        """Candidate text grew after retrieval (REFINE): rebuild the sub-query from its latest text."""
        idx = sq.candidate_keys.index(cand.key) if cand.key in sq.candidate_keys else 0
        if idx < len(sq.need_texts):
            sq.need_texts[idx] = cand.need_text
        for c in cand.constraints:
            if c not in sq.constraints:
                sq.constraints.append(c)
        sq.label = make_label(cand) if idx == 0 else sq.label
        sq.typed_constraints = [c.to_public() for c in cand.typed_constraints]
        sq.query = self.query_for(sq.need_texts, sq.constraints, sq.anchor, sq.facet_constraints, sq.text)
        sq.footprint, sq.footprint_quality = self.footprint_with_quality(
            RetrievalQuery.build(" ; ".join(sq.need_texts), "", sq.constraints)
        )
