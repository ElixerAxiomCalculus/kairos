"""Claim-Dependency Delta Maintenance (CDDM): answers are materialized views over the evidence graph.

On a late constraint k:
1. affected(sq)   — k's facet dimension exists in sq's evidence (or its sibling sections), k replaces a
                    slot sq depends on, or k shifts sq's probe *within sq's own documents*.
2. delta retrieval — only affected sub-queries are re-retrieved, with k as a facet filter.
3. plan(claim)    — KEEP (sub-query unaffected / general rule not overridden), REVISE (facet conflict,
                    replaced slot, or overridden by more specific evidence), RETRACT (no replacement).
                    ADD new constraint-specific evidence.
4. Only REVISE/ADD claims are synthesized; KEEP claims are carried verbatim with their citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kairos.corpus.ingest import Corpus
from kairos.retrieval.index import HybridIndex, RetrievalQuery
from kairos.schemas.models import Claim, SubQuery
from kairos.streaming.frame import Frame
from kairos.textutil import content_terms, jaccard, sentences

CUE_TERMS = {"actually", "instead", "only", "rather", "assume", "switch", "sorry", "mean", "change", "make"}


@dataclass
class Constraint:
    text: str
    facets: dict[str, str] = field(default_factory=dict)
    slots: list[dict] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return "|".join(sorted(self.terms)) + "#" + "|".join(f"{k}={v}" for k, v in sorted(self.facets.items()))

    @property
    def query_text(self) -> str:
        return " ".join(self.terms)

    def to_public(self) -> dict:
        return {"text": self.text, "facets": self.facets, "slots": self.slots, "terms": self.terms}


@dataclass
class AffectedSubQuery:
    sub_query: SubQuery
    reasons: list[str]
    replaced_slots: list[dict] = field(default_factory=list)


def extract_constraint(frame: Frame, corpus: Corpus) -> Constraint | None:
    facets: dict[str, str] = {}
    slots = []
    for slot in frame.slots.values():
        slots.append(slot.to_public())
        if slot.kind == "qualifier":
            facets[slot.key] = slot.value
    terms = [t for t in content_terms(frame.analysis_text) if t not in CUE_TERMS]
    if not terms and not slots:
        return None
    return Constraint(frame.raw_text.strip(), facets, slots, list(dict.fromkeys(terms)))


class DeltaEngine:
    def __init__(self, index: HybridIndex, cfg: dict, anchor_weight: float) -> None:
        self.index = index
        self.corpus = index.corpus
        self.cfg = cfg
        self.anchor_weight = anchor_weight

    def affected(self, sub_queries: list[SubQuery], constraint: Constraint,
                 claim_evidence: dict[str, list[str]] | None = None) -> tuple[list[AffectedSubQuery], list[dict]]:
        """claim_evidence: sub-query id -> chunk ids its *claims* cite. The answer depends on those chunks,
        not on every retrieved hit, so dependency is checked against them (falls back to all hits)."""
        affected, unaffected = [], []
        for sq in sub_queries:
            reasons: list[str] = []
            cited = (claim_evidence or {}).get(sq.id) or sq.evidence
            evidence = [self.corpus.chunk(cid) for cid in cited if self.corpus.chunk(cid)]
            for group in constraint.facets:
                if any(ch.facets.get(group) for ch in evidence):
                    reasons.append("FACET_IN_EVIDENCE")
                elif any(self.corpus.siblings_with_group(ch.section_key, group) for ch in evidence):
                    reasons.append("FACET_IN_SIBLING_SECTIONS")
            replaced = []
            for slot in constraint.slots:
                if slot["kind"] != "place":
                    continue
                for place in self.corpus.places:
                    if place != slot["value"].lower() and re.search(rf"\b{place}\b", sq.text, re.I):
                        replaced.append({"from": place, "to": slot["value"]})
            if replaced:
                reasons.append("SLOT_REPLACED")
            new_q = self.constrained_query(sq, constraint, replaced)
            shadow = self.index.search(new_q, top_k=3).relevant_hits
            docs = {ch.doc_id for ch in evidence}
            specific = [h.chunk_id for h in shadow if h.chunk_id not in cited
                        and self.corpus.chunk(h.chunk_id).doc_id in docs
                        and self._chunk_carries(h.chunk_id, constraint)
                        and self._scope_compatible(sq, h.chunk_id, constraint)]
            if specific:
                reasons.append("CONSTRAINT_SPECIFIC_EVIDENCE")
            if reasons:
                affected.append(AffectedSubQuery(sq, sorted(set(reasons)), replaced))
            else:
                unaffected.append({"sub_query_id": sq.id, "reasons": ["NO_DEPENDENCY_ON_CONSTRAINT"],
                                   "shadow_top": [h.chunk_id for h in shadow]})
        return affected, unaffected

    def _chunk_carries(self, chunk_id: str, constraint: Constraint) -> bool:
        """Does this chunk state something specific to the constraint (facet value or constraint term)?"""
        ch = self.corpus.chunk(chunk_id)
        if ch is None:
            return False
        if any(ch.facets.get(g) == v for g, v in constraint.facets.items()):
            return True
        return bool(set(constraint.terms) & set(content_terms(f"{ch.section_path} {ch.text}")) - {"event"})

    def _scope_compatible(self, sq: SubQuery, chunk_id: str, constraint: Constraint) -> bool:
        """Require a constraint-specific shadow hit to address this intent's local subject.

        This prevents a broad document-level facet from invalidating unrelated sibling claims
        (for example, an international-travel change must not churn non-reimbursable expenses).
        """
        if not self.cfg.get("scope_guard", True):
            return True
        chunk = self.corpus.chunk(chunk_id)
        if chunk is None:
            return False
        sq_text = sq.text.lower()
        chunk_text = f"{chunk.section_path} {chunk.text}".lower()
        negative = bool(re.search(r"\b(?:not|non[- ]?reimbursable|excluded|ineligible|without)\b", sq_text))
        if negative and not re.search(r"\b(?:not|non[- ]?reimbursable|excluded|ineligible|without)\b", chunk_text):
            return False
        from kairos.synthesis.base import focus_terms

        generic = set(constraint.terms) | {"event", "events", "travel", "policy", "rule", "expense", "expenses"}
        local = set(focus_terms(self.index, sq.need_texts, k=3)) - generic
        present = set(content_terms(chunk_text))
        return bool(local & present)

    def constrained_query(self, sq: SubQuery, constraint: Constraint, replaced: list[dict]) -> RetrievalQuery:
        needs = list(sq.need_texts)
        anchor = sq.anchor
        for rep in replaced:
            needs = [re.sub(rf"\b{rep['from']}\b", rep["to"], n, flags=re.I) for n in needs]
            anchor = re.sub(rf"\b{rep['from']}\b", rep["to"], anchor, flags=re.I)
        facets = {**sq.facet_constraints, **constraint.facets}
        constraints = [*sq.constraints, constraint.query_text]
        return RetrievalQuery.build(" ; ".join(needs), anchor, constraints, self.anchor_weight, facets,
                                    f"{' / '.join(needs)} [{'; '.join(constraints)}]")

    def apply_to_subquery(self, a: AffectedSubQuery, constraint: Constraint) -> None:
        sq = a.sub_query
        for rep in a.replaced_slots:
            sq.need_texts = [re.sub(rf"\b{rep['from']}\b", rep["to"], n, flags=re.I) for n in sq.need_texts]
            sq.anchor = re.sub(rf"\b{rep['from']}\b", rep["to"], sq.anchor, flags=re.I)
        if constraint.query_text and constraint.query_text not in sq.constraints:
            sq.constraints.append(constraint.query_text)
        sq.facet_constraints.update(constraint.facets)
        sq.query = self.constrained_query(sq, Constraint(constraint.text, {}, [], []), [])
        sq.query.facet_constraints = dict(sq.facet_constraints)

    def plan(self, claims: list[Claim], affected: dict[str, AffectedSubQuery], constraint: Constraint,
             new_chunk_ids: dict[str, list[str]]) -> dict[str, dict]:
        plan: dict[str, dict] = {}
        for claim in claims:
            a = affected.get(claim.sub_query_id)
            if a is None:
                plan[claim.claim_id] = {"action": "KEEP", "reasons": ["SUBQUERY_UNAFFECTED"]}
                continue
            reasons = []
            chunks = [self.corpus.chunk(c) for c in claim.evidence if self.corpus.chunk(c)]
            for group, value in constraint.facets.items():
                if any(ch.facets.get(group) not in (None, value) for ch in chunks):
                    reasons.append("FACET_CONFLICT")
            if "SLOT_REPLACED" in a.reasons:
                old_places = {r["from"] for r in a.replaced_slots}
                if any(re.search(rf"\b{p}\b", claim.text + " " + " ".join(ch.text for ch in chunks), re.I)
                       for p in old_places):
                    reasons.append("REPLACED_SLOT_IN_EVIDENCE")
            if not reasons and self._overridden(claim, chunks, new_chunk_ids.get(claim.sub_query_id, []), constraint):
                reasons.append("OVERRIDDEN_BY_SPECIFIC_EVIDENCE")
            if reasons:
                plan[claim.claim_id] = {"action": "REVISE", "reasons": reasons}
            else:
                plan[claim.claim_id] = {"action": "KEEP", "reasons": ["GENERAL_RULE_STILL_APPLIES"]}
        return plan

    def _overridden(self, claim: Claim, chunks, new_ids: list[str], constraint: Constraint) -> bool:
        """A general (unqualified) claim is overridden when a constraint-specific chunk in the same parent
        section, or with high topical overlap, states a different number for the same unit."""
        from kairos.grounding.firewall import _num_units

        claim_nums = _num_units(claim.quote or claim.text)
        claim_terms = {t for t in content_terms(claim.quote or claim.text) if not t[0].isdigit()}
        for cid in new_ids:
            ch = self.corpus.chunk(cid)
            if ch is None or cid in claim.evidence:
                continue
            specific = any(ch.facets.get(g) == v for g, v in constraint.facets.items())
            if not specific:
                continue
            same_parent = any(ch.parent_key and ch.parent_key == c.parent_key for c in chunks)
            for sent in sentences(ch.text):
                sterms = {t for t in content_terms(sent) if not t[0].isdigit()}
                if not (same_parent or jaccard(claim_terms, sterms) >= 0.3):
                    continue
                nums = _num_units(sent)
                units = {u for _, u in claim_nums} & {u for _, u in nums} - {""}
                for u in units:
                    if {n for n, x in claim_nums if x == u} != {n for n, x in nums if x == u}:
                        return True
        return False

    @staticmethod
    def match_lineage(old: list[Claim], new_texts: list[str]) -> list[int | None]:
        """Assign each new draft to the most similar revised claim (lineage), greedily."""
        taken: set[int] = set()
        out: list[int | None] = []
        for text in new_texts:
            best, best_val = None, -1.0
            for i, c in enumerate(old):
                if i in taken:
                    continue
                val = jaccard(content_terms(text), content_terms(c.text))
                if val > best_val:
                    best, best_val = i, val
            if best is not None:
                taken.add(best)
            out.append(best)
        return out
