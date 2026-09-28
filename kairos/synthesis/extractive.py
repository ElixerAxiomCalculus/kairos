"""Extractive claim synthesizer: every claim is an evidence sentence (grounded by construction).

This is the default backend because it is deterministic, needs no API key, and makes the
grounding firewall's job verifiable. Generative backends (synthesis/llm.py) plug into the same
interface and are held to the same firewall.
"""

from __future__ import annotations

from kairos.retrieval.index import Hit, HybridIndex
from kairos.schemas.models import SubQuery
from kairos.synthesis.base import (
    ClaimDraft,
    SynthesisResult,
    claim_text_for,
    context_tokens,
    evidence_covers_focus,
    score_sentences,
)
from kairos.textutil import jaccard, lower_tokens


class ExtractiveSynthesizer:
    name = "extractive"

    def __init__(self, index: HybridIndex, cfg: dict) -> None:
        self.index = index
        self.cfg = cfg

    def synthesize(self, sq: SubQuery, hits: list[Hit], n: int | None = None,
                   avoid_facets: dict[str, str] | None = None, prefer_terms: set[str] | None = None,
                   exclude_quotes: set[str] | None = None) -> SynthesisResult:
        n = n or self.cfg["sentences_per_subquery"]
        ctx = context_tokens(self.index, hits)
        tokens = {"llm_prompt": 0, "llm_completion": 0}
        if not hits or hits[0].rerank_score < self.cfg["tau_sufficiency"]:
            return SynthesisResult([], self.gap_text(sq), tokens, ctx)
        covered, focus = evidence_covers_focus(self.index, sq.need_texts, hits)
        if not covered:
            return SynthesisResult([], self.gap_text(sq, focus), tokens, ctx)
        ranked = score_sentences(self.index, sq.query, hits, avoid_facets, prefer_terms)
        ranked = [s for s in ranked if s.relevance >= self.cfg["tau_sentence"]
                  and s.text not in (exclude_quotes or set())]
        if not ranked:
            return SynthesisResult([], self.gap_text(sq), tokens, ctx)
        chosen = [ranked[0]]
        while len(chosen) < n:
            best, best_val = None, -1.0
            for cand in ranked:
                if cand in chosen:
                    continue
                redundancy = max(jaccard(lower_tokens(cand.text), lower_tokens(c.text)) for c in chosen)
                val = 0.7 * cand.relevance - 0.3 * redundancy
                if redundancy < 0.6 and val > best_val:
                    best, best_val = cand, val
            if best is None:
                break
            chosen.append(best)
        drafts = [ClaimDraft(claim_text_for(s), s.text, [s.chunk_id], s.relevance, sq.id) for s in chosen]
        return SynthesisResult(drafts, None, tokens, ctx)

    @staticmethod
    def gap_text(sq: SubQuery, missing: list[str] | None = None) -> str:
        detail = f" (no evidence mentions {', '.join(missing)})" if missing else ""
        return f"Insufficient evidence in the provided corpus to verify: {sq.label}{detail}."
