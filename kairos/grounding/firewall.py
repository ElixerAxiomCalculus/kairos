"""Grounding firewall: no claim reaches the answer without passing Claim → Evidence → Citation checks.

Checks, in order:
1. Citation validity  — every cited chunk exists in the corpus index (sha1 match) AND was retrieved
                        in this session. Anything else is a fabricated/foreign ID and is dropped.
2. Quote              — the quoted span is a verbatim substring of a cited chunk.
3. Anchors            — numbers, amounts and proper nouns in the claim appear in the cited text.
4. Support            — lexical (IDF-weighted term entailment) or cross-encoder NLI.
5. Conflict           — same-topic evidence from another document states a different number
                        under a compatible scope (e.g. '72 hours' vs '24 hours').
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kairos.corpus.ingest import Chunk
from kairos.retrieval.index import HybridIndex
from kairos.schemas.models import Claim
from kairos.textutil import anchors, content_set, jaccard, sentences, tokens

NUM_UNIT_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|[a-z]+)?", re.I)


@dataclass
class Verification:
    claim_id: str
    verdict: str
    support: float
    anchors_ok: bool
    quote_ok: bool
    invalid_ids: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    missing_anchors: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "claim_id": self.claim_id, "verdict": self.verdict, "support": round(self.support, 4),
            "anchors_ok": self.anchors_ok, "quote_ok": self.quote_ok, "invalid_ids": self.invalid_ids,
            "conflicts": self.conflicts, "missing_anchors": self.missing_anchors,
        }


def _num_units(text: str) -> set[tuple[str, str]]:
    out = set()
    for m in NUM_UNIT_RE.finditer(text):
        num = m.group(1).replace(",", "")
        unit = (m.group(2) or "").lower()
        if unit in {"working", "calendar"}:
            unit = "days"
        out.add((num, unit.rstrip("s")))
    return out


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class GroundingFirewall:
    def __init__(self, index: HybridIndex, cfg: dict) -> None:
        self.index = index
        self.cfg = cfg
        self.nli = None
        if cfg.get("nli") == "cross-encoder":
            from sentence_transformers import CrossEncoder

            self.nli = CrossEncoder(cfg["nli_model"])

    def support_score(self, claim_text: str, chunks: list[Chunk]) -> float:
        if self.nli is not None:
            best = 0.0
            for ch in chunks:
                for sent in sentences(ch.text) or [ch.text]:
                    logits = self.nli.predict([(sent, claim_text)])[0]
                    import numpy as np

                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                    best = max(best, float(probs[1]))  # label order: contradiction, entailment, neutral
            return best
        terms = content_set(claim_text)
        if not terms:
            return 0.0
        available: set[str] = set()
        for ch in chunks:
            available |= content_set(ch.index_text)
        idf = self.index.bm25.idf_of
        total = sum(idf(t) for t in terms)
        return sum(idf(t) for t in terms if t in available) / total if total else 0.0

    def verify(self, claim: Claim, retrieved: set[str], peer_chunk_ids: list[str]) -> Verification:
        corpus = self.index.corpus
        invalid = [cid for cid in claim.evidence if cid not in retrieved or corpus.chunk(cid) is None]
        valid = [cid for cid in claim.evidence if cid not in invalid]
        claim.evidence = valid
        if not valid:
            verdict = "INVALID_CITATION" if invalid else "UNSUPPORTED"
            return Verification(claim.claim_id, verdict, 0.0, False, False, invalid)
        chunks = [corpus.chunk(cid) for cid in valid]
        cited_text = " ".join(f"{c.section_title} {c.text}" for c in chunks)
        quote_ok = bool(claim.quote) and any(_norm_ws(claim.quote) in _norm_ws(c.text) for c in chunks)
        claim_anchors = anchors(claim.text)
        cited_anchor_text = cited_text.lower().replace(",", "")
        missing = sorted(a for a in claim_anchors if a.replace(" ", "") not in cited_anchor_text)
        anchors_ok = not missing
        support = self.support_score(claim.text, chunks)
        conflicts = self._conflicts(claim, chunks, peer_chunk_ids)

        if support >= self.cfg["tau_support"] and anchors_ok:
            verdict = "CONFLICTING" if conflicts else "SUPPORTED"
        elif support >= self.cfg["tau_partial"] or (quote_ok and not anchors_ok):
            verdict = "PARTIAL"
        else:
            verdict = "UNSUPPORTED"
        return Verification(claim.claim_id, verdict, support, anchors_ok, quote_ok, invalid, conflicts, missing)

    def _conflicts(self, claim: Claim, cited: list[Chunk], peer_chunk_ids: list[str]) -> list[dict]:
        claim_nums = _num_units(claim.quote or claim.text)
        if not claim_nums:
            return []
        claim_terms = {t for t in content_set(claim.quote or claim.text) if not any(ch.isdigit() for ch in t)}
        cited_docs = {c.doc_id for c in cited}
        cited_facets = [c.facets for c in cited]
        out = []
        for cid in peer_chunk_ids:
            peer = self.index.corpus.chunk(cid)
            if peer is None or peer.doc_id in cited_docs:
                continue
            if any(_facet_clash(peer.facets, f) for f in cited_facets):
                continue
            for sent in sentences(peer.text):
                sent_terms = {t for t in content_set(sent) if not any(ch.isdigit() for ch in t)}
                if jaccard(claim_terms, sent_terms) < self.cfg["conflict_overlap"]:
                    continue
                peer_nums = _num_units(sent)
                shared_units = {u for _, u in claim_nums} & {u for _, u in peer_nums} - {""}
                for unit in shared_units:
                    a = {n for n, u in claim_nums if u == unit}
                    b = {n for n, u in peer_nums if u == unit}
                    if a and b and not (a & b):
                        out.append({"chunk_id": cid, "doc_id": peer.doc_id, "sentence": sent, "unit": unit,
                                    "claim_values": sorted(a), "peer_values": sorted(b),
                                    "peer_effective_date": peer.effective_date})
                        break
        return out

    def repair(self, claim: Claim) -> bool:
        """Extractive fallback: replace a weak claim with its best-matching cited sentence."""
        best, best_overlap = None, 0.0
        for cid in claim.evidence:
            chunk = self.index.corpus.chunk(cid)
            if chunk is None:
                continue
            for sent in sentences(chunk.text):
                ov = jaccard(tokens(sent.lower()), tokens(claim.text.lower()))
                if ov > best_overlap:
                    best, best_overlap = sent, ov
        if best is None:
            return False
        claim.text, claim.quote, claim.repaired = best, best, True
        return True


def _facet_clash(a: dict[str, str], b: dict[str, str]) -> bool:
    return any(k in b and b[k] != v for k, v in a.items())


def find_contradictions(index: HybridIndex, chunk_ids: list[str], overlap: float) -> list[dict]:
    """Pairs of retrieved chunks from different documents, compatible scope, that state different numbers
    for the same unit about the same topic (e.g. headcount changes '72 hours' vs '24 hours')."""
    corpus = index.corpus
    chunks = [corpus.chunk(c) for c in chunk_ids if corpus.chunk(c)]
    out = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a, b = chunks[i], chunks[j]
            if a.doc_id == b.doc_id or _facet_clash(a.facets, b.facets):
                continue
            for sa in sentences(a.text):
                ta = {t for t in content_set(sa) if not any(ch.isdigit() for ch in t)}
                na = _num_units(sa)
                if not na:
                    continue
                for sb in sentences(b.text):
                    tb = {t for t in content_set(sb) if not any(ch.isdigit() for ch in t)}
                    if jaccard(ta, tb) < overlap:
                        continue
                    nb = _num_units(sb)
                    units = {u for _, u in na} & {u for _, u in nb} - {""}
                    for u in units:
                        va = {n for n, x in na if x == u}
                        vb = {n for n, x in nb if x == u}
                        if va and vb and not (va & vb):
                            newer = a if (a.effective_date or "") >= (b.effective_date or "") else b
                            out.append({
                                "chunks": [a.chunk_id, b.chunk_id], "documents": [a.doc_id, b.doc_id],
                                "unit": u, "values": [sorted(va), sorted(vb)], "sentences": [sa, sb],
                                "more_recent": newer.chunk_id, "effective_dates": [a.effective_date, b.effective_date],
                            })
                            break
    return out
