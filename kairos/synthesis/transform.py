"""Presentation transforms: re-render an existing answer from its claim set — zero retrieval.

The transformation firewall guarantees a transform adds no facts:
  * every output claim derives from a parent claim,
  * citations are a subset of the parent's citations,
  * numeric anchors are a subset of the source claim's anchors.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from kairos.schemas.models import AnswerBlock, AnswerVersion, Claim
from kairos.textutil import anchors

NUMERIC = re.compile(r"\d")


@dataclass
class TransformOutcome:
    ok: bool
    blocks: list[AnswerBlock] = field(default_factory=list)
    claims: dict[str, Claim] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    reason: str | None = None
    ops: list[str] = field(default_factory=list)


def _best_claim(version: AnswerVersion, ids: list[str]) -> str:
    rank = {"SUPPORTED": 0, "CONFLICTING": 1, "PARTIAL": 2}
    return sorted(ids, key=lambda c: (rank.get(version.claims[c].verdict, 3), -version.claims[c].score))[0]


class TransformHandler:
    def __init__(self, llm=None) -> None:
        self.llm = llm  # LLMSynthesizer or None

    def apply(self, version: AnswerVersion, spec: dict, next_claim_id) -> TransformOutcome:
        ops = list(spec.get("ops", []))
        notes: list[str] = []
        blocks = [copy.deepcopy(b) for b in version.blocks]
        claims = {cid: copy.deepcopy(version.claims[cid]) for cid in version.claim_ids}
        tokens = {"llm_prompt": 0, "llm_completion": 0}

        generative = [op for op in ops if op in ("translate", "simplify", "rephrase")]
        if generative and self.llm is None:
            if ops == ["translate"] or (generative == ["translate"] and "condense" not in ops and "bullets" not in ops):
                return TransformOutcome(False, reason="Translation needs a generative backend "
                                        "(synthesis.backend=gemini|ollama); no version was created.", ops=ops)
            notes.append(f"{', '.join(generative)} needs a generative backend; applied the extractive "
                         "equivalent (condensed claims) instead.")
            if "condense" not in ops:
                ops.append("condense")

        if "condense" in ops:
            for b in blocks:
                if b.claim_ids:
                    b.claim_ids = [_best_claim(version, b.claim_ids)]
        if "bullets" in ops:
            n = spec.get("bullets") or len(blocks)
            blocks = self._bulletize(blocks, max(1, n))

        if generative and self.llm is not None:
            source = [claims[c] for b in blocks for c in b.claim_ids]
            instruction = {
                "translate": f"Translate each claim into {spec.get('language') or 'the requested language'}. "
                             "Keep numbers as Arabic digits and keep proper names.",
                "simplify": "Rewrite each claim in plain, simple language.",
                "rephrase": "Rephrase each claim more clearly.",
            }[generative[0]]
            rewritten, usage = self.llm.transform(source, instruction)
            tokens = {"llm_prompt": usage.get("llm_prompt", 0), "llm_completion": usage.get("llm_completion", 0)}
            mapping: dict[str, str] = {}
            new_claims: dict[str, Claim] = {}
            for item in rewritten:
                src = claims.get(item.get("source_claim_id", ""))
                if src is None:
                    continue
                c = copy.deepcopy(src)
                c.claim_id = next_claim_id()
                c.text = item["text"].strip()
                c.status = "TRANSFORMED"
                c.derived_from = [src.claim_id]
                mapping[src.claim_id] = c.claim_id
                new_claims[c.claim_id] = c
            for b in blocks:
                b.claim_ids = [mapping.get(cid, cid) for cid in b.claim_ids]
            claims = {**claims, **new_claims}

        used = {c for b in blocks for c in b.claim_ids}
        claims = {cid: c for cid, c in claims.items() if cid in used}
        for c in claims.values():
            if c.status not in ("TRANSFORMED",):
                c.status = "KEEP"
        return TransformOutcome(True, blocks, claims, notes, tokens, ops=ops)

    @staticmethod
    def _bulletize(blocks: list[AnswerBlock], n: int) -> list[AnswerBlock]:
        content = [b for b in blocks if b.claim_ids or b.uncertainty]
        if not content:
            return blocks
        if len(content) < n:
            # Fewer topics than requested bullets: split claims (not topics) across bullets.
            claims = [(b, cid) for b in content for cid in b.claim_ids]
            n = min(n, max(1, len(claims)))
            buckets: list[list[tuple[AnswerBlock, str]]] = [[] for _ in range(n)]
            for i, item in enumerate(claims):
                buckets[min(i * n // len(claims), n - 1)].append(item)
            return [AnswerBlock(
                sub_query_id=bucket[0][0].sub_query_id, title=bucket[0][0].title, style="bullet",
                claim_ids=[cid for _, cid in bucket],
                uncertainty=(bucket[-1][0].uncertainty if k == n - 1 else None),
            ) for k, bucket in enumerate(buckets) if bucket]
        n = min(n, len(content))
        groups: list[list[AnswerBlock]] = [[] for _ in range(n)]
        for i, b in enumerate(content):
            groups[min(i * n // len(content), n - 1)].append(b)
        out = []
        for g in groups:
            out.append(AnswerBlock(
                sub_query_id=g[0].sub_query_id if len(g) == 1 else None,
                title=" + ".join(b.title for b in g),
                style="bullet",
                claim_ids=[c for b in g for c in b.claim_ids],
                uncertainty=" ".join(b.uncertainty for b in g if b.uncertainty) or None,
            ))
        return out


def verify_transform(parent: AnswerVersion, outcome: TransformOutcome) -> dict:
    parent_ids = set(parent.claim_ids)
    parent_citations = {e for c in parent.claims.values() for e in c.evidence}
    violations = []
    for c in outcome.claims.values():
        sources = [c.claim_id] if c.claim_id in parent_ids else c.derived_from
        if not sources or not all(s in parent_ids for s in sources):
            violations.append({"claim_id": c.claim_id, "rule": "NOT_DERIVED_FROM_PARENT"})
            continue
        if not set(c.evidence) <= parent_citations:
            violations.append({"claim_id": c.claim_id, "rule": "NEW_CITATION"})
        src_anchors = set().union(*(anchors(parent.claims[s].text) for s in sources))
        new_nums = {a for a in anchors(c.text) if NUMERIC.search(a)} - {a for a in src_anchors if NUMERIC.search(a)}
        if new_nums:
            violations.append({"claim_id": c.claim_id, "rule": "NEW_NUMERIC_ANCHOR", "anchors": sorted(new_nums)})
    return {
        "passed": not violations,
        "violations": violations,
        "checks": ["DERIVED_FROM_PARENT", "CITATIONS_SUBSET", "NO_NEW_NUMERIC_ANCHORS"],
        "citations_subset": all(set(c.evidence) <= parent_citations for c in outcome.claims.values()),
    }
