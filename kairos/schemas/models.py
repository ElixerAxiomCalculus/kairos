"""Core domain objects held in ephemeral session state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from kairos.retrieval.index import Hit, RetrievalQuery


@dataclass
class SubQuery:
    id: str                              # "I01"
    utterance_id: str
    label: str
    need_texts: list[str]
    constraints: list[str]
    anchor: str
    facet_constraints: dict[str, str] = field(default_factory=dict)
    candidate_keys: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    dependency_type: str | None = None   # soft | hard
    footprint: list[str] = field(default_factory=list)
    status: str = "PENDING"              # PENDING | RETRIEVED | STALE
    retrieval_count: int = 0
    hits: list[Hit] = field(default_factory=list)
    query: RetrievalQuery | None = None
    merged_from: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts = [" / ".join(self.need_texts)]
        if self.constraints:
            parts.append("[" + "; ".join(self.constraints) + "]")
        return " ".join(parts)

    @property
    def evidence(self) -> list[str]:
        return [h.chunk_id for h in self.hits]

    def to_public(self) -> dict:
        return {
            "id": self.id, "utterance_id": self.utterance_id, "label": self.label, "text": self.text,
            "need_texts": self.need_texts, "constraints": self.constraints, "anchor": self.anchor,
            "facet_constraints": self.facet_constraints, "candidate_keys": self.candidate_keys,
            "depends_on": self.depends_on, "dependency_type": self.dependency_type,
            "footprint": self.footprint, "status": self.status, "retrieval_count": self.retrieval_count,
            "merged_from": self.merged_from,
        }


@dataclass
class Claim:
    claim_id: str
    lineage_id: str
    text: str
    sub_query_id: str
    evidence: list[str]
    quote: str
    score: float
    verdict: str = "PENDING"             # SUPPORTED | PARTIAL | CONFLICTING | UNSUPPORTED | INVALID_CITATION
    support: float = 0.0
    anchors_ok: bool = True
    quote_ok: bool = True
    status: str = "NEW"                  # NEW | KEEP | REVISED | ADDED | TRANSFORMED
    derived_from: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    conditioned_on: list[str] = field(default_factory=list)
    repaired: bool = False
    version_created: int = 0

    def to_public(self) -> dict:
        return asdict(self)


@dataclass
class AnswerBlock:
    sub_query_id: str | None
    title: str
    style: str                           # paragraph | bullet
    claim_ids: list[str]
    uncertainty: str | None = None


@dataclass
class AnswerVersion:
    version: int
    parent_version: int | None
    trigger: str                         # NEW_INFO | CONSTRAINT | TRANSFORM | ELABORATE
    kind: str                            # SEMANTIC | PRESENTATION
    utterance_id: str
    query_version: int
    blocks: list[AnswerBlock]
    claims: dict[str, Claim]
    uncertainty: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    diff: dict = field(default_factory=dict)
    constraint: dict | None = None
    transform: dict | None = None
    retrievals_used: list[str] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    counterfactual: dict = field(default_factory=dict)
    created_t: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def claim_ids(self) -> list[str]:
        return [cid for b in self.blocks for cid in b.claim_ids]

    def rendering(self) -> str:
        lines = []
        for block in self.blocks:
            texts = [f"{self.claims[c].text} [{', '.join(self.claims[c].evidence)}]" for c in block.claim_ids]
            if block.uncertainty:
                texts.append(block.uncertainty)
            if block.style == "bullet":
                lines.append("- " + " ".join(texts))
            else:
                lines.append(" ".join(texts))
        return "\n".join(lines)

    def to_public(self) -> dict:
        return {
            "version": self.version, "parent_version": self.parent_version, "trigger": self.trigger,
            "kind": self.kind, "utterance_id": self.utterance_id, "query_version": self.query_version,
            "blocks": [asdict(b) for b in self.blocks],
            "claims": [self.claims[c].to_public() for c in self.claim_ids],
            "uncertainty": self.uncertainty, "conflicts": self.conflicts, "diff": self.diff,
            "constraint": self.constraint, "transform": self.transform,
            "retrievals_used": self.retrievals_used, "tokens": self.tokens,
            "counterfactual": self.counterfactual, "created_t": round(self.created_t, 3),
            "notes": self.notes, "rendering": self.rendering(),
        }
