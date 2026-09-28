"""Session evidence graph: SubQuery → Chunk → Claim → AnswerVersion, with typed edges.

The graph is the dependency structure the delta engine walks: a late constraint invalidates
exactly the claims reachable from the sub-queries/chunks it touches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Edge:
    src: str
    dst: str
    type: str          # DECOMPOSES_TO RETRIEVED SUPPORTS CONTRADICTS DUPLICATES CITES CONTAINS DERIVED_FROM CONDITIONED_ON RETRACTED_BY
    attrs: dict = field(default_factory=dict)

    def to_public(self) -> dict:
        return {"src": self.src, "dst": self.dst, "type": self.type, "attrs": self.attrs}


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[Edge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()

    def add_node(self, node_id: str, kind: str, **attrs) -> None:
        node = self.nodes.setdefault(node_id, {"id": node_id, "kind": kind})
        node.update(attrs)

    def add_edge(self, src: str, dst: str, type_: str, **attrs) -> Edge | None:
        key = (src, dst, type_)
        if key in self._edge_keys:
            for e in self.edges:
                if (e.src, e.dst, e.type) == key:
                    e.attrs.update(attrs)
                    return None
        self._edge_keys.add(key)
        edge = Edge(src, dst, type_, attrs)
        self.edges.append(edge)
        return edge

    def out(self, src: str, type_: str | None = None) -> list[Edge]:
        return [e for e in self.edges if e.src == src and (type_ is None or e.type == type_)]

    def into(self, dst: str, type_: str | None = None) -> list[Edge]:
        return [e for e in self.edges if e.dst == dst and (type_ is None or e.type == type_)]

    # Queries the delta engine and UI rely on -------------------------------------------------
    def claims_of_subquery(self, sq_node: str) -> list[str]:
        return [e.dst for e in self.edges if e.type == "PRODUCES" and e.src == sq_node]

    def unsupported_claims(self) -> list[str]:
        return [n for n, d in self.nodes.items() if d["kind"] == "claim" and not self.into(n, "SUPPORTS")]

    def snapshot(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": [e.to_public() for e in self.edges]}


def sq_node(sq_id: str) -> str:
    return f"SQ:{sq_id}"


def chunk_node(chunk_id: str) -> str:
    return f"CH:{chunk_id}"


def claim_node(claim_id: str) -> str:
    return f"CL:{claim_id}"


def version_node(version: int) -> str:
    return f"AV:{version}"
