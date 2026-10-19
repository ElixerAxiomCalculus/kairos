"""Intent-conditioned evidence alignment for the V2.3 stopping dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kairos.textutil import jaccard, rbo


class AlignmentStatus(StrEnum):
    SUPPORTED_GOLD_EVIDENCE = "SUPPORTED_GOLD_EVIDENCE"
    RELEVANT_BUT_CHANGED_EVIDENCE = "RELEVANT_BUT_CHANGED_EVIDENCE"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    UNDEFINED_ALIGNMENT = "UNDEFINED_ALIGNMENT"
    MISSING_GOLD_ANNOTATION = "MISSING_GOLD_ANNOTATION"
    # V2.4 evaluation states.  Historical names above remain stable so frozen
    # V2.3 artifacts can still be read without reinterpretation.
    MISSING_GOLD = "MISSING_GOLD"
    UNDEFINED_INTENT_ALIGNMENT = "UNDEFINED_INTENT_ALIGNMENT"
    UNSUPPORTED_DOMAIN = "UNSUPPORTED_DOMAIN"
    NO_ACCEPTED_EVIDENCE = "NO_ACCEPTED_EVIDENCE"
    NO_FINAL_REFERENCE = "NO_FINAL_REFERENCE"
    INCOMPATIBLE_CONSTRAINTS = "INCOMPATIBLE_CONSTRAINTS"


@dataclass(frozen=True)
class EvidenceIdentity:
    chunks: tuple[str, ...]
    sections: tuple[str, ...]
    documents: tuple[str, ...]

    def public(self) -> dict[str, list[str]]:
        return {"chunks": list(self.chunks), "sections": list(self.sections), "documents": list(self.documents)}


def evidence_identity(ctx, result) -> EvidenceIdentity:
    chunks = tuple(hit.chunk_id for hit in result.relevant_hits)
    sections = tuple(sorted({ctx.index.section_of(chunk_id) for chunk_id in chunks}))
    documents = tuple(sorted({ctx.corpus.chunk(chunk_id).doc_id for chunk_id in chunks if ctx.corpus.chunk(chunk_id)}))
    return EvidenceIdentity(chunks, sections, documents)


def gold_for_stream(stream: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = stream.get("gold_intents") or []
    if annotated:
        return annotated
    sections = stream.get("gold_sections") or []
    documents = stream.get("gold_documents") or []
    if not sections and not documents:
        return []
    return [{
        "intent_id": f"{stream['stream_id']}:gold:aggregate",
        "gold_sections": list(sections),
        "gold_documents": list(documents),
        "annotation_scope": "aggregate_stream_fallback",
    }]


def matching_gold_intents(identity: EvidenceIdentity, gold_intents: list[dict[str, Any]]) -> list[str]:
    sections, documents = set(identity.sections), set(identity.documents)
    matches = []
    for item in gold_intents:
        if sections & set(item.get("gold_sections", [])) or documents & set(item.get("gold_documents", [])):
            matches.append(str(item.get("intent_id", "UNNAMED_GOLD_INTENT")))
    return sorted(set(matches))


def classify_final_alignment(
    *,
    identity: EvidenceIdentity | None,
    gold_intents: list[dict[str, Any]],
    domain_available: bool,
    lineage_defined: bool,
) -> AlignmentStatus:
    if not lineage_defined:
        return AlignmentStatus.UNDEFINED_ALIGNMENT
    if not gold_intents:
        return AlignmentStatus.MISSING_GOLD_ANNOTATION
    if not domain_available:
        return AlignmentStatus.OUT_OF_DOMAIN
    if identity is None or not identity.chunks:
        return AlignmentStatus.NO_RELEVANT_EVIDENCE
    if matching_gold_intents(identity, gold_intents):
        return AlignmentStatus.SUPPORTED_GOLD_EVIDENCE
    return AlignmentStatus.RELEVANT_BUT_CHANGED_EVIDENCE


def classify_v2_4_label_state(
    *,
    identity: EvidenceIdentity | None,
    gold_intents: list[dict[str, Any]],
    domain_available: bool,
    lineage_defined: bool,
    final_reference_available: bool,
    constraints_compatible: bool | None,
) -> AlignmentStatus:
    """Lossless V2.4 label-state classification; unknown never means negative."""
    if not lineage_defined:
        return AlignmentStatus.UNDEFINED_INTENT_ALIGNMENT
    if not final_reference_available:
        return AlignmentStatus.NO_FINAL_REFERENCE
    if not gold_intents:
        return AlignmentStatus.MISSING_GOLD
    if not domain_available:
        return AlignmentStatus.UNSUPPORTED_DOMAIN
    if constraints_compatible is False:
        return AlignmentStatus.INCOMPATIBLE_CONSTRAINTS
    if identity is None or not identity.chunks:
        return AlignmentStatus.NO_ACCEPTED_EVIDENCE
    if matching_gold_intents(identity, gold_intents):
        return AlignmentStatus.SUPPORTED_GOLD_EVIDENCE
    return AlignmentStatus.RELEVANT_BUT_CHANGED_EVIDENCE


def survival(prefix: EvidenceIdentity, final: EvidenceIdentity | None) -> dict[str, float | int | None]:
    """Report chunk and section survival separately; never invent a final reference."""
    if final is None:
        return {
            "accepted_chunk_rbo": None,
            "accepted_section_jaccard": None,
            "survival_score": None,
            "survival_label": None,
        }
    chunk_rbo = rbo(list(prefix.chunks), list(final.chunks), 0.9) if prefix.chunks and final.chunks else 0.0
    section_jaccard = jaccard(set(prefix.sections), set(final.sections)) if prefix.sections or final.sections else 0.0
    score = 0.55 * chunk_rbo + 0.45 * section_jaccard
    return {
        "accepted_chunk_rbo": round(chunk_rbo, 6),
        "accepted_section_jaccard": round(section_jaccard, 6),
        "survival_score": round(score, 6),
        "survival_label": int(score >= 0.7),
    }
