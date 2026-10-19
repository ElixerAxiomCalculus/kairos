"""Event envelope shared by every pipeline stage (and mirrored by web/src/schemas/events.ts)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "SessionStarted",
    "TranscriptReconciled",
    "TranscriptChunkReceived",
    "QueryFrameUpdated",
    "TurnClassified",
    "ProbeExecuted",
    "PrefetchDecision",
    "PrefetchQueued",
    "PrefetchStarted",
    "PrefetchCompleted",
    "PrefetchSuperseded",
    "PrefetchDiscarded",
    "EvidenceValidationStarted",
    "EvidenceValidated",
    "EvidenceInvalidated",
    "EvidenceCommitted",
    "RetrievalDecision",
    "SubQueriesGenerated",
    "RetrievalStarted",
    "RetrievalCompleted",
    "EvidenceFused",
    "DeltaPlanComputed",
    "ClaimsGenerated",
    "GroundingVerified",
    "RetrievalSuppressed",
    "TransformVerified",
    "TransformUnavailable",
    "AnswerVersionCreated",
    "TelemetryEmitted",
    "SessionEnded",
]

RetrievalTrigger = Literal["EARLY", "END_OF_UTTERANCE", "REFINE", "DELTA", "NONE"]

class Event(BaseModel):
    event_id: str
    parent_event_ids: list[str] = Field(default_factory=list)
    timestamp: float                      # session clock, seconds
    wall_time: str
    session_id: str
    event_type: EventType
    utterance_id: str | None = None
    query_version: int | None = None
    answer_version: int | None = None
    retrieval_trigger: RetrievalTrigger | None = None
    sub_query_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    token_cost: dict[str, int] = Field(default_factory=dict)
    confidence: float | None = None
    citations: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


def wall_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
