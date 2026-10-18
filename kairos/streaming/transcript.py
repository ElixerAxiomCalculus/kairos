"""Authoritative coordination for mutable browser speech-recognition hypotheses."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kairos.engine import ChunkIn, SessionEngine


def normalize_transcript(text: str) -> str:
    """Normalize transport whitespace without changing the words reported by ASR."""
    return re.sub(r"\s+", " ", text).strip()


class TranscriptSnapshot(BaseModel):
    """A complete best-known transcript for one still-open conversational turn."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["transcript_snapshot"]
    utterance_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    revision: int = Field(ge=1)
    text: str
    asr_final: bool = False
    end_of_turn: bool = False
    source: Literal["microphone"] = "microphone"


@dataclass
class SnapshotAck:
    utterance_id: str
    revision: int
    status: Literal["accepted", "duplicate", "stale", "finalized"]
    committed_text: str
    provisional_text: str
    reconciled: bool = False
    finalized: bool = False

    def to_wire(self) -> dict:
        return {"type": "transcript_ack", **self.__dict__}


@dataclass
class _UtteranceTranscript:
    utterance_id: str
    last_revision: int = 0
    authoritative_text: str = ""
    previous_tokens: list[str] = field(default_factory=list)
    committed_tokens: list[str] = field(default_factory=list)
    last_fingerprint: tuple | None = None
    finalized: bool = False


def _common_prefix_len(left: list[str], right: list[str]) -> int:
    size = min(len(left), len(right))
    idx = 0
    while idx < size and left[idx] == right[idx]:
        idx += 1
    return idx


class TranscriptIngestionCoordinator:
    """Converts mutable cumulative snapshots into safe append-only engine chunks.

    Prefixes are emitted only after two observations agree, while a configurable
    tail remains provisional. If a later ASR correction crosses the committed
    boundary, SessionEngine restores the pre-utterance domain checkpoint and
    emits TranscriptReconciled before reprocessing the authoritative snapshot.
    """

    def __init__(self, engine: SessionEngine, cfg: dict) -> None:
        self.engine = engine
        self.mutable_tail_tokens = max(0, int(cfg.get("mutable_tail_tokens", 2)))
        self.max_utterance_chars = max(1, int(cfg.get("max_utterance_chars", 4000)))
        self._utterances: dict[str, _UtteranceTranscript] = {}
        self._active_utterance_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def active_utterance_id(self) -> str | None:
        return self._active_utterance_id

    async def ingest(self, snapshot: TranscriptSnapshot) -> SnapshotAck:
        async with self._lock:
            text = normalize_transcript(snapshot.text)
            if len(text) > self.max_utterance_chars:
                raise ValueError(f"transcript exceeds {self.max_utterance_chars} characters")

            state = self._utterances.get(snapshot.utterance_id)
            fingerprint = (text, snapshot.asr_final, snapshot.end_of_turn, snapshot.source)
            if state is not None:
                if snapshot.revision < state.last_revision:
                    return self._ack(state, snapshot.revision, "stale")
                if snapshot.revision == state.last_revision:
                    if fingerprint == state.last_fingerprint:
                        return self._ack(state, snapshot.revision, "duplicate")
                    raise ValueError("revision was already used with a different transcript snapshot")
                if state.finalized:
                    raise ValueError("utterance has already been finalized")
            else:
                if self._active_utterance_id and self._active_utterance_id != snapshot.utterance_id:
                    raise ValueError(f"utterance {self._active_utterance_id!r} is still active")
                state = _UtteranceTranscript(snapshot.utterance_id)
                self._utterances[snapshot.utterance_id] = state
                self._active_utterance_id = snapshot.utterance_id
                await self.engine.begin_revisable_utterance(snapshot.utterance_id)

            tokens = text.split() if text else []
            committed_len = len(state.committed_tokens)
            crosses_committed = tokens[:committed_len] != state.committed_tokens
            reconciled = False

            if crosses_committed:
                await self.engine.reconcile_utterance(snapshot.utterance_id, text)
                state.committed_tokens = list(tokens)
                reconciled = True
            else:
                if snapshot.asr_final or snapshot.end_of_turn:
                    stable_len = len(tokens)
                elif state.previous_tokens:
                    agreed = _common_prefix_len(state.previous_tokens, tokens)
                    stable_len = max(committed_len, agreed - self.mutable_tail_tokens)
                else:
                    stable_len = committed_len

                if stable_len > committed_len:
                    delta = " ".join(tokens[committed_len:stable_len])
                    await self.engine.on_chunk(ChunkIn(
                        utterance_id=snapshot.utterance_id,
                        text=delta,
                        final=False,
                        chunk_id=f"{snapshot.utterance_id}.r{snapshot.revision}",
                    ))
                    state.committed_tokens.extend(tokens[committed_len:stable_len])

            state.last_revision = snapshot.revision
            state.last_fingerprint = fingerprint
            state.authoritative_text = text
            state.previous_tokens = list(tokens)

            if snapshot.end_of_turn:
                if tokens and len(state.committed_tokens) < len(tokens):
                    delta = " ".join(tokens[len(state.committed_tokens):])
                    await self.engine.on_chunk(ChunkIn(
                        utterance_id=snapshot.utterance_id,
                        text=delta,
                        final=False,
                        chunk_id=f"{snapshot.utterance_id}.r{snapshot.revision}.final",
                    ))
                    state.committed_tokens = list(tokens)
                if tokens:
                    await self.engine.end_utterance(snapshot.utterance_id)
                await self.engine.finish_revisable_utterance(snapshot.utterance_id)
                state.finalized = True
                self._active_utterance_id = None
                return self._ack(state, snapshot.revision, "finalized", reconciled=True if reconciled else False)

            return self._ack(state, snapshot.revision, "accepted", reconciled=reconciled)

    def _ack(self, state: _UtteranceTranscript, revision: int, status: str,
             reconciled: bool = False) -> SnapshotAck:
        committed = " ".join(state.committed_tokens)
        provisional = state.authoritative_text[len(committed):].lstrip() if committed else state.authoritative_text
        return SnapshotAck(
            utterance_id=state.utterance_id,
            revision=revision,
            status=status,
            committed_text=committed,
            provisional_text=provisional,
            reconciled=reconciled,
            finalized=state.finalized or status == "finalized",
        )

    async def close(self) -> None:
        async with self._lock:
            for utterance_id, state in list(self._utterances.items()):
                if not state.finalized:
                    await self.engine.finish_revisable_utterance(utterance_id)
            self._utterances.clear()
            self._active_utterance_id = None
