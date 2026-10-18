"""Ephemeral session store: in-memory only, idle TTL, LRU cap, explicit purge."""

from __future__ import annotations

import time
import secrets
from collections import OrderedDict

from kairos.engine import EngineContext, SessionEngine
from kairos.streaming.transcript import TranscriptIngestionCoordinator


class SessionStore:
    def __init__(self, ctx: EngineContext, ttl_s: float, max_sessions: int) -> None:
        self.ctx = ctx
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, SessionEngine] = OrderedDict()
        self._ingestion: dict[str, TranscriptIngestionCoordinator] = {}
        self._tokens: dict[str, str] = {}

    def create(self, session_id: str | None = None) -> SessionEngine:
        self.evict_expired()
        while len(self._sessions) >= self.max_sessions:
            evicted_id, _ = self._sessions.popitem(last=False)
            self._ingestion.pop(evicted_id, None)
            self._tokens.pop(evicted_id, None)
        engine = SessionEngine(self.ctx, session_id)
        self._sessions[engine.id] = engine
        self._ingestion[engine.id] = TranscriptIngestionCoordinator(engine, self.ctx.cfg.get("live", {}))
        self._tokens[engine.id] = secrets.token_urlsafe(32)
        engine.start()
        return engine

    def register(self, engine: SessionEngine) -> None:
        """Adopt an engine created by the caller (so it can subscribe to events before start())."""
        self.evict_expired()
        while len(self._sessions) >= self.max_sessions:
            session_id, _ = self._sessions.popitem(last=False)
            self._ingestion.pop(session_id, None)
            self._tokens.pop(session_id, None)
        self._sessions[engine.id] = engine
        self._ingestion[engine.id] = TranscriptIngestionCoordinator(engine, self.ctx.cfg.get("live", {}))
        self._tokens[engine.id] = secrets.token_urlsafe(32)
        engine.start()

    def get(self, session_id: str) -> SessionEngine | None:
        self.evict_expired()
        engine = self._sessions.get(session_id)
        if engine is not None:
            self._sessions.move_to_end(session_id)
        return engine

    async def end(self, session_id: str) -> bool:
        engine = self._sessions.pop(session_id, None)
        if engine is None:
            return False
        ingestion = self._ingestion.pop(session_id, None)
        self._tokens.pop(session_id, None)
        if ingestion is not None:
            await ingestion.close()
        await engine.end()
        return True

    def ingestion(self, session_id: str) -> TranscriptIngestionCoordinator | None:
        return self._ingestion.get(session_id)

    def token_for(self, session_id: str) -> str | None:
        return self._tokens.get(session_id)

    def authorized(self, session_id: str, token: str | None) -> bool:
        expected = self._tokens.get(session_id)
        return bool(expected and token and secrets.compare_digest(expected, token))

    def evict_expired(self) -> list[str]:
        now = time.time()
        expired = [sid for sid, e in self._sessions.items() if now - e.last_active > self.ttl_s]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._ingestion.pop(sid, None)
            self._tokens.pop(sid, None)
        return expired

    def __len__(self) -> int:
        return len(self._sessions)
