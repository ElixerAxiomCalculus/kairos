"""In-process event bus. Every stage publishes here; sinks fan out to JSONL, SQLite and WebSockets."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from kairos.schemas.events import Event


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        self._subscribers.append(fn)

        def unsubscribe() -> None:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

        return unsubscribe

    def publish(self, event: Event) -> None:
        for fn in list(self._subscribers):
            fn(event)

    def delete_session(self, session_id: str) -> dict[str, bool]:
        """Delete one session from persistent sinks that support authorized deletion."""
        results: dict[str, bool] = {}
        for subscriber in list(self._subscribers):
            delete = getattr(subscriber, "delete_session", None)
            if callable(delete):
                results[type(subscriber).__name__] = bool(delete(session_id))
        return results


class JsonlSink:
    """Append-only trace, one file per session. This file is the benchmark's source of truth."""

    def __init__(self, directory: Path, include_content: bool = True) -> None:
        self.directory = directory
        self.include_content = include_content
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        data = _event_data(event, self.include_content)
        with self._lock, open(self.directory / f"{event.session_id}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def delete_session(self, session_id: str) -> bool:
        path = self.directory / f"{session_id}.jsonl"
        with self._lock:
            if path.exists():
                path.unlink()
        return not path.exists()


class SqliteSink:
    def __init__(self, path: Path, include_content: bool = True) -> None:
        self.include_content = include_content
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events (event_id TEXT, session_id TEXT, event_type TEXT,"
            " ts REAL, utterance_id TEXT, latency_ms REAL, body TEXT, PRIMARY KEY (session_id, event_id))"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_events_session ON events(session_id, ts)")
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        data = _event_data(event, self.include_content)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)",
                (event.event_id, event.session_id, event.event_type, event.timestamp, event.utterance_id,
                 event.latency_ms, json.dumps(data, ensure_ascii=False)),
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            self._conn.commit()
            remaining = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        return remaining == 0


class QueueSink:
    """Bridges bus events into an asyncio queue (used by the WebSocket hub)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, session_id: str) -> None:
        self.loop, self.queue, self.session_id = loop, queue, session_id
        self.overflowed = False

    def _put(self, event: Event) -> None:
        if self.overflowed:
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.overflowed = True
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(None)

    def __call__(self, event: Event) -> None:
        if event.session_id != self.session_id:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            self._put(event)
        else:
            self.loop.call_soon_threadsafe(self._put, event)


_SAFE_ENUM_KEYS = {
    "action", "cache_status", "controller_mode", "controller_variant", "decision", "decision_threshold",
    "dependency_type", "domain", "evidence_status", "kind", "mode", "policy_variant", "reason", "reasons",
    "retrieval_type", "state", "status", "trigger", "turn_type", "verdict",
}
_SAFE_ENUM = re.compile(r"^[A-Z][A-Z0-9_.:-]{0,95}$")
_SAFE_HASH = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{32,64}$")


def _opaque(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"<redacted:sha256:{digest}>"


def _redact_value(value, key: str = ""):
    """Default-deny redaction for persisted telemetry.

    Unknown scalar strings, strings nested in arrays, arbitrary metadata keys,
    provider/error messages, query fragments, titles, and structured constraints
    are all replaced by irreversible short digests.  Only explicit operational
    enums and content checksums survive.
    """
    if isinstance(value, dict):
        return {str(child): _redact_value(item, str(child).lower()) for child, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, key) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, key) for item in value]
    if isinstance(value, str):
        if (key in _SAFE_ENUM_KEYS and _SAFE_ENUM.fullmatch(value)) or (
            ("hash" in key or "checksum" in key or "fingerprint" in key) and _SAFE_HASH.fullmatch(value)
        ):
            return value
        return _opaque(value)
    return value


def _event_data(event: Event, include_content: bool) -> dict:
    data = event.model_dump()
    if include_content:
        return data
    data["utterance_id"] = _redact_value(data.get("utterance_id"), "utterance_id")
    data["payload"] = _redact_value(data.get("payload", {}), "payload")
    return data


def _redact(payload: dict) -> dict:
    """Compatibility wrapper used by tests and downstream privacy checks."""
    return _redact_value(payload, "payload")
