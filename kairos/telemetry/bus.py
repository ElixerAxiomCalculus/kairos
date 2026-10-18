"""In-process event bus. Every stage publishes here; sinks fan out to JSONL, SQLite and WebSockets."""

from __future__ import annotations

import asyncio
import json
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


class JsonlSink:
    """Append-only trace, one file per session. This file is the benchmark's source of truth."""

    def __init__(self, directory: Path, include_content: bool = True) -> None:
        self.directory = directory
        self.include_content = include_content
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        data = event.model_dump()
        if not self.include_content:
            data["payload"] = _redact(data["payload"])
        with self._lock, open(self.directory / f"{event.session_id}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")


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
        data = event.model_dump()
        if not self.include_content:
            data["payload"] = _redact(data["payload"])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)",
                (event.event_id, event.session_id, event.event_type, event.timestamp, event.utterance_id,
                 event.latency_ms, json.dumps(data, ensure_ascii=False)),
            )
            self._conn.commit()


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


def _redact(payload: dict) -> dict:
    sensitive = {
        "text", "quote", "rendering", "utterance_text", "raw_text", "analysis_text", "normalized",
        "authoritative_text", "query", "query_text", "need", "anchor", "label", "title", "value",
        "from", "to", "constraints", "facet_constraints", "facets", "terms", "repairs", "spans",
        "candidates", "slots", "content_terms", "novel_terms", "slot_terms",
    }
    out = {}
    for key, value in payload.items():
        lower_key = key.lower()
        if key in sensitive or any(marker in lower_key for marker in ("text", "query", "quote", "render")):
            out[key] = f"<redacted:{len(str(value))} chars>"
        elif isinstance(value, dict):
            out[key] = _redact(value)
        elif isinstance(value, list):
            out[key] = [_redact(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out
