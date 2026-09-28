"""Deterministic replay of timestamped transcript scripts (demo/scripts/*.json)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kairos.config import resolve_path
from kairos.engine import ChunkIn, EngineContext, SessionEngine
from kairos.schemas.events import Event

SCRIPTS_DIR = resolve_path("demo/scripts")


def list_scripts() -> list[dict]:
    out = []
    for p in sorted(SCRIPTS_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": data["id"], "title": data["title"], "description": data.get("description", ""),
                    "chunks": len(data["chunks"]),
                    "duration_s": max(c["t1"] for c in data["chunks"])})
    return out


def load_script(script_id: str) -> dict:
    path = SCRIPTS_DIR / f"{script_id}.json"
    if not path.exists():
        raise FileNotFoundError(script_id)
    return json.loads(path.read_text(encoding="utf-8"))


async def run_script(engine: SessionEngine, script: dict, speed: float | None = None,
                     end_session: bool = False) -> list[Event]:
    """Feed a script into an engine. speed=None: as fast as possible (timestamps still from the script);
    speed=1.0: real time. Timestamps in emitted events are always session-clock seconds."""
    engine.start()
    clock_start = asyncio.get_running_loop().time()
    for chunk in script["chunks"]:
        if speed:
            due = chunk["t1"] / speed
            delay = due - (asyncio.get_running_loop().time() - clock_start)
            if delay > 0:
                await asyncio.sleep(delay)
        await engine.on_chunk(ChunkIn(**chunk))
    if end_session:
        await engine.end()
    return list(engine.events)


def run_script_sync(ctx: EngineContext, script_id: str, session_id: str | None = None) -> list[Event]:
    engine = SessionEngine(ctx, session_id or f"replay-{script_id}")
    return asyncio.run(run_script(engine, load_script(script_id)))


def dump_events(events: list[Event], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.model_dump() for e in events], ensure_ascii=False, indent=1), encoding="utf-8")
