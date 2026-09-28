"""FastAPI app: REST for status/benchmarks, WebSocket for live sessions, static UI.

WebSocket protocol (/ws):
  server → {"type": "hello", "session_id", "corpus", "config"}
  server → {"type": "event", "event": {...}}          one per engine event
  server → {"type": "error", "message"}
  client → {"type": "chunk", "utterance_id", "text", "t0"?, "t1"?, "final"?}
  client → {"type": "utterance_end", "utterance_id"}
  client → {"type": "replay", "script_id", "speed"?}  server paces a demo script in real time
  client → {"type": "end_session"}
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from kairos import __version__
from kairos.config import Config, resolve_path
from kairos.engine import ChunkIn, EngineContext
from kairos.replay import list_scripts, load_script
from kairos.session.store import SessionStore
from kairos.telemetry.bus import JsonlSink, QueueSink, SqliteSink
from kairos.telemetry.validator import validate

WEB_DIST = resolve_path("web/dist")


def create_app(cfg: Config) -> FastAPI:
    ctx = EngineContext(cfg)
    tel = cfg["telemetry"]
    ctx.bus.subscribe(JsonlSink(resolve_path(tel["trace_dir"]), tel["trace_content"]))
    ctx.bus.subscribe(SqliteSink(resolve_path(tel["sqlite_path"])))
    store = SessionStore(ctx, cfg["session"]["ttl_s"], cfg["session"]["max_sessions"])

    app = FastAPI(title="Kairos — Streaming Live RAG", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.ctx, app.state.store = ctx, store

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "sessions": len(store)}

    @app.get("/api/corpus")
    def corpus() -> dict:
        return {"status": "READY", **ctx.corpus_status()}

    @app.get("/api/config")
    def config() -> dict:
        return {**ctx.config_summary(), "stability_model": ctx.cfg["controller"]["stability_model"]}

    @app.get("/api/scripts")
    def scripts() -> list[dict]:
        return list_scripts()

    @app.get("/api/scripts/{script_id}")
    def script(script_id: str) -> dict:
        try:
            return load_script(script_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"unknown script {script_id}") from exc

    @app.get("/api/sessions/{session_id}/trace")
    def trace(session_id: str) -> dict:
        engine = store.get(session_id)
        if engine is None:
            raise HTTPException(404, "session not found or expired")
        events = [e.model_dump() for e in engine.events]
        return {"events": events, "validation": validate(engine.events)}

    @app.delete("/api/sessions/{session_id}")
    async def end_session(session_id: str) -> dict:
        return {"ended": await store.end(session_id)}

    def _latest(kind: str) -> JSONResponse:
        path = resolve_path(cfg["bench"]["results_dir"]) / kind / "latest.json"
        if not path.exists():
            return JSONResponse({"status": "NOT_MEASURED", "targets": cfg["bench"]["targets"]})
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/benchmark/latest")
    def benchmark_latest() -> JSONResponse:
        return _latest("benchmark")

    @app.get("/api/experiments/latest")
    def experiments_latest() -> JSONResponse:
        return _latest("experiments")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        requested = websocket.query_params.get("session_id")
        engine = store.get(requested) if requested else None
        loop = asyncio.get_running_loop()
        sink = None
        if engine is None:
            # subscribe before start() so SessionStarted reaches the client
            from kairos.engine import SessionEngine

            engine = SessionEngine(ctx)
            sink = QueueSink(loop, queue, engine.id)
            unsubscribe = ctx.bus.subscribe(sink)
            store.register(engine)
        else:
            sink = QueueSink(loop, queue, engine.id)
            unsubscribe = ctx.bus.subscribe(sink)
        await websocket.send_json({"type": "hello", "session_id": engine.id, "corpus": ctx.corpus_status(),
                                   "config": ctx.config_summary()})

        async def pump() -> None:
            while True:
                ev = await queue.get()
                await websocket.send_json({"type": "event", "event": ev.model_dump()})

        async def run_replay(script_id: str, speed: float) -> None:
            data = load_script(script_id)
            start = loop.time()
            offset = engine.now() + 0.5  # the script's clock starts half a second after "now"
            for chunk in data["chunks"]:
                due = chunk["t1"] / speed
                delay = due - (loop.time() - start)
                if delay > 0:
                    await asyncio.sleep(delay)
                c = dict(chunk)
                c["t0"] = offset + c["t0"]
                c["t1"] = offset + c["t1"]
                c["utterance_id"] = f"{script_id}:{c['utterance_id']}"
                await engine.on_chunk(ChunkIn(**c))

        pump_task = asyncio.create_task(pump())
        replay_task: asyncio.Task | None = None
        try:
            while True:
                msg = await websocket.receive_json()
                kind = msg.get("type")
                try:
                    if kind == "chunk":
                        payload = {k: v for k, v in msg.items() if k != "type"}
                        if payload.get("t1") is None:
                            payload["t1"] = engine.wall_elapsed()
                        await engine.on_chunk(ChunkIn(**payload))
                    elif kind == "utterance_end":
                        await engine.end_utterance(msg["utterance_id"])
                    elif kind == "replay":
                        if replay_task and not replay_task.done():
                            replay_task.cancel()
                        replay_task = asyncio.create_task(run_replay(msg["script_id"], float(msg.get("speed", 1.0))))
                    elif kind == "end_session":
                        await store.end(engine.id)
                        await asyncio.sleep(0.05)
                        await websocket.close()
                        return
                    else:
                        await websocket.send_json({"type": "error", "message": f"unknown message type {kind!r}"})
                except (ValidationError, KeyError, FileNotFoundError) as exc:
                    await websocket.send_json({"type": "error", "message": f"bad message: {exc}"})
        except WebSocketDisconnect:
            pass
        finally:
            unsubscribe()
            pump_task.cancel()
            if replay_task:
                replay_task.cancel()

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = WEB_DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_ui() -> JSONResponse:
            return JSONResponse({"message": "UI not built. Run `npm --prefix web ci && npm --prefix web run build`, "
                                            "or use the API at /docs."})

    return app


def run(cfg: Config, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


def dist_exists() -> bool:
    return Path(WEB_DIST / "index.html").exists()
