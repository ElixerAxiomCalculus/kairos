"""FastAPI app: REST for status/benchmarks, WebSocket for live sessions, static UI.

WebSocket protocol (/ws):
  server → {"type": "hello", "session_id", "resume_token", "corpus", "config"}
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
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from kairos import __version__
from kairos.config import Config, resolve_path
from kairos.engine import ChunkIn, EngineContext
from kairos.replay import list_scripts, load_script
from kairos.session.store import SessionStore
from kairos.streaming.transcript import TranscriptSnapshot
from kairos.telemetry.bus import JsonlSink, QueueSink, SqliteSink
from kairos.telemetry.validator import validate

WEB_DIST = resolve_path("web/dist")


def create_app(cfg: Config) -> FastAPI:
    ctx = EngineContext(cfg)
    tel = cfg["telemetry"]
    ctx.bus.subscribe(JsonlSink(resolve_path(tel["trace_dir"]), tel["trace_content"]))
    ctx.bus.subscribe(SqliteSink(resolve_path(tel["sqlite_path"]), tel["trace_content"]))
    store = SessionStore(ctx, cfg["session"]["ttl_s"], cfg["session"]["max_sessions"])

    app = FastAPI(title="Kairos — Streaming Live RAG", version=__version__)
    # Split hosting (UI on another origin): KAIROS_CORS_ORIGINS=https://your-site.netlify.app
    origins = [o.strip() for o in os.environ.get(
        "KAIROS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "DELETE"],
                       allow_headers=["Content-Type", "X-Kairos-Session-Token"])
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
    def trace(session_id: str, x_kairos_session_token: str | None = Header(default=None)) -> dict:
        if not store.authorized(session_id, x_kairos_session_token):
            raise HTTPException(403, "invalid session capability")
        engine = store.get(session_id)
        if engine is None:
            raise HTTPException(404, "session not found or expired")
        active = engine.active_events()
        events = [e.model_dump() for e in active]
        return {"events": events, "validation": validate(active)}

    @app.delete("/api/sessions/{session_id}")
    async def end_session(session_id: str, x_kairos_session_token: str | None = Header(default=None)) -> dict:
        if store.get(session_id) is not None and not store.authorized(session_id, x_kairos_session_token):
            raise HTTPException(403, "invalid session capability")
        return {"ended": await store.end(session_id)}

    def _latest(kind: str) -> JSONResponse:
        path = resolve_path(cfg["bench"]["results_dir"]) / kind / "latest.json"
        if not path.exists():
            return JSONResponse({"status": "NOT_MEASURED", "targets": cfg["bench"]["targets"]})
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/benchmark/latest")
    def benchmark_latest() -> JSONResponse:
        return _latest("benchmark")

    @app.get("/api/semantic-benchmark/latest")
    def semantic_benchmark_latest() -> JSONResponse:
        return _latest("semantic")

    @app.get("/api/hardening-benchmark/latest")
    def hardening_benchmark_latest() -> JSONResponse:
        return _latest("hardening")

    @app.get("/api/experiments/latest")
    def experiments_latest() -> JSONResponse:
        return _latest("experiments")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        host = websocket.headers.get("host", "")
        same_origin = bool(origin and origin.split("://", 1)[-1] == host)
        if origin and not same_origin and "*" not in origins and origin not in origins:
            await websocket.close(code=1008, reason="origin not allowed")
            return
        requested = websocket.query_params.get("session_id")
        resume_token = websocket.query_params.get("resume_token")
        if requested and store.get(requested) is not None and not store.authorized(requested, resume_token):
            await websocket.close(code=1008, reason="invalid session capability")
            return
        await websocket.accept()
        live_cfg = cfg.get("live", {})
        queue: asyncio.Queue = asyncio.Queue(maxsize=int(live_cfg.get("outbound_queue_size", 256)))
        engine = store.get(requested) if requested else None
        resumed = engine is not None
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
        ingestion = store.ingestion(engine.id)
        send_lock = asyncio.Lock()

        async def send_json(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        await send_json({"type": "hello", "session_id": engine.id, "resume_token": store.token_for(engine.id),
                         "resumed": resumed,
                         "corpus": ctx.corpus_status(), "config": ctx.config_summary()})

        if resumed:
            after_event_id = websocket.query_params.get("after_event_id")
            active = engine.active_events()
            start = 0
            if after_event_id:
                match = next((i for i, event in enumerate(active) if event.event_id == after_event_id), None)
                start = match + 1 if match is not None else 0
            for event in active[start:]:
                await send_json({"type": "event", "event": event.model_dump()})

        async def pump() -> None:
            while True:
                ev = await queue.get()
                if ev is None:
                    await websocket.close(code=1013, reason="client event queue overflow; reconnect to resume")
                    return
                await send_json({"type": "event", "event": ev.model_dump()})

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
                raw = await websocket.receive_text()
                if len(raw.encode("utf-8")) > int(live_cfg.get("max_message_bytes", 16384)):
                    await websocket.close(code=1009, reason="message too large")
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await send_json({"type": "error", "message": "bad message: invalid JSON"})
                    continue
                if not isinstance(msg, dict):
                    await send_json({"type": "error", "message": "bad message: message must be a JSON object"})
                    continue
                kind = msg.get("type")
                try:
                    if kind == "chunk":
                        payload = {k: v for k, v in msg.items() if k != "type"}
                        if payload.get("t1") is None:
                            payload["t1"] = engine.wall_elapsed()
                        await engine.on_chunk(ChunkIn(**payload))
                    elif kind == "transcript_snapshot":
                        if ingestion is None:
                            raise ValueError("transcript ingestion is unavailable for this session")
                        snapshot = TranscriptSnapshot.model_validate(msg)
                        ack = await ingestion.ingest(snapshot)
                        await send_json(ack.to_wire())
                    elif kind == "utterance_end":
                        await engine.end_utterance(msg["utterance_id"])
                    elif kind == "replay":
                        if replay_task and not replay_task.done():
                            replay_task.cancel()
                        speed = float(msg.get("speed", 1.0))
                        if speed <= 0 or speed > 16:
                            raise ValueError("replay speed must be greater than 0 and at most 16")
                        replay_task = asyncio.create_task(run_replay(msg["script_id"], speed))
                    elif kind == "end_session":
                        await store.end(engine.id)
                        await asyncio.sleep(0.05)
                        await websocket.close()
                        return
                    else:
                        await send_json({"type": "error", "message": f"unknown message type {kind!r}"})
                except ValidationError as exc:
                    detail = "; ".join(f"{'.'.join(map(str, issue['loc']))}: {issue['msg']}"
                                       for issue in exc.errors(include_input=False))
                    await send_json({"type": "error", "message": f"bad message: {detail}"})
                except (KeyError, FileNotFoundError, ValueError, TypeError) as exc:
                    await send_json({"type": "error", "message": f"bad message: {exc}"})
        except (WebSocketDisconnect, RuntimeError):
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
