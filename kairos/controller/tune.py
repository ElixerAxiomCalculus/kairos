"""Label-free tuning of controller cost weights on the TRAINING streams (never on dev/test).

Objective: maximise the fraction of multi-chunk utterances that retrieve before the speaker finishes,
subject to premature retrievals (early evidence that does not survive to the final query) ≤ max_premature.
Default cap 0.20: premature evidence is not an error in this design — REFINE re-retrieves when the
final query would change results — so the cap bounds refinement cost, which is reported separately.
Both quantities are computed from retrieval results alone, so no gold annotation is needed.
"""

from __future__ import annotations

import asyncio
import itertools

from kairos.config import Config
from kairos.controller.train import load_streams
from kairos.engine import ChunkIn, EngineContext, SessionEngine
from kairos.retrieval.index import RetrievalQuery
from kairos.textutil import lower_tokens

GRID = {
    "controller.lambda_latency": [0.25, 0.35, 0.45, 0.5, 0.55, 0.6],
    "controller.theta_min": [0.3, 0.35, 0.4, 0.45, 0.5],
    "controller.rho_min": [0.3, 0.5],
    "controller.kappa_min": [0.45, 0.6],
}


async def _run(ctx: EngineContext, chunks: list[str], i: int):
    engine = SessionEngine(ctx, f"tune-{i}")
    engine.start()
    t = 1.0
    for k, text in enumerate(chunks):
        dur = 0.3 * len(lower_tokens(text)) + 0.1
        await engine.on_chunk(ChunkIn(utterance_id="u1", text=text, t0=t, t1=t + dur, final=k == len(chunks) - 1))
        t += dur
    return engine, t - 0.0


def evaluate(cfg: Config, streams: list[list[str]]) -> dict:
    ctx = EngineContext(cfg)
    early_utts, eligible, survivals = 0, 0, []
    for i, chunks in enumerate(streams):
        engine, _ = asyncio.run(_run(ctx, chunks, i))
        chunk_evs = [e for e in engine.events if e.event_type == "TranscriptChunkReceived"]
        t_end = chunk_evs[-1].timestamp
        if len(chunks) >= 2:
            eligible += 1
        started = [e for e in engine.events if e.event_type == "RetrievalStarted"]
        if any(e.timestamp < t_end - 1e-6 for e in started) and len(chunks) >= 2:
            early_utts += 1
        frames = [e for e in engine.events if e.event_type == "QueryFrameUpdated"]
        final = {c["key"]: c for c in frames[-1].payload["candidates"]}
        anchor = frames[-1].payload["anchor"]
        gen = [e for e in engine.events if e.event_type == "SubQueriesGenerated"]
        for e in engine.events:
            if e.event_type != "RetrievalCompleted" or e.timestamp >= t_end - 1e-6:
                continue
            keys = [k for g in gen for sq in g.payload["sub_queries"] if sq["id"] == e.sub_query_id
                    for k in sq["candidate_keys"]]
            cands = [final[k] for k in dict.fromkeys(keys) if k in final]
            if not cands:
                continue
            q = RetrievalQuery.build(" ; ".join(c["need"] for c in cands), anchor if cands[0]["key"] != "k0" else "",
                                     [x for c in cands for x in c["constraints"]])
            fin = {h.chunk_id for h in ctx.index.search(q).hits}
            got = set(e.chunk_ids)
            survivals.append(len(got & fin) / len(got) if got else 1.0)
    premature = sum(1 for s in survivals if s < 0.5) / len(survivals) if survivals else 0.0
    return {"early_rate": early_utts / eligible if eligible else 0.0, "premature_rate": premature,
            "early_retrievals": len(survivals)}


def tune(cfg: Config, max_premature: float = 0.20) -> dict:
    streams = load_streams()
    keys = list(GRID)
    results = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        overrides = dict(zip(keys, values))
        r = evaluate(cfg.with_overrides(overrides), streams)
        results.append({"overrides": overrides, **r})
    feasible = [r for r in results if r["premature_rate"] <= max_premature]
    best = max(feasible or results, key=lambda r: (r["early_rate"], -r["premature_rate"],
                                                    -r["overrides"]["controller.lambda_latency"]))
    return {"best": best, "max_premature": max_premature, "grid_size": len(results),
            "top5": sorted(results, key=lambda r: (-r["early_rate"], r["premature_rate"]))[:5]}
