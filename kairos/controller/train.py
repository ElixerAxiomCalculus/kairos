"""Self-supervised training of the stability model ŝ (algorithm A2).

For every training utterance we replay it chunk by chunk and record the controller's signal vector for
each candidate need at each chunk. The label needs no annotation:

    y_t = 1[ RBO(top-k(retrieve(q_t)), top-k(retrieve(q_final))) ≥ τ_y ]

i.e. "evidence retrieved on this prefix survives to the end of the utterance". A logistic regression on
[ρ, χ, κ, σ, 1-π] then gives a calibrated ŝ. Training streams live in benchmark/datasets/train and are
disjoint from the dev sessions used for evaluation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from kairos.config import Config, resolve_path
from kairos.controller.policy import FEATURES, save_model
from kairos.engine import ChunkIn, EngineContext, SessionEngine
from kairos.retrieval.index import RetrievalQuery
from kairos.textutil import lower_tokens, rbo

TRAIN_FILE = resolve_path("benchmark/datasets/train/utterances.txt")


def load_streams(path: Path = TRAIN_FILE) -> list[list[str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append([c.strip() for c in line.split("|") if c.strip()])
    return out


async def _collect(ctx: EngineContext, chunks: list[str], idx: int) -> list[tuple[dict, str, str]]:
    engine = SessionEngine(ctx, f"train-{idx}")
    engine.start()
    t = 1.0
    for i, text in enumerate(chunks):
        dur = 0.3 * len(lower_tokens(text)) + 0.1
        await engine.on_chunk(ChunkIn(utterance_id="u1", text=text, t0=t, t1=t + dur, final=i == len(chunks) - 1))
        t += dur
    rows = []
    frames = [e for e in engine.events if e.event_type == "QueryFrameUpdated"]
    final = {c["key"]: c for c in frames[-1].payload["candidates"]} if frames else {}
    anchor = frames[-1].payload["anchor"] if frames else ""
    decisions = [e for e in engine.events if e.event_type == "RetrievalDecision"]
    for d in decisions[:-1]:  # the last decision is taken at utterance end (label trivially 1)
        for c in d.payload["candidates"]:
            if c["key"] in final and "rho" in c["signals"]:
                rows.append((c["signals"], c["text"], _query_text(final[c["key"]], anchor)))
    return rows


def _query_text(cand: dict, anchor: str) -> str:
    return cand["text"] + (f" || {anchor}" if cand["key"] != "k0" else "")


def train(cfg: Config, tau_y: float = 0.7, out: str | None = None) -> dict:
    ctx = EngineContext(cfg.with_overrides({"controller.stability_model": ""}))
    X, y = [], []
    for i, stream in enumerate(load_streams()):
        for sig, text_t, final_text in asyncio.run(_collect(ctx, stream, i)):
            now_q = RetrievalQuery.build(*text_t.split(" || ")) if " || " in text_t else RetrievalQuery.build(text_t)
            fin = final_text.split(" || ")
            fin_q = RetrievalQuery.build(fin[0], fin[1] if len(fin) > 1 else "")
            a = [h.chunk_id for h in ctx.index.search(now_q).hits]
            b = [h.chunk_id for h in ctx.index.search(fin_q).hits]
            X.append([sig[f] for f in FEATURES])
            y.append(int(rbo(a, b, 0.9) >= tau_y))
    from sklearn.linear_model import LogisticRegression

    Xa, ya = np.array(X), np.array(y)
    if len(set(ya)) < 2:
        raise RuntimeError("training labels are all one class; add more varied training streams")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(Xa, ya)
    weights = {f: round(float(w), 4) for f, w in zip(FEATURES, model.coef_[0])}
    weights["bias"] = round(float(model.intercept_[0]), 4)
    acc = float((model.predict(Xa) == ya).mean())
    meta = {"trained_on": TRAIN_FILE.relative_to(resolve_path(".")).as_posix(), "examples": int(len(ya)),
            "positive_rate": round(float(ya.mean()), 4), "train_accuracy": round(acc, 4), "tau_y": tau_y,
            "features": list(FEATURES), "label": "RBO(top-k now, top-k final) >= tau_y"}
    path = resolve_path(out or cfg["controller"]["stability_model"])
    save_model(path, weights, meta)
    return {"path": str(path), "weights": weights, **meta}
