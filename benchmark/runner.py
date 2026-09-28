"""Benchmark runner: replays annotated sessions through the engine and measures every gate from the
event traces alone (the same JSONL the UI and judges see).

Honesty rules baked in:
* Every number written here is MEASURED on the named dataset split; targets live in config and are
  reported alongside, never substituted.
* The grounding judge is independent of the firewall (different criterion, no IDF weighting, verbatim /
  token-recall based). It is still automatic; a human audit sheet is exported for spot-checking.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from kairos.config import Config, resolve_path
from kairos.engine import ChunkIn, EngineContext, SessionEngine
from kairos.retrieval.index import RetrievalQuery
from kairos.telemetry.validator import validate
from kairos.textutil import anchors, content_terms, lower_tokens

DATASETS = resolve_path("benchmark/datasets")
SECONDS_PER_WORD = 0.3
CHUNK_OVERHEAD = 0.1
UTTERANCE_GAP = 2.5


def load_dataset(split: str) -> dict:
    return json.loads((DATASETS / split / "sessions.json").read_text(encoding="utf-8"))


def timed_chunks(session: dict) -> list[dict]:
    """Assign deterministic timestamps to authored chunks."""
    t = 1.0
    out = []
    for u in session["utterances"]:
        for i, text in enumerate(u["chunks"]):
            dur = SECONDS_PER_WORD * len(lower_tokens(text)) + CHUNK_OVERHEAD
            out.append({"utterance_id": u["id"], "t0": round(t, 3), "t1": round(t + dur, 3), "text": text,
                        "final": i == len(u["chunks"]) - 1})
            t += dur
        t += UTTERANCE_GAP
    return out


async def replay_session(ctx: EngineContext, session: dict) -> list:
    engine = SessionEngine(ctx, f"bench-{session['id']}")
    engine.start()
    for c in timed_chunks(session):
        await engine.on_chunk(ChunkIn(**c))
    return list(engine.events)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
    return round(vals[k], 3)


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


class Judge:
    """Independent automatic grounding judge (not the firewall)."""

    def __init__(self, ctx: EngineContext) -> None:
        self.corpus = ctx.corpus

    def supports(self, claim_text: str, quote: str, chunk_id: str) -> bool:
        chunk = self.corpus.chunk(chunk_id)
        if chunk is None:
            return False
        text = f"{chunk.section_title}. {chunk.text}"
        low = " ".join(text.split()).lower()
        claim_nums = {a for a in anchors(claim_text) if any(ch.isdigit() for ch in a)}
        if not all(n in low.replace(",", "") for n in claim_nums):
            return False
        if quote and " ".join(quote.split()).lower() in low:
            quote_terms = set(content_terms(quote))
            claim_terms = set(content_terms(claim_text))
            if claim_terms and len(claim_terms & quote_terms) / len(claim_terms) >= 0.6:
                return True
        terms = content_terms(claim_text)
        if not terms:
            return False
        chunk_terms = set(content_terms(text))
        return sum(t in chunk_terms for t in terms) / len(terms) >= 0.85


def evaluate(ctx: EngineContext, dataset: dict, traces: dict[str, list]) -> dict:
    judge = Judge(ctx)
    m: dict[str, list] = {k: [] for k in (
        "eligible", "early", "lead", "survival", "turn_ok", "transform_turns", "transform_retrievals",
        "intent_matched_pred", "intent_pred", "intent_matched_gold", "intent_gold", "count_ok", "recall_hits",
        "abstain_ok", "abstain_n", "claims", "claims_supported", "citations", "citations_supported",
        "claims_with_support", "fabricated", "delta_ratio", "refine_correct", "churn", "update_tokens",
        "update_full_tokens", "ret_latency", "post_latency", "e2e_latency", "llm_tokens", "context_tokens",
        "unnecessary", "elaborate_suppressed", "rr", "update_retrievals", "full_retrievals")}
    per_session = []
    for session in dataset["sessions"]:
        events = [e.model_dump() if hasattr(e, "model_dump") else e for e in traces[session["id"]]]
        chunks = timed_chunks(session)
        sq_to_gold: dict[str, str] = {}
        detail = {"id": session["id"], "utterances": []}
        for u in session["utterances"]:
            uid = u["id"]
            uev = [e for e in events if e["utterance_id"] == uid]
            u_chunks = [c for c in chunks if c["utterance_id"] == uid]
            t_end = u_chunks[-1]["t1"]
            t_start = u_chunks[0]["t0"]
            turn_evs = [e for e in uev if e["event_type"] == "TurnClassified"]
            predicted_turn = turn_evs[-1]["payload"]["turn_type"] if turn_evs else None
            m["turn_ok"].append(predicted_turn == u["type"])
            started = [e for e in uev if e["event_type"] == "RetrievalStarted"]
            completed = [e for e in uev if e["event_type"] == "RetrievalCompleted"]
            m["ret_latency"].extend(e["latency_ms"] for e in completed if e["latency_ms"] is not None)
            tel = next((e for e in uev if e["event_type"] == "TelemetryEmitted"), None)
            if tel and tel["payload"].get("post_utterance_latency_ms") is not None:
                m["post_latency"].append(tel["payload"]["post_utterance_latency_ms"])
            av = next((e for e in uev if e["event_type"] == "AnswerVersionCreated"), None)
            if av:
                m["e2e_latency"].append(round((av["timestamp"] - t_start) * 1000, 3))
                tc = av["token_cost"] or {}
                m["llm_tokens"].append(tc.get("llm_prompt", 0) + tc.get("llm_completion", 0))
                m["context_tokens"].append(tc.get("context_est", 0))
            ud = {"id": uid, "type": u["type"], "predicted_turn": predicted_turn,
                  "retrievals": len(started), "answer_version": av["answer_version"] if av else None}

            if u["type"] in ("TRANSFORM",):
                m["transform_turns"].append(1)
                m["transform_retrievals"].append(len(started))
                m["unnecessary"].append(len(started))
            if u["type"] == "ELABORATE":
                m["elaborate_suppressed"].append(any(e["event_type"] == "RetrievalSuppressed" for e in uev))

            if u["type"] == "NEW_INFO":
                first = min((e["timestamp"] for e in started), default=None)
                eligible = u.get("eligible_chunk", len(u["chunks"])) < len(u["chunks"])
                is_early = first is not None and first < t_end - 1e-6
                if eligible:
                    m["eligible"].append(1)
                    m["early"].append(1 if is_early else 0)
                if is_early:
                    m["lead"].append(round(t_end - first, 3))
                ud.update({"first_retrieval_t": first, "utterance_end_t": t_end, "early": is_early,
                           "eligible_for_early": eligible})
                # survival of early evidence vs retrieval on the final candidate text
                sq_gen = [e for e in uev if e["event_type"] == "SubQueriesGenerated"]
                frames = [e for e in uev if e["event_type"] == "QueryFrameUpdated"]
                final_frame = frames[-1]["payload"] if frames else None
                for e in completed:
                    if e["timestamp"] >= t_end - 1e-6 or not final_frame:
                        continue
                    keys = []
                    for g in sq_gen:
                        for sq in g["payload"]["sub_queries"]:
                            if sq["id"] == e["sub_query_id"]:
                                keys = sq["candidate_keys"]
                    cands = [c for c in final_frame["candidates"] if c["key"] in keys]
                    if not cands:
                        continue
                    anchor = final_frame["anchor"] if cands[0]["key"] != "k0" else ""
                    q = RetrievalQuery.build(" ; ".join(c["need"] for c in cands), anchor,
                                             [x for c in cands for x in c["constraints"]])
                    final_ids = {h.chunk_id for h in ctx.index.search(q).hits}
                    early_ids = set(e["chunk_ids"])
                    m["survival"].append(len(early_ids & final_ids) / len(early_ids) if early_ids else 1.0)

                # intents: last retrieval per sub-query
                last_hits: dict[str, list[str]] = {}
                for e in completed:
                    last_hits[e["sub_query_id"]] = [ctx.index.section_of(c) for c in e["chunk_ids"]][:5]
                answerable = [g for g in u["intents"] if not g.get("unanswerable")]
                pairs = []
                for sid, secs in last_hits.items():
                    for g in answerable:
                        ranks = [secs.index(x) for x in g["gold_sections"] if x in secs]
                        if ranks:
                            pairs.append((min(ranks), sid, g["id"]))
                pairs.sort()  # best-ranked gold evidence first, then one-to-one greedy assignment
                used_s, used_g = set(), set()
                for _, sid, gid in pairs:
                    if sid not in used_s and gid not in used_g:
                        used_s.add(sid)
                        used_g.add(gid)
                        sq_to_gold[sid] = gid
                for g in answerable:
                    best = min((secs.index(x) for secs in last_hits.values() for x in g["gold_sections"]
                                if x in secs), default=None)
                    m["rr"].append(0.0 if best is None else 1.0 / (best + 1))
                m["intent_pred"].append(len(last_hits))
                m["intent_matched_pred"].append(len(used_s))
                m["intent_gold"].append(len(answerable))
                m["intent_matched_gold"].append(len(used_g))
                m["count_ok"].append(len(last_hits) == len(u["intents"]))
                m["recall_hits"].append(len(used_g))
                ud.update({"sub_queries": list(last_hits), "matched": {s: sq_to_gold.get(s) for s in last_hits}})
                if any(g.get("unanswerable") for g in u["intents"]):
                    m["abstain_n"].append(1)
                    unmatched = [s for s in last_hits if s not in used_s]
                    unc = {x["sub_query_id"] for x in (av["payload"]["uncertainty"] if av else [])}
                    m["abstain_ok"].append(1 if any(s in unc for s in unmatched) else 0)

            if u["type"] == "CONSTRAINT":
                plan = next((e for e in uev if e["event_type"] == "DeltaPlanComputed"), None)
                cf = (av["payload"].get("counterfactual") or {}) if av else {}
                if cf:
                    # measured: sub-queries actually re-retrieved and context actually processed in this turn
                    re_retrieved = {e["sub_query_id"] for e in started}
                    total = cf["full_restart_retrievals"]
                    m["delta_ratio"].append(len(re_retrieved) / total if total else 0.0)
                    m["update_retrievals"].append(len(started))
                    m["full_retrievals"].append(total)
                    m["update_tokens"].append(sum((e["token_cost"] or {}).get("context_est", 0)
                                                  for e in uev if e["event_type"] == "ClaimsGenerated"))
                    m["update_full_tokens"].append(cf["full_restart_context_tokens_est"])
                    predicted = {sq_to_gold.get(sid) for sid in re_retrieved}
                    predicted.discard(None)
                    m["refine_correct"].append(predicted == set(u.get("affected_intents", [])))
                    ud.update({"affected_predicted": sorted(predicted), "affected_gold": u.get("affected_intents")})
                if av:
                    # churn: claims of intents the constraint does NOT affect (per gold) whose text changed
                    gold_affected = set(u.get("affected_intents", []))
                    prev = [e for e in events if e["event_type"] == "AnswerVersionCreated"
                            and e["answer_version"] == av["payload"]["parent_version"]]
                    if prev:
                        new_texts = {c["text"] for c in av["payload"]["claims"]}
                        for c in prev[0]["payload"]["claims"]:
                            gold = sq_to_gold.get(c["sub_query_id"])
                            if gold is not None and gold not in gold_affected:
                                m["churn"].append(0 if c["text"] in new_texts else 1)
            if av and av["payload"]["kind"] == "SEMANTIC":
                for c in av["payload"]["claims"]:
                    if c["status"] == "KEEP":
                        continue  # judged when first emitted
                    m["claims"].append(1)
                    fabricated = [x for x in c["evidence"] if ctx.corpus.chunk(x) is None]
                    m["fabricated"].append(len(fabricated))
                    sup = [judge.supports(c["text"], c["quote"], x) for x in c["evidence"]]
                    m["citations"].extend([1] * len(sup))
                    m["citations_supported"].extend(int(s) for s in sup)
                    m["claims_with_support"].append(int(any(sup)))
                    m["claims_supported"].append(int(any(sup)))
            detail["utterances"].append(ud)
        per_session.append(detail)

    validation = validate([e for t in traces.values() for e in t])
    s = sum
    metrics = {
        "early_retrieval_rate": _ratio(s(m["early"]), s(m["eligible"])),
        "eligible_utterances": s(m["eligible"]),
        "retrieval_lead_s_median": round(statistics.median(m["lead"]), 3) if m["lead"] else None,
        "evidence_survival_rate": round(statistics.mean(m["survival"]), 4) if m["survival"] else None,
        "premature_retrieval_rate": _ratio(sum(1 for x in m["survival"] if x < 0.5), len(m["survival"])),
        "false_retrieval_rate": _ratio(sum(1 for x in m["transform_retrievals"] if x > 0), len(m["transform_turns"])),
        "unnecessary_retrieval_count": s(m["unnecessary"]),
        "elaborate_suppression_rate": _ratio(s(m["elaborate_suppressed"]), len(m["elaborate_suppressed"])),
        "turn_classification_accuracy": _ratio(s(m["turn_ok"]), len(m["turn_ok"])),
        "intent_precision": _ratio(s(m["intent_matched_pred"]), s(m["intent_pred"])),
        "intent_recall": _ratio(s(m["intent_matched_gold"]), s(m["intent_gold"])),
        "intent_count_accuracy": _ratio(s(m["count_ok"]), len(m["count_ok"])),
        "gold_section_mrr": round(statistics.mean(m["rr"]), 4) if m["rr"] else None,
        "abstention_accuracy": _ratio(s(m["abstain_ok"]), s(m["abstain_n"])),
        "grounded_claim_rate": _ratio(s(m["claims_supported"]), s(m["claims"])),
        "citation_precision": _ratio(s(m["citations_supported"]), s(m["citations"])),
        "citation_recall": _ratio(s(m["claims_with_support"]), s(m["claims"])),
        "claims_judged": s(m["claims"]),
        "fabricated_doc_ids": s(m["fabricated"]),
        "delta_retrieval_ratio": round(statistics.mean(m["delta_ratio"]), 4) if m["delta_ratio"] else None,
        "refinement_correctness": _ratio(s(m["refine_correct"]), len(m["refine_correct"])),
        "unnecessary_claim_churn": s(m["churn"]),
        "update_retrievals": s(m["update_retrievals"]),
        "full_restart_retrievals": s(m["full_retrievals"]),
        "update_context_tokens_est": s(m["update_tokens"]),
        "full_restart_context_tokens_est": s(m["update_full_tokens"]),
        "retrieval_latency_ms_mean": round(statistics.mean(m["ret_latency"]), 3) if m["ret_latency"] else None,
        "retrieval_latency_ms_p95": _pct(m["ret_latency"], 0.95),
        "post_utterance_latency_ms_p50": _pct(m["post_latency"], 0.5),
        "post_utterance_latency_ms_p95": _pct(m["post_latency"], 0.95),
        "end_to_end_latency_ms_p50": _pct(m["e2e_latency"], 0.5),
        "llm_tokens_total": s(m["llm_tokens"]),
        "context_tokens_est_total": s(m["context_tokens"]),
        "telemetry_coverage": validation["coverage"],
    }
    return {"metrics": metrics, "validation": validation, "per_session": per_session}


LATENCY_KEYS = ("latency", "tokens_est_total")


def comparable(metrics: dict) -> dict:
    """Metrics that must be bit-identical across runs (wall-clock timings are excluded)."""
    return {k: v for k, v in metrics.items() if "latency" not in k and "lead" not in k}


def run_benchmark(cfg: Config, split: str = "dev", repeats: int = 2, label: str = "default",
                  write: bool = True) -> dict:
    dataset = load_dataset(split)
    runs = []
    ctx = None
    for _ in range(repeats):
        ctx = EngineContext(cfg)  # fresh context: nothing carries over between runs
        t0 = time.perf_counter()
        traces = {s["id"]: asyncio.run(replay_session(ctx, s)) for s in dataset["sessions"]}
        result = evaluate(ctx, dataset, traces)
        result["wall_s"] = round(time.perf_counter() - t0, 2)
        runs.append((result, traces))
    first = runs[0][0]
    deltas = []
    for other, _ in runs[1:]:
        a, b = comparable(first["metrics"]), comparable(other["metrics"])
        deltas.append(max((abs((a[k] or 0) - (b[k] or 0)) for k in a if isinstance(a[k], (int, float))), default=0.0))
    first["metrics"]["reproducibility_delta"] = round(max(deltas), 6) if deltas else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "status": "MEASURED",
        "run_id": run_id,
        "label": label,
        "dataset": {"split": split, "name": dataset["name"], "sessions": len(dataset["sessions"]),
                    "note": dataset["note"]},
        "config": ctx.config_summary(),
        "stability_model": cfg["controller"]["stability_model"],
        "repeats": repeats,
        "metrics": first["metrics"],
        "targets": cfg["bench"]["targets"],
        "gates": gate_table(first["metrics"], cfg["bench"]["targets"]),
        "validation": {k: v for k, v in first["validation"].items() if k != "failures"},
        "per_session": first["per_session"],
        "wall_s": first["wall_s"],
        "judge": "automatic lexical judge (verbatim quote / token recall + numeric anchors); independent of the "
                 "firewall's IDF-weighted support score. Human audit pending — see audit_sheet.csv.",
    }
    if write:
        out = resolve_path(cfg["bench"]["results_dir"]) / "benchmark"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{run_id}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        _audit_sheet(out / "audit_sheet.csv", runs[0][1])
        traces_dir = out / "traces" / run_id
        traces_dir.mkdir(parents=True, exist_ok=True)
        for sid, evs in runs[0][1].items():
            (traces_dir / f"{sid}.jsonl").write_text(
                "\n".join(json.dumps(e.model_dump(), ensure_ascii=False) for e in evs), encoding="utf-8")
    return report


def gate_table(metrics: dict, targets: dict) -> list[dict]:
    ops = {">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "<": lambda a, b: a < b,
           "<=": lambda a, b: a <= b}
    out = []
    for key, t in targets.items():
        value = metrics.get(key)
        status = "NOT_MEASURED" if value is None else ("PASS" if ops[t["op"]](value, t["value"]) else "FAIL")
        out.append({"gate": t["gate"], "metric": key, "target": f"{t['op']} {t['value']}", "measured": value,
                    "status": status})
    return out


def _audit_sheet(path: Path, traces: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "answer_version", "claim_id", "claim", "citations", "quote", "firewall_verdict",
                    "human_supported(Y/N)", "notes"])
        for sid, evs in traces.items():
            for e in evs:
                if e.event_type != "AnswerVersionCreated":
                    continue
                for c in e.payload["claims"]:
                    if c["status"] == "KEEP":
                        continue
                    w.writerow([sid, e.answer_version, c["claim_id"], c["text"], " ".join(c["evidence"]),
                                c["quote"], c["verdict"], "", ""])
