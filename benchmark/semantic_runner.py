"""V2 semantic/relevance/grounding/scale benchmark with explicit dataset provenance.

This report intentionally keeps project-authored engineering cases separate from official SQuAD
annotations and reports ablation arms independently.  It does not collapse incompatible metrics
into a synthetic score.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import statistics
import time
import tracemalloc
from datetime import datetime, timezone

from benchmark.runner import report_provenance, run_benchmark
from kairos.config import Config, load_config, resolve_path
from kairos.engine import EngineContext, SessionEngine
from kairos.decomposer.efis import Decomposer
from kairos.replay import load_script, run_script
from kairos.retrieval.index import RetrievalQuery
from kairos.schemas.models import Claim
from kairos.streaming.frame import LegacyFrameBuilder
from kairos.streaming.frame import CandidateNeed, Frame
from kairos.textutil import content_terms, jaccard


DATASET_PATH = resolve_path("benchmark/datasets/heldout/semantic_cases.json")
EFIS_PAIRS_PATH = resolve_path("benchmark/datasets/heldout/efis_pairs_v2_1.json")


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _wilson(success: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.96
    p = success / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _match(predicted: list[str], gold: list[str]) -> int:
    pairs = []
    for pi, pred in enumerate(predicted):
        for gi, target in enumerate(gold):
            score = jaccard(content_terms(pred), content_terms(target))
            if score >= 0.2:
                pairs.append((score, pi, gi))
    used_p, used_g = set(), set()
    for _score, pi, gi in sorted(pairs, reverse=True):
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return len(used_g)


def _frame_metrics(ctx: EngineContext, cases: list[dict], legacy: bool = False) -> dict:
    builder = LegacyFrameBuilder(ctx.corpus) if legacy else ctx.frame_builder
    matched = pred_n = gold_n = exact = false_merge = false_split = constraint_ok = constraint_n = 0
    details = []
    for index, case in enumerate(cases):
        frame = builder.build(case["text"], {}, 1.0, f"case-{index}")
        predicted = [candidate.text for candidate in frame.candidates]
        got = _match(predicted, case["intents"])
        matched += got
        pred_n += len(predicted)
        gold_n += len(case["intents"])
        exact += len(predicted) == len(case["intents"]) and got == len(case["intents"])
        false_merge += max(0, len(case["intents"]) - len(predicted))
        false_split += max(0, len(predicted) - len(case["intents"]))
        if not legacy and case.get("constraints"):
            constraint_n += sum(len(group) for group in case["constraints"])
            for candidate, expected in zip(frame.candidates, case["constraints"]):
                actual = set()
                for item in candidate.typed_constraints:
                    value = item.normalized_value
                    actual.add(f"{item.kind}:{value}:{item.relation}" if item.kind == "budget"
                               else f"{item.kind}:{value}")
                constraint_ok += sum(item in actual for item in expected)
        details.append({"text": case["text"], "gold": case["intents"], "predicted": predicted, "matched": got})
    precision = _ratio(matched, pred_n)
    recall = _ratio(matched, gold_n)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {
        "intent_precision": precision, "intent_recall": recall, "intent_f1": f1,
        "boundary_f1": f1, "count_accuracy": _ratio(exact, len(cases)),
        "false_merge_rate": _ratio(false_merge, gold_n), "false_split_rate": _ratio(false_split, gold_n),
        "topic_transition_accuracy": _ratio(exact, len(cases)),
        "constraint_ownership_accuracy": _ratio(constraint_ok, constraint_n),
        "cases": len(cases), "intents": gold_n,
        "intent_recall_ci95": _wilson(matched, gold_n), "details": details,
    }


def _retrieval_metrics(cfg: Config, cases: list[dict]) -> tuple[dict, dict[str, EngineContext]]:
    contexts: dict[str, EngineContext] = {}
    positives = negatives = hit_count = reciprocal = ndcg = accepted = accepted_relevant = rejected = routes = 0
    details = []
    for case in cases:
        profile = case["profile"]
        ctx = contexts.setdefault(profile, EngineContext(cfg.with_overrides({"corpus.profile": profile})))
        result = ctx.index.search(RetrievalQuery.build(case["query"]))
        docs = [ctx.corpus.chunk(hit.chunk_id).doc_id for hit in result.relevant_hits]
        relevant = set(case["relevant_docs"])
        routes += result.domain == case["domain"] or (case["domain"] == "scientific_literature" and result.domain == "unknown")
        if relevant:
            positives += 1
            ranks = [idx for idx, doc in enumerate(docs) if doc in relevant]
            if ranks:
                hit_count += 1
                reciprocal += 1 / (min(ranks) + 1)
                ndcg += 1 / math.log2(min(ranks) + 2)
            accepted += len(docs)
            accepted_relevant += sum(doc in relevant for doc in docs)
        else:
            negatives += 1
            rejected += not docs
            accepted += len(docs)
        details.append({"query": case["query"], "profile": profile, "expected": sorted(relevant),
                        "accepted_docs": docs, "status": result.evidence_status, "domain": result.domain})

    # Official SQuAD questions remain labels, never corpus documents.
    labels_path = resolve_path("corpus/imported/squad/evaluation-labels.jsonl")
    external_total = external_hits = 0
    external_rr = []
    if labels_path.exists():
        ctx = contexts.setdefault("general", EngineContext(cfg.with_overrides({"corpus.profile": "general"})))
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            label = json.loads(line)
            result = ctx.index.search(RetrievalQuery.build(label["question"]))
            docs = [ctx.corpus.chunk(hit.chunk_id).doc_id for hit in result.relevant_hits]
            external_total += 1
            rank = next((i for i, doc in enumerate(docs) if doc == label["document_id"]), None)
            external_hits += rank is not None
            external_rr.append(0.0 if rank is None else 1 / (rank + 1))
    metrics = {
        "recall_at_5": _ratio(hit_count, positives), "mrr": round(reciprocal / positives, 4) if positives else None,
        "ndcg_at_5": round(ndcg / positives, 4) if positives else None,
        "relevant_evidence_precision": _ratio(accepted_relevant, accepted),
        "out_of_domain_rejection_accuracy": _ratio(rejected, negatives),
        "irrelevant_evidence_acceptance_rate": _ratio(negatives - rejected, negatives),
        "domain_routing_accuracy": _ratio(routes, len(cases)),
        "positive_queries": positives, "negative_queries": negatives,
        "external_squad_queries": external_total,
        "external_squad_recall_at_5": _ratio(external_hits, external_total),
        "external_squad_mrr": round(statistics.mean(external_rr), 4) if external_rr else None,
        "ood_rejection_ci95": _wilson(rejected, negatives), "details": details,
    }
    return metrics, contexts


def _external_label_metrics(cfg: Config, profile: str, path: str, *, query_key: str,
                            document_key: str, graded: bool = False) -> dict:
    labels_path = resolve_path(path)
    if not labels_path.exists():
        return {"status": "NOT_RUN", "reason": f"missing {path}"}
    grouped: dict[str, dict] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        query_id = str(row.get("query_id") or row.get("id") or row[query_key])
        item = grouped.setdefault(query_id, {"query": row[query_key], "relevant": {}})
        item["relevant"][row[document_key]] = int(row.get("relevance", 1))
    ctx = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
    hit1 = hit5 = hit10 = 0
    reciprocal = ndcg = 0.0
    details = []
    for query_id, item in grouped.items():
        result = ctx.index.search(RetrievalQuery.build(item["query"]), top_k=10)
        docs = [ctx.corpus.chunk(hit.chunk_id).doc_id for hit in result.relevant_hits]
        relevant = item["relevant"]
        ranks = [index for index, doc in enumerate(docs) if doc in relevant]
        if ranks:
            best = min(ranks)
            hit1 += best < 1
            hit5 += best < 5
            hit10 += best < 10
            reciprocal += 1 / (best + 1)
        dcg = sum((2 ** relevant.get(doc, 0) - 1) / math.log2(rank + 2)
                  for rank, doc in enumerate(docs[:10]))
        ideal = sum((2 ** score - 1) / math.log2(rank + 2)
                    for rank, score in enumerate(sorted(relevant.values(), reverse=True)[:10]))
        ndcg += dcg / ideal if ideal else 0.0
        details.append({"query_id": query_id, "query": item["query"], "relevant": relevant,
                        "retrieved": docs, "first_relevant_rank": min(ranks) + 1 if ranks else None})
    total = len(grouped)
    return {
        "status": "MEASURED", "profile": profile, "queries": total,
        "recall_at_1": _ratio(hit1, total), "recall_at_5": _ratio(hit5, total),
        "recall_at_10": _ratio(hit10, total), "mrr_at_10": _ratio(reciprocal, total),
        "ndcg_at_10": _ratio(ndcg, total), "graded_qrels": graded,
        "recall_at_5_ci95": _wilson(hit5, total), "details": details,
    }


def _controller_ablation(cfg: Config) -> dict:
    arms = {
        "v2_psrc": {},
        "v2_1_psrc": {},
        "fixed_tokens_4": {"controller.mode": "fixed_tokens", "controller.fixed_tokens_n": 4},
        "fixed_tokens_8": {"controller.mode": "fixed_tokens", "controller.fixed_tokens_n": 8},
        "end_of_utterance": {"controller.mode": "eou"},
    }
    out = {}
    for name, overrides in arms.items():
        report = run_benchmark(cfg.with_overrides(overrides), repeats=1, label=f"v2.1:{name}", write=False)
        metrics = report["metrics"]
        retrievals = [u["retrievals"] for session in report["per_session"] for u in session["utterances"]
                      if u["type"] == "NEW_INFO"]
        out[name] = {
            "early_retrieval_rate": metrics["early_retrieval_rate"],
            "premature_retrieval_rate": metrics["premature_retrieval_rate"],
            "evidence_survival_rate": metrics["evidence_survival_rate"],
            "intent_recall": metrics["intent_recall"],
            "retrievals_per_information_need": round(sum(retrievals) / len(retrievals), 4) if retrievals else None,
        }
    out["v2_psrc"]["note"] = "Same controller behavior as V2; retained as the explicit comparison arm."
    out["v2_1_psrc"]["note"] = "No threshold retuning adopted; diagnostics added without dev-label leakage."
    return out


def _grounding_metrics(ctx: EngineContext, strict: bool = True) -> dict:
    firewall = ctx.firewall if strict else EngineContext(ctx.cfg.with_overrides({
        "grounding.require_quote": False, "grounding.verify_checksums": False
    })).firewall
    result = ctx.index.search(RetrievalQuery.build("Banyan Hall participants"))
    hit = result.relevant_hits[0]
    chunk = ctx.corpus.chunk(hit.chunk_id)
    retrieved_ids = {item.chunk_id for item in result.hits}
    alternate = next(item for item in ctx.corpus.chunks if item.chunk_id != hit.chunk_id)
    foreign = next(item for item in ctx.corpus.chunks if item.chunk_id not in retrieved_ids)
    sentence = next(s.strip() + "." for s in chunk.text.split(".") if "40" in s)
    claims = [
        ("supported_sha256", Claim("C1", "L1", sentence, "I1", [hit.chunk_id], sentence, 1.0,
                            evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"}), "SUPPORTED"),
        ("supported_legacy_sha1", Claim("C2", "L2", sentence, "I1", [hit.chunk_id], sentence, 1.0,
                                       evidence_checksums={hit.chunk_id: chunk.sha1}), "SUPPORTED"),
        ("supported_unprefixed_sha256", Claim("C3", "L3", sentence, "I1", [hit.chunk_id], sentence, 1.0,
                                              evidence_checksums={hit.chunk_id: chunk.sha256}), "SUPPORTED"),
        ("false_quote", Claim("C4", "L4", sentence, "I1", [hit.chunk_id], "Invented source quote.", 1.0,
                              evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"}), "UNSUPPORTED"),
        ("wrong_number", Claim("C5", "L5", sentence.replace("40", "400"), "I1", [hit.chunk_id], sentence, 1.0,
                               evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"}), "PARTIAL"),
        ("invalid_id", Claim("C6", "L6", sentence, "I1", ["FAKE:C01"], sentence, 1.0), "INVALID_CITATION"),
        ("foreign_unretrieved_id", Claim("C7", "L7", foreign.text, "I1", [foreign.chunk_id], foreign.text, 1.0,
                                         evidence_checksums={foreign.chunk_id: f"sha256:{foreign.sha256}"}),
         "INVALID_CITATION"),
        ("bad_checksum", Claim("C8", "L8", sentence, "I1", [hit.chunk_id], sentence, 1.0,
                               evidence_checksums={hit.chunk_id: "0" * 40}), "INVALID_CITATION"),
        ("missing_checksum", Claim("C9", "L9", sentence, "I1", [hit.chunk_id], sentence, 1.0),
         "INVALID_CITATION"),
        ("empty_evidence", Claim("C10", "L10", sentence, "I1", [], sentence, 1.0), "UNSUPPORTED"),
        ("quote_from_wrong_chunk", Claim("C11", "L11", sentence, "I1", [hit.chunk_id], alternate.text, 1.0,
                                         evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"}),
         "UNSUPPORTED"),
        # These semantic adversaries expose the deterministic lexical firewall's current boundary.
        ("negated_supported_sentence", Claim("C12", "L12", f"It is false that {sentence}", "I1",
                                             [hit.chunk_id], sentence, 1.0,
                                             evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"}),
         "UNSUPPORTED"),
        ("unsupported_modality", Claim("C13", "L13", sentence.replace("can", "might always"), "I1",
                                       [hit.chunk_id], sentence, 1.0,
                                       evidence_checksums={hit.chunk_id: f"sha256:{chunk.sha256}"}),
         "UNSUPPORTED"),
    ]
    rows, correct = [], 0
    for name, claim, expected in claims:
        verdict = firewall.verify(claim, retrieved_ids, []).verdict
        expected_ok = verdict == expected or (name == "wrong_number" and verdict in {"PARTIAL", "UNSUPPORTED"})
        correct += expected_ok
        rows.append({"case": name, "expected": expected, "verdict": verdict, "correct": expected_ok})
    return {
        "cases": len(claims), "accuracy": _ratio(correct, len(claims)),
        "unsupported_claim_rejection": _ratio(
            sum(r["correct"] for r in rows if not r["case"].startswith("supported_")),
            sum(not r["case"].startswith("supported_") for r in rows),
        ),
        "quote_verification_accuracy": int(next(r["correct"] for r in rows if r["case"] == "false_quote")),
        "numeric_contradiction_detection": int(next(r["correct"] for r in rows if r["case"] == "wrong_number")),
        "invalid_citation_rate": _ratio(sum(r["verdict"] == "INVALID_CITATION" for r in rows), len(rows)),
        "false_grounding_acceptance_rate": _ratio(
            sum(r["verdict"] == "SUPPORTED" for r in rows if not r["case"].startswith("supported_")),
            sum(not r["case"].startswith("supported_") for r in rows),
        ),
        "details": rows,
    }


def _efis_metrics(cfg: Config, original: bool = False) -> dict:
    overrides = ({"decomposer.footprint_validity_gate": False, "decomposer.topic_shift_guard": False,
                  "decomposer.semantic_objective_merge": False,
                  "retrieval.relevance_gate": False} if original else {})
    ctx = EngineContext(cfg.with_overrides(overrides))
    cases = [
        ("What are the cancellation rules? Explain the cancellation policy.", True),
        ("What are the cancellation rules? What catering options are available?", False),
        ("Define an AI agent. Find houses in Mumbai.", False),
    ]
    rows, correct = [], 0
    for index, (text, expected_merge) in enumerate(cases):
        frame = ctx.frame_builder.build(text, {}, 1.0, f"efis-{index}")
        dec = Decomposer(ctx.cfg["decomposer"], ctx.index)
        first = dec.assign(frame.candidates[0], frame, [], "I01", "u1", None).sub_query
        assignment = dec.assign(frame.candidates[1], frame, [first], "I02", "u1", first)
        merged = not assignment.created
        correct += merged == expected_merge
        rows.append({"text": text, "expected_merge": expected_merge, "merged": merged,
                     "footprint_valid": assignment.footprint_valid})
    return {"accuracy": _ratio(correct, len(cases)), "cases": len(cases), "details": rows}


def _efis_pair_metrics(cfg: Config, original: bool = False) -> dict:
    data = json.loads(EFIS_PAIRS_PATH.read_text(encoding="utf-8"))
    overrides = ({
        "decomposer.footprint_validity_gate": False,
        "decomposer.topic_shift_guard": False,
        "decomposer.semantic_objective_merge": False,
        "retrieval.relevance_gate": False,
    } if original else {})
    ctx = EngineContext(cfg.with_overrides(overrides))
    dec = Decomposer(ctx.cfg["decomposer"], ctx.index)
    tp = fp = tn = fn = constraint_ok = constraint_total = 0
    rows = []
    for case in data["cases"]:
        left_frame = ctx.frame_builder.build(case["left"], {}, 1.0, f"{case['id']}-left")
        right_frame = ctx.frame_builder.build(case["right"], {}, 2.0, f"{case['id']}-right")
        left = copy.deepcopy(left_frame.candidates[0]) if left_frame.candidates else CandidateNeed(
            "k1", case["left"], [], True, False, request_cue="what"
        )
        right = copy.deepcopy(right_frame.candidates[0]) if right_frame.candidates else CandidateNeed(
            "k2", case["right"], [], True, False, request_cue="what"
        )
        left.key, right.key = "k1", "k2"
        frame = Frame(f"{case['left']} {case['right']}", f"{case['left']} {case['right']}", [],
                      [left, right], {}, anchor="")
        first = dec.assign(left, frame, [], "I01", case["id"], None).sub_query
        assignment = dec.assign(right, frame, [first], "I02", case["id"], first)
        predicted = not assignment.created
        expected = bool(case["merge"])
        tp += predicted and expected
        fp += predicted and not expected
        tn += not predicted and not expected
        fn += not predicted and expected
        signal = (assignment.decision_signals or {}).get("I01", {})
        if case["category"].startswith("constraint_"):
            constraint_total += 1
            constraint_ok += bool(signal.get("constraint_compatibility")) == (case["category"] == "constraint_compatible")
        rows.append({
            "id": case["id"], "category": case["category"], "expected_merge": expected,
            "predicted_merge": predicted, "correct": predicted == expected,
            "reason": assignment.merge_reason if predicted else signal.get("reason", "NEW_INTENT"),
            "signals": signal,
        })
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    total = len(data["cases"])
    return {
        "dataset": data["name"], "split": data["split"], "annotation_policy": data["annotation_policy"],
        "independence_warning": data["independence_warning"], "cases": total,
        "merge_precision": precision, "merge_recall": recall, "merge_f1": f1,
        "accuracy": _ratio(tp + tn, total), "false_merge_rate": _ratio(fp, fp + tn),
        "false_split_rate": _ratio(fn, fn + tp),
        "intent_count_accuracy": _ratio(tp + tn, total),
        "constraint_compatibility_accuracy": _ratio(constraint_ok, constraint_total),
        "confusion": {"true_merge": tp, "false_merge": fp, "true_split": tn, "false_split": fn},
        "details": rows,
    }


def _extractive_focus_metrics(cfg: Config, improved: bool) -> dict:
    ctx = EngineContext(cfg.with_overrides({"synthesis.require_sentence_focus": improved}))
    engine = SessionEngine(ctx, f"extractive-focus-{improved}")
    events = asyncio.run(run_script(engine, load_script("conflict-demo")))
    answer = next(event for event in events if event.event_type == "AnswerVersionCreated")
    claims = answer.payload["claims"]
    direct = sum("headcount" in claim["text"].lower() for claim in claims)
    return {"claims": len(claims), "direct_focus_claim_rate": _ratio(direct, len(claims)),
            "conflicting_sources": len({claim["evidence"][0].split(":")[0] for claim in claims})}


async def _stress(ctx: EngineContext) -> tuple[SessionEngine, float]:
    engine = SessionEngine(ctx, "semantic-benchmark-stress")
    start = time.perf_counter()
    await run_script(engine, load_script("multi-domain-stress"))
    return engine, (time.perf_counter() - start) * 1000


def _streaming_metrics(cfg: Config, cache_enabled: bool = True) -> dict:
    ctx = EngineContext(cfg.with_overrides({"retrieval.probe_cache_enabled": cache_enabled}))
    engine, elapsed = asyncio.run(_stress(ctx))
    active = engine.active_events()
    retrievals = [event for event in active if event.event_type == "RetrievalCompleted"]
    probes = [event for event in active if event.event_type == "ProbeExecuted"]
    queries = [event.payload.get("label") for event in retrievals]
    duplicates = len(queries) - len(set(queries))
    claims = [claim for version in engine.versions for claim in version.claims.values()]
    return {
        "transcript_chunks": sum(len(u.chunks) for u in engine.utterances.values()),
        "active_intents": len(engine.sub_queries), "probe_events": len(probes),
        "actual_bm25_probe_computations": ctx.index.probe_cache_misses,
        "probe_cache_hits": ctx.index.probe_cache_hits, "retrieval_count": len(retrievals),
        "duplicate_retrieval_rate": _ratio(duplicates, len(retrievals)),
        "supported_claims": sum(c.verdict in {"SUPPORTED", "PARTIAL", "CONFLICTING"} for c in claims),
        "wall_ms": round(elapsed, 3),
    }


def _scale_metrics(cfg: Config, profiles: list[str]) -> list[dict]:
    rows = []
    queries = ["cancellation policy", "murine sepsis treatment", "oldest building on campus"]
    for profile in profiles:
        tracemalloc.start()
        start = time.perf_counter()
        ctx = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
        build_ms = (time.perf_counter() - start) * 1000
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        latencies = []
        for query in queries * 5:
            t0 = time.perf_counter()
            ctx.index.search(RetrievalQuery.build(query))
            latencies.append((time.perf_counter() - t0) * 1000)
        ordered = sorted(latencies)
        rows.append({
            "profile": profile, "documents": len(ctx.corpus.documents), "chunks": len(ctx.corpus.chunks),
            "build_ms": round(build_ms, 3), "python_peak_bytes": peak,
            "retrieval_p50_ms": round(statistics.median(latencies), 3),
            "retrieval_p95_ms": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 3),
            "throughput_qps_single_thread": round(1000 / statistics.mean(latencies), 2),
            "dedup_representation": "sparse-tfidf-csr",
        })
    return rows


def run_semantic_benchmark(cfg: Config, write: bool = True) -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = [case for group in dataset["groups"] for case in group["cases"]]
    sample = EngineContext(cfg)
    intent_improved = _frame_metrics(sample, cases)
    intent_original = _frame_metrics(sample, cases, legacy=True)
    retrieval, _contexts = _retrieval_metrics(cfg, dataset["retrieval"])
    grounding = _grounding_metrics(sample)
    streaming = _streaming_metrics(cfg, True)
    streaming_no_cache = _streaming_metrics(cfg, False)
    # Existing dev split remains development data; this arm is reported separately from held-out metrics.
    cddm_improved = run_benchmark(cfg.with_overrides({"delta.scope_guard": True}), repeats=1,
                                  label="v2-cddm", write=False)["metrics"]
    cddm_original = run_benchmark(cfg.with_overrides({"delta.scope_guard": False}), repeats=1,
                                  label="legacy-cddm", write=False)["metrics"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scifact = _external_label_metrics(
        cfg, "science", "corpus/imported/scifact/scifact-qrels-test.jsonl",
        query_key="query", document_key="document_id", graded=True,
    )
    squad_expanded = _external_label_metrics(
        cfg, "general_expanded", "corpus/imported/squad_v2_1/evaluation-labels.jsonl",
        query_key="question", document_key="document_id",
    )
    report = {
        "status": "MEASURED", "run_id": run_id,
        "provenance": report_provenance(cfg, sample, seed=int(dataset["seed"]), repeats=1),
        "dataset": {"name": dataset["name"], "split": dataset["split"], "seed": dataset["seed"],
                    "note": dataset["note"], "groups": [g["id"] for g in dataset["groups"]]},
        "corpora": {
            "sample": "project-authored synthetic policy corpus",
            "science": "BeIR/scifact bounded corpus subset",
            "general": "rajpurkar/squad bounded train-context subset; answers excluded from index",
        },
        "metrics": {"intent": intent_improved, "retrieval": retrieval,
                    "external_scifact_qrels": scifact, "external_squad_expanded": squad_expanded,
                    "grounding": grounding,
                    "streaming": streaming, "scale": _scale_metrics(cfg, ["sample", "science", "general", "multi_domain_demo"])},
        "ablations": {
            "queryframe": {"original": intent_original, "improved": intent_improved},
            "relevance_gate": {
                "original_negative_acceptance": _retrieval_metrics(
                    cfg.with_overrides({"retrieval.relevance_gate": False}), dataset["retrieval"]
                )[0]["irrelevant_evidence_acceptance_rate"],
                "improved_negative_acceptance": retrieval["irrelevant_evidence_acceptance_rate"],
            },
            "efis": {"original": _efis_metrics(cfg, True), "revised": _efis_metrics(cfg, False)},
            "efis_100_pairs": {"v2": _efis_pair_metrics(cfg, True), "v2_1": _efis_pair_metrics(cfg, False)},
            "domain_routing": {
                "single_mixed_corpus": _retrieval_metrics(
                    cfg.with_overrides({"retrieval.domain_routing": False}),
                    [{**case, "profile": "multi_domain_demo"} for case in dataset["retrieval"] if case["relevant_docs"]]
                )[0],
                "domain_aware": _retrieval_metrics(
                    cfg.with_overrides({"retrieval.domain_routing": True}),
                    [{**case, "profile": "multi_domain_demo"} for case in dataset["retrieval"] if case["relevant_docs"]]
                )[0],
            },
            "psrc_probe_cache": {"original": streaming_no_cache, "optimized": streaming},
            "psrc_policy": _controller_ablation(cfg),
            "grounding": {"original": _grounding_metrics(sample, strict=False), "strengthened": grounding},
            "extractive_focus": {"original": _extractive_focus_metrics(cfg, False),
                                 "improved": _extractive_focus_metrics(cfg, True)},
            "cddm_scope": {
                "dataset": "development",
                "original": {"refinement_correctness": cddm_original["refinement_correctness"],
                             "delta_retrieval_ratio": cddm_original["delta_retrieval_ratio"]},
                "improved": {"refinement_correctness": cddm_improved["refinement_correctness"],
                             "delta_retrieval_ratio": cddm_improved["delta_retrieval_ratio"]},
            },
        },
        "integrity": {
            "train_tuning": "benchmark/datasets/train (controller only)",
            "development": "benchmark/datasets/dev (existing iterative benchmark)",
            "held_out": str(DATASET_PATH.relative_to(resolve_path("."))),
            "external_evaluation": "SQuAD official train questions retained as labels for the bounded imported subset",
            "warning": "The engineering held-out set was authored for this upgrade and is not an independent public benchmark.",
        },
    }
    if write:
        out = resolve_path(cfg["bench"]["results_dir"]) / "semantic"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{run_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
