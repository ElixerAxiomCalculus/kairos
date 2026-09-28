"""Ablation experiments. Each arm replays the same dev sessions with one component swapped.

A  controller:     PSRC vs fixed-token triggers (N=4, N=8) vs end-of-utterance only
B  retrieval:      hybrid (+rerank) vs dense-only vs sparse-only vs hybrid without reranker
C  refinement:     delta maintenance vs full pipeline restart on late constraints
D  grounding:      evidence graph + firewall vs flat context — requires a generative backend; with the
                   extractive default every claim is a verbatim quote, so the comparison is not
                   meaningful and the arm is reported as NOT_RUN rather than faked.
E  decomposition:  EFIS vs no decomposition (whole utterance as one query)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from benchmark.runner import run_benchmark
from kairos.config import Config, resolve_path

EXPERIMENTS = {
    "A": {
        "title": "Predictive controller vs fixed-threshold retrieval",
        "hypothesis": "PSRC retrieves early on more utterances than end-of-utterance retrieval with fewer "
                      "retrievals per utterance than fixed token triggers, at a bounded premature rate.",
        "metrics": ["early_retrieval_rate", "premature_retrieval_rate", "evidence_survival_rate",
                    "retrievals_per_utterance", "intent_recall"],
        "arms": {
            "psrc": {},
            "fixed_tokens_4": {"controller.mode": "fixed_tokens", "controller.fixed_tokens_n": 4},
            "fixed_tokens_8": {"controller.mode": "fixed_tokens", "controller.fixed_tokens_n": 8},
            "end_of_utterance": {"controller.mode": "eou"},
        },
    },
    "B": {
        "title": "Hybrid retrieval vs dense-only vs sparse-only",
        "hypothesis": "Hybrid fusion finds gold sections at least as often as either retriever alone; the "
                      "reranker's absolute relevance is what makes abstention possible.",
        "metrics": ["gold_section_mrr", "intent_recall", "intent_precision", "abstention_accuracy",
                    "grounded_claim_rate", "retrieval_latency_ms_mean"],
        "arms": {
            "hybrid_rerank": {},
            "dense_only": {"retrieval.mode": "dense"},
            "sparse_only": {"retrieval.mode": "sparse"},
            "hybrid_no_rerank": {"retrieval.reranker": "none"},
        },
    },
    "C": {
        "title": "Delta refinement vs full pipeline restart",
        "hypothesis": "Delta maintenance re-retrieves fewer sub-queries and processes less context than a "
                      "full restart, with zero churn in unaffected claims.",
        "metrics": ["delta_retrieval_ratio", "update_retrievals", "full_restart_retrievals",
                    "update_context_tokens_est", "unnecessary_claim_churn", "refinement_correctness",
                    "grounded_claim_rate"],
        "arms": {
            "delta": {},
            "full_restart": {"delta.refinement": "full_restart"},
        },
    },
    "E": {
        "title": "Evidence-footprint decomposition vs no decomposition",
        "hypothesis": "EFIS identifies more gold intents than a single whole-utterance query.",
        "metrics": ["intent_recall", "intent_precision", "intent_count_accuracy", "grounded_claim_rate"],
        "arms": {
            "efis": {},
            "single_query": {"decomposer.mode": "single"},
        },
    },
}

NOT_RUN = {
    "D": {
        "title": "Evidence graph + grounding firewall vs flat context concatenation",
        "hypothesis": "With a generative synthesizer, closed evidence handles and the firewall remove "
                      "fabricated IDs and reduce unsupported claims versus free-form citations over flat context.",
        "status": "NOT_RUN",
        "reason": "Requires a generative backend (configs/gemini.yaml). The extractive default emits verbatim "
                  "quotes, so this comparison would be vacuous. Run: python -m kairos ablate --only D --config "
                  "configs/gemini.yaml",
    }
}


def _retrievals_per_utterance(report: dict) -> float | None:
    counts = [u["retrievals"] for s in report["per_session"] for u in s["utterances"] if u["type"] == "NEW_INFO"]
    return round(sum(counts) / len(counts), 3) if counts else None


def run_ablations(cfg: Config, only: list[str] | None = None, write: bool = True) -> dict:
    out = {"status": "MEASURED", "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
           "dataset": "dev", "experiments": {}}
    for key, exp in EXPERIMENTS.items():
        if only and key not in only:
            continue
        arms = {}
        for arm, overrides in exp["arms"].items():
            report = run_benchmark(cfg.with_overrides(overrides), repeats=1, label=f"{key}:{arm}", write=False)
            metrics = dict(report["metrics"])
            metrics["retrievals_per_utterance"] = _retrievals_per_utterance(report)
            arms[arm] = {"overrides": overrides, "metrics": {m: metrics.get(m) for m in exp["metrics"]}}
        out["experiments"][key] = {"title": exp["title"], "hypothesis": exp["hypothesis"], "status": "MEASURED",
                                   "metrics": exp["metrics"], "arms": arms}
    for key, exp in NOT_RUN.items():
        if not only or key in only:
            out["experiments"][key] = exp
    if write:
        d = resolve_path(cfg["bench"]["results_dir"]) / "experiments"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{out['run_id']}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
