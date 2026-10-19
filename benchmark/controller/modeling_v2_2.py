"""Policy-independent feature extraction, training, and calibration for PSRC V2.2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmark.controller.dataset_v2_2 import ROOT, SPLITS, load_split
from benchmark.controller.metrics import calibration_metrics
from kairos.config import Config, resolve_path
from kairos.controller.cold_start import assess_cold_start, critical_constraint_risk
from kairos.controller.intent_readiness import assess_intent_readiness
from kairos.controller.policy import FEATURES, StabilityModel, clarity, save_model
from kairos.engine import EngineContext
from kairos.retrieval.index import RetrievalQuery
from kairos.streaming.frame import completeness, continuation_hazard
from kairos.textutil import content_terms, jaccard, rbo

EXTENDED_FEATURES = (
    "rho", "chi", "kappa", "sigma", "cont", "first_observation", "cold_start_confidence",
    "query_informativeness", "rank_separation", "objective_clarity", "intent_readiness",
    "constraint_readiness", "constraint_revision_risk", "evidence_relevance",
    "domain_availability", "prefix_length", "candidate_age", "last_slot_change",
    "accepted_evidence_overlap",
)
MODEL_ROOT = resolve_path("models/controller_v2_2")


def _portable_path(path: Path) -> str:
    """Store repository-relative paths so artifacts replay outside this workstation."""
    try:
        return path.resolve().relative_to(resolve_path(".").resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _corpus_source_hash(ctx: EngineContext) -> str:
    return hashlib.sha256("\n".join(f"{chunk.chunk_id}:{chunk.sha256}" for chunk in ctx.corpus.chunks).encode()).hexdigest()


def _gold_overlap(ctx: EngineContext, result, stream: dict) -> float | None:
    sections = set(stream.get("gold_sections", []))
    documents = set(stream.get("gold_documents", []))
    if not sections and not documents:
        return None
    accepted = result.relevant_hits
    matched = any(
        ctx.index.section_of(hit.chunk_id) in sections or ctx.corpus.chunk(hit.chunk_id).doc_id in documents
        for hit in accepted
    )
    return float(matched)


def _survival(ctx: EngineContext, prefix_result, final_result, stream: dict) -> dict:
    prefix_ids = [hit.chunk_id for hit in prefix_result.relevant_hits]
    final_ids = [hit.chunk_id for hit in final_result.relevant_hits]
    prefix_sections = {ctx.index.section_of(item) for item in prefix_ids}
    final_sections = {ctx.index.section_of(item) for item in final_ids}
    rank_score = rbo(prefix_ids, final_ids, 0.9) if prefix_ids and final_ids else 0.0
    section_score = jaccard(prefix_sections, final_sections) if prefix_sections or final_sections else 0.0
    gold_score = _gold_overlap(ctx, prefix_result, stream)
    if gold_score is None:
        composite = 0.56 * rank_score + 0.44 * section_score
        definition = "0.56*accepted_chunk_RBO + 0.44*accepted_section_Jaccard"
    else:
        composite = 0.45 * rank_score + 0.35 * section_score + 0.20 * gold_score
        definition = "0.45*accepted_chunk_RBO + 0.35*accepted_section_Jaccard + 0.20*independent_gold_overlap"
    return {
        "ranking_rbo": round(rank_score, 4), "section_overlap": round(section_score, 4),
        "gold_overlap": gold_score, "survival_score": round(composite, 4),
        "survival_label": int(composite >= 0.7), "target_definition": definition,
    }


def extract_split(cfg: Config, split: str, out_dir: Path | None = None) -> list[dict]:
    streams = load_split(split)
    contexts: dict[str, EngineContext] = {}
    rows: list[dict] = []
    for stream in streams:
        profile = stream["corpus_profile"]
        if profile not in contexts:
            contexts[profile] = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
        ctx = contexts[profile]
        final_query = RetrievalQuery.build(stream["full_text"])
        final_result = ctx.index.search(final_query)
        previous_ranking: list[str] = []
        previous_accepted: list[str] = []
        previous_slots: dict = {}
        for index, (prefix, timestamp) in enumerate(zip(stream["prefixes"], stream["prefix_timestamps"]), start=1):
            frame = ctx.frame_builder.build(prefix, previous_slots, timestamp, f"{stream['stream_id']}:{index}")
            candidate = max(frame.candidates, key=lambda item: len(item.text), default=None)
            if candidate is None:
                continue
            query = RetrievalQuery.build(candidate.need_text, frame.anchor if candidate.key != "k0" else "",
                                         candidate.constraints, 0.4)
            probe = ctx.index.probe(query.terms)
            ranking = [item for item, _ in probe]
            result = ctx.index.search(query)
            accepted = [hit.chunk_id for hit in result.relevant_hits]
            first = index == 1
            rho = 0.5 if first else (rbo(ranking, previous_ranking) if ranking else 0.0)
            chi = clarity([score for _, score in probe])
            final = index == len(stream["prefixes"])
            kappa = completeness(candidate.text, candidate.closed, final)
            pi = continuation_hazard(candidate.text, candidate.closed, final)
            unchanged = sum(
                key in previous_slots and previous_slots[key].value == slot.value for key, slot in frame.slots.items()
            )
            sigma = unchanged / len(frame.slots) if frame.slots else 0.5
            cold = assess_cold_start(ctx.index, query, candidate, probe, is_final=final, cfg={})
            intent = assess_intent_readiness(candidate, {"kappa": kappa, "pi": pi}, is_final=final)
            idfs = [ctx.index.bm25.idf_of(term) for term in query.terms if term in ctx.index.bm25.idf]
            max_idf = max(ctx.index.bm25.idf.values(), default=1.0)
            informativeness = (sum(idfs) / len(idfs) / max_idf) if idfs else 0.0
            relevance = max((hit.rerank_score for hit in result.relevant_hits), default=0.0)
            _, domain_confidence, domain_available = ctx.index.route_domain(query.dense_text)
            feature_values = {
                "rho": rho, "chi": chi, "kappa": kappa, "sigma": sigma, "cont": 1.0 - pi,
                "first_observation": float(first), "cold_start_confidence": cold.confidence,
                "query_informativeness": informativeness, "rank_separation": cold.rank_separation,
                "objective_clarity": cold.objective_clarity, "intent_readiness": intent.score,
                "constraint_readiness": cold.constraint_readiness,
                "constraint_revision_risk": critical_constraint_risk(candidate),
                "evidence_relevance": relevance, "domain_availability": float(domain_available) * domain_confidence,
                "prefix_length": min(len(content_terms(prefix)) / 16.0, 1.0),
                "candidate_age": min((index - 1) / 5.0, 1.0),
                "last_slot_change": float(any(
                    key not in previous_slots or previous_slots[key].value != slot.value for key, slot in frame.slots.items()
                )),
                "accepted_evidence_overlap": rbo(accepted, previous_accepted) if previous_accepted and accepted else 0.0,
            }
            target = _survival(ctx, result, final_result, stream)
            rows.append({
                "stream_id": stream["stream_id"], "session_id": stream["session_id"],
                "utterance_id": stream["utterance_id"], "question_family": stream["question_family"],
                "source_dataset": stream["source_dataset"], "source_question_id": stream["source_question_id"],
                "corpus_profile": profile, "split": split, "prefix_index": index,
                "prefix_text": prefix, "candidate_id": candidate.key, "candidate_text": candidate.text,
                "final_text": stream["full_text"], "is_final": final,
                "rho_available": not first, "features": {key: round(float(value), 6) for key, value in feature_values.items()},
                "accepted_evidence": accepted, "final_accepted_evidence": [hit.chunk_id for hit in final_result.relevant_hits],
                "gold_sections": stream.get("gold_sections", []), "gold_documents": stream.get("gold_documents", []),
                **target,
            })
            previous_ranking, previous_accepted, previous_slots = ranking, accepted, frame.slots
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"features_{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return rows


def _matrix(rows: list[dict], features: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(feature, 0.0)) for feature in features] for row in rows])


def _fit(rows: list[dict], features: tuple[str, ...], *, c: float) -> tuple[dict, object]:
    from sklearn.linear_model import LogisticRegression

    usable = [row for row in rows if not row["is_final"]]
    x = _matrix(usable, features)
    y = np.asarray([row["survival_label"] for row in usable])
    if len(set(y.tolist())) < 2:
        raise RuntimeError("V2.2 survival labels contain only one class")
    model = LogisticRegression(C=c, max_iter=2000, random_state=42, class_weight="balanced").fit(x, y)
    weights = {feature: float(value) for feature, value in zip(features, model.coef_[0])}
    weights["bias"] = float(model.intercept_[0])
    return weights, model


def _predict(weights: dict[str, float], features: tuple[str, ...], rows: list[dict]) -> list[float]:
    out = []
    for row in rows:
        z = weights["bias"] + sum(weights[name] * float(row["features"].get(name, 0.0)) for name in features)
        out.append(1.0 / (1.0 + math.exp(-max(min(z, 40), -40))))
    return out


def train_models(cfg: Config, *, artifact_dir: Path | None = None) -> dict:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_dir = artifact_dir or resolve_path("runs/bench/controller_v2_2/modeling")
    train_rows = extract_split(cfg, "train", artifact_dir)
    calibration_rows = extract_split(cfg, "calibration", artifact_dir)
    development_rows = extract_split(cfg, "development", artifact_dir)
    train_fit = [row for row in train_rows if not row["is_final"]]
    calibration_fit = [row for row in calibration_rows if not row["is_final"]]
    development_fit = [row for row in development_rows if not row["is_final"]]

    historical = StabilityModel.load(cfg["controller"])
    historical_prob = [historical.predict(row["features"]) for row in development_fit]
    labels_dev = [row["survival_label"] for row in development_fit]
    models: dict[str, dict] = {
        "M0_historical": {"metrics": calibration_metrics(labels_dev, historical_prob), "source": historical.source}
    }
    specifications = {
        "M1_retrained_historical": (tuple(FEATURES), 1.0),
        "M2_extended_logistic": (EXTENDED_FEATURES, 1.0),
        "M3_regularized_logistic": (EXTENDED_FEATURES, 0.25),
    }
    fitted: dict[str, tuple[dict, tuple[str, ...]]] = {}
    for name, (features, c_value) in specifications.items():
        weights, _ = _fit(train_rows, features, c=c_value)
        probabilities = _predict(weights, features, development_fit)
        path = MODEL_ROOT / f"{name}.json"
        meta = {
            "feature_names": list(features), "trained_on": "controller_v2_2/train",
            "training_groups": len({row["question_family"] for row in train_fit}),
            "examples": len(train_fit), "positive_rate": round(sum(row["survival_label"] for row in train_fit) / len(train_fit), 4),
            "regularization_C": c_value, "target": "V2.2 composite accepted-evidence survival >= 0.7",
            "dataset_hash": _hash([row["stream_id"] for row in train_rows]),
        }
        save_model(path, {key: round(value, 8) for key, value in weights.items()}, meta)
        fitted[name] = (weights, features)
        models[name] = {"path": _portable_path(path), "metrics": calibration_metrics(labels_dev, probabilities), **meta}

    # Calibration is fitted only on the separate calibration groups.
    from sklearn.linear_model import LogisticRegression

    base_weights, base_features = fitted["M3_regularized_logistic"]
    cal_raw = _predict(base_weights, base_features, calibration_fit)
    cal_labels = np.asarray([row["survival_label"] for row in calibration_fit])
    logits = np.asarray([math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6))) for p in cal_raw]).reshape(-1, 1)
    if len(set(cal_labels.tolist())) >= 2:
        calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(logits, cal_labels)
        slope = float(calibrator.coef_[0][0])
        intercept = float(calibrator.intercept_[0])
    else:
        slope, intercept = 1.0, 0.0
    dev_raw = _predict(base_weights, base_features, development_fit)
    dev_calibrated = [1.0 / (1.0 + math.exp(-(slope * math.log(p / (1 - p)) + intercept))) for p in dev_raw]
    m4_path = MODEL_ROOT / "M4_calibrated_logistic.json"
    save_model(m4_path, {key: round(value, 8) for key, value in base_weights.items()}, {
        "feature_names": list(base_features), "trained_on": "controller_v2_2/train",
        "calibration": {"enabled": True, "method": "platt", "slope": slope, "intercept": intercept},
        "calibration_split": "controller_v2_2/calibration", "calibration_groups": len({row["question_family"] for row in calibration_fit}),
        "target": "V2.2 composite accepted-evidence survival >= 0.7",
    })
    models["M4_calibrated_logistic"] = {
        "path": _portable_path(m4_path), "metrics": calibration_metrics(labels_dev, dev_calibrated),
        "calibration": {"method": "platt", "slope": slope, "intercept": intercept},
    }
    models["M5_isotonic"] = {
        "status": "SKIPPED_INSUFFICIENT_INDEPENDENT_CALIBRATION_GROUPS" if len({row["question_family"] for row in calibration_fit}) < 100 else "ELIGIBLE_NOT_SELECTED",
        "calibration_groups": len({row["question_family"] for row in calibration_fit}),
    }
    cold_calibration_rows = [row for row in calibration_fit if row["features"].get("first_observation") == 1.0]
    cold_dev_rows = [row for row in development_fit if row["features"].get("first_observation") == 1.0]
    cold_cfg = {"enabled": False, "method": "identity", "slope": 1.0, "intercept": 0.0}
    if len({row["survival_label"] for row in cold_calibration_rows}) >= 2:
        cold_scores = np.asarray([
            math.log(min(max(row["features"]["cold_start_confidence"], 1e-6), 1 - 1e-6) /
                     (1 - min(max(row["features"]["cold_start_confidence"], 1e-6), 1 - 1e-6)))
            for row in cold_calibration_rows
        ]).reshape(-1, 1)
        cold_y = np.asarray([row["survival_label"] for row in cold_calibration_rows])
        cold_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(cold_scores, cold_y)
        cold_cfg = {"enabled": True, "method": "platt", "slope": float(cold_model.coef_[0][0]),
                    "intercept": float(cold_model.intercept_[0])}
    cold_path = MODEL_ROOT / "cold_start_calibration.json"
    cold_path.write_text(json.dumps({
        "calibration": cold_cfg, "fit_split": "controller_v2_2/calibration",
        "groups": len({row["question_family"] for row in cold_calibration_rows}),
        "examples": len(cold_calibration_rows),
    }, indent=2), encoding="utf-8")
    cold_probabilities = []
    for row in cold_dev_rows:
        score = min(max(row["features"]["cold_start_confidence"], 1e-6), 1 - 1e-6)
        if cold_cfg["enabled"]:
            value = cold_cfg["slope"] * math.log(score / (1 - score)) + cold_cfg["intercept"]
            score = 1.0 / (1.0 + math.exp(-value))
        cold_probabilities.append(score)
    result = {
        "status": "MEASURED", "models": models,
        "samples": {"train_prefixes": len(train_fit), "calibration_prefixes": len(calibration_fit), "development_prefixes": len(development_fit)},
        "groups": {split: len({row["question_family"] for row in values}) for split, values in {
            "train": train_fit, "calibration": calibration_fit, "development": development_fit}.items()},
        "policy_independent": True, "future_text_features": [],
        "cold_start_calibration": {
            "path": _portable_path(cold_path), "configuration": cold_cfg,
            "metrics": calibration_metrics([row["survival_label"] for row in cold_dev_rows], cold_probabilities),
            "development_first_observations": len(cold_dev_rows),
        },
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
