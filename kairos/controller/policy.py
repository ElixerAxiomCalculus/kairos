"""Probe-Stability Retrieval Control (PSRC).

For every candidate need we compute signals on each transcript chunk:

  ρ  result stability   RBO of BM25 probe rankings between consecutive observations
  χ  clarity            1 - normalised entropy of the probe score distribution
  κ  completeness       syntactic/semantic closure of the clause
  σ  slot stability     fraction of the clause's slots unchanged at this chunk
  π  continuation       probability that more speech changes the clause
  ν  novelty            1 - max cosine to queries already retrieved in this session

ŝ = sigmoid(w·[ρ, χ, κ, σ, 1-π] + b) estimates P(evidence retrieved now survives to the end).
Retrieving now beats waiting when ŝ ≥ θ = 1 - λ_L·H / (λ_C·C_r + λ_N·N), where H is the latency
that retrieving now can hide (bounded by the expected remaining speech).
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kairos.config import resolve_path
from kairos.controller.calibration import calibrate_probability
from kairos.controller.cold_start import assess_cold_start
from kairos.controller.evidence_cache import EvidenceCache, query_fingerprint
from kairos.controller.intent_readiness import assess_intent_readiness
from kairos.controller.utility import assess_utility
from kairos.retrieval.index import HybridIndex, RetrievalQuery
from kairos.streaming.frame import CandidateNeed, Frame, completeness, continuation_hazard
from kairos.textutil import content_terms, rbo

FEATURES = ("rho", "chi", "kappa", "sigma", "cont")


class StabilityModel:
    def __init__(self, weights: dict[str, float], source: str, *,
                 features: list[str] | tuple[str, ...] = FEATURES, calibration: dict | None = None) -> None:
        self.weights = weights
        self.source = source
        self.features = tuple(features)
        self.calibration = calibration or {}

    @classmethod
    def load(cls, cfg: dict) -> "StabilityModel":
        path = resolve_path(cfg["stability_model"]) if cfg.get("stability_model") else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(data["weights"], f"trained:{path.name}",
                       features=data.get("feature_names", data.get("features", FEATURES)),
                       calibration=data.get("calibration", {}))
        return cls(dict(cfg["default_weights"]), "default_weights")

    def predict_raw(self, feats: dict[str, float]) -> float:
        z = self.weights["bias"] + sum(self.weights[f] * float(feats.get(f) or 0.0) for f in self.features)
        return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))

    def predict(self, feats: dict[str, float]) -> float:
        return calibrate_probability(self.predict_raw(feats), self.calibration)


@dataclass
class CandidateState:
    key: str
    observations: int = 0
    last_text: str = ""
    last_probe: list[str] = field(default_factory=list)
    s_hat_prev: float = 0.0
    sub_query_id: str | None = None
    retrieved_text: str | None = None
    retrieved_probe: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    retrievals: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class CandidateDecision:
    key: str
    action: str                  # WAIT | RETRIEVE | REFINE | HOLD | SUPPRESS
    trigger: str                 # EARLY | END_OF_UTTERANCE | REFINE | NONE
    reasons: list[str]
    signals: dict[str, float]
    probe: list[str]


def clarity(scores: list[float], temperature: float = 0.25) -> float:
    if not scores or scores[0] <= 0:
        return 0.0
    if len(scores) == 1:
        return 1.0
    s = np.array(scores) / (scores[0] * temperature)
    p = np.exp(s - s.max())
    p /= p.sum()
    h = -(p * np.log(p + 1e-12)).sum()
    return float(max(0.0, 1.0 - h / math.log(len(scores))))


class RetrievalController:
    def __init__(self, cfg: dict, index: HybridIndex) -> None:
        self.cfg = cfg
        self.index = index
        self.model = StabilityModel.load(cfg)
        self.pipeline_latency_s = cfg["pipeline_latency_prior_s"]
        reuse_cfg = cfg.get("evidence_reuse", {})
        self.evidence_cache = EvidenceCache(max_entries=reuse_cfg.get("max_entries", 64),
                                            ttl_s=reuse_cfg.get("ttl_s", 300.0))
        self.corpus_profile = str(cfg.get("_corpus_profile", reuse_cfg.get("corpus_profile", "sample")))
        self.corpus_version = hashlib.sha256(
            "\n".join(f"{chunk.chunk_id}:{chunk.sha256}" for chunk in index.chunks).encode()
        ).hexdigest()

    def fingerprint(self, query: RetrievalQuery, *, objective: str | None = None,
                    typed_constraints: list[dict] | None = None) -> str:
        relevant_keys = ("mode", "dense_backend", "dense_model", "lsa_dims", "bm25_k1", "bm25_b",
                         "rrf_k", "candidates", "top_k", "reranker")
        index_cfg = {key: self.index.cfg.get(key) for key in relevant_keys}
        gate_cfg = {key: value for key, value in self.index.cfg.items() if str(key).startswith("relevance_")}
        return query_fingerprint(
            query, corpus_profile=self.corpus_profile, corpus_version=self.corpus_version,
            index_configuration=index_cfg, intent_objective=objective,
            typed_constraints=typed_constraints, retrieval_mode=self.index.cfg.get("mode"),
            relevance_gate_configuration=gate_cfg,
        )

    @property
    def policy_variant(self) -> str:
        return str(self.cfg.get("policy_variant", "baseline"))

    # --------------------------------------------------------------- signals
    def query_for(self, cand: CandidateNeed, frame: Frame, anchor_weight: float) -> RetrievalQuery:
        first_key = frame.candidates[0].key if frame.candidates else cand.key
        anchor = frame.anchor if cand.key != first_key else ""
        return RetrievalQuery.build(cand.need_text, anchor, cand.constraints, anchor_weight)

    def observe_pipeline_latency(self, seconds: float) -> None:
        """EMA of measured retrieval→verified-claim latency; feeds H in the threshold.
        Off by default: with the extractive backend the pipeline takes ~10 ms, which would teach the
        controller that early retrieval hides nothing. The prior models a generative deployment."""
        if self.cfg.get("adapt_latency", False):
            self.pipeline_latency_s = 0.8 * self.pipeline_latency_s + 0.2 * seconds

    def threshold(self, remaining_s: float) -> tuple[float, float]:
        cfg = self.cfg
        hidden = min(self.pipeline_latency_s, remaining_s)
        theta = 1.0 - cfg["lambda_latency"] * hidden / (cfg["cost_retrieval"] + cfg["cost_noise"])
        return float(min(max(theta, cfg["theta_min"]), cfg["theta_max"])), hidden

    def signals(self, cand: CandidateNeed, state: CandidateState, frame: Frame, is_final: bool,
                utterance_end_cue: bool, committed_queries: list[str], anchor_weight: float,
                now: float) -> tuple[dict, list[str]]:
        q = self.query_for(cand, frame, anchor_weight)
        probe = self.index.probe(q.terms)
        keys = [k for k, _ in probe]
        if state.observations == 0:
            terms = content_terms(cand.text)
            loo = dict(q.terms)
            if terms:
                loo.pop(terms[-1], None)
            prev_keys = [k for k, _ in self.index.probe(loo)]
        elif cand.text == state.last_text:
            prev_keys = keys
        else:
            prev_keys = state.last_probe
        rho = rbo(keys, prev_keys) if keys else 0.0
        chi = clarity([s for _, s in probe])
        kappa = completeness(cand.text, cand.closed, is_final)
        pi = continuation_hazard(cand.text, cand.closed, is_final)
        cand_low = cand.text.lower()
        slot_vals = [s for s in frame.slots.values() if s.value.lower() in cand_low]
        if slot_vals:
            sigma = sum(1 for s in slot_vals if s.last_changed_t < now) / len(slot_vals)
        else:
            sigma = 0.5
        if committed_queries:
            vec = self.index.encode(q.dense_text)
            sims = [float(np.dot(vec, self.index.encode(c))) for c in committed_queries]
            nu = 1.0 - max(sims)
        else:
            nu = 1.0
        p_end = 1.0 if is_final else (0.6 if utterance_end_cue else 0.0)
        remaining = (1.0 - p_end) * self.cfg["remaining_prior_s"]
        feats = {"rho": rho, "chi": chi, "kappa": kappa, "sigma": sigma, "cont": 1.0 - pi}
        cold_cfg = self.cfg.get("cold_start", {})
        need_extended = any(feature not in FEATURES for feature in self.model.features)
        assessment = assess_cold_start(self.index, q, cand, probe, is_final=is_final, cfg=cold_cfg)
        readiness = assess_intent_readiness(cand, {**feats, "pi": pi}, is_final=is_final)
        values = readiness.signals()
        intent_score = values.pop("score")
        extended = {
            "first_observation": float(state.observations == 0),
            "cold_start_confidence": assessment.confidence,
            "query_informativeness": assessment.informativeness,
            "rank_separation": assessment.rank_separation,
            "objective_clarity": assessment.objective_clarity,
            "intent_readiness": intent_score,
            "constraint_readiness": assessment.constraint_readiness,
            "constraint_revision_risk": assessment.constraint_risk,
            "evidence_relevance": assessment.evidence_relevance,
            "domain_availability": assessment.domain_compatibility,
            "prefix_length": min(len(content_terms(cand.text)) / 16.0, 1.0),
            "candidate_age": min(state.observations / 5.0, 1.0),
            "last_slot_change": float(any(slot.last_changed_t >= now for slot in slot_vals)),
            "accepted_evidence_overlap": rbo(keys, state.retrieved_probe) if state.retrieved_probe else 0.0,
        }
        model_features = {**feats, **extended}
        raw_s_hat = self.model.predict_raw(model_features)
        model_output = calibrate_probability(raw_s_hat, self.model.calibration)
        # Historical config-level calibration remains available for legacy experiments.
        s_hat = calibrate_probability(model_output, self.cfg.get("calibration", {}))
        theta, hidden = self.threshold(remaining)
        sig = {**feats, "pi": pi, "nu": max(0.0, nu), "s_hat": s_hat, "theta": theta,
               "s_hat_raw": raw_s_hat, "remaining_s": remaining, "hidden_latency_s": hidden,
               "rho_available": float(state.observations > 0), **({key: extended[key] for key in self.model.features
                                                                  if key in extended} if need_extended else {})}
        if cold_cfg.get("enabled", False) or state.observations == 0:
            sig.update({f"cold_{key}": value for key, value in assessment.signals().items()})
        intent_cfg = self.cfg.get("intent_completion", {})
        if intent_cfg.get("enabled", False) or need_extended:
            sig["intent_readiness"] = intent_score
            sig.update({f"intent_{key}": value for key, value in values.items()})
        utility_cfg = self.cfg.get("utility", {})
        if utility_cfg.get("enabled", False):
            sig.update(assess_utility(s_hat, hidden, utility_cfg).signals())
        return {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in sig.items()}, keys

    # --------------------------------------------------------------- policy
    def decide(self, cand: CandidateNeed, state: CandidateState, sig: dict, probe: list[str], frame: Frame,
               is_final: bool, words_since_retrieval: int, budget_left: int, query: RetrievalQuery) -> CandidateDecision:
        cfg = self.cfg
        mode = cfg["mode"]
        reasons: list[str] = []

        if cand.status in ("INCOMPLETE", "CONSTRAINT_ONLY") and sig["kappa"] < cfg["kappa_min"]:
            return CandidateDecision(cand.key, "WAIT", "NONE", ["CLARIFICATION_REQUIRED", "INCOMPLETE_CLAUSE"],
                                     sig, probe)

        if state.sub_query_id is not None:
            if cand.text == state.retrieved_text:
                return CandidateDecision(cand.key, "HOLD", "NONE", ["ALREADY_RETRIEVED"], sig, probe)
            if mode == "fixed_tokens":
                if words_since_retrieval >= cfg["fixed_tokens_n"] or is_final:
                    return CandidateDecision(cand.key, "REFINE", "REFINE", ["FIXED_TOKEN_TRIGGER"], sig, probe)
                return CandidateDecision(cand.key, "HOLD", "NONE", ["BELOW_TOKEN_COUNT"], sig, probe)
            retrieved_low = (state.retrieved_text or "").lower()
            revised = [s for s in frame.slots.values()
                       if s.revised_from and s.revised_from.lower() in retrieved_low and s.value.lower() not in retrieved_low]
            if revised and budget_left > 0:
                sig["slot_revised"] = 1.0
                return CandidateDecision(cand.key, "REFINE", "REFINE", ["SLOT_REVISED"], sig, probe)
            # Shadow checks may reuse only exact provenance/constraint-compatible validated results.
            reuse_cfg = cfg.get("evidence_reuse", {})
            typed = [item.to_public() for item in cand.typed_constraints]
            fingerprint = self.fingerprint(query, objective=cand.need_text, typed_constraints=typed)
            shadow_result = self.evidence_cache.get(fingerprint, self.corpus_version) if reuse_cfg.get("enabled", False) else None
            if shadow_result is None:
                shadow_result = self.index.search(query)
                sig["shadow_retrieval"] = 1.0
                sig["shadow_search_executed"] = 1.0
                if reuse_cfg.get("enabled", False):
                    self.evidence_cache.put(fingerprint, shadow_result, self.corpus_version, validated=True)
            else:
                sig["shadow_retrieval"] = 0.0
                sig["shadow_search_executed"] = 0.0
                sig["cached_shadow_result"] = 1.0
                sig["cache_status"] = "REUSABLE"
            shadow = [h.chunk_id for h in shadow_result.relevant_hits]
            shift = rbo(shadow, state.retrieved_ids or state.retrieved_probe, 0.9)
            sig["rho_vs_retrieved"] = round(shift, 4)
            if shift < cfg["tau_refine"] and budget_left > 0 and (cand.closed or is_final or sig["s_hat"] >= sig["theta"]):
                return CandidateDecision(cand.key, "REFINE", "REFINE", ["RESULTS_WOULD_CHANGE"], sig, probe)
            return CandidateDecision(cand.key, "HOLD", "NONE", ["CONSTRAINT_ABSORBED"], sig, probe)

        if not query.terms:
            return CandidateDecision(cand.key, "WAIT", "NONE", ["NO_CONTENT"], sig, probe)

        if mode == "eou":
            if is_final:
                return CandidateDecision(cand.key, "RETRIEVE", "END_OF_UTTERANCE", ["END_OF_UTTERANCE"], sig, probe)
            return CandidateDecision(cand.key, "WAIT", "NONE", ["EOU_MODE"], sig, probe)
        if mode == "fixed_tokens":
            if words_since_retrieval >= cfg["fixed_tokens_n"]:
                return CandidateDecision(cand.key, "RETRIEVE", "EARLY", ["FIXED_TOKEN_TRIGGER"], sig, probe)
            if is_final:
                return CandidateDecision(cand.key, "RETRIEVE", "END_OF_UTTERANCE", ["END_OF_UTTERANCE"], sig, probe)
            return CandidateDecision(cand.key, "WAIT", "NONE", ["BELOW_TOKEN_COUNT"], sig, probe)

        # PSRC
        if sig["nu"] < cfg["nu_min"] and sig["kappa"] >= cfg["kappa_min"]:
            return CandidateDecision(cand.key, "SUPPRESS", "NONE", ["DUPLICATE_OF_PRIOR_RETRIEVAL"], sig, probe)
        if is_final:
            return CandidateDecision(cand.key, "RETRIEVE", "END_OF_UTTERANCE", ["END_OF_UTTERANCE"], sig, probe)
        threshold = sig["theta"] + (cfg["hysteresis"] if state.retrievals else 0.0)
        ok_stable = sig["s_hat"] >= threshold
        ok_rho = sig["rho"] >= cfg["rho_min"] or sig["remaining_s"] <= self.pipeline_latency_s
        ok_kappa = sig["kappa"] >= cfg["kappa_min"]
        cold_cfg = cfg.get("cold_start", {})
        cold_ready = (
            cold_cfg.get("enabled", False)
            and state.observations == 0
            and (sig.get("cold_probability") if sig.get("cold_probability") is not None
                 else sig.get("cold_confidence", 0.0)) >= float(cold_cfg.get("threshold", 0.68))
            and sig.get("cold_evidence_relevance", 0.0) >= float(cold_cfg.get("min_relevance", 0.45))
            and sig.get("cold_domain_compatibility", 0.0) > 0.0
            and sig.get("cold_constraint_readiness", 0.0) >= float(cold_cfg.get("min_constraint_readiness", 0.75))
            and (str(cold_cfg.get("variant", "CS0_original_heuristic")).startswith("CS0")
                 or sig.get("cold_constraint_risk", 1.0) <= float(cold_cfg.get("max_constraint_risk", 0.45)))
            and sig["kappa"] >= float(cold_cfg.get("kappa_min", 0.30))
        )
        intent_cfg = cfg.get("intent_completion", {})
        intent_ready = (
            intent_cfg.get("enabled", False)
            and sig.get("intent_readiness", 0.0) >= float(intent_cfg.get("threshold", 0.76))
            and sig.get("intent_independent", 0.0) >= 0.75
            and sig.get("intent_constraint_readiness", 0.0) >= 0.75
        )
        utility_allowed = not cfg.get("utility", {}).get("enabled", False) or bool(sig.get("utility_allowed", 0.0))
        if cold_ready and utility_allowed and budget_left > 0:
            return CandidateDecision(cand.key, "RETRIEVE", "EARLY",
                                     ["COLD_START_CONFIDENCE", "COLD_START_RETRIEVE"], sig, probe)
        guarded_bypass = (
            intent_cfg.get("guarded_first_observation_bypass", False)
            and state.observations == 0 and intent_ready and ok_stable and ok_kappa
            and sig.get("cold_evidence_relevance", 0.0) >= float(intent_cfg.get("min_evidence_relevance", 0.45))
            and sig.get("cold_constraint_risk", 1.0) <= float(intent_cfg.get("max_constraint_risk", 0.35))
            and sig.get("cold_domain_compatibility", 0.0) > 0.0
        )
        if guarded_bypass and utility_allowed and budget_left > 0:
            return CandidateDecision(cand.key, "RETRIEVE", "EARLY",
                                     ["INTENT_READY", "FIRST_OBSERVATION_HISTORY_BYPASS"], sig, probe)
        if intent_ready and ok_stable and ok_rho and ok_kappa and utility_allowed and budget_left > 0:
            return CandidateDecision(cand.key, "RETRIEVE", "EARLY",
                                     ["INTENT_LEVEL_READY", "REQUEST_COMPLETED"], sig, probe)
        if ok_stable and ok_rho and ok_kappa and utility_allowed and budget_left > 0:
            reasons = ["STABLE_ABOVE_THRESHOLD"]
            if sig["rho"] >= 0.8:
                reasons.append("RESULTS_PLATEAU")
            if cand.closed:
                reasons.append("CLAUSE_CLOSED")
            return CandidateDecision(cand.key, "RETRIEVE", "EARLY", reasons, sig, probe)
        if not ok_kappa:
            reasons.append("INCOMPLETE_CLAUSE")
        if not ok_stable:
            reasons.append("BELOW_THRESHOLD")
        if not ok_rho:
            reasons.append("RESULTS_STILL_MOVING")
        if sig["chi"] < 0.15:
            reasons.append("AMBIGUOUS_PROBE")
        if budget_left <= 0:
            reasons.append("BUDGET_EXHAUSTED")
        if cfg.get("utility", {}).get("enabled", False) and not utility_allowed:
            reasons.append("UTILITY_OR_RISK_REJECTED")
        if cold_cfg.get("enabled", False) and state.observations == 0 and not cold_ready:
            reasons.append("COLD_START_REJECTED")
        return CandidateDecision(cand.key, "WAIT", "NONE", reasons, sig, probe)


def save_model(path: Path, weights: dict[str, float], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weights": weights, **meta}, indent=2), encoding="utf-8")
