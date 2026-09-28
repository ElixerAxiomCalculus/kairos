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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kairos.config import resolve_path
from kairos.retrieval.index import HybridIndex, RetrievalQuery
from kairos.streaming.frame import CandidateNeed, Frame, completeness, continuation_hazard
from kairos.textutil import content_terms, rbo

FEATURES = ("rho", "chi", "kappa", "sigma", "cont")


class StabilityModel:
    def __init__(self, weights: dict[str, float], source: str) -> None:
        self.weights = weights
        self.source = source

    @classmethod
    def load(cls, cfg: dict) -> "StabilityModel":
        path = resolve_path(cfg["stability_model"]) if cfg.get("stability_model") else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(data["weights"], f"trained:{path.name}")
        return cls(dict(cfg["default_weights"]), "default_weights")

    def predict(self, feats: dict[str, float]) -> float:
        z = self.weights["bias"] + sum(self.weights[f] * feats[f] for f in FEATURES)
        return 1.0 / (1.0 + math.exp(-z))


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

    # --------------------------------------------------------------- signals
    def query_for(self, cand: CandidateNeed, frame: Frame, anchor_weight: float) -> RetrievalQuery:
        anchor = frame.anchor if cand.key != "k0" else ""
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
        s_hat = self.model.predict(feats)
        theta, hidden = self.threshold(remaining)
        sig = {**feats, "pi": pi, "nu": max(0.0, nu), "s_hat": s_hat, "theta": theta,
               "remaining_s": remaining, "hidden_latency_s": hidden}
        return {k: round(float(v), 4) for k, v in sig.items()}, keys

    # --------------------------------------------------------------- policy
    def decide(self, cand: CandidateNeed, state: CandidateState, sig: dict, probe: list[str], frame: Frame,
               is_final: bool, words_since_retrieval: int, budget_left: int, query: RetrievalQuery) -> CandidateDecision:
        cfg = self.cfg
        mode = cfg["mode"]
        reasons: list[str] = []

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
            # Shadow check: would the grown clause retrieve different evidence than what we hold?
            shadow = [h.chunk_id for h in self.index.search(query).hits]
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
        if ok_stable and ok_rho and ok_kappa and budget_left > 0:
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
        return CandidateDecision(cand.key, "WAIT", "NONE", reasons, sig, probe)


def save_model(path: Path, weights: dict[str, float], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weights": weights, **meta}, indent=2), encoding="utf-8")
