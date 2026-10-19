"""Session orchestrator: transcript chunks in, causally-linked events out.

One SessionEngine per session. All state is ephemeral and lives on this object; ending the session
drops it. Every stage emits an Event whose parent_event_ids point at the event that caused it, so
each answer version can be traced back to the transcript chunks that produced it.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel

from kairos.config import Config, resolve_path
from kairos.controller.caper import CaperManager, PreparedEvidence
from kairos.controller.features import candidate_slot_snapshot
from kairos.controller.policy import CandidateState, RetrievalController
from kairos.controller.turns import TurnClassifier, TurnResult
from kairos.corpus.ingest import load_corpus
from kairos.decomposer.efis import Decomposer
from kairos.delta.engine import AffectedSubQuery, Constraint, DeltaEngine, extract_constraint
from kairos.evidence.graph import EvidenceGraph, chunk_node, claim_node, sq_node, version_node
from kairos.grounding.firewall import GroundingFirewall, find_contradictions
from kairos.retrieval.index import HybridIndex, SearchResult
from kairos.schemas.events import Event, wall_now
from kairos.schemas.models import AnswerBlock, AnswerVersion, Claim, SubQuery
from kairos.streaming.frame import CandidateNeed, Frame, FrameBuilder, Slot
from kairos.synthesis.base import ClaimDraft, SynthesisResult, context_tokens
from kairos.synthesis.extractive import ExtractiveSynthesizer
from kairos.synthesis.transform import TransformHandler, verify_transform
from kairos.telemetry.bus import EventBus
from kairos.textutil import content_set, content_terms, jaccard, lower_tokens


class ChunkIn(BaseModel):
    utterance_id: str
    text: str
    t0: float | None = None
    t1: float | None = None
    final: bool = False
    chunk_id: str | None = None


class EngineContext:
    """Process-wide, read-only resources shared by all sessions (index, models, config)."""

    def __init__(self, cfg: Config, bus: EventBus | None = None) -> None:
        self.cfg = cfg
        corpus_cfg = cfg["corpus"]
        profile = corpus_cfg.get("profile", "sample")
        configured_paths = corpus_cfg.get("paths")
        if configured_paths is None and profile != "sample":
            configured_paths = corpus_cfg.get("profiles", {}).get(profile)
        roots = [resolve_path(p) for p in (configured_paths or [corpus_cfg["path"]])]
        self.corpus = load_corpus(roots, profile=profile)
        self.corpus.places.update(str(place).lower() for place in corpus_cfg.get("places", []))
        self.index = HybridIndex(self.corpus, cfg["retrieval"])
        self.frame_builder = FrameBuilder(self.corpus, set(corpus_cfg.get("places", [])))
        self.turns = TurnClassifier(self.corpus)
        self.decomposer = Decomposer(cfg["decomposer"], self.index)
        self.extractive = ExtractiveSynthesizer(self.index, cfg["synthesis"])
        self.llm = None
        if cfg["synthesis"]["backend"] in ("gemini", "ollama"):
            from kairos.synthesis.llm import LLMSynthesizer

            self.llm = LLMSynthesizer(self.index, cfg["synthesis"], self.extractive)
        self.synth = self.llm or self.extractive
        self.firewall = GroundingFirewall(self.index, cfg["grounding"])
        self.delta = DeltaEngine(self.index, cfg["delta"], cfg["decomposer"]["anchor_weight"])
        self.transforms = TransformHandler(self.llm)
        self.bus = bus or EventBus()
        self.index_built_at = wall_now()

    def corpus_status(self) -> dict:
        return {**self.index.status(), "indexed_at": self.index_built_at,
                "documents_list": list(self.corpus.documents.values())}

    def config_summary(self) -> dict:
        c = self.cfg
        return {
            "controller_mode": c["controller"]["mode"],
            "controller_policy_variant": c["controller"].get("policy_variant", "baseline"),
            "caper_enabled": bool(c["controller"].get("caper", {}).get("enabled", False)),
            "retrieval_mode": c["retrieval"]["mode"],
            "reranker": c["retrieval"]["reranker"],
            "synthesis_backend": getattr(self.synth, "name", "extractive"),
            "grounding_nli": c["grounding"]["nli"],
            "refinement": c["delta"]["refinement"],
            "corpus_profile": c["corpus"].get("profile", "sample"),
            "persistent_trace_content": bool(c.get("telemetry", {}).get("trace_content", False)),
        }


@dataclass
class UtteranceState:
    id: str
    chunks: list[dict] = field(default_factory=list)
    text: str = ""
    closed: bool = False
    turn: TurnResult | None = None
    frame: Frame | None = None
    slots: dict[str, Slot] = field(default_factory=dict)
    cstate: dict[str, CandidateState] = field(default_factory=dict)
    sq_ids: list[str] = field(default_factory=list)
    reused_sq_ids: list[str] = field(default_factory=list)
    words_since_retrieval: int = 0
    retrievals: int = 0
    first_retrieval_t: float | None = None
    last_chunk_t: float = 0.0
    last_chunk_event: str | None = None
    last_frame_event: str | None = None
    decision_events: list[str] = field(default_factory=list)
    fused_events: list[str] = field(default_factory=list)
    suppressed_event: str | None = None
    delta_key: str | None = None
    delta_affected: dict[str, AffectedSubQuery] = field(default_factory=dict)
    delta_unaffected: list[dict] = field(default_factory=list)
    delta_new_ids: dict[str, list[str]] = field(default_factory=dict)
    delta_event: str | None = None
    tokens: dict = field(default_factory=lambda: {"llm_prompt": 0, "llm_completion": 0, "context_est": 0})


@dataclass
class _RevisionCheckpoint:
    event_index: int
    utterances: dict[str, UtteranceState]
    order: list[str]
    sub_queries: dict[str, SubQuery]
    claims: dict[str, Claim]
    versions: list[AnswerVersion]
    retrieved: set[str]
    committed: list[tuple[str, str]]
    vocab: set[str]
    graph: EvidenceGraph


def _priority(action: str) -> int:
    return {"REFINE": 4, "RETRIEVE": 3, "SUPPRESS": 2, "WAIT": 1, "HOLD": 0}[action]


class SessionEngine:
    def __init__(self, ctx: EngineContext, session_id: str | None = None) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.id = session_id or f"s-{uuid.uuid4().hex}"
        controller_cfg = copy.deepcopy(self.cfg["controller"])
        controller_cfg["_corpus_profile"] = self.cfg["corpus"].get("profile", "sample")
        self.controller = RetrievalController(controller_cfg, ctx.index)
        self.graph = EvidenceGraph()
        self.utterances: dict[str, UtteranceState] = {}
        self.order: list[str] = []
        self.sub_queries: dict[str, SubQuery] = {}
        self.claims: dict[str, Claim] = {}
        self.versions: list[AnswerVersion] = []
        self.retrieved: set[str] = set()
        self.committed: list[tuple[str, str]] = []   # (utterance_id, dense query text)
        self.vocab: set[str] = set()
        self.events: list[Event] = []
        self.query_version = 0
        self._seq = 0
        self._sq_seq = 0
        self._claim_seq = 0
        self._lineage_seq = 0
        self._base_t = 0.0
        self._base_perf = time.perf_counter()
        self._wall_start = time.perf_counter()
        self.last_active = time.time()
        self.ended = False
        self._lock = asyncio.Lock()
        self._revision_checkpoints: dict[str, _RevisionCheckpoint] = {}
        self._invalidated_event_ids: set[str] = set()
        self.caper = CaperManager(
            session_id=self.id,
            index=self.ctx.index,
            cache=self.controller.evidence_cache,
            corpus_version=self.controller.corpus_version,
            cfg=controller_cfg.get("caper", {}),
            emit=self.emit,
        )

    # ------------------------------------------------------------------ plumbing
    def now(self) -> float:
        return self._base_t + (time.perf_counter() - self._base_perf)

    def _set_clock(self, t: float) -> None:
        last = self.events[-1].timestamp if self.events else 0.0
        self._base_t = max(t, last)
        self._base_perf = time.perf_counter()

    def wall_elapsed(self) -> float:
        return time.perf_counter() - self._wall_start

    def emit(self, event_type: str, parents: list[str | None] | None = None, **fields) -> Event:
        self._seq += 1
        ts = fields.pop("t", None)
        ev = Event(
            event_id=f"evt_{self._seq:04d}",
            parent_event_ids=[p for p in (parents or []) if p],
            timestamp=round(ts if ts is not None else self.now(), 4),
            wall_time=wall_now(),
            session_id=self.id,
            event_type=event_type,
            **fields,
        )
        self.events.append(ev)
        self.ctx.bus.publish(ev)
        return ev

    def _next_sq_id(self) -> str:
        self._sq_seq += 1
        return f"I{self._sq_seq:02d}"

    def _next_claim_id(self) -> str:
        self._claim_seq += 1
        return f"C{self._claim_seq:02d}"

    def _next_lineage(self) -> str:
        self._lineage_seq += 1
        return f"L{self._lineage_seq:02d}"

    def latest_version(self) -> AnswerVersion | None:
        return self.versions[-1] if self.versions else None

    def active_events(self) -> list[Event]:
        """Events that still describe authoritative state after ASR reconciliation."""
        return [e for e in self.events if e.event_id not in self._invalidated_event_ids]

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> Event:
        return self.emit("SessionStarted", payload={
            "corpus": self.ctx.corpus_status(), "config": self.ctx.config_summary(),
            "stability_model": self.controller.model.source,
            "caper": {"enabled": self.caper.enabled, "status": "EXPERIMENTAL" if self.caper.enabled else "DISABLED"},
        }, t=0.0)

    async def end(self) -> Event:
        for u in list(self.utterances.values()):
            if not u.closed:
                await self._commit(u)
        await self.caper.close()
        ev = self.emit("SessionEnded", payload={
            "utterances": len(self.utterances), "answer_versions": len(self.versions),
            "events": self._seq + 1, "purged": True, "caper": self.caper.snapshot()["stats"],
        })
        self.ended = True
        self.utterances.clear()
        self.sub_queries.clear()
        self.claims.clear()
        self.versions.clear()
        self.graph = EvidenceGraph()
        self.retrieved.clear()
        self.committed.clear()
        self.vocab.clear()
        self._revision_checkpoints.clear()
        self._invalidated_event_ids.clear()
        self.controller.evidence_cache.invalidate_session(self.id)
        return ev

    async def begin_revisable_utterance(self, utterance_id: str) -> None:
        """Capture domain state before a mutable ASR utterance reaches the engine."""
        async with self._lock:
            if utterance_id in self._revision_checkpoints:
                return
            if any(uid != utterance_id for uid in self._revision_checkpoints):
                raise ValueError("another revisable utterance is already active")
            self._revision_checkpoints[utterance_id] = _RevisionCheckpoint(
                event_index=len(self.events),
                utterances=copy.deepcopy(self.utterances),
                order=list(self.order),
                sub_queries=copy.deepcopy(self.sub_queries),
                claims=copy.deepcopy(self.claims),
                versions=copy.deepcopy(self.versions),
                retrieved=set(self.retrieved),
                committed=list(self.committed),
                vocab=set(self.vocab),
                graph=copy.deepcopy(self.graph),
            )

    async def reconcile_utterance(self, utterance_id: str, authoritative_text: str) -> None:
        """Roll back one open ASR turn and reprocess its corrected authoritative text.

        The append-only event log retains an explicit invalidation record for auditability;
        consumers remove the listed superseded events from their current projection.
        """
        async with self._lock:
            checkpoint = self._revision_checkpoints.get(utterance_id)
            if checkpoint is None:
                raise ValueError("utterance is not open for transcript reconciliation")
            invalidated = [e.event_id for e in self.events[checkpoint.event_index:]
                           if e.event_id not in self._invalidated_event_ids]
            self._invalidated_event_ids.update(invalidated)
            self.utterances = copy.deepcopy(checkpoint.utterances)
            self.order = list(checkpoint.order)
            self.sub_queries = copy.deepcopy(checkpoint.sub_queries)
            self.claims = copy.deepcopy(checkpoint.claims)
            self.versions = copy.deepcopy(checkpoint.versions)
            self.retrieved = set(checkpoint.retrieved)
            self.committed = list(checkpoint.committed)
            self.vocab = set(checkpoint.vocab)
            self.graph = copy.deepcopy(checkpoint.graph)
            prior = self.events[:checkpoint.event_index]
            parent = next((e.event_id for e in reversed(prior)
                           if e.event_id not in self._invalidated_event_ids), None)
            ev = self.emit("TranscriptReconciled", [parent], utterance_id=utterance_id, payload={
                "invalidated_event_ids": invalidated,
                "authoritative_text": authoritative_text,
                "reason": "ASR_CORRECTION_CROSSED_COMMITTED_PREFIX",
            })
            self.caper.invalidate_utterance(
                utterance_id, ev.event_id, "ASR_CORRECTION_CROSSED_COMMITTED_PREFIX"
            )
            if authoritative_text:
                u = UtteranceState(utterance_id, last_chunk_event=ev.event_id)
                self.utterances[utterance_id] = u
                self.order.append(utterance_id)
                await self._on_chunk(ChunkIn(
                    utterance_id=utterance_id,
                    text=authoritative_text,
                    final=False,
                    chunk_id=f"{utterance_id}.reconciled.{self._seq + 1}",
                ))

    async def finish_revisable_utterance(self, utterance_id: str) -> None:
        async with self._lock:
            self._revision_checkpoints.pop(utterance_id, None)

    async def end_utterance(self, utterance_id: str) -> None:
        async with self._lock:
            u = self.utterances.get(utterance_id)
            if u and not u.closed:
                self._set_clock(max(u.last_chunk_t, self.now()))
                await self._finalize_decisions(u)
                await self._commit(u)

    # ------------------------------------------------------------------ chunk path
    async def on_chunk(self, chunk: ChunkIn) -> None:
        async with self._lock:
            await self._on_chunk(chunk)

    async def _on_chunk(self, c: ChunkIn) -> None:
        self.last_active = time.time()
        u = self.utterances.get(c.utterance_id)
        if u is None:
            for other in self.utterances.values():
                if not other.closed:
                    await self._finalize_decisions(other)
                    await self._commit(other)
            u = UtteranceState(c.utterance_id)
            self.utterances[c.utterance_id] = u
            self.order.append(c.utterance_id)
        if u.closed:
            return
        t1 = c.t1 if c.t1 is not None else self.wall_elapsed()
        t0 = c.t0 if c.t0 is not None else (u.last_chunk_t or t1)
        self._set_clock(t1)
        seq = len(u.chunks) + 1
        chunk_id = c.chunk_id or f"{u.id}.c{seq}"
        u.text = f"{u.text} {c.text}".strip()
        u.chunks.append({"chunk_id": chunk_id, "seq": seq, "t0": t0, "t1": t1, "text": c.text, "final": c.final})
        u.last_chunk_t = t1
        u.words_since_retrieval += len(lower_tokens(c.text))
        ev_chunk = self.emit("TranscriptChunkReceived", [u.last_chunk_event], utterance_id=u.id, t=t1, payload={
            "chunk_id": chunk_id, "seq": seq, "t0": round(t0, 3), "t1": round(t1, 3), "text": c.text,
            "final": c.final, "utterance_text": u.text,
        })
        u.last_chunk_event = ev_chunk.event_id

        t_stage = time.perf_counter()
        previous_frame = u.frame
        frame = self.ctx.frame_builder.build(u.text, u.slots, t1, chunk_id, previous_frame)
        u.frame = frame
        u.slots = frame.slots
        self.query_version += 1
        ev_frame = self.emit("QueryFrameUpdated", [ev_chunk.event_id], utterance_id=u.id,
                             query_version=self.query_version,
                             latency_ms=round((time.perf_counter() - t_stage) * 1000, 3), payload={
            "raw_text": frame.raw_text, "analysis_text": frame.analysis_text, "normalized": frame.normalized(),
            "anchor": frame.anchor, "repairs": frame.repairs,
            "spans": [{"text": s.text, "kind": s.kind, "closed": s.closed, "soft": s.soft,
                       "start": s.start, "end": s.end, "request_cue": s.request_cue} for s in frame.spans],
            "candidates": [x.to_public() for x in frame.candidates],
            "slots": [s.to_public() for s in frame.slots.values()],
        })
        u.last_frame_event = ev_frame.event_id

        turn = self.ctx.turns.classify(u.text, bool(self.versions), self.vocab, c.final)
        if u.turn is None or u.turn.turn_type != turn.turn_type or c.final:
            ev_turn = self.emit("TurnClassified", [ev_frame.event_id], utterance_id=u.id,
                                query_version=self.query_version, confidence=turn.confidence,
                                payload=turn.to_public())
            parent = ev_turn.event_id
        else:
            parent = ev_frame.event_id
        u.turn = turn

        if turn.turn_type == "TRANSFORM":
            self._decision_suppress(u, parent, "PRESENTATION_TRANSFORMATION", turn)
        elif turn.turn_type == "CONSTRAINT":
            await self._constraint_stream(u, frame, parent, c.final)
        elif turn.turn_type == "ELABORATE":
            self._emit_decision(u, parent, "WAIT", "NONE", ["AWAITING_END_OF_UTTERANCE"], {}, [], turn)
        else:
            await self._new_info_stream(u, frame, parent, c.final)

        if c.final:
            await self._commit(u)

    # ------------------------------------------------------------------ decisions
    def _emit_decision(self, u: UtteranceState, parent: str | None, decision: str, trigger: str,
                       reasons: list[str], signals: dict, candidates: list[dict], turn: TurnResult,
                       extra_parents: list[str] | None = None) -> Event:
        ev = self.emit("RetrievalDecision", [parent, *(extra_parents or [])], utterance_id=u.id,
                       query_version=self.query_version, retrieval_trigger=trigger,
                       answer_version=self.latest_version().version if self.versions else None,
                       confidence=signals.get("s_hat"), payload={
            "decision": decision, "reasons": reasons, "signals": signals, "candidates": candidates,
            "turn_type": turn.turn_type, "controller_mode": self.cfg["controller"]["mode"],
            "policy_variant": self.controller.policy_variant,
            "stability_model": self.controller.model.source,
            "budget_left": self.cfg["controller"]["budget_per_utterance"] - u.retrievals,
            "decision_provenance": [{
                "controller_variant": self.controller.policy_variant,
                "model_version": self.controller.model.source,
                "query_version": self.query_version,
                "candidate_id": candidate.get("key"),
                "information_need_id": candidate.get("sub_query_id"),
                "feature_values": candidate.get("signals", {}),
                "predicted_survival": candidate.get("signals", {}).get("s_hat_raw"),
                "calibrated_probability": candidate.get("signals", {}).get("s_hat"),
                "decision_threshold": candidate.get("signals", {}).get("theta"),
                "cold_start_confidence": candidate.get("signals", {}).get("cold_confidence"),
                "intent_readiness": candidate.get("signals", {}).get("intent_readiness"),
                "constraint_risk": candidate.get("signals", {}).get("cold_constraint_risk"),
                "evidence_relevance": candidate.get("signals", {}).get("cold_evidence_relevance"),
                "cache_status": candidate.get("signals", {}).get("cache_status", "MISS_OR_DISABLED"),
                "decision": candidate.get("action"),
                "decision_reasons": candidate.get("reasons", []),
            } for candidate in candidates],
        })
        u.decision_events.append(ev.event_id)
        return ev

    def _decision_suppress(self, u: UtteranceState, parent: str | None, reason: str, turn: TurnResult) -> None:
        ev = self._emit_decision(u, parent, "SUPPRESS", "NONE", [reason], {}, [], turn)
        if u.suppressed_event is None:
            base = self.latest_version()
            reused = sorted({e for c in base.claims.values() for e in c.evidence}) if base else []
            sev = self.emit("RetrievalSuppressed", [ev.event_id], utterance_id=u.id,
                            query_version=self.query_version,
                            answer_version=base.version if base else None, retrieval_trigger="NONE",
                            chunk_ids=reused, document_ids=sorted({self.ctx.corpus.chunk(x).doc_id for x in reused
                                                                  if self.ctx.corpus.chunk(x)}), payload={
                "reason": reason, "reused_answer_version": base.version if base else None,
                "evidence_reused": len(reused), "skipped": ["VECTOR_SEARCH", "BM25", "RERANKER", "FUSION"],
                "transform": turn.transform,
            })
            u.suppressed_event = sev.event_id

    async def _new_info_stream(self, u: UtteranceState, frame: Frame, parent: str | None, is_final: bool) -> None:
        ctl = self.controller
        aw = self.cfg["decomposer"]["anchor_weight"]
        prior = [q for uid, q in self.committed if uid != u.id]
        end_cue = frame.raw_text.rstrip().endswith((".", "?", "!"))
        budget_left = self.cfg["controller"]["budget_per_utterance"] - u.retrievals
        t_probe = time.perf_counter()
        decisions = []
        candidates = frame.candidates
        if self.cfg["decomposer"].get("mode") == "single" and candidates:
            # Ablation E: no decomposition — the whole utterance is one retrieval query.
            candidates = [CandidateNeed("k0", " ".join(c.text for c in candidates), [],
                                        candidates[-1].closed, False)]
        for cand in candidates:
            st = u.cstate.setdefault(cand.key, CandidateState(cand.key))
            q = ctl.query_for(cand, frame, aw)
            sig, probe = ctl.signals(cand, st, frame, is_final, end_cue, prior, aw, now=self._base_t)
            dec = ctl.decide(cand, st, sig, probe, frame, is_final, u.words_since_retrieval, budget_left, q)
            st.observations += 1
            st.last_text = cand.text
            st.last_probe = probe
            st.last_query_text = q.dense_text
            st.last_slots = candidate_slot_snapshot(cand, frame)
            st.s_hat_prev = sig["s_hat"]
            st.history.append({"t": round(self._base_t, 3), "text": cand.text, **sig})
            decisions.append((cand, st, dec))
        probe_ms = (time.perf_counter() - t_probe) * 1000
        ev_probe = self.emit("ProbeExecuted", [parent], utterance_id=u.id, query_version=self.query_version,
                             latency_ms=round(probe_ms, 3), payload={
            "probes": [{"candidate": d.key, "sections": d.probe[:5]} for _, _, d in decisions],
            "retriever": "bm25-probe",
            "cache": self.ctx.index.status()["probe_cache"],
        })
        if self.caper.enabled:
            for cand, _st, dec in decisions:
                query = ctl.query_for(cand, frame, aw)
                self.caper.observe(
                    utterance_id=u.id,
                    intent_id=cand.key,
                    query_version=self.query_version,
                    query=query,
                    typed_constraints=[item.to_public() for item in cand.typed_constraints],
                    probe=dec.probe,
                    signals=dec.signals,
                    is_final=is_final,
                    parent_event_id=ev_probe.event_id,
                    timestamp=self.now(),
                    allow_prefetch=dec.action not in {"RETRIEVE", "REFINE"},
                )
        if not decisions:
            self._emit_decision(u, ev_probe.event_id, "WAIT", "NONE", ["NO_CANDIDATE_NEED"], {}, [], u.turn)
            return
        top = max(decisions, key=lambda x: (_priority(x[2].action), x[2].signals.get("s_hat", 0)))
        frame_action = top[2].action if top[2].action != "HOLD" else "WAIT"
        cand_pub = [{"key": c.key, "text": c.text, "action": d.action, "trigger": d.trigger, "reasons": d.reasons,
                     "signals": d.signals, "sub_query_id": st.sub_query_id} for c, st, d in decisions]
        ev_dec = self._emit_decision(u, ev_probe.event_id, frame_action, top[2].trigger, top[2].reasons,
                                     top[2].signals, cand_pub, u.turn)
        await self._act(u, frame, decisions, ev_dec)

    async def _act(self, u: UtteranceState, frame: Frame, decisions, ev_dec: Event) -> None:
        jobs: list[tuple[SubQuery, str]] = []
        assignments = []
        first_sq = self.sub_queries[u.sq_ids[0]] if u.sq_ids else None
        for cand, st, dec in decisions:
            if dec.action == "RETRIEVE":
                existing = [self.sub_queries[i] for i in u.sq_ids]
                a = self.ctx.decomposer.assign(cand, frame, existing, f"I{self._sq_seq + 1:02d}", u.id, first_sq)
                if a.created:
                    self._sq_seq += 1
                    self.sub_queries[a.sub_query.id] = a.sub_query
                    u.sq_ids.append(a.sub_query.id)
                    first_sq = first_sq or a.sub_query
                st.sub_query_id = a.sub_query.id
                trigger = dec.trigger if a.created else "REFINE"
                if all(j[0].id != a.sub_query.id for j in jobs):
                    jobs.append((a.sub_query, trigger))
                assignments.append({"candidate": cand.key, "sub_query_id": a.sub_query.id, "created": a.created,
                                    "merged_into": a.merged_into, "footprint_overlap": a.footprint_overlap,
                                    "footprint_valid": a.footprint_valid,
                                    "standalone_status": a.standalone_status,
                                    "merge_reason": a.merge_reason,
                                    "decision_signals": a.decision_signals or {}})
            elif dec.action == "REFINE":
                sq = self.sub_queries[st.sub_query_id]
                self.ctx.decomposer.refresh(sq, cand)
                if all(j[0].id != sq.id for j in jobs):
                    jobs.append((sq, "REFINE"))
                assignments.append({"candidate": cand.key, "sub_query_id": sq.id, "created": False,
                                    "merged_into": None, "refined": True})
            elif dec.action == "SUPPRESS":
                prior = self._most_similar_prior_sq(cand, frame)
                if prior is not None:
                    st.sub_query_id = prior.id
                    st.retrieved_text = cand.text
                    if prior.id not in u.reused_sq_ids:
                        u.reused_sq_ids.append(prior.id)
                    self.emit("RetrievalSuppressed", [ev_dec.event_id], utterance_id=u.id,
                              query_version=self.query_version, sub_query_id=prior.id, retrieval_trigger="NONE",
                              chunk_ids=prior.evidence, payload={
                        "reason": "DUPLICATE_OF_PRIOR_RETRIEVAL", "reused_sub_query": prior.id,
                        "evidence_reused": len(prior.evidence), "skipped": ["VECTOR_SEARCH", "BM25", "RERANKER"],
                    })
        if not jobs:
            return
        ev_sq = self.emit("SubQueriesGenerated", [ev_dec.event_id], utterance_id=u.id,
                          query_version=self.query_version, payload={
            "sub_queries": [self.sub_queries[i].to_public() for i in u.sq_ids],
            "assignments": assignments, "active": [j[0].id for j in jobs],
            "parallel": len(jobs) > 1,
        })
        await self._retrieve(u, jobs, ev_sq.event_id)
        for cand, st, dec in decisions:
            if dec.action in ("RETRIEVE", "REFINE"):
                st.retrieved_text = cand.text
                st.retrieved_probe = dec.probe
                if st.sub_query_id in self.sub_queries:
                    st.retrieved_ids = list(self.sub_queries[st.sub_query_id].evidence)
                    st.retrieved_sections = sorted({
                        self.ctx.index.chunks[self.ctx.index.pos[chunk_id]].section_key
                        for chunk_id in st.retrieved_ids if chunk_id in self.ctx.index.pos
                    })
                st.retrievals += 1
        u.words_since_retrieval = 0

    def _most_similar_prior_sq(self, cand, frame) -> SubQuery | None:
        q = self.controller.query_for(cand, frame, self.cfg["decomposer"]["anchor_weight"])
        vec = self.ctx.index.encode(q.dense_text)
        best, best_sim = None, -1.0
        for sq in self.sub_queries.values():
            if sq.query is None:
                continue
            sim = float(vec @ self.ctx.index.encode(sq.query.dense_text))
            if sim > best_sim:
                best, best_sim = sq, sim
        return best

    async def _finalize_decisions(self, u: UtteranceState) -> None:
        """An utterance ended without a final-flagged chunk: run the end-of-utterance decision pass."""
        if u.frame is None or u.turn is None:
            return
        if u.turn.turn_type == "NEW_INFO":
            await self._new_info_stream(u, u.frame, u.last_frame_event, True)
        elif u.turn.turn_type == "CONSTRAINT":
            await self._constraint_stream(u, u.frame, u.last_frame_event, True)

    # ------------------------------------------------------------------ retrieval
    def _timed_search(
        self,
        sq: SubQuery,
        trigger: str,
        prepared: PreparedEvidence | None = None,
    ) -> tuple[SearchResult, float, float, str, str, str | None]:
        start = time.perf_counter()
        reuse_cfg = self.cfg["controller"].get("evidence_reuse", {})
        fingerprint = self.controller.fingerprint(
            sq.query, objective=" ".join(sq.need_texts), typed_constraints=sq.typed_constraints
        )
        res = copy.deepcopy(prepared.result) if prepared is not None else None
        reuse_kind = "CAPER_VALIDATED" if prepared is not None else "NONE"
        if res is None and reuse_cfg.get("enabled", False):
            res = self.controller.evidence_cache.get(fingerprint, self.controller.corpus_version)
            if res is not None:
                reuse_kind = "VALIDATED_CACHE"
        if res is None:
            retrieval_type = {
                "REFINE": "REFINEMENT",
                "DELTA": "REFINEMENT",
                "END_OF_UTTERANCE": "FINAL_FALLBACK",
            }.get(trigger, "PRIMARY")
            res = self.ctx.index.search(sq.query, execution_context={
                "retrieval_type": retrieval_type,
                "session_id": self.id,
                "intent_id": sq.id,
                "query_version": self.query_version,
            })
            if reuse_cfg.get("enabled", False):
                # Admission already ran inside HybridIndex.search; grounding still runs downstream.
                self.controller.evidence_cache.put(
                    fingerprint, res, self.controller.corpus_version, validated=True
                )
        return res, start, time.perf_counter(), reuse_kind, fingerprint, (
            prepared.evidence_id if prepared is not None else None
        )

    async def _retrieve(self, u: UtteranceState, jobs: list[tuple[SubQuery, str]], parent: str | None) -> list[Event]:
        prepared: dict[str, PreparedEvidence] = {}
        if self.caper.enabled:
            for sq, trigger in jobs:
                wait_s = self.caper.finalization_wait_s if trigger == "END_OF_UTTERANCE" else 0.0
                record = await self.caper.validated_for(sq.candidate_keys, wait_s=wait_s)
                if record is not None:
                    prepared[sq.id] = record
        started: dict[str, Event] = {}
        for sq, trig in jobs:
            started[sq.id] = self.emit("RetrievalStarted", [parent], utterance_id=u.id,
                                       query_version=self.query_version, sub_query_id=sq.id,
                                       retrieval_trigger=trig, payload={
                "query": sq.query.text, "terms": {k: round(v, 2) for k, v in sq.query.terms.items()},
                "facet_constraints": sq.query.facet_constraints, "mode": self.cfg["retrieval"]["mode"],
                "label": sq.label, "parallel_group": [j[0].id for j in jobs],
                "prepared_evidence_id": prepared[sq.id].evidence_id if sq.id in prepared else None,
                "full_search_planned": sq.id not in prepared,
            })
        base_t, base_perf = self.now(), time.perf_counter()
        if u.first_retrieval_t is None:
            u.first_retrieval_t = base_t
        results = await asyncio.gather(*(
            asyncio.to_thread(self._timed_search, sq, trigger, prepared.get(sq.id))
            for sq, trigger in jobs
        ))
        completed: list[Event] = []
        all_ids: list[str] = []
        all_candidate_ids: list[str] = []
        for (sq, trig), (res, start, end, reuse_kind, fingerprint, prepared_id) in zip(jobs, results):
            if sq.id in prepared:
                self.caper.mark_consumed(prepared[sq.id])
            sq.candidate_hits = res.hits
            sq.hits = res.relevant_hits
            sq.status = "RETRIEVED"
            sq.evidence_status = res.evidence_status
            sq.selected_domain = res.domain
            sq.retrieval_count += 1
            candidate_ids = [h.chunk_id for h in res.hits]
            ids = [h.chunk_id for h in res.relevant_hits]
            all_ids.extend(ids)
            all_candidate_ids.extend(candidate_ids)
            self.retrieved.update(candidate_ids)
            self.committed.append((u.id, sq.query.dense_text))
            u.retrievals += 1
            self.graph.add_node(sq_node(sq.id), "sub_query", label=sq.label, text=sq.text)
            for h in res.hits:
                ch = self.ctx.corpus.chunk(h.chunk_id)
                self.graph.add_node(chunk_node(h.chunk_id), "chunk", doc_id=ch.doc_id, section=ch.section_number,
                                    evidence_status=h.relevance_status, accepted=h.accepted,
                                    source_dataset=ch.source_dataset, corpus_profile=ch.corpus_profile)
                self.graph.add_edge(sq_node(sq.id), chunk_node(h.chunk_id), "RETRIEVED", rank=h.rank,
                                    rerank=round(h.rerank_score, 4), trigger=trig, accepted=h.accepted,
                                    evidence_status=h.relevance_status)
            for dup, canon in res.duplicates:
                self.graph.add_edge(chunk_node(dup), chunk_node(canon), "DUPLICATES")
            end_off = (end - base_perf)
            hits_pub = []
            for h in res.hits:
                ch = self.ctx.corpus.chunk(h.chunk_id)
                hits_pub.append({**h.to_public(), **ch.to_public(self.cfg["telemetry"]["trace_content"])})
            ev = self.emit("RetrievalCompleted", [started[sq.id].event_id], utterance_id=u.id,
                           query_version=self.query_version, sub_query_id=sq.id, retrieval_trigger=trig,
                           document_ids=sorted({self.ctx.corpus.chunk(cid).doc_id for cid in candidate_ids}),
                           chunk_ids=candidate_ids,
                           latency_ms=round((end - start) * 1000, 3),
                           confidence=round(res.relevant_hits[0].rerank_score, 4) if res.relevant_hits else 0.0,
                           t=base_t + end_off, payload={
                "hits": hits_pub, "duplicates": res.duplicates, "filtered_by_facet": res.filtered,
                "stage_ms": {k: round(v, 3) for k, v in res.stage_ms.items()}, "mode": res.mode,
                "evidence_status": res.evidence_status, "accepted_chunk_ids": ids,
                "candidate_chunk_ids": candidate_ids, "selected_domain": res.domain,
                "domain_confidence": round(res.domain_confidence, 4),
                "started_offset_ms": round((start - base_perf) * 1000, 3),
                "ended_offset_ms": round(end_off * 1000, 3), "label": sq.label,
                "search_executed": reuse_kind == "NONE",
                "search_id": res.search_id if reuse_kind == "NONE" else None,
                "retrieval_type": res.retrieval_type if reuse_kind == "NONE" else "REUSE",
                "cache_status": reuse_kind if reuse_kind != "NONE" else "MISS_STORED",
                "prepared_evidence_id": prepared_id,
                "query_fingerprint": fingerprint,
                "cache": self.controller.evidence_cache.status(),
            })
            completed.append(ev)
        contradictions = find_contradictions(self.ctx.index, list(dict.fromkeys(all_ids)),
                                             self.cfg["grounding"]["conflict_overlap"])
        for con in contradictions:
            a, b = con["chunks"]
            self.graph.add_edge(chunk_node(a), chunk_node(b), "CONTRADICTS", unit=con["unit"])
        shared = sorted({cid for cid in all_ids if all_ids.count(cid) > 1})
        fused = self.emit("EvidenceFused", [e.event_id for e in completed], utterance_id=u.id,
                          query_version=self.query_version, chunk_ids=list(dict.fromkeys(all_candidate_ids)),
                          document_ids=sorted({self.ctx.corpus.chunk(x).doc_id for x in all_ids}), payload={
            "sub_queries": [sq.id for sq, _ in jobs], "evidence_count": len(set(all_ids)),
            "candidate_evidence_count": len(set(all_candidate_ids)),
            "accepted_evidence_count": len(set(all_ids)),
            "shared_across_sub_queries": shared, "contradictions": contradictions,
            "duplicates_collapsed": sum(len(item[0].duplicates) for item in results),
        })
        u.fused_events.append(fused.event_id)
        return completed

    # ------------------------------------------------------------------ constraint (delta) streaming
    async def _constraint_stream(self, u: UtteranceState, frame: Frame, parent: str | None, is_final: bool) -> None:
        constraint = extract_constraint(frame, self.ctx.corpus)
        has_anchor = constraint is not None and (constraint.facets or any(
            s["kind"] in ("place", "number", "proper") for s in constraint.slots))
        if constraint is None or (not has_anchor and not is_final):
            self._emit_decision(u, parent, "WAIT", "NONE", ["CONSTRAINT_INCOMPLETE"], {}, [], u.turn)
            return
        if u.delta_key == constraint.key:
            self._emit_decision(u, parent, "WAIT", "NONE", ["DELTA_ALREADY_APPLIED"], {}, [], u.turn)
            return
        base = self.latest_version()
        if base is None:
            return
        if self.cfg["delta"]["refinement"] == "full_restart":
            ev = self._emit_decision(u, parent, "RETRIEVE", "END_OF_UTTERANCE" if is_final else "EARLY",
                                     ["LATE_CONSTRAINT", "FULL_RESTART_ABLATION"], {}, [], u.turn)
            if is_final:
                u.delta_key = constraint.key
                u.delta_event = ev.event_id
            return
        sqs = [self.sub_queries[b.sub_query_id] for b in base.blocks if b.sub_query_id in self.sub_queries]
        cited: dict[str, list[str]] = {}
        for c in base.claims.values():
            cited.setdefault(c.sub_query_id, []).extend(e for e in c.evidence if e not in cited.get(c.sub_query_id, []))
        affected, unaffected = self.ctx.delta.affected(sqs, constraint, cited)
        ev = self._emit_decision(u, parent, "REFINE", "DELTA", ["LATE_CONSTRAINT"],
                                 {"affected": len(affected), "total": len(sqs)},
                                 [{"sub_query_id": a.sub_query.id, "reasons": a.reasons} for a in affected], u.turn)
        u.delta_key = constraint.key
        u.delta_affected = {a.sub_query.id: a for a in affected}
        u.delta_unaffected = unaffected
        u.delta_event = ev.event_id
        if not affected:
            return
        before = {a.sub_query.id: set(a.sub_query.evidence) for a in affected}
        for a in affected:
            self.ctx.delta.apply_to_subquery(a, constraint)
        await self._retrieve(u, [(a.sub_query, "DELTA") for a in affected], ev.event_id)
        u.delta_new_ids = {sid: [c for c in self.sub_queries[sid].evidence if c not in before[sid]] for sid in before}

    # ------------------------------------------------------------------ commit
    async def _commit(self, u: UtteranceState) -> None:
        if u.closed:
            return
        u.closed = True
        turn = u.turn or TurnResult("NEW_INFO", 0.0)
        t_commit = time.perf_counter()
        if turn.turn_type == "TRANSFORM":
            version = await self._commit_transform(u, turn)
        elif turn.turn_type == "CONSTRAINT" and self.versions:
            version = await self._commit_constraint(u)
        elif turn.turn_type == "ELABORATE" and self.versions:
            version = await self._commit_elaborate(u, turn)
        else:
            version = await self._commit_new_info(u)
        self.vocab |= content_set(u.text)
        if version is not None:
            for c in version.claims.values():
                self.vocab |= content_set(c.text)
            for b in version.blocks:
                if b.sub_query_id in self.sub_queries:
                    self.vocab |= content_set(self.sub_queries[b.sub_query_id].text)
        commit_s = time.perf_counter() - t_commit
        if version is not None and version.kind == "SEMANTIC":
            mean_ret = [e.latency_ms or 0 for e in self.active_events()
                        if e.event_type == "RetrievalCompleted" and e.utterance_id == u.id]
            self.controller.observe_pipeline_latency(commit_s + (sum(mean_ret) / len(mean_ret) / 1000 if mean_ret else 0))
        self._telemetry_rollup(u, version)

    def _claims_from_drafts(self, drafts: list[ClaimDraft], version_no: int) -> list[Claim]:
        out = []
        for d in drafts:
            checksums = {cid: f"sha256:{self.ctx.corpus.chunk(cid).sha256}"
                         for cid in d.evidence if self.ctx.corpus.chunk(cid)}
            out.append(Claim(self._next_claim_id(), self._next_lineage(), d.text, d.sub_query_id, list(d.evidence),
                             d.quote, round(d.score, 4), version_created=version_no,
                             evidence_checksums=checksums))
        return out

    def _verify(self, claims: list[Claim], parent: str | None, u: UtteranceState) -> tuple[list[Claim], Event]:
        t = time.perf_counter()
        results = []
        fabricated = 0
        for c in claims:
            peers = self.sub_queries[c.sub_query_id].evidence if c.sub_query_id in self.sub_queries else []
            v = self.ctx.firewall.verify(c, self.retrieved, peers)
            fabricated += len(v.invalid_ids)
            repaired = False
            if v.verdict in ("PARTIAL", "UNSUPPORTED") and c.evidence:
                if self.ctx.firewall.repair(c):
                    v = self.ctx.firewall.verify(c, self.retrieved, peers)
                    repaired = True
            c.verdict, c.support, c.anchors_ok, c.quote_ok = v.verdict, round(v.support, 4), v.anchors_ok, v.quote_ok
            c.conflicts_with = [x["chunk_id"] for x in v.conflicts]
            results.append({**v.to_public(), "repaired": repaired, "text": c.text, "evidence": c.evidence,
                            "sub_query_id": c.sub_query_id})
        companions = self._conflict_companions(claims, fabricated_guard=results)
        claims.extend(companions)
        passed = [c for c in claims if c.verdict in ("SUPPORTED", "CONFLICTING", "PARTIAL")]
        summary = {
            "checked": len(claims),
            "supported": sum(c.verdict == "SUPPORTED" for c in claims),
            "partial": sum(c.verdict == "PARTIAL" for c in claims),
            "conflicting": sum(c.verdict == "CONFLICTING" for c in claims),
            "unsupported": sum(c.verdict == "UNSUPPORTED" for c in claims),
            "invalid_citation": sum(c.verdict == "INVALID_CITATION" for c in claims),
            "fabricated_ids_blocked": fabricated,
            "citations_valid": sum(len(c.evidence) for c in passed),
            "emitted": len(passed),
        }
        if summary["invalid_citation"]:
            summary["status"] = "INVALID_GROUNDING"
        elif summary["checked"] == 0 or summary["emitted"] == 0:
            summary["status"] = "INSUFFICIENT_EVIDENCE"
        elif summary["conflicting"]:
            summary["status"] = "CONFLICTING"
        elif summary["partial"] or summary["unsupported"]:
            summary["status"] = "PARTIAL"
        else:
            summary["status"] = "SUPPORTED"
        ev = self.emit("GroundingVerified", [parent], utterance_id=u.id, query_version=self.query_version,
                       latency_ms=round((time.perf_counter() - t) * 1000, 3),
                       citations=sorted({e for c in passed for e in c.evidence}),
                       confidence=round(sum(c.support for c in passed) / len(passed), 4) if passed else 0.0,
                       payload={"results": results, "summary": summary, "checks": [
                           "CITATION_IN_SESSION_RETRIEVAL", "CORPUS_SHA_MATCH", "VERBATIM_QUOTE", "ANCHORS",
                           "SUPPORT_" + self.cfg["grounding"]["nli"].upper(), "NUMERIC_CONFLICT"]})
        if self.caper.enabled:
            self.caper.commit_grounded(
                {chunk_id for claim in passed for chunk_id in claim.evidence},
                ev.event_id,
                self.query_version,
            )
        return passed, ev

    def _conflict_notes(self, claims: list[Claim]) -> list[dict]:
        notes = []
        for c in claims:
            if not c.conflicts_with:
                continue
            mine = self.ctx.corpus.chunk(c.evidence[0]) if c.evidence else None
            other = self.ctx.corpus.chunk(c.conflicts_with[0])
            if not mine or not other:
                continue
            newer = mine if (mine.effective_date or "") >= (other.effective_date or "") else other
            notes.append({
                "claim_id": c.claim_id, "conflicts_with": c.conflicts_with,
                "documents": [mine.doc_id, other.doc_id],
                "effective_dates": [mine.effective_date, other.effective_date],
                "more_recent_document": newer.doc_id,
                "note": f"Sources disagree: {mine.doc_id} ({mine.effective_date}) vs {other.doc_id} "
                        f"({other.effective_date}); {newer.doc_id} is more recent.",
            })
        return notes

    def _order_claims(self, claims: list[Claim]) -> list[str]:
        """Keep synthesis order, but among conflicting claims put the most recent source first."""
        def date(c: Claim) -> str:
            ch = self.ctx.corpus.chunk(c.evidence[0]) if c.evidence else None
            return (ch.effective_date or "") if ch else ""
        normal = [c for c in claims if c.verdict != "CONFLICTING"]
        conflicting = sorted([c for c in claims if c.verdict == "CONFLICTING"], key=date, reverse=True)
        return [c.claim_id for c in normal + conflicting]

    def _conflict_companions(self, claims: list[Claim], fabricated_guard: list[dict]) -> list[Claim]:
        """Contradictory evidence is never silently resolved: for each CONFLICTING claim, the opposing
        sentence (from a chunk retrieved in this session) is added as its own cited, verified claim."""
        out: list[Claim] = []
        present = {c.quote for c in claims}
        for c in list(claims):
            if c.verdict != "CONFLICTING":
                continue
            peers = self.sub_queries[c.sub_query_id].evidence if c.sub_query_id in self.sub_queries else []
            v0 = self.ctx.firewall.verify(copy.deepcopy(c), self.retrieved, peers)
            for con in v0.conflicts:
                if con["sentence"] in present:
                    continue
                other = Claim(self._next_claim_id(), self._next_lineage(), con["sentence"], c.sub_query_id,
                              [con["chunk_id"]], con["sentence"], c.score, version_created=c.version_created,
                              evidence_checksums={
                                  con["chunk_id"]: f"sha256:{self.ctx.corpus.chunk(con['chunk_id']).sha256}"
                              })
                v = self.ctx.firewall.verify(other, self.retrieved, peers)
                other.verdict = "CONFLICTING" if v.verdict in ("SUPPORTED", "CONFLICTING") else v.verdict
                other.support, other.anchors_ok, other.quote_ok = round(v.support, 4), v.anchors_ok, v.quote_ok
                other.conflicts_with = list(c.evidence)
                present.add(con["sentence"])
                out.append(other)
                fabricated_guard.append({**v.to_public(), "verdict": other.verdict, "repaired": False,
                                         "text": other.text, "evidence": other.evidence,
                                         "sub_query_id": other.sub_query_id, "companion_of": c.claim_id})
        return out

    def _synthesize(self, jobs: list[tuple[SubQuery, dict]]) -> list[tuple[SubQuery, SynthesisResult, float]]:
        out = []
        for sq, kwargs in jobs:
            t = time.perf_counter()
            res = self.ctx.synth.synthesize(sq, sq.hits, **kwargs)
            out.append((sq, res, (time.perf_counter() - t) * 1000))
        return out

    def _emit_claims(self, u: UtteranceState, parent: list[str | None], results, claims_by_sq: dict[str, list[Claim]]) -> Event:
        tokens = {"llm_prompt": 0, "llm_completion": 0}
        ctx_tokens = 0
        for _, res, _ in results:
            for k in tokens:
                tokens[k] += int(res.tokens.get(k, 0))
            ctx_tokens += res.context_tokens_est
        for k in tokens:
            u.tokens[k] += tokens[k]
        u.tokens["context_est"] += ctx_tokens
        return self.emit("ClaimsGenerated", parent, utterance_id=u.id, query_version=self.query_version,
                         latency_ms=round(sum(ms for _, _, ms in results), 3),
                         token_cost={**tokens, "context_est": ctx_tokens}, payload={
            "backend": getattr(self.ctx.synth, "name", "extractive"),
            "per_sub_query": [{
                "sub_query_id": sq.id, "backend": res.backend, "gap": res.gap, "latency_ms": round(ms, 3),
                "claims": [{"claim_id": c.claim_id, "lineage_id": c.lineage_id, "text": c.text,
                            "evidence": c.evidence, "quote": c.quote, "score": c.score, "status": c.status,
                            "derived_from": c.derived_from}
                           for c in claims_by_sq.get(sq.id, [])],
            } for sq, res, ms in results],
        })

    def _compose(self, u: UtteranceState, trigger: str, kind: str, blocks: list[AnswerBlock],
                 claims: list[Claim], parents: list[str | None], diff: dict | None = None, **extra) -> AnswerVersion:
        base = self.latest_version()
        number = len(self.versions) + 1
        snapshot = {c.claim_id: copy.deepcopy(c) for c in claims}
        for c in claims:
            self.claims[c.claim_id] = c
        version = AnswerVersion(
            version=number, parent_version=base.version if base else None, trigger=trigger, kind=kind,
            utterance_id=u.id, query_version=self.query_version, blocks=blocks, claims=snapshot,
            uncertainty=[{"sub_query_id": b.sub_query_id, "text": b.uncertainty} for b in blocks if b.uncertainty],
            conflicts=self._conflict_notes(claims),
            diff=diff or {}, created_t=self.now(),
            retrievals_used=sorted({e for c in claims for e in c.evidence}), **extra,
        )
        self.versions.append(version)
        vn = version_node(number)
        self.graph.add_node(vn, "answer_version", answer_kind=kind, trigger=trigger)
        if base:
            self.graph.add_edge(vn, version_node(base.version), "DERIVED_FROM")
        for c in claims:
            self.graph.add_node(claim_node(c.claim_id), "claim", verdict=c.verdict, lineage=c.lineage_id)
            self.graph.add_edge(vn, claim_node(c.claim_id), "CONTAINS")
            self.graph.add_edge(sq_node(c.sub_query_id), claim_node(c.claim_id), "PRODUCES")
            for e in c.evidence:
                self.graph.add_edge(chunk_node(e), claim_node(c.claim_id), "SUPPORTS", support=c.support)
            for src in c.derived_from:
                self.graph.add_edge(claim_node(c.claim_id), claim_node(src), "DERIVED_FROM")
        self.emit("AnswerVersionCreated", parents, utterance_id=u.id, query_version=self.query_version,
                  answer_version=number, citations=version.retrievals_used,
                  document_ids=sorted({self.ctx.corpus.chunk(e).doc_id for e in version.retrievals_used
                                       if self.ctx.corpus.chunk(e)}),
                  token_cost=dict(version.tokens), payload=version.to_public())
        return version

    async def _commit_new_info(self, u: UtteranceState) -> AnswerVersion | None:
        sqs = [self.sub_queries[i] for i in u.sq_ids] + [self.sub_queries[i] for i in u.reused_sq_ids
                                                          if i not in u.sq_ids]
        parent_events = u.fused_events[-3:] or [u.decision_events[-1] if u.decision_events else u.last_chunk_event]
        if not sqs:
            return None
        n_version = len(self.versions) + 1
        results = await asyncio.to_thread(self._synthesize, [(sq, {}) for sq in sqs])
        claims_by_sq = {sq.id: self._claims_from_drafts(res.drafts, n_version) for sq, res, _ in results}
        ev_claims = self._emit_claims(u, parent_events, results, claims_by_sq)
        all_claims = [c for cl in claims_by_sq.values() for c in cl]
        passed, ev_ground = self._verify(all_claims, ev_claims.event_id, u)
        passed_ids = {c.claim_id for c in passed}
        blocks = []
        for sq, res, _ in results:
            ids = self._order_claims([c for c in passed if c.sub_query_id == sq.id])
            gap = res.gap if not ids else None
            if not ids and not gap:
                gap = self.ctx.extractive.gap_text(sq)
            blocks.append(AnswerBlock(sq.id, sq.label, "paragraph", ids, gap))
        return self._compose(u, "NEW_INFO", "SEMANTIC", blocks, passed, [ev_ground.event_id],
                             diff={"kept": [], "revised": [], "added": [c.claim_id for c in passed], "retracted": []},
                             tokens=dict(u.tokens))

    async def _commit_constraint(self, u: UtteranceState) -> AnswerVersion | None:
        frame = u.frame
        if u.delta_key is None:
            await self._constraint_stream(u, frame, u.last_frame_event, True)
        if self.cfg["delta"]["refinement"] == "full_restart":
            return await self._commit_full_restart(u)
        base = self.latest_version()
        constraint = extract_constraint(frame, self.ctx.corpus) or Constraint(u.text)
        affected = u.delta_affected
        base_claims = [base.claims[c] for c in base.claim_ids]
        plan = self.ctx.delta.plan(base_claims, affected, constraint, u.delta_new_ids)
        all_sq_ids = [b.sub_query_id for b in base.blocks if b.sub_query_id]
        counterfactual = self._counterfactual(all_sq_ids, list(affected))
        ev_plan = self.emit("DeltaPlanComputed", [u.delta_event, *u.fused_events[-3:]], utterance_id=u.id,
                            query_version=self.query_version, answer_version=base.version,
                            retrieval_trigger="DELTA", payload={
            "constraint": constraint.to_public(),
            "affected_sub_queries": [{"sub_query_id": sid, "reasons": a.reasons} for sid, a in affected.items()],
            "unaffected_sub_queries": u.delta_unaffected,
            "plan": plan, "base_version": base.version, "counterfactual": counterfactual,
            "claims_invalidated": sum(1 for p in plan.values() if p["action"] != "KEEP"),
            "claims_total": len(plan),
        })
        n_version = len(self.versions) + 1
        jobs = []
        for sid, a in affected.items():
            revise = [c for c in base_claims if c.sub_query_id == sid and plan[c.claim_id]["action"] == "REVISE"]
            keep_quotes = {c.quote for c in base_claims if c.sub_query_id == sid and plan[c.claim_id]["action"] == "KEEP"}
            n = len(revise) + self.cfg["delta"]["max_added_per_subquery"]
            jobs.append((a.sub_query, {"n": n, "avoid_facets": constraint.facets,
                                       "prefer_terms": set(constraint.terms), "exclude_quotes": keep_quotes}))
        results = await asyncio.to_thread(self._synthesize, jobs)
        new_claims: dict[str, list[Claim]] = {}
        diff = {"kept": [], "revised": [], "added": [], "retracted": []}
        replacement_of: dict[str, Claim] = {}
        for sq, res, _ in results:
            revise = [c for c in base_claims if c.sub_query_id == sq.id and plan[c.claim_id]["action"] == "REVISE"]
            drafts = res.drafts
            lineage = self.ctx.delta.match_lineage(revise, [d.text for d in drafts])
            made: list[Claim] = []
            added = 0
            for d, idx in zip(drafts, lineage):
                c = self._claims_from_drafts([d], n_version)[0]
                if idx is not None:
                    old = revise[idx]
                    c.lineage_id, c.status, c.derived_from = old.lineage_id, "REVISED", [old.claim_id]
                    replacement_of[old.claim_id] = c
                else:
                    chunk = self.ctx.corpus.chunk(d.evidence[0]) if d.evidence else None
                    specific = chunk is not None and (
                        any(chunk.facets.get(g) == v for g, v in constraint.facets.items())
                        or bool(set(constraint.terms) & content_set(d.quote)))
                    if not specific or added >= self.cfg["delta"]["max_added_per_subquery"]:
                        continue
                    c.status = "ADDED"
                    added += 1
                c.conditioned_on = [constraint.query_text]
                made.append(c)
            new_claims[sq.id] = made
        ev_claims = self._emit_claims(u, [ev_plan.event_id], results, new_claims)
        fresh = [c for cl in new_claims.values() for c in cl]
        passed, ev_ground = self._verify(fresh, ev_claims.event_id, u)
        passed_ids = {c.claim_id for c in passed}

        kept_texts = [base.claims[cid].text for cid in base.claim_ids if plan.get(cid, {}).get("action") == "KEEP"]

        def duplicates_existing(c: Claim, texts: list[str]) -> bool:
            terms = content_terms(c.text)
            return any(jaccard(terms, content_terms(t)) >= 0.8 for t in texts)

        blocks: list[AnswerBlock] = []
        final_claims: list[Claim] = []
        for b in base.blocks:
            ids: list[str] = []
            for cid in b.claim_ids:
                old = base.claims[cid]
                action = plan.get(cid, {"action": "KEEP"})["action"]
                if action == "KEEP":
                    kept = copy.deepcopy(old)
                    kept.status = "KEEP"
                    ids.append(cid)
                    final_claims.append(kept)
                    diff["kept"].append(cid)
                else:
                    rep = replacement_of.get(cid)
                    if rep is not None and duplicates_existing(rep, kept_texts):
                        diff["retracted"].append({"claim_id": cid, "reasons": [*plan[cid]["reasons"],
                                                                               "SUPERSEDED_BY_EXISTING_CLAIM"]})
                    elif rep is not None and rep.claim_id in passed_ids:
                        ids.append(rep.claim_id)
                        final_claims.append(rep)
                        kept_texts.append(rep.text)
                        diff["revised"].append({"from": cid, "to": rep.claim_id, "reasons": plan[cid]["reasons"]})
                    else:
                        diff["retracted"].append({"claim_id": cid, "reasons": plan[cid]["reasons"]})
                        self.graph.add_edge(claim_node(cid), f"CON:{u.id}", "RETRACTED_BY")
            for c in new_claims.get(b.sub_query_id or "", []):
                if c.status == "ADDED" and c.claim_id in passed_ids and not duplicates_existing(c, kept_texts):
                    kept_texts.append(c.text)
                    ids.append(c.claim_id)
                    final_claims.append(c)
                    diff["added"].append(c.claim_id)
            uncertainty = b.uncertainty
            sq = self.sub_queries.get(b.sub_query_id or "")
            if not ids and sq is not None:
                uncertainty = self.ctx.extractive.gap_text(sq)
            blocks.append(AnswerBlock(b.sub_query_id, sq.label if sq else b.title, "paragraph", ids, uncertainty))
        self.graph.add_node(f"CON:{u.id}", "constraint", text=constraint.text)
        for sid in affected:
            self.graph.add_edge(sq_node(sid), f"CON:{u.id}", "CONDITIONED_ON")
        return self._compose(u, "CONSTRAINT", "SEMANTIC", blocks, final_claims, [ev_ground.event_id, ev_plan.event_id],
                             diff=diff, constraint=constraint.to_public(), counterfactual=counterfactual,
                             tokens=dict(u.tokens))

    async def _commit_full_restart(self, u: UtteranceState) -> AnswerVersion | None:
        """Ablation C baseline: re-run every sub-query with the constraint, regenerate every claim."""
        base = self.latest_version()
        constraint = extract_constraint(u.frame, self.ctx.corpus) or Constraint(u.text)
        sqs = [self.sub_queries[b.sub_query_id] for b in base.blocks if b.sub_query_id in self.sub_queries]
        for sq in sqs:
            self.ctx.delta.apply_to_subquery(AffectedSubQuery(sq, ["FULL_RESTART"]), constraint)
        await self._retrieve(u, [(sq, "REFINE") for sq in sqs], u.delta_event or u.last_frame_event)
        n_version = len(self.versions) + 1
        results = await asyncio.to_thread(self._synthesize, [(sq, {"avoid_facets": constraint.facets})
                                                             for sq in sqs])
        claims_by_sq = {sq.id: self._claims_from_drafts(res.drafts, n_version) for sq, res, _ in results}
        ev_claims = self._emit_claims(u, u.fused_events[-1:], results, claims_by_sq)
        all_claims = [c for cl in claims_by_sq.values() for c in cl]
        passed, ev_ground = self._verify(all_claims, ev_claims.event_id, u)
        passed_ids = {c.claim_id for c in passed}
        blocks = []
        for sq, res, _ in results:
            ids = [c.claim_id for c in claims_by_sq[sq.id] if c.claim_id in passed_ids]
            blocks.append(AnswerBlock(sq.id, sq.label, "paragraph", ids,
                                      None if ids else (res.gap or self.ctx.extractive.gap_text(sq))))
        return self._compose(u, "CONSTRAINT", "SEMANTIC", blocks, passed, [ev_ground.event_id],
                             diff={"kept": [], "revised": [], "added": [c.claim_id for c in passed],
                                   "retracted": list(base.claim_ids)},
                             constraint=constraint.to_public(), tokens=dict(u.tokens),
                             counterfactual=self._counterfactual([sq.id for sq in sqs], [sq.id for sq in sqs]))

    async def _commit_transform(self, u: UtteranceState, turn: TurnResult) -> AnswerVersion | None:
        base = self.latest_version()
        if base is None:
            return None
        if u.suppressed_event is None:
            self._decision_suppress(u, u.last_frame_event, "PRESENTATION_TRANSFORMATION", turn)
        t = time.perf_counter()
        try:
            outcome = await asyncio.to_thread(self.ctx.transforms.apply, base, turn.transform, self._next_claim_id)
        except Exception as exc:  # generative transform failure must not break the session
            self.emit("TransformUnavailable", [u.suppressed_event], utterance_id=u.id,
                      answer_version=base.version, payload={"reason": f"transform failed: {exc}",
                                                            "ops": turn.transform.get("ops")})
            return None
        if not outcome.ok:
            self.emit("TransformUnavailable", [u.suppressed_event], utterance_id=u.id,
                      answer_version=base.version, payload={"reason": outcome.reason, "ops": outcome.ops})
            return None
        check = verify_transform(base, outcome)
        for k in ("llm_prompt", "llm_completion"):
            u.tokens[k] += int(outcome.tokens.get(k, 0))
        ev_check = self.emit("TransformVerified", [u.suppressed_event], utterance_id=u.id,
                             answer_version=base.version, latency_ms=round((time.perf_counter() - t) * 1000, 3),
                             token_cost=dict(outcome.tokens), citations=sorted({e for c in outcome.claims.values()
                                                                                for e in c.evidence}),
                             payload={**check, "ops": outcome.ops, "notes": outcome.notes,
                                      "claims": list(outcome.claims)})
        claims = [c for c in outcome.claims.values()]
        if not check["passed"]:
            bad = {v["claim_id"] for v in check["violations"]}
            claims = [c for c in claims if c.claim_id not in bad]
            for b in outcome.blocks:
                b.claim_ids = [c for c in b.claim_ids if c not in bad]
        diff = {"kept": [c.claim_id for c in claims if c.status == "KEEP"],
                "revised": [{"from": c.derived_from[0], "to": c.claim_id, "reasons": ["PRESENTATION"]}
                            for c in claims if c.status == "TRANSFORMED"],
                "added": [], "retracted": [cid for cid in base.claim_ids
                                           if cid not in outcome.claims and not any(cid in c.derived_from for c in claims)]}
        return self._compose(u, "TRANSFORM", "PRESENTATION", outcome.blocks, claims, [ev_check.event_id],
                             diff=diff, transform={"ops": outcome.ops, **{k: v for k, v in turn.transform.items()
                                                                          if k != "ops"}},
                             notes=outcome.notes, tokens=dict(u.tokens))

    async def _commit_elaborate(self, u: UtteranceState, turn: TurnResult) -> AnswerVersion | None:
        base = self.latest_version()
        terms = set(content_terms(u.text))
        sqs = [self.sub_queries[b.sub_query_id] for b in base.blocks if b.sub_query_id in self.sub_queries]
        targets = [sq for sq in sqs if terms & set(content_terms(sq.text + " " + sq.label))] or sqs
        existing_quotes = {c.quote for c in base.claims.values()}
        trial = await asyncio.to_thread(self._synthesize, [(sq, {"n": 2, "exclude_quotes": existing_quotes})
                                                           for sq in targets])
        sufficient = any(res.drafts for _, res, _ in trial)
        if sufficient:
            self._decision_suppress(u, u.last_frame_event, "SUFFICIENT_SESSION_EVIDENCE", turn)
            results = trial
            parents = [u.suppressed_event]
        else:
            ev = self._emit_decision(u, u.last_frame_event, "RETRIEVE", "END_OF_UTTERANCE",
                                     ["SESSION_EVIDENCE_INSUFFICIENT"], {}, [], turn)
            for sq in targets:
                sq.query.terms.update({t: 1.0 for t in terms})
            await self._retrieve(u, [(sq, "REFINE") for sq in targets], ev.event_id)
            results = await asyncio.to_thread(self._synthesize, [(sq, {"n": 2, "exclude_quotes": existing_quotes})
                                                                 for sq in targets])
            parents = u.fused_events[-1:]
        n_version = len(self.versions) + 1
        claims_by_sq = {sq.id: self._claims_from_drafts(res.drafts, n_version) for sq, res, _ in results}
        for cl in claims_by_sq.values():
            for c in cl:
                c.status = "ADDED"
        ev_claims = self._emit_claims(u, parents, results, claims_by_sq)
        passed, ev_ground = self._verify([c for cl in claims_by_sq.values() for c in cl], ev_claims.event_id, u)
        passed_ids = {c.claim_id for c in passed}
        blocks, final_claims = [], []
        for b in base.blocks:
            ids = list(b.claim_ids)
            for cid in ids:
                kept = copy.deepcopy(base.claims[cid])
                kept.status = "KEEP"
                final_claims.append(kept)
            for c in claims_by_sq.get(b.sub_query_id or "", []):
                if c.claim_id in passed_ids:
                    ids.append(c.claim_id)
                    final_claims.append(c)
            blocks.append(AnswerBlock(b.sub_query_id, b.title, "paragraph", ids, b.uncertainty))
        return self._compose(u, "ELABORATE", "SEMANTIC", blocks, final_claims, [ev_ground.event_id],
                             diff={"kept": list(base.claim_ids), "revised": [], "added": sorted(passed_ids),
                                   "retracted": []}, tokens=dict(u.tokens))

    # ------------------------------------------------------------------ accounting
    def _counterfactual(self, all_sq_ids: list[str], affected_ids: list[str]) -> dict:
        idx = self.ctx.index
        full_ctx = sum(context_tokens(idx, self.sub_queries[s].hits) for s in all_sq_ids if s in self.sub_queries)
        delta_ctx = sum(context_tokens(idx, self.sub_queries[s].hits) for s in affected_ids if s in self.sub_queries)
        out = {
            "full_restart_retrievals": len(all_sq_ids),
            "delta_retrievals": len(affected_ids),
            "full_restart_context_tokens_est": full_ctx,
            "delta_context_tokens_est": delta_ctx,
            "retrieval_savings_ratio": round(1 - len(affected_ids) / len(all_sq_ids), 4) if all_sq_ids else 0.0,
        }
        if self.ctx.llm is not None:
            out["full_restart_llm_prompt_tokens_est"] = sum(
                self.ctx.llm.prompt_tokens_estimate(self.sub_queries[s], self.sub_queries[s].hits)
                for s in all_sq_ids if s in self.sub_queries)
        return out

    def _telemetry_rollup(self, u: UtteranceState, version: AnswerVersion | None) -> None:
        events = self.active_events()
        ret_events = [e for e in events if e.utterance_id == u.id and e.event_type == "RetrievalCompleted"]
        latencies = [e.latency_ms or 0.0 for e in ret_events]
        early = [e for e in events if e.utterance_id == u.id and e.event_type == "RetrievalStarted"
                 and e.timestamp < u.last_chunk_t]
        answer_t = version.created_t if version else None
        self.emit("TelemetryEmitted", [self.events[-1].event_id], utterance_id=u.id,
                  query_version=self.query_version, answer_version=version.version if version else None,
                  token_cost=dict(u.tokens), payload={
            "turn_type": u.turn.turn_type if u.turn else None,
            "retrievals": len(ret_events),
            "early_retrievals": len(early),
            "first_retrieval_t": round(u.first_retrieval_t, 4) if u.first_retrieval_t is not None else None,
            "utterance_end_t": round(u.last_chunk_t, 4),
            "retrieval_lead_s": round(u.last_chunk_t - u.first_retrieval_t, 4) if u.first_retrieval_t is not None else None,
            "retrieval_latency_ms_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "post_utterance_latency_ms": round((answer_t - u.last_chunk_t) * 1000, 3) if answer_t else None,
            "sub_queries": list(u.sq_ids), "suppressed": u.suppressed_event is not None,
            "answer_version": version.version if version else None,
            "claims": len(version.claim_ids) if version else 0,
            "pipeline_latency_ema_s": round(self.controller.pipeline_latency_s, 4),
            "actual_search_cost": self.ctx.index.search_cost_summary(session_id=self.id),
            "caper": self.caper.snapshot()["stats"],
        })
