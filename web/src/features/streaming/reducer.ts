import type { WireEvent } from "@/schemas/events";
import type {
  AnswerBlock,
  AnswerDiff,
  AnswerVersion,
  CandidateDecision,
  Claim,
  ClaimStatus,
  ControllerState,
  CorpusStatus,
  Counterfactual,
  DeltaPlan,
  EvidenceChunk,
  EvidenceRelationship,
  GroundingResult,
  Intent,
  QueryVersion,
  RetrievalDecision,
  RetrievalTrigger,
  Session,
  SessionConfig,
  Signals,
  Slot,
  Stage,
  SuppressionInfo,
  TelemetryEvent,
  TelemetryTotals,
  TranscriptChunk,
  TurnType,
  Utterance,
  Verdict,
} from "@/types/domain";

/**
 * The whole UI state is a pure fold over backend events: view_n = reduce(view_{n-1}, event_n).
 * That gives deterministic replay, stepping backwards (refold) and "jump to V2" for free, and
 * guarantees the UI never shows anything that no event reported.
 */
export interface EngineView {
  session: Session;
  utterances: Record<string, Utterance>;
  utteranceOrder: string[];
  chunks: Record<string, TranscriptChunk>;
  chunkOrder: string[];
  queryVersions: QueryVersion[];
  controller: RetrievalDecision | null;
  decisions: RetrievalDecision[];
  intents: Record<string, Intent>;
  intentOrder: string[];
  evidence: Record<string, EvidenceChunk>;
  relationships: EvidenceRelationship[];
  contradictions: { chunks: [string, string]; documents: [string, string]; unit: string; moreRecent: string }[];
  claims: Record<string, Claim>;
  versions: AnswerVersion[];
  grounding: GroundingResult[];
  delta: DeltaPlan | null;
  suppression: SuppressionInfo | null;
  transformUnavailable: { reason: string; timestamp: number } | null;
  telemetry: TelemetryTotals;
  events: TelemetryEvent[];
  activeRetrievals: Record<string, { trigger: RetrievalTrigger; startedAt: number; eventId: string }>;
  stage: Stage;
  lastEventAt: number;
  malformed: number;
}

export function initialView(): EngineView {
  return {
    session: { id: null, status: "IDLE" },
    utterances: {},
    utteranceOrder: [],
    chunks: {},
    chunkOrder: [],
    queryVersions: [],
    controller: null,
    decisions: [],
    intents: {},
    intentOrder: [],
    evidence: {},
    relationships: [],
    contradictions: [],
    claims: {},
    versions: [],
    grounding: [],
    delta: null,
    suppression: null,
    transformUnavailable: null,
    telemetry: {
      retrievalLatencies: [],
      postUtteranceLatencies: [],
      retrievalCount: 0,
      earlyRetrievalCount: 0,
      refineCount: 0,
      deltaCount: 0,
      probeCount: 0,
      suppressedCount: 0,
      llmTokens: 0,
      contextTokensEst: 0,
      documentsTouched: 0,
      lastLeadSeconds: null,
    },
    events: [],
    activeRetrievals: {},
    stage: "idle",
    lastEventAt: 0,
    malformed: 0,
  };
}

type P = Record<string, unknown>;
const num = (v: unknown): number | undefined => (typeof v === "number" ? v : undefined);
const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
const arr = <T = unknown>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);
const rec = (v: unknown): P => (v && typeof v === "object" && !Array.isArray(v) ? (v as P) : {});

function toTelemetryEvent(ev: WireEvent): TelemetryEvent {
  return {
    id: ev.event_id,
    eventType: ev.event_type,
    timestamp: ev.timestamp,
    sessionId: ev.session_id,
    parentIds: ev.parent_event_ids,
    utteranceId: ev.utterance_id ?? null,
    queryVersion: ev.query_version ?? null,
    answerVersion: ev.answer_version ?? null,
    retrievalTrigger: ev.retrieval_trigger ?? null,
    subQueryId: ev.sub_query_id ?? null,
    documentIds: ev.document_ids,
    chunkIds: ev.chunk_ids,
    latency: ev.latency_ms ?? null,
    tokenCost: ev.token_cost,
    confidence: ev.confidence ?? null,
    citations: ev.citations,
    payload: ev.payload,
    raw: ev,
  };
}

function toClaim(c: P): Claim {
  return {
    id: String(c.claim_id),
    lineageId: String(c.lineage_id ?? c.claim_id),
    text: String(c.text ?? ""),
    quote: String(c.quote ?? ""),
    subQueryId: String(c.sub_query_id ?? ""),
    evidenceIds: arr<string>(c.evidence),
    supportStatus: (str(c.verdict) ?? "PENDING") as Verdict,
    support: num(c.support) ?? 0,
    status: (str(c.status) ?? "NEW") as ClaimStatus,
    derivedFrom: arr<string>(c.derived_from),
    conflictsWith: arr<string>(c.conflicts_with),
    versionCreated: num(c.version_created) ?? 0,
  };
}

function signalsOf(p: unknown): Signals {
  const s = rec(p);
  const out: Signals = {};
  for (const k of ["rho", "chi", "kappa", "sigma", "cont", "pi", "nu", "s_hat", "theta", "remaining_s",
    "hidden_latency_s", "rho_vs_retrieved"] as const) {
    const v = num(s[k]);
    if (v !== undefined) out[k] = v;
  }
  return out;
}

function stageFor(type: string, prev: Stage, decision?: ControllerState): Stage {
  switch (type) {
    case "TranscriptChunkReceived":
      return prev === "idle" || prev === "answer" || prev === "suppress" ? "listening" : prev;
    case "RetrievalDecision":
      if (decision === "SUPPRESS") return "suppress";
      if (decision === "REFINE") return "refine";
      if (decision === "RETRIEVE") return "retrieving";
      return prev;
    case "RetrievalSuppressed":
      return "suppress";
    case "DeltaPlanComputed":
      return "refine";
    case "AnswerVersionCreated":
      return "answer";
    default:
      return prev;
  }
}

export function applyEvent(view: EngineView, ev: WireEvent): EngineView {
  const p = ev.payload as P;
  const te = toTelemetryEvent(ev);
  let v: EngineView = { ...view, events: [...view.events, te], lastEventAt: ev.timestamp };

  switch (ev.event_type) {
    case "SessionStarted": {
      v.session = {
        id: ev.session_id,
        status: "ACTIVE",
        startedAt: ev.wall_time,
        corpus: rec(p.corpus) as unknown as CorpusStatus,
        config: rec(p.config) as unknown as SessionConfig,
        stabilityModel: str(p.stability_model),
      };
      break;
    }
    case "SessionEnded": {
      v.session = { ...v.session, status: "ENDED" };
      break;
    }
    case "TranscriptChunkReceived": {
      const uid = ev.utterance_id ?? "u?";
      const id = str(p.chunk_id) ?? ev.event_id;
      const chunk: TranscriptChunk = {
        id,
        eventId: ev.event_id,
        utteranceId: uid,
        sequence: num(p.seq) ?? 0,
        timestamp: num(p.t1) ?? ev.timestamp,
        t0: num(p.t0) ?? ev.timestamp,
        text: str(p.text) ?? "",
        final: Boolean(p.final),
      };
      v.chunks = { ...v.chunks, [id]: chunk };
      v.chunkOrder = [...v.chunkOrder, id];
      const existing = v.utterances[uid];
      const utt: Utterance = existing
        ? { ...existing, chunkIds: [...existing.chunkIds, id], text: str(p.utterance_text) ?? existing.text,
            endT: chunk.timestamp, closed: chunk.final }
        : { id: uid, chunkIds: [id], text: str(p.utterance_text) ?? chunk.text, closed: chunk.final,
            startT: chunk.t0, endT: chunk.timestamp };
      v.utterances = { ...v.utterances, [uid]: utt };
      if (!existing) v.utteranceOrder = [...v.utteranceOrder, uid];
      break;
    }
    case "QueryFrameUpdated": {
      const q: QueryVersion = {
        version: ev.query_version ?? v.queryVersions.length + 1,
        utteranceId: ev.utterance_id ?? "",
        normalized: str(p.normalized) ?? "",
        slots: arr<Slot>(p.slots),
        candidates: arr<P>(p.candidates).map((c) => ({ key: String(c.key), text: String(c.text), closed: Boolean(c.closed) })),
        repairs: arr(p.repairs),
        timestamp: ev.timestamp,
      };
      v.queryVersions = [...v.queryVersions, q];
      break;
    }
    case "TurnClassified": {
      const uid = ev.utterance_id;
      if (uid && v.utterances[uid]) {
        v.utterances = {
          ...v.utterances,
          [uid]: { ...v.utterances[uid], turnType: str(p.turn_type) as TurnType, turnConfidence: num(p.confidence) },
        };
      }
      break;
    }
    case "ProbeExecuted": {
      v.telemetry = { ...v.telemetry, probeCount: v.telemetry.probeCount + arr(p.probes).length };
      break;
    }
    case "RetrievalDecision": {
      const state = (str(p.decision) ?? "WAIT") as ControllerState;
      const signals = signalsOf(p.signals);
      const decision: RetrievalDecision = {
        eventId: ev.event_id,
        state,
        confidence: ev.confidence ?? signals.s_hat,
        reasons: arr<string>(p.reasons),
        signals,
        queryVersion: ev.query_version ?? undefined,
        utteranceId: ev.utterance_id ?? undefined,
        timestamp: ev.timestamp,
        trigger: (ev.retrieval_trigger ?? "NONE") as RetrievalTrigger,
        turnType: str(p.turn_type) as TurnType,
        candidates: arr<P>(p.candidates).map((c) => ({
          key: String(c.key ?? c.sub_query_id ?? ""),
          text: String(c.text ?? ""),
          action: String(c.action ?? ""),
          trigger: (str(c.trigger) ?? "NONE") as RetrievalTrigger,
          reasons: arr<string>(c.reasons),
          signals: signalsOf(c.signals),
          sub_query_id: str(c.sub_query_id) ?? null,
        })) as CandidateDecision[],
      };
      v.controller = decision;
      v.decisions = [...v.decisions, decision];
      // attach the signals to the latest chunk of this utterance
      const utt = ev.utterance_id ? v.utterances[ev.utterance_id] : undefined;
      const lastChunkId = utt?.chunkIds[utt.chunkIds.length - 1];
      if (lastChunkId && v.chunks[lastChunkId]) {
        const c = v.chunks[lastChunkId];
        v.chunks = {
          ...v.chunks,
          [lastChunkId]: {
            ...c,
            semanticCompleteness: signals.kappa ?? c.semanticCompleteness,
            intentConfidence: signals.chi ?? c.intentConfidence,
            entityStability: signals.sigma ?? c.entityStability,
            novelty: signals.nu ?? c.novelty,
            informationGain: signals.rho ?? c.informationGain,
            stability: signals.s_hat ?? c.stability,
            threshold: signals.theta ?? c.threshold,
            decision: state,
            retrievalFired: c.retrievalFired || state === "RETRIEVE" || state === "REFINE",
          },
        };
      }
      break;
    }
    case "SubQueriesGenerated": {
      const intents = { ...v.intents };
      const order = [...v.intentOrder];
      for (const sq of arr<P>(p.sub_queries)) {
        const id = String(sq.id);
        const prev = intents[id];
        intents[id] = {
          id,
          label: String(sq.label ?? id),
          normalizedQuery: String(sq.text ?? ""),
          utteranceId: String(sq.utterance_id ?? ev.utterance_id ?? ""),
          dependencies: arr<string>(sq.depends_on),
          dependencyType: str(sq.dependency_type) ?? null,
          retrievalStatus: prev?.retrievalStatus ?? "PENDING",
          retrievalCount: prev?.retrievalCount ?? 0,
          documentIds: prev?.documentIds ?? [],
          chunkIds: prev?.chunkIds ?? [],
          lastLatencyMs: prev?.lastLatencyMs,
          lastTrigger: prev?.lastTrigger,
          footprint: arr<string>(sq.footprint),
          createdAt: prev?.createdAt ?? ev.timestamp,
          confidence: prev?.confidence,
          facetConstraints: rec(sq.facet_constraints) as Record<string, string>,
        };
        if (!prev) order.push(id);
      }
      v.intents = intents;
      v.intentOrder = order;
      break;
    }
    case "RetrievalStarted": {
      const id = ev.sub_query_id ?? "";
      const trigger = (ev.retrieval_trigger ?? "NONE") as RetrievalTrigger;
      v.activeRetrievals = { ...v.activeRetrievals, [id]: { trigger, startedAt: ev.timestamp, eventId: ev.event_id } };
      if (v.intents[id]) {
        v.intents = {
          ...v.intents,
          [id]: {
            ...v.intents[id],
            retrievalStatus: trigger === "REFINE" || trigger === "DELTA" ? "REFINING" : "RETRIEVING",
            normalizedQuery: str(p.query) ?? v.intents[id].normalizedQuery,
            label: str(p.label) ?? v.intents[id].label,
            facetConstraints: rec(p.facet_constraints) as Record<string, string>,
          },
        };
      }
      const t = v.telemetry;
      const utt = ev.utterance_id ? v.utterances[ev.utterance_id] : undefined;
      const early = trigger === "EARLY" || (trigger === "REFINE" && utt !== undefined && !utt.closed);
      v.telemetry = {
        ...t,
        retrievalCount: t.retrievalCount + 1,
        earlyRetrievalCount: t.earlyRetrievalCount + (trigger === "EARLY" ? 1 : 0),
        refineCount: t.refineCount + (trigger === "REFINE" ? 1 : 0),
        deltaCount: t.deltaCount + (trigger === "DELTA" ? 1 : 0),
      };
      void early;
      break;
    }
    case "RetrievalCompleted": {
      const id = ev.sub_query_id ?? "";
      const trigger = (ev.retrieval_trigger ?? "NONE") as RetrievalTrigger;
      const { [id]: _done, ...rest } = v.activeRetrievals;
      void _done;
      v.activeRetrievals = rest;
      const evidence = { ...v.evidence };
      const rels = [...v.relationships];
      const currentVersion = v.versions.length ? v.versions[v.versions.length - 1].id : undefined;
      for (const h of arr<P>(p.hits)) {
        const cid = String(h.chunk_id);
        const prev = evidence[cid];
        evidence[cid] = {
          chunkId: cid,
          documentId: String(h.doc_id ?? cid.split(":")[0]),
          docTitle: String(h.doc_title ?? ""),
          section: String(h.section ?? ""),
          sectionTitle: String(h.section_title ?? ""),
          sectionPath: String(h.section_path ?? ""),
          text: str(h.text) ?? prev?.text,
          denseScore: num(h.dense_score),
          sparseScore: num(h.sparse_score),
          fusedScore: num(h.fused_score),
          rerankScore: num(h.rerank_score),
          facets: rec(h.facets) as Record<string, string>,
          effectiveDate: str(h.effective_date) ?? null,
          docType: str(h.doc_type) ?? null,
          intents: Array.from(new Set([...(prev?.intents ?? []), id])),
          firstSeen: prev?.firstSeen ?? ev.timestamp,
        };
        rels.push({
          id: `${ev.event_id}:${cid}`,
          type: "RETRIEVED",
          source: `SQ:${id}`,
          target: `CH:${cid}`,
          trigger,
          weight: num(h.rerank_score),
          rank: num(h.rank),
          answerVersion: currentVersion,
        });
      }
      for (const pair of arr<unknown[]>(p.duplicates)) {
        rels.push({ id: `${ev.event_id}:dup:${pair[0]}`, type: "DUPLICATES", source: `CH:${pair[0]}`, target: `CH:${pair[1]}` });
      }
      v.evidence = evidence;
      v.relationships = rels;
      if (v.intents[id]) {
        const docs = Array.from(new Set(ev.document_ids));
        v.intents = {
          ...v.intents,
          [id]: {
            ...v.intents[id],
            retrievalStatus: "RETRIEVED",
            retrievalCount: v.intents[id].retrievalCount + 1,
            documentIds: docs,
            chunkIds: ev.chunk_ids,
            lastLatencyMs: ev.latency_ms ?? undefined,
            lastTrigger: trigger,
            confidence: ev.confidence ?? undefined,
          },
        };
      }
      const docsTouched = new Set(Object.values(evidence).map((e) => e.documentId)).size;
      v.telemetry = {
        ...v.telemetry,
        retrievalLatencies: [...v.telemetry.retrievalLatencies,
          { t: ev.timestamp, ms: ev.latency_ms ?? 0, label: `${id} ${trigger}` }],
        documentsTouched: docsTouched,
      };
      break;
    }
    case "EvidenceFused": {
      const cons = arr<P>(p.contradictions).map((c) => ({
        chunks: arr<string>(c.chunks) as [string, string],
        documents: arr<string>(c.documents) as [string, string],
        unit: String(c.unit ?? ""),
        moreRecent: String(c.more_recent ?? ""),
      }));
      if (cons.length) {
        const known = new Set(v.contradictions.map((c) => c.chunks.join("|")));
        const fresh = cons.filter((c) => !known.has(c.chunks.join("|")));
        v.contradictions = [...v.contradictions, ...fresh];
        v.relationships = [...v.relationships, ...fresh.map((c) => ({
          id: `${ev.event_id}:con:${c.chunks.join("|")}`,
          type: "CONTRADICTS" as const,
          source: `CH:${c.chunks[0]}`,
          target: `CH:${c.chunks[1]}`,
        }))];
      }
      break;
    }
    case "DeltaPlanComputed": {
      v.delta = {
        eventId: ev.event_id,
        utteranceId: ev.utterance_id ?? "",
        baseVersion: num(p.base_version) ?? 0,
        constraint: rec(p.constraint) as DeltaPlan["constraint"],
        affected: arr(p.affected_sub_queries),
        unaffected: arr(p.unaffected_sub_queries),
        plan: rec(p.plan) as DeltaPlan["plan"],
        counterfactual: rec(p.counterfactual) as unknown as Counterfactual,
        claimsInvalidated: num(p.claims_invalidated) ?? 0,
        claimsTotal: num(p.claims_total) ?? 0,
        timestamp: ev.timestamp,
      };
      break;
    }
    case "ClaimsGenerated": {
      const tc = ev.token_cost;
      v.telemetry = {
        ...v.telemetry,
        llmTokens: v.telemetry.llmTokens + (tc.llm_prompt ?? 0) + (tc.llm_completion ?? 0),
        contextTokensEst: v.telemetry.contextTokensEst + (tc.context_est ?? 0),
      };
      break;
    }
    case "GroundingVerified": {
      const s = rec(p.summary);
      v.grounding = [...v.grounding, {
        eventId: ev.event_id,
        checked: num(s.checked) ?? 0,
        supported: num(s.supported) ?? 0,
        partial: num(s.partial) ?? 0,
        conflicting: num(s.conflicting) ?? 0,
        unsupported: num(s.unsupported) ?? 0,
        invalidCitation: num(s.invalid_citation) ?? 0,
        fabricatedIdsBlocked: num(s.fabricated_ids_blocked) ?? 0,
        citationsValid: num(s.citations_valid) ?? 0,
        emitted: num(s.emitted) ?? 0,
        status: (str(s.status) ?? "PASS") as "PASS" | "BLOCKED",
        results: arr(p.results),
        checks: arr<string>(p.checks),
        timestamp: ev.timestamp,
      }];
      break;
    }
    case "RetrievalSuppressed": {
      v.suppression = {
        eventId: ev.event_id,
        utteranceId: ev.utterance_id ?? "",
        reason: str(p.reason) ?? "",
        reusedAnswerVersion: num(p.reused_answer_version) ?? null,
        evidenceReused: num(p.evidence_reused) ?? 0,
        skipped: arr<string>(p.skipped),
        timestamp: ev.timestamp,
      };
      v.telemetry = { ...v.telemetry, suppressedCount: v.telemetry.suppressedCount + 1 };
      break;
    }
    case "TransformUnavailable": {
      v.transformUnavailable = { reason: str(p.reason) ?? "", timestamp: ev.timestamp };
      break;
    }
    case "AnswerVersionCreated": {
      const claims = arr<P>(p.claims).map(toClaim);
      const diff = rec(p.diff) as unknown as AnswerDiff;
      const changed = [...(diff.added ?? []), ...(diff.revised ?? []).map((r) => r.to)];
      const version: AnswerVersion = {
        id: ev.answer_version ?? v.versions.length + 1,
        parentVersion: num(p.parent_version) ?? null,
        queryVersion: ev.query_version ?? 0,
        utteranceId: ev.utterance_id ?? "",
        trigger: (str(p.trigger) ?? "NEW_INFO") as TurnType,
        kind: (str(p.kind) ?? "SEMANTIC") as "SEMANTIC" | "PRESENTATION",
        blocks: arr<AnswerBlock>(p.blocks),
        claims,
        changedClaims: changed,
        unchangedClaims: diff.kept ?? [],
        diff: { kept: diff.kept ?? [], revised: diff.revised ?? [], added: diff.added ?? [], retracted: diff.retracted ?? [] },
        uncertainty: arr(p.uncertainty),
        conflicts: arr(p.conflicts),
        constraint: (p.constraint as AnswerVersion["constraint"]) ?? null,
        transform: (p.transform as AnswerVersion["transform"]) ?? null,
        counterfactual: rec(p.counterfactual) as AnswerVersion["counterfactual"],
        notes: arr<string>(p.notes),
        tokens: rec(p.tokens) as AnswerVersion["tokens"],
        timestamp: ev.timestamp,
        eventId: ev.event_id,
      };
      const claimMap = { ...v.claims };
      const rels = [...v.relationships];
      for (const c of claims) {
        claimMap[c.id] = c;
        for (const e of c.evidenceIds) {
          rels.push({ id: `${ev.event_id}:sup:${e}:${c.id}`, type: "SUPPORTS", source: `CH:${e}`, target: `CL:${c.id}`,
            weight: c.support, answerVersion: version.id });
        }
        for (const d of c.derivedFrom) {
          rels.push({ id: `${ev.event_id}:der:${c.id}`, type: "DERIVED_FROM", source: `CL:${d}`, target: `CL:${c.id}`,
            answerVersion: version.id });
        }
        rels.push({ id: `${ev.event_id}:con:${c.id}`, type: "CONTAINS", source: `CL:${c.id}`, target: `AV:${version.id}`,
          answerVersion: version.id });
      }
      v.claims = claimMap;
      v.relationships = rels;
      v.versions = [...v.versions, version];
      if (v.grounding.length) {
        const g = v.grounding[v.grounding.length - 1];
        if (g.answerVersion === undefined) v.grounding = [...v.grounding.slice(0, -1), { ...g, answerVersion: version.id }];
      }
      break;
    }
    case "TelemetryEmitted": {
      const post = num(p.post_utterance_latency_ms);
      v.telemetry = {
        ...v.telemetry,
        postUtteranceLatencies: post !== undefined
          ? [...v.telemetry.postUtteranceLatencies, { t: ev.timestamp, ms: post, label: ev.utterance_id ?? "" }]
          : v.telemetry.postUtteranceLatencies,
        lastLeadSeconds: num(p.retrieval_lead_s) ?? v.telemetry.lastLeadSeconds,
      };
      break;
    }
    default:
      break;
  }

  const decision = ev.event_type === "RetrievalDecision" ? v.controller?.state : undefined;
  v = { ...v, stage: stageFor(ev.event_type, view.stage, decision) };
  return v;
}

export function foldEvents(events: WireEvent[], upto = events.length): EngineView {
  let view = initialView();
  for (let i = 0; i < upto; i++) view = applyEvent(view, events[i]);
  return view;
}
