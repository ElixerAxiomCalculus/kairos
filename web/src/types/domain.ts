/**
 * Frontend domain model. Every field is derived from backend events (see features/streaming/reducer.ts);
 * nothing here is synthesized. Optional fields are absent until the corresponding event arrives.
 */

export type ControllerState = "WAIT" | "RETRIEVE" | "SUPPRESS" | "REFINE";
export type RetrievalTrigger = "EARLY" | "END_OF_UTTERANCE" | "REFINE" | "DELTA" | "NONE";
export type TurnType = "NEW_INFO" | "CONSTRAINT" | "TRANSFORM" | "ELABORATE";
export type Verdict = "SUPPORTED" | "PARTIAL" | "CONFLICTING" | "UNSUPPORTED" | "INVALID_CITATION" | "PENDING";
export type ClaimStatus = "NEW" | "KEEP" | "REVISED" | "ADDED" | "TRANSFORMED";
export type ConnectionState = "LIVE" | "RECONNECTING" | "REPLAY" | "OFFLINE" | "MOCK";

export interface Signals {
  rho?: number; // result stability (RBO of probes)
  chi?: number; // clarity of the probe score distribution
  kappa?: number; // clause completeness
  sigma?: number; // slot stability
  cont?: number; // 1 - continuation hazard
  pi?: number;
  nu?: number; // novelty vs prior retrievals
  s_hat?: number; // calibrated stability
  theta?: number; // adaptive threshold
  remaining_s?: number;
  hidden_latency_s?: number;
  rho_vs_retrieved?: number;
}

export interface CorpusStatus {
  documents: number;
  chunks: number;
  dense_backend: string;
  embedding_model: string;
  sparse_index: string;
  reranker: string;
  indexed_at?: string;
  facet_dimensions?: Record<string, string[]>;
  documents_list?: { doc_id: string; title: string; doc_type?: string; effective_date?: string; chunks: number }[];
}

export interface SessionConfig {
  controller_mode: string;
  retrieval_mode: string;
  reranker: string;
  synthesis_backend: string;
  grounding_nli: string;
  refinement: string;
}

export interface Session {
  id: string | null;
  status: "IDLE" | "ACTIVE" | "ENDED";
  startedAt?: string;
  corpus?: CorpusStatus;
  config?: SessionConfig;
  stabilityModel?: string;
}

export interface TranscriptChunk {
  id: string;
  eventId: string;
  utteranceId: string;
  sequence: number;
  timestamp: number; // session clock seconds (t1)
  t0: number;
  text: string;
  final: boolean;
  // controller signals evaluated on this chunk (decisive candidate)
  semanticCompleteness?: number;
  intentConfidence?: number;
  entityStability?: number;
  novelty?: number;
  informationGain?: number;
  stability?: number;
  threshold?: number;
  decision?: ControllerState;
  retrievalFired?: boolean;
}

export interface Utterance {
  id: string;
  chunkIds: string[];
  text: string;
  turnType?: TurnType;
  turnConfidence?: number;
  closed: boolean;
  startT: number;
  endT: number;
}

export interface Slot {
  key: string;
  kind: string;
  value: string;
  revised_from?: string | null;
}

export interface QueryVersion {
  version: number;
  utteranceId: string;
  normalized: string;
  slots: Slot[];
  candidates: { key: string; text: string; closed: boolean }[];
  repairs: { kind: string; from: string; to: string }[];
  timestamp: number;
}

export interface CandidateDecision {
  key: string;
  text: string;
  action: string;
  trigger: RetrievalTrigger;
  reasons: string[];
  signals: Signals;
  sub_query_id?: string | null;
}

export interface RetrievalDecision {
  eventId: string;
  state: ControllerState;
  confidence?: number;
  reasons: string[];
  signals: Signals;
  queryVersion?: number;
  utteranceId?: string;
  timestamp: number;
  trigger: RetrievalTrigger;
  turnType?: TurnType;
  candidates: CandidateDecision[];
}

export type IntentRetrievalStatus = "PENDING" | "RETRIEVING" | "RETRIEVED" | "REFINING" | "SUPPRESSED";

export interface Intent {
  id: string;
  label: string;
  normalizedQuery: string;
  utteranceId: string;
  confidence?: number;
  dependencies: string[];
  dependencyType?: string | null;
  retrievalStatus: IntentRetrievalStatus;
  retrievalCount: number;
  documentIds: string[];
  chunkIds: string[];
  lastLatencyMs?: number;
  lastTrigger?: RetrievalTrigger;
  footprint: string[];
  createdAt: number;
  facetConstraints: Record<string, string>;
}

export interface EvidenceChunk {
  chunkId: string;
  documentId: string;
  docTitle: string;
  section: string;
  sectionTitle: string;
  sectionPath: string;
  text?: string;
  denseScore?: number;
  sparseScore?: number;
  fusedScore?: number;
  rerankScore?: number;
  facets: Record<string, string>;
  effectiveDate?: string | null;
  docType?: string | null;
  intents: string[];
  firstSeen: number;
}

export type RelationType = "RETRIEVED" | "SUPPORTS" | "CONTRADICTS" | "DUPLICATES" | "DERIVED_FROM" | "CONTAINS";

export interface EvidenceRelationship {
  id: string;
  type: RelationType;
  source: string; // graph node id
  target: string;
  trigger?: RetrievalTrigger;
  weight?: number;
  rank?: number;
  answerVersion?: number;
}

export interface Claim {
  id: string;
  lineageId: string;
  text: string;
  quote: string;
  subQueryId: string;
  evidenceIds: string[];
  supportStatus: Verdict;
  support: number;
  status: ClaimStatus;
  derivedFrom: string[];
  conflictsWith: string[];
  versionCreated: number;
}

export interface AnswerBlock {
  sub_query_id: string | null;
  title: string;
  style: "paragraph" | "bullet";
  claim_ids: string[];
  uncertainty?: string | null;
}

export interface AnswerDiff {
  kept: string[];
  revised: { from: string; to: string; reasons: string[] }[];
  added: string[];
  retracted: (string | { claim_id: string; reasons: string[] })[];
}

export interface Counterfactual {
  full_restart_retrievals: number;
  delta_retrievals: number;
  full_restart_context_tokens_est: number;
  delta_context_tokens_est: number;
  retrieval_savings_ratio: number;
  full_restart_llm_prompt_tokens_est?: number;
}

export interface AnswerVersion {
  id: number;
  parentVersion: number | null;
  queryVersion: number;
  utteranceId: string;
  trigger: TurnType;
  kind: "SEMANTIC" | "PRESENTATION";
  blocks: AnswerBlock[];
  claims: Claim[];
  changedClaims: string[];
  unchangedClaims: string[];
  diff: AnswerDiff;
  uncertainty: { sub_query_id: string | null; text: string }[];
  conflicts: { claim_id: string; conflicts_with: string[]; note?: string; more_recent_document?: string }[];
  constraint?: { text: string; facets: Record<string, string>; terms: string[] } | null;
  transform?: { ops: string[]; bullets?: number; language?: string | null } | null;
  counterfactual?: Counterfactual | Record<string, never>;
  notes: string[];
  tokens: { llm_prompt?: number; llm_completion?: number; context_est?: number };
  timestamp: number;
  eventId: string;
}

export interface GroundingResult {
  answerVersion?: number;
  eventId: string;
  checked: number;
  supported: number;
  partial: number;
  conflicting: number;
  unsupported: number;
  invalidCitation: number;
  fabricatedIdsBlocked: number;
  citationsValid: number;
  emitted: number;
  status: "PASS" | "BLOCKED";
  results: {
    claim_id: string;
    verdict: Verdict;
    support: number;
    anchors_ok: boolean;
    quote_ok: boolean;
    invalid_ids: string[];
    text: string;
    evidence: string[];
    repaired: boolean;
  }[];
  checks: string[];
  timestamp: number;
}

export interface DeltaPlan {
  eventId: string;
  utteranceId: string;
  baseVersion: number;
  constraint: { text: string; facets: Record<string, string>; terms: string[] };
  affected: { sub_query_id: string; reasons: string[] }[];
  unaffected: { sub_query_id: string; reasons: string[] }[];
  plan: Record<string, { action: "KEEP" | "REVISE"; reasons: string[] }>;
  counterfactual: Counterfactual;
  claimsInvalidated: number;
  claimsTotal: number;
  timestamp: number;
}

export interface SuppressionInfo {
  eventId: string;
  utteranceId: string;
  reason: string;
  reusedAnswerVersion?: number | null;
  evidenceReused: number;
  skipped: string[];
  timestamp: number;
}

export interface LatencyMetric {
  t: number;
  ms: number;
  label: string;
}

export interface TelemetryTotals {
  retrievalLatencies: LatencyMetric[];
  postUtteranceLatencies: LatencyMetric[];
  retrievalCount: number;
  earlyRetrievalCount: number;
  refineCount: number;
  deltaCount: number;
  probeCount: number;
  suppressedCount: number;
  llmTokens: number;
  contextTokensEst: number;
  documentsTouched: number;
  lastLeadSeconds?: number | null;
}

export interface TelemetryEvent {
  id: string;
  eventType: string;
  timestamp: number;
  sessionId: string;
  parentIds: string[];
  utteranceId?: string | null;
  queryVersion?: number | null;
  answerVersion?: number | null;
  retrievalTrigger?: RetrievalTrigger | null;
  subQueryId?: string | null;
  documentIds: string[];
  chunkIds: string[];
  latency?: number | null;
  tokenCost: Record<string, number>;
  confidence?: number | null;
  citations: string[];
  payload: Record<string, unknown>;
  raw: unknown;
}

export interface BenchmarkMetric {
  key: string;
  label: string;
  gate?: string;
  target?: string;
  measured?: number | null;
  status: "PASS" | "FAIL" | "NOT_MEASURED" | "INFO";
  unit?: "ratio" | "ms" | "s" | "count" | "tokens";
}

export type Stage = "idle" | "listening" | "retrieving" | "answer" | "refine" | "suppress";
