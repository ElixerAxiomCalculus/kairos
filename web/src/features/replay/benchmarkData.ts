import { z } from "zod";
import {
  loadBenchmarkSnapshot,
  loadControllerBenchmarkSnapshot,
  loadHardeningBenchmarkSnapshot,
  loadSemanticBenchmarkSnapshot,
} from "@/mocks";
import { apiUrl } from "@/services/apiBase";

/** Benchmark reports are produced by `python -m kairos bench`; the UI only displays them. */
const GateSchema = z.object({
  gate: z.string(),
  metric: z.string(),
  target: z.string(),
  measured: z.number().nullable(),
  status: z.enum(["PASS", "FAIL", "NOT_MEASURED"]),
});

export const BenchmarkReportSchema = z.object({
  status: z.enum(["MEASURED", "NOT_MEASURED"]),
  run_id: z.string().optional(),
  dataset: z.object({ split: z.string(), name: z.string(), sessions: z.number(), note: z.string() }).optional(),
  config: z.record(z.unknown()).optional(),
  repeats: z.number().optional(),
  metrics: z.record(z.number().nullable()).optional(),
  targets: z.record(z.object({ op: z.string(), value: z.number(), gate: z.string() })).optional(),
  gates: z.array(GateSchema).optional(),
  judge: z.string().optional(),
  per_session: z.array(z.unknown()).optional(),
});

export type BenchmarkReport = z.infer<typeof BenchmarkReportSchema>;
export type Source = "BACKEND" | "BUNDLED_SNAPSHOT" | "NONE";

const SemanticBenchmarkReportSchema = z.object({
  status: z.enum(["MEASURED", "NOT_MEASURED"]),
  run_id: z.string().optional(),
  dataset: z.object({
    name: z.string(),
    split: z.string(),
    seed: z.number(),
    note: z.string(),
    groups: z.array(z.string()),
  }).optional(),
  corpora: z.record(z.string()).optional(),
  metrics: z.object({
    intent: z.object({
      intent_precision: z.number(),
      intent_recall: z.number(),
      intent_f1: z.number(),
      boundary_f1: z.number(),
      count_accuracy: z.number(),
      false_merge_rate: z.number(),
      false_split_rate: z.number(),
      topic_transition_accuracy: z.number(),
      constraint_ownership_accuracy: z.number(),
      cases: z.number(),
      intents: z.number(),
      intent_recall_ci95: z.tuple([z.number(), z.number()]),
    }).passthrough(),
    retrieval: z.object({
      recall_at_5: z.number(),
      mrr: z.number(),
      ndcg_at_5: z.number(),
      relevant_evidence_precision: z.number(),
      out_of_domain_rejection_accuracy: z.number(),
      irrelevant_evidence_acceptance_rate: z.number(),
      domain_routing_accuracy: z.number(),
      positive_queries: z.number(),
      negative_queries: z.number(),
      external_squad_queries: z.number(),
      external_squad_recall_at_5: z.number(),
      external_squad_mrr: z.number(),
    }).passthrough(),
    external_scifact_qrels: z.object({
      status: z.string(), queries: z.number(), recall_at_5: z.number().nullable(), mrr_at_10: z.number().nullable(),
    }).passthrough().optional(),
    external_squad_expanded: z.object({
      status: z.string(), queries: z.number(), recall_at_5: z.number().nullable(), mrr_at_10: z.number().nullable(),
    }).passthrough().optional(),
    grounding: z.object({
      cases: z.number(),
      accuracy: z.number(),
      unsupported_claim_rejection: z.number(),
      quote_verification_accuracy: z.number(),
      numeric_contradiction_detection: z.number(),
      false_grounding_acceptance_rate: z.number(),
    }).passthrough(),
    streaming: z.object({
      transcript_chunks: z.number(),
      active_intents: z.number(),
      actual_bm25_probe_computations: z.number(),
      probe_cache_hits: z.number(),
      retrieval_count: z.number(),
      duplicate_retrieval_rate: z.number(),
    }).passthrough(),
    scale: z.array(z.object({
      profile: z.string(),
      documents: z.number(),
      chunks: z.number(),
      build_ms: z.number(),
      python_peak_bytes: z.number(),
      retrieval_p50_ms: z.number(),
      retrieval_p95_ms: z.number(),
      throughput_qps_single_thread: z.number(),
    }).passthrough()).optional(),
  }).optional(),
  ablations: z.object({
    efis_100_pairs: z.object({
      v2: z.object({ merge_precision: z.number(), merge_recall: z.number(), merge_f1: z.number(), accuracy: z.number(), false_merge_rate: z.number(), cases: z.number() }).passthrough(),
      v2_1: z.object({ merge_precision: z.number(), merge_recall: z.number(), merge_f1: z.number(), accuracy: z.number(), false_merge_rate: z.number(), cases: z.number() }).passthrough(),
    }).optional(),
  }).passthrough().optional(),
  integrity: z.record(z.string()).optional(),
});

export type SemanticBenchmarkReport = z.infer<typeof SemanticBenchmarkReportSchema>;

const HardeningBenchmarkReportSchema = z.object({
  status: z.enum(["MEASURED", "NOT_MEASURED"]),
  run_id: z.string().optional(),
  dataset: z.object({ name: z.string(), revision: z.string(), license: z.string() }).passthrough().optional(),
  scale: z.array(z.object({
    documents: z.number(), chunks: z.number(), index_build_ms: z.number(), python_traced_peak_bytes: z.number(),
    search_p50_ms: z.number().nullable(), search_p95_ms: z.number().nullable(), search_p99_ms: z.number().nullable(),
    single_session_qps: z.number(), index_persistence_status: z.string(),
  }).passthrough()).optional(),
  concurrency: z.array(z.object({
    sessions: z.number(), wall_ms: z.number(), sessions_per_second: z.number(), session_isolation: z.boolean(),
  }).passthrough()).optional(),
  soak: z.object({ rounds: z.number(), sessions_per_round: z.number(), duration_s: z.number(), qualification: z.string() }).passthrough().optional(),
});
export type HardeningBenchmarkReport = z.infer<typeof HardeningBenchmarkReportSchema>;

const ControllerMetricsSchema = z.object({
  early_retrieval_rate: z.number(),
  useful_early_retrieval_rate: z.number(),
  premature_retrieval_rate: z.number().nullable(),
  evidence_survival_rate: z.number().nullable(),
  full_searches: z.number(),
  duplicate_retrievals_avoided: z.number(),
});

const ControllerBenchmarkReportSchema = z.object({
  status: z.literal("MEASURED"),
  run_id: z.string(),
  dataset_hash: z.string(),
  variants: z.record(z.object({ metrics: ControllerMetricsSchema }).passthrough()),
  selection: z.object({ selected: z.string(), promoted: z.boolean() }).passthrough(),
  held_out: z.object({
    status: z.string(),
    selected_variant: z.string(),
    metrics: ControllerMetricsSchema.extend({
      useful_early_group_bootstrap_ci95: z.tuple([z.number(), z.number()]),
      telemetry_coverage: z.number(),
      grounded_claim_rate: z.number(),
    }).passthrough(),
  }).passthrough(),
});
export type ControllerBenchmarkReport = z.infer<typeof ControllerBenchmarkReportSchema>;

async function fetchJson(url: string): Promise<unknown | null> {
  try {
    const res = await fetch(apiUrl(url));
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function loadBenchmark(): Promise<{ report: BenchmarkReport | null; source: Source; error?: string }> {
  const live = await fetchJson("/api/benchmark/latest");
  if (live) {
    const parsed = BenchmarkReportSchema.safeParse(live);
    if (parsed.success && parsed.data.status === "MEASURED") return { report: parsed.data, source: "BACKEND" };
    if (!parsed.success) return { report: null, source: "NONE", error: "Backend returned a malformed benchmark report." };
  }
  const snap = BenchmarkReportSchema.safeParse(await loadBenchmarkSnapshot());
  return snap.success ? { report: snap.data, source: "BUNDLED_SNAPSHOT" } : { report: null, source: "NONE" };
}

export async function loadSemanticBenchmark(): Promise<{ report: SemanticBenchmarkReport | null; source: Source; error?: string }> {
  const live = await fetchJson("/api/semantic-benchmark/latest");
  if (live) {
    const parsed = SemanticBenchmarkReportSchema.safeParse(live);
    if (parsed.success && parsed.data.status === "MEASURED") return { report: parsed.data, source: "BACKEND" };
    if (!parsed.success) return { report: null, source: "NONE", error: "Backend returned a malformed semantic benchmark report." };
  }
  const snap = SemanticBenchmarkReportSchema.safeParse(await loadSemanticBenchmarkSnapshot());
  return snap.success ? { report: snap.data, source: "BUNDLED_SNAPSHOT" } : { report: null, source: "NONE" };
}

export async function loadHardeningBenchmark(): Promise<{ report: HardeningBenchmarkReport | null; source: Source; error?: string }> {
  const live = await fetchJson("/api/hardening-benchmark/latest");
  if (live) {
    const parsed = HardeningBenchmarkReportSchema.safeParse(live);
    if (parsed.success && parsed.data.status === "MEASURED") return { report: parsed.data, source: "BACKEND" };
    if (!parsed.success) return { report: null, source: "NONE", error: "Backend returned a malformed hardening report." };
  }
  const snap = HardeningBenchmarkReportSchema.safeParse(await loadHardeningBenchmarkSnapshot());
  return snap.success ? { report: snap.data, source: "BUNDLED_SNAPSHOT" } : { report: null, source: "NONE" };
}

export async function loadControllerBenchmark(): Promise<{ report: ControllerBenchmarkReport | null; source: Source; error?: string }> {
  const live = await fetchJson("/api/controller-benchmark/latest");
  if (live) {
    const parsed = ControllerBenchmarkReportSchema.safeParse(live);
    if (parsed.success) return { report: parsed.data, source: "BACKEND" };
    return { report: null, source: "NONE", error: "Backend returned a malformed controller benchmark report." };
  }
  const snap = ControllerBenchmarkReportSchema.safeParse(await loadControllerBenchmarkSnapshot());
  return snap.success ? { report: snap.data, source: "BUNDLED_SNAPSHOT" } : { report: null, source: "NONE" };
}

export interface ExperimentsReport {
  status: string;
  run_id?: string;
  dataset?: string;
  experiments: Record<string, {
    title: string;
    hypothesis: string;
    status: "MEASURED" | "NOT_RUN";
    metrics?: string[];
    reason?: string;
    arms?: Record<string, { overrides: Record<string, unknown>; metrics: Record<string, number | null> }>;
  }>;
}

export async function loadExperiments(): Promise<{ report: ExperimentsReport | null; source: Source }> {
  const live = (await fetchJson("/api/experiments/latest")) as ExperimentsReport | null;
  if (live && live.experiments) return { report: live, source: "BACKEND" };
  try {
    const mod = await import("@/mocks/fixtures/experiments.latest.json");
    return { report: mod.default as unknown as ExperimentsReport, source: "BUNDLED_SNAPSHOT" };
  } catch {
    return { report: null, source: "NONE" };
  }
}
