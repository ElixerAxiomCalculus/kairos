import { z } from "zod";
import { loadBenchmarkSnapshot } from "@/mocks";

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

async function fetchJson(url: string): Promise<unknown | null> {
  try {
    const res = await fetch(url);
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
    if (parsed.success) return { report: parsed.data, source: "BACKEND" };
    return { report: null, source: "NONE", error: "Backend returned a malformed benchmark report." };
  }
  const snap = BenchmarkReportSchema.safeParse(await loadBenchmarkSnapshot());
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
