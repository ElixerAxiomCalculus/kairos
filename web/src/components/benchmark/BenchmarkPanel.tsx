import { createColumnHelper, flexRender, getCoreRowModel, getSortedRowModel, useReactTable, type SortingState } from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { StatusPill } from "@/components/common/StatusPill";
import type { BenchmarkReport, SemanticBenchmarkReport } from "@/features/replay/benchmarkData";
import { DASH, ms, ratio } from "@/utils/format";
import styles from "@/routes/routes.module.css";

type Unit = "ratio" | "ms" | "count" | "tokens" | "s";
interface MetricDef {
  key: string;
  label: string;
  unit: Unit;
  gate?: string;
  explain: string;
}

/** Catalogue of what is measured. Targets come from the report (config), never from this file. */
export const METRICS: MetricDef[] = [
  { key: "early_retrieval_rate", label: "Early Retrieval Rate", unit: "ratio", gate: "G2", explain: "Eligible utterances whose first retrieval started before the speaker finished." },
  { key: "premature_retrieval_rate", label: "Premature Retrieval Rate", unit: "ratio", explain: "Early retrievals whose evidence did not survive to the final query (repaired by REFINE)." },
  { key: "false_retrieval_rate", label: "False Retrieval Rate", unit: "ratio", gate: "G5", explain: "Presentation-only turns that triggered retrieval." },
  { key: "intent_recall", label: "Intent Identification", unit: "ratio", gate: "G3", explain: "Gold intents matched by a sub-query whose top-5 evidence contains a gold section." },
  { key: "intent_precision", label: "Intent Precision", unit: "ratio", explain: "Sub-queries that match a gold intent." },
  { key: "intent_count_accuracy", label: "Intent Count Accuracy", unit: "ratio", explain: "Utterances decomposed into exactly the gold number of intents." },
  { key: "gold_section_mrr", label: "Gold Section MRR", unit: "ratio", explain: "Mean reciprocal rank of the first gold section." },
  { key: "citation_precision", label: "Citation Precision", unit: "ratio", gate: "G4", explain: "Citations that support their claim (independent automatic judge)." },
  { key: "citation_recall", label: "Citation Recall", unit: "ratio", explain: "Claims with at least one supporting citation." },
  { key: "grounded_claim_rate", label: "Grounded Claim Rate", unit: "ratio", gate: "G4", explain: "Emitted factual claims judged supported." },
  { key: "fabricated_doc_ids", label: "Fabricated Doc IDs", unit: "count", gate: "G4", explain: "Citations that do not resolve to a corpus chunk." },
  { key: "abstention_accuracy", label: "Abstention Accuracy", unit: "ratio", explain: "Unanswerable intents answered with an uncertainty signal." },
  { key: "delta_retrieval_ratio", label: "Delta Retrieval Ratio", unit: "ratio", gate: "G5", explain: "Sub-queries re-retrieved on a late constraint / sub-queries in the answer." },
  { key: "refinement_correctness", label: "Refinement Correctness", unit: "ratio", explain: "Constraint turns whose re-retrieved intents equal the gold affected set." },
  { key: "unnecessary_claim_churn", label: "Unnecessary Claim Churn", unit: "count", explain: "Claims of unaffected intents whose text changed after a constraint." },
  { key: "unnecessary_retrieval_count", label: "Unnecessary Retrieval Count", unit: "count", explain: "Retrievals on presentation-only turns." },
  { key: "update_context_tokens_est", label: "Answer Update Cost", unit: "tokens", explain: "Context tokens processed for delta updates (estimate)." },
  { key: "full_restart_context_tokens_est", label: "Full-Restart Cost", unit: "tokens", explain: "Counterfactual context tokens had the pipeline restarted (estimate)." },
  { key: "retrieval_latency_ms_mean", label: "Avg Retrieval Latency", unit: "ms", explain: "Mean hybrid retrieval wall time." },
  { key: "post_utterance_latency_ms_p50", label: "Post-Utterance Latency p50", unit: "ms", explain: "Final chunk → answer version." },
  { key: "end_to_end_latency_ms_p50", label: "End-to-End Latency p50", unit: "ms", explain: "First chunk → answer version (includes speaking time)." },
  { key: "llm_tokens_total", label: "Token Cost (LLM)", unit: "tokens", explain: "LLM tokens across the run (0 with the extractive backend)." },
  { key: "telemetry_coverage", label: "Telemetry Coverage", unit: "ratio", gate: "G6", explain: "Lineage invariants satisfied / checked." },
  { key: "reproducibility_delta", label: "Reproducibility Δ", unit: "count", gate: "G1", explain: "Max difference of non-timing metrics across repeated runs." },
];

export function formatMetric(v: number | null | undefined, unit: Unit): string {
  if (v === null || v === undefined) return DASH;
  if (unit === "ratio") return ratio(v, 1);
  if (unit === "ms") return ms(v);
  return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(3);
}

interface Row {
  gate: string;
  metric: string;
  target: string;
  measured: number | null;
  status: "PASS" | "FAIL" | "NOT_MEASURED";
}
const col = createColumnHelper<Row>();

export function GateTable({ report }: { report: BenchmarkReport | null }) {
  const [sorting, setSorting] = useState<SortingState>([{ id: "gate", desc: false }]);
  const data = useMemo<Row[]>(() => {
    if (report?.gates) return report.gates;
    return Object.entries(report?.targets ?? {}).map(([metric, t]) => ({
      gate: t.gate, metric, target: `${t.op} ${t.value}`, measured: null, status: "NOT_MEASURED" as const,
    }));
  }, [report]);
  const columns = useMemo(() => [
    col.accessor("gate", { header: "GATE" }),
    col.accessor("metric", { header: "METRIC", cell: (c) => METRICS.find((m) => m.key === c.getValue())?.label ?? c.getValue() }),
    col.accessor("target", { header: "TARGET", cell: (c) => <span className="mono">TARGET {c.getValue()}</span> }),
    col.accessor("measured", {
      header: "MEASURED",
      cell: (c) => {
        const def = METRICS.find((m) => m.key === c.row.original.metric);
        return <span className="mono">MEASURED {formatMetric(c.getValue(), def?.unit ?? "count")}</span>;
      },
    }),
    col.accessor("status", {
      header: "STATUS",
      cell: (c) => <StatusPill state={c.getValue() === "PASS" ? "PASS" : c.getValue() === "FAIL" ? "FAIL" : "WAIT"} label={c.getValue().replace("_", " ")} />,
    }),
  ], []);
  const table = useReactTable({ data, columns, state: { sorting }, onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel() });
  return (
    <table className={styles.table} data-testid="gate-table">
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th key={h.id} onClick={h.column.getToggleSortingHandler()} aria-sort={h.column.getIsSorted() === "asc" ? "ascending" : h.column.getIsSorted() === "desc" ? "descending" : "none"}>
                {flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((r) => (
          <tr key={r.id}>
            {r.getVisibleCells().map((c) => <td key={c.id}>{flexRender(c.column.columnDef.cell, c.getContext())}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function MetricGrid({ report }: { report: BenchmarkReport | null }) {
  const measured = report?.status === "MEASURED" ? report.metrics ?? {} : {};
  const targets = report?.targets ?? {};
  return (
    <div className={styles.metrics} data-testid="metric-grid">
      {METRICS.map((m) => {
        const v = measured[m.key];
        const t = targets[m.key];
        return (
          <div key={m.key} className={styles.metric} title={m.explain}>
            <span className="px muted">{m.label}{m.gate ? ` · ${m.gate}` : ""}</span>
            <span className={styles.big}>{formatMetric(v, m.unit)}</span>
            <div className={styles.metricRow}>
              <span>{t ? `TARGET ${t.op} ${m.unit === "ratio" ? ratio(t.value) : t.value}` : "TARGET —"}</span>
              <span>{v === undefined || v === null ? "NOT MEASURED" : "MEASURED"}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

const SEMANTIC_METRICS: Array<{
  label: string;
  value: (report: SemanticBenchmarkReport) => number | undefined;
  detail: (report: SemanticBenchmarkReport) => string;
}> = [
  { label: "Intent F1", value: (r) => r.metrics?.intent.intent_f1, detail: (r) => `${r.metrics?.intent.intents ?? 0} intents / ${r.metrics?.intent.cases ?? 0} cases` },
  { label: "Boundary F1", value: (r) => r.metrics?.intent.boundary_f1, detail: () => "deterministic speech boundaries" },
  { label: "Constraint Ownership", value: (r) => r.metrics?.intent.constraint_ownership_accuracy, detail: () => "intent-scoped slots" },
  { label: "Retrieval Recall@5", value: (r) => r.metrics?.retrieval.recall_at_5, detail: (r) => `${r.metrics?.retrieval.positive_queries ?? 0} positive profile queries` },
  { label: "Relevant Evidence Precision", value: (r) => r.metrics?.retrieval.relevant_evidence_precision, detail: () => "accepted top-k evidence" },
  { label: "OOD Rejection", value: (r) => r.metrics?.retrieval.out_of_domain_rejection_accuracy, detail: (r) => `${r.metrics?.retrieval.negative_queries ?? 0} negative queries` },
  { label: "SQuAD Recall@5", value: (r) => r.metrics?.retrieval.external_squad_recall_at_5, detail: (r) => `${r.metrics?.retrieval.external_squad_queries ?? 0} held-aside labels` },
  { label: "Grounding Accuracy", value: (r) => r.metrics?.grounding.accuracy, detail: (r) => `${r.metrics?.grounding.cases ?? 0} adversarial cases` },
  { label: "False Grounding Acceptance", value: (r) => r.metrics?.grounding.false_grounding_acceptance_rate, detail: () => "lower is better" },
];

export function SemanticMetricGrid({ report }: { report: SemanticBenchmarkReport }) {
  return (
    <div className={styles.metrics} data-testid="semantic-metric-grid">
      {SEMANTIC_METRICS.map((metric) => (
        <div key={metric.label} className={styles.metric} title={metric.detail(report)}>
          <span className="px muted">{metric.label}</span>
          <span className={styles.big}>{formatMetric(metric.value(report), "ratio")}</span>
          <div className={styles.metricRow}><span>{metric.detail(report)}</span><span>MEASURED</span></div>
        </div>
      ))}
    </div>
  );
}
