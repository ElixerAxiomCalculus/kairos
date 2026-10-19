import { useEffect, useState } from "react";
import { GateTable, MetricGrid, SemanticMetricGrid } from "@/components/benchmark/BenchmarkPanel";
import { EmptyState } from "@/components/common/EmptyState";
import {
  loadBenchmark,
  loadControllerBenchmark,
  loadHardeningBenchmark,
  loadSemanticBenchmark,
  type BenchmarkReport,
  type ControllerBenchmarkReport,
  type HardeningBenchmarkReport,
  type SemanticBenchmarkReport,
  type Source,
} from "@/features/replay/benchmarkData";
import styles from "./routes.module.css";

type ReportState<T> = { report: T | null; source: Source; error?: string };

const corpora = [
  {
    name: "SciFact via BEIR",
    use: "12 documents / 29 chunks; 1 intersecting official test qrel (labels not indexed)",
    license: "CC BY-SA 4.0",
    dataset: "https://huggingface.co/datasets/BeIR/scifact",
    citation: "Wadden et al. (EMNLP 2020); Thakur et al. (NeurIPS 2021)",
    paper: "https://aclanthology.org/2020.emnlp-main.609/",
  },
  {
    name: "SQuAD 1.1",
    use: "21-context expanded profile / 41 chunks; 48 Q&A labels excluded from retrieval",
    license: "CC BY-SA 4.0",
    dataset: "https://huggingface.co/datasets/rajpurkar/squad",
    citation: "Rajpurkar et al. (EMNLP 2016)",
    paper: "https://aclanthology.org/D16-1264/",
  },
];

/** Target vs measured, never blurred. Provenance of every number is on screen. */
export default function BenchmarkRoute() {
  const [state, setState] = useState<ReportState<BenchmarkReport> | null>(null);
  const [semanticState, setSemanticState] = useState<ReportState<SemanticBenchmarkReport> | null>(null);
  const [hardeningState, setHardeningState] = useState<ReportState<HardeningBenchmarkReport> | null>(null);
  const [controllerState, setControllerState] = useState<ReportState<ControllerBenchmarkReport> | null>(null);
  useEffect(() => {
    void Promise.all([loadBenchmark(), loadSemanticBenchmark(), loadHardeningBenchmark(), loadControllerBenchmark()]).then(([development, semantic, hardening, controller]) => {
      setState(development);
      setSemanticState(semantic);
      setHardeningState(hardening);
      setControllerState(controller);
    });
  }, []);
  const r = state?.report ?? null;
  const semantic = semanticState?.report ?? null;
  const hardening = hardeningState?.report ?? null;
  const controller = controllerState?.report ?? null;
  const measured = r?.status === "MEASURED";
  return (
    <main className={styles.page} data-testid="benchmark-page">
      <div className={styles.pageHead}>
        <div>
          <span className="px muted">BENCHMARK MODE</span>
          <h1 className={styles.pageTitle}>Evaluation <em>gates</em></h1>
        </div>
        <div className={styles.provenance} data-testid="benchmark-provenance">
          {state === null ? "loading…" : measured ? (
            <>
              MEASURED · run {r?.run_id} · {r?.dataset?.name} ({r?.dataset?.split}, {r?.dataset?.sessions} sessions) · {r?.repeats} repeats
              <br />source: {state.source === "BACKEND" ? "live backend reports" : "bundled snapshots of the last measured runs"}
            </>
          ) : "NOT MEASURED · run python -m kairos bench"}
        </div>
      </div>
      {state?.error ? <div className="px" style={{ color: "var(--status-fail)" }}>{state.error}</div> : null}
      {semanticState?.error ? <div className="px" style={{ color: "var(--status-fail)" }}>{semanticState.error}</div> : null}
      {hardeningState?.error ? <div className="px" style={{ color: "var(--status-fail)" }}>{hardeningState.error}</div> : null}
      {controllerState?.error ? <div className="px" style={{ color: "var(--status-fail)" }}>{controllerState.error}</div> : null}
      {state && !r ? <EmptyState label="No benchmark report">Run <code>python -m kairos bench</code> to measure.</EmptyState> : null}
      {r ? (
        <>
          <section className="panel" style={{ padding: 0 }}>
            <header className="panel-head"><span className="px secondary">CURRENT V2.1 DEVELOPMENT GATES · TARGET VS MEASURED</span></header>
            <div style={{ overflowX: "auto" }}><GateTable report={r} /></div>
          </section>
          <MetricGrid report={r} />
          <p className={styles.note}>
            {r.dataset?.note} {r.judge ? `Judge: ${r.judge}` : ""} The dev set is small and was authored alongside the
            system; treat these as development measurements, not a claim about the official benchmark.
          </p>
        </>
      ) : null}
      {controller ? (
        <section className="panel" style={{ padding: 0 }} data-testid="controller-v2-2-benchmark">
          <header className="panel-head">
            <span className="px secondary">V2.2 PSRC RESEARCH · RUN {controller.run_id}</span>
          </header>
          <div style={{ overflowX: "auto" }}>
            <table className={styles.table}>
              <thead><tr><th>POLICY / SPLIT</th><th>USEFUL EARLY</th><th>PREMATURE</th><th>SURVIVAL</th><th>FULL SEARCHES</th><th>OUTCOME</th></tr></thead>
              <tbody>
                {([
                  ["P0 · original PSRC", controller.variants.P0_original_psrc?.metrics, "SELECTED"],
                  ["P3 · constraint-aware cold start", controller.variants.P3_constraint_cold_start?.metrics, "NOT PROMOTED"],
                  ["P6 · validated evidence reuse", controller.variants.P6_evidence_reuse?.metrics, "NOT PROMOTED"],
                  ["Frozen held-out · P0", controller.held_out.metrics, "POST-SELECTION"],
                ] as const).map(([label, metrics, outcome]) => metrics ? (
                  <tr key={label}>
                    <td>{label}</td><td>{metrics.useful_early_retrieval_rate.toFixed(4)}</td>
                    <td>{metrics.premature_retrieval_rate?.toFixed(4) ?? "—"}</td>
                    <td>{metrics.evidence_survival_rate?.toFixed(4) ?? "—"}</td>
                    <td>{metrics.full_searches}{metrics.duplicate_retrievals_avoided ? ` (${metrics.duplicate_retrievals_avoided} avoided)` : ""}</td>
                    <td>{outcome}</td>
                  </tr>
                ) : null)}
              </tbody>
            </table>
          </div>
          <p className={styles.note} style={{ padding: "0 var(--sp-4) var(--sp-4)", margin: 0 }}>
            Development policies use 36 grouped streams; the frozen holdout has 8 project-authored streams and a useful-early
            group-bootstrap 95% interval of {controller.held_out.metrics.useful_early_group_bootstrap_ci95.join("–")}.
            No V2.2 arm passed every correctness, G2, survival, and search-cost filter, so original PSRC remains the default.
            These are small-corpus research measurements, not external benchmark claims.
          </p>
        </section>
      ) : null}
      {semantic ? (
        <>
          <section className="panel" style={{ padding: 0 }} data-testid="semantic-benchmark">
            <header className="panel-head">
              <span className="px secondary">V2.1 SEMANTIC RELIABILITY · RUN {semantic.run_id}</span>
            </header>
            <div style={{ padding: "var(--sp-4)" }}>
              <SemanticMetricGrid report={semantic} />
              <p className={styles.note}>
                {semantic.dataset?.note} Intent recall 95% CI: {semantic.metrics?.intent.intent_recall_ci95.join("–")}.
                Relevant-evidence precision and expanded SQuAD retrieval are shown even though they are below 1.0;
                the official SciFact value has only one intersecting query and must not be generalized.
              </p>
            </div>
          </section>
          <section className="panel" style={{ padding: 0 }} data-testid="corpus-attribution">
            <header className="panel-head"><span className="px secondary">RAG CORPUS · ATTRIBUTION AND LICENSES</span></header>
            <div style={{ overflowX: "auto" }}>
              <table className={styles.table}>
                <thead><tr><th>DATASET</th><th>INDEXED SUBSET</th><th>LICENSE</th><th>CITATION</th></tr></thead>
                <tbody>
                  {corpora.map((corpus) => (
                    <tr key={corpus.name}>
                      <td><a href={corpus.dataset} target="_blank" rel="noreferrer">{corpus.name}</a></td>
                      <td>{corpus.use}</td>
                      <td>{corpus.license}</td>
                      <td><a href={corpus.paper} target="_blank" rel="noreferrer">{corpus.citation}</a></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className={styles.note} style={{ padding: "0 var(--sp-4) var(--sp-4)", margin: 0 }}>
              The project-authored sample corpus contains 8 fictional-policy documents / 37 chunks. External subsets retain
              immutable revisions and SHA-256 provenance; SQuAD questions/answers and SciFact qrels are evaluation labels,
              never indexed evidence.
            </p>
          </section>
        </>
      ) : semanticState ? (
        <EmptyState label="No semantic benchmark report">Run <code>python -m kairos semantic-bench</code> to measure.</EmptyState>
      ) : null}
      {hardening ? (
        <section className="panel" style={{ padding: 0 }} data-testid="hardening-benchmark">
          <header className="panel-head"><span className="px secondary">V2.1 SCALE AND CONCURRENCY · RUN {hardening.run_id}</span></header>
          <div style={{ overflowX: "auto" }}>
            <table className={styles.table}>
              <thead><tr><th>DOCUMENTS</th><th>CHUNKS</th><th>BUILD</th><th>PEAK PYTHON MEMORY</th><th>SEARCH p50 / p95 / p99</th><th>QPS</th></tr></thead>
              <tbody>{hardening.scale?.map((row) => (
                <tr key={row.documents}>
                  <td>{row.documents.toLocaleString()}</td><td>{row.chunks.toLocaleString()}</td>
                  <td>{(row.index_build_ms / 1000).toFixed(2)} s</td>
                  <td>{(row.python_traced_peak_bytes / 1024 / 1024).toFixed(1)} MiB</td>
                  <td>{row.search_p50_ms?.toFixed(1) ?? "—"} / {row.search_p95_ms?.toFixed(1) ?? "—"} / {row.search_p99_ms?.toFixed(1) ?? "—"} ms</td>
                  <td>{row.single_session_qps.toFixed(2)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className={styles.note} style={{ padding: "0 var(--sp-4) var(--sp-4)" }}>
            Pinned official {hardening.dataset?.name} rows. Concurrency isolation: {hardening.concurrency?.every((row) => row.session_isolation) ? "PASS" : "FAIL"}.
            {" "}{hardening.soak?.qualification}. Disk index size is not measured because persistence is not implemented.
          </p>
        </section>
      ) : null}
    </main>
  );
}
