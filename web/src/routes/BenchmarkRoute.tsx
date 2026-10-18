import { useEffect, useState } from "react";
import { GateTable, MetricGrid, SemanticMetricGrid } from "@/components/benchmark/BenchmarkPanel";
import { EmptyState } from "@/components/common/EmptyState";
import {
  loadBenchmark,
  loadSemanticBenchmark,
  type BenchmarkReport,
  type SemanticBenchmarkReport,
  type Source,
} from "@/features/replay/benchmarkData";
import styles from "./routes.module.css";

type ReportState<T> = { report: T | null; source: Source; error?: string };

const corpora = [
  {
    name: "SciFact via BEIR",
    use: "12 documents / 29 chunks",
    license: "CC BY-SA 4.0",
    dataset: "https://huggingface.co/datasets/BeIR/scifact",
    citation: "Wadden et al. (EMNLP 2020); Thakur et al. (NeurIPS 2021)",
    paper: "https://aclanthology.org/2020.emnlp-main.609/",
  },
  {
    name: "SQuAD 1.1",
    use: "8 contexts / 16 chunks; 12 Q&A labels excluded from retrieval",
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
  useEffect(() => {
    void Promise.all([loadBenchmark(), loadSemanticBenchmark()]).then(([development, semantic]) => {
      setState(development);
      setSemanticState(semantic);
    });
  }, []);
  const r = state?.report ?? null;
  const semantic = semanticState?.report ?? null;
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
      {state && !r ? <EmptyState label="No benchmark report">Run <code>python -m kairos bench</code> to measure.</EmptyState> : null}
      {r ? (
        <>
          <section className="panel" style={{ padding: 0 }}>
            <header className="panel-head"><span className="px secondary">CURRENT V2 DEVELOPMENT GATES · TARGET VS MEASURED</span></header>
            <div style={{ overflowX: "auto" }}><GateTable report={r} /></div>
          </section>
          <MetricGrid report={r} />
          <p className={styles.note}>
            {r.dataset?.note} {r.judge ? `Judge: ${r.judge}` : ""} The dev set is small and was authored alongside the
            system; treat these as development measurements, not a claim about the official benchmark.
          </p>
        </>
      ) : null}
      {semantic ? (
        <>
          <section className="panel" style={{ padding: 0 }} data-testid="semantic-benchmark">
            <header className="panel-head">
              <span className="px secondary">V2 SEMANTIC RELIABILITY · RUN {semantic.run_id}</span>
            </header>
            <div style={{ padding: "var(--sp-4)" }}>
              <SemanticMetricGrid report={semantic} />
              <p className={styles.note}>
                {semantic.dataset?.note} Intent recall 95% CI: {semantic.metrics?.intent.intent_recall_ci95.join("–")}.
                Relevant-evidence precision and SQuAD retrieval are shown even though they are below 1.0.
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
              revision and SHA-256 provenance; SQuAD questions and answers are evaluation labels, never indexed evidence.
            </p>
          </section>
        </>
      ) : semanticState ? (
        <EmptyState label="No semantic benchmark report">Run <code>python -m kairos semantic-bench</code> to measure.</EmptyState>
      ) : null}
    </main>
  );
}
