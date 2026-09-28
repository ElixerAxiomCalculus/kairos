import { useEffect, useState } from "react";
import { GateTable, MetricGrid } from "@/components/benchmark/BenchmarkPanel";
import { EmptyState } from "@/components/common/EmptyState";
import { loadBenchmark, type BenchmarkReport, type Source } from "@/features/replay/benchmarkData";
import styles from "./routes.module.css";

/** Target vs measured, never blurred. Provenance of every number is on screen. */
export default function BenchmarkRoute() {
  const [state, setState] = useState<{ report: BenchmarkReport | null; source: Source; error?: string } | null>(null);
  useEffect(() => {
    void loadBenchmark().then(setState);
  }, []);
  const r = state?.report ?? null;
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
              <br />source: {state.source === "BACKEND" ? "live backend /api/benchmark/latest" : "bundled snapshot of the last measured run"}
            </>
          ) : "NOT MEASURED · run python -m kairos bench"}
        </div>
      </div>
      {state?.error ? <div className="px" style={{ color: "var(--status-fail)" }}>{state.error}</div> : null}
      {state && !r ? <EmptyState label="No benchmark report">Run <code>python -m kairos bench</code> to measure.</EmptyState> : null}
      {r ? (
        <>
          <section className="panel" style={{ padding: 0 }}>
            <header className="panel-head"><span className="px secondary">GATES · TARGET VS MEASURED</span></header>
            <div style={{ overflowX: "auto" }}><GateTable report={r} /></div>
          </section>
          <MetricGrid report={r} />
          <p className={styles.note}>
            {r.dataset?.note} {r.judge ? `Judge: ${r.judge}` : ""} The dev set is small and was authored alongside the
            system; treat these as development measurements, not a claim about the official benchmark.
          </p>
        </>
      ) : null}
    </main>
  );
}
