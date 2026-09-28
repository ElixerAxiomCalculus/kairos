import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatMetric, METRICS } from "@/components/benchmark/BenchmarkPanel";
import { StatusPill } from "@/components/common/StatusPill";
import { loadExperiments, type ExperimentsReport, type Source } from "@/features/replay/benchmarkData";
import styles from "./routes.module.css";

const CHART_METRIC: Record<string, string> = {
  A: "early_retrieval_rate",
  B: "gold_section_mrr",
  C: "update_retrievals",
  E: "intent_recall",
};

const unitOf = (key: string) => METRICS.find((m) => m.key === key)?.unit ?? "count";

/** Ablations: hypothesis, arms, measured metrics. Experiments that were not run show empty slots. */
export default function ExperimentsRoute() {
  const [state, setState] = useState<{ report: ExperimentsReport | null; source: Source } | null>(null);
  useEffect(() => {
    void loadExperiments().then(setState);
  }, []);
  const exps = state?.report?.experiments ?? {};
  return (
    <main className={styles.page} data-testid="experiments-page">
      <div className={styles.pageHead}>
        <div>
          <span className="px muted">ABLATION STUDIES</span>
          <h1 className={styles.pageTitle}>What each component <em>buys</em></h1>
        </div>
        <div className={styles.provenance}>
          {state?.report ? <>MEASURED · run {state.report.run_id} · {state.report.dataset}<br />source: {state.source === "BACKEND" ? "live backend" : "bundled snapshot"}</> : "NOT MEASURED · run python -m kairos ablate"}
        </div>
      </div>
      <div className={styles.experiments}>
        {Object.entries(exps).map(([key, exp]) => {
          const arms = Object.entries(exp.arms ?? {});
          const metric = CHART_METRIC[key];
          const chart = arms.map(([name, a]) => ({ name, value: a.metrics[metric] ?? 0, missing: a.metrics[metric] === null }));
          return (
            <article key={key} className={styles.exp} data-testid={`experiment-${key}`}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="px muted">EXPERIMENT {key}</span>
                <StatusPill state={exp.status === "MEASURED" ? "PASS" : "WAIT"} label={exp.status.replace("_", " ")} />
              </div>
              <h3>{exp.title}</h3>
              <p className={styles.hyp}><span className="px muted">HYPOTHESIS </span>{exp.hypothesis}</p>
              {exp.status === "MEASURED" && metric ? (
                <div style={{ height: 150 }} aria-label={`${metric} by arm`}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chart} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
                      <CartesianGrid vertical={false} stroke="var(--border)" />
                      <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--text-muted)", fontFamily: "var(--font-mono)" }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
                      <Tooltip cursor={{ fill: "var(--surface-hover)" }} contentStyle={{ background: "var(--surface-raised)", border: "1px solid var(--border-strong)", fontSize: 12 }}
                        formatter={(v: number) => [formatMetric(v, unitOf(metric)), METRICS.find((m) => m.key === metric)?.label ?? metric]} />
                      <Bar dataKey="value" fill="var(--text-secondary)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : null}
              {exp.status === "MEASURED" ? (
                <div style={{ overflowX: "auto" }}>
                  <table className={styles.table}>
                    <thead><tr><th>ARM</th>{(exp.metrics ?? []).map((m) => <th key={m}>{METRICS.find((d) => d.key === m)?.label ?? m.replace(/_/g, " ")}</th>)}</tr></thead>
                    <tbody>
                      {arms.map(([name, a]) => (
                        <tr key={name}>
                          <td className="mono">{name}</td>
                          {(exp.metrics ?? []).map((m) => <td key={m} className="mono">{formatMetric(a.metrics[m], unitOf(m))}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className={styles.slot}>
                  <div className="px">RESULT SLOT · NOT RUN</div>
                  <p style={{ margin: "6px 0 0", fontSize: 13 }}>{exp.reason}</p>
                </div>
              )}
            </article>
          );
        })}
      </div>
      <p className={styles.note}>Each arm replays the same dev sessions with one component swapped. Interpretation belongs in the brief: where Kairos loses (for example fixed 4-token triggers retrieve early more often, at a higher premature rate and more retrievals), it is reported rather than hidden.</p>
    </main>
  );
}
