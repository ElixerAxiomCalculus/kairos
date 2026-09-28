import { useMemo } from "react";
import { Panel } from "@/components/common/Panel";
import { useStream } from "@/stores/streamStore";
import type { Counterfactual } from "@/types/domain";
import { STATUS_VAR } from "@/tokens/motion";
import { DASH, int, ms, pad, ratio, score } from "@/utils/format";
import { GroundingFirewall } from "./GroundingFirewall";
import { LatencySparkline } from "./LatencySparkline";
import { MetricCard } from "./MetricCard";
import styles from "./telemetry.module.css";

/** Live telemetry as typography + sparklines. Every number maps to an event field; absent -> "—". */
export function TelemetryGrid({ className }: { className?: string }) {
  const t = useStream((s) => s.view.telemetry);
  const versions = useStream((s) => s.view.versions);
  const grounding = useStream((s) => s.view.grounding);
  const intents = useStream((s) => s.view.intentOrder.length);
  const controller = useStream((s) => s.view.controller);
  const queryVersions = useStream((s) => s.view.queryVersions.length);
  const events = useStream((s) => s.view.events);
  const suppression = useStream((s) => s.view.suppression);

  const latest = versions.at(-1);
  const lastRet = t.retrievalLatencies.at(-1);
  const lastPost = t.postUtteranceLatencies.at(-1);
  const g = grounding.at(-1) ?? null;
  const transform = useMemo(() => {
    if (latest?.kind !== "PRESENTATION") return null;
    return [...events].reverse().find((e) => e.eventType === "TransformVerified") ?? null;
  }, [events, latest]);
  const cf = (latest?.trigger === "CONSTRAINT" ? latest.counterfactual : undefined) as Counterfactual | undefined;
  const hasCf = cf && typeof cf.full_restart_retrievals === "number";
  const groundScore = g && g.checked ? g.emitted / g.checked : undefined;

  return (
    <Panel label="TELEMETRY" meta={events.length ? `${events.length} events` : undefined} className={className} testId="panel-telemetry">
      <div className={styles.column}>
        {events.length === 0 ? <div className="px muted" style={{ padding: "4px 0" }}>SESSION TELEMETRY INITIALIZING</div> : null}
        <div className={styles.grid}>
          <MetricCard label="Retrieval latency" value={lastRet ? ms(lastRet.ms) : DASH} testId="metric-retrieval-latency"
            detail={lastRet?.label} hint="Wall time of the last hybrid retrieval (BM25 + dense + fusion + rerank + dedup).">
            <LatencySparkline data={t.retrievalLatencies} color={STATUS_VAR.RETRIEVE} />
          </MetricCard>
          <MetricCard label="Post-utterance" value={lastPost ? ms(lastPost.ms) : DASH} testId="metric-post-latency"
            detail={t.lastLeadSeconds !== null && t.lastLeadSeconds !== undefined && t.lastLeadSeconds > 0 ? `retrieved ${t.lastLeadSeconds.toFixed(2)} s before end` : "speech end → answer"}
            hint="Time from the final transcript chunk to the answer version (end-to-end after speech).">
            <LatencySparkline data={t.postUtteranceLatencies} />
          </MetricCard>
          <MetricCard label="Tokens" value={int(t.llmTokens)} unit="llm" testId="metric-tokens"
            detail={`context ≈ ${int(t.contextTokensEst)} tok (est)`}
            hint="LLM prompt+completion tokens reported by the synthesis backend (0 in extractive mode), plus an estimate of evidence context processed." />
          <MetricCard label="Retrievals" value={int(t.retrievalCount)} testId="metric-retrievals"
            detail={`early ${t.earlyRetrievalCount} · refine ${t.refineCount} · delta ${t.deltaCount}`} />
          <MetricCard label="Grounding" value={groundScore !== undefined ? ratio(groundScore) : DASH} testId="metric-grounding"
            accent={g ? (g.status === "PASS" ? STATUS_VAR.PASS : STATUS_VAR.FAIL) : undefined}
            detail={g ? `${g.emitted}/${g.checked} claims · ${g.citationsValid} citations` : undefined} />
          <MetricCard label="Delta ratio" value={hasCf ? `${cf.delta_retrievals}/${cf.full_restart_retrievals}` : DASH}
            detail={hasCf ? `saved ${ratio(cf.retrieval_savings_ratio)} vs full restart` : "no late constraint yet"} testId="metric-delta" />
          <MetricCard label="Intents · docs" value={`${intents} · ${t.documentsTouched}`} detail={`${t.probeCount} probes (BM25)`} />
          <MetricCard label="Versions" value={`Q${pad(queryVersions)} · V${pad(latest?.id ?? 0)}`}
            detail={`ŝ ${score(controller?.confidence)} · reuse ${suppression ? suppression.evidenceReused : 0} chunks`} />
        </div>
        <GroundingFirewall result={g} transform={transform} />
      </div>
    </Panel>
  );
}
