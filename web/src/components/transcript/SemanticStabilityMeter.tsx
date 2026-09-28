import * as d3 from "d3";
import { motion } from "framer-motion";
import { memo, useMemo } from "react";
import { MicroBar } from "@/components/common/MicroBar";
import type { RetrievalDecision } from "@/types/domain";
import { STATUS_VAR, transition } from "@/tokens/motion";
import { score } from "@/utils/format";
import styles from "./transcript.module.css";

const W = 260;
const TRACE_H = 46;
const GAUGE_H = 22;

interface Props {
  decisions: RetrievalDecision[]; // current utterance, in order
  current: RetrievalDecision | null;
}

/**
 * An instrument, not a progress bar: an oscilloscope trace of ŝ across chunks with the adaptive
 * threshold θ, trigger ticks where retrieval fired, and a gauge showing ŝ against θ now.
 */
function SemanticStabilityMeterImpl({ decisions, current }: Props) {
  const withSignal = decisions.filter((d) => d.signals.s_hat !== undefined);
  const sHat = current?.signals.s_hat;
  const theta = current?.signals.theta ?? withSignal.at(-1)?.signals.theta;
  const x = useMemo(
    () => d3.scaleLinear().domain([0, Math.max(1, withSignal.length - 1)]).range([4, W - 4]),
    [withSignal.length],
  );
  const y = d3.scaleLinear().domain([0, 1]).range([TRACE_H - 3, 3]);
  const path = d3
    .line<RetrievalDecision>()
    .x((_, i) => x(i))
    .y((d) => y(d.signals.s_hat ?? 0))
    .curve(d3.curveMonotoneX)(withSignal);
  const stepTheta = d3
    .line<RetrievalDecision>()
    .x((_, i) => x(i))
    .y((d) => y(d.signals.theta ?? 0))
    .curve(d3.curveStepAfter)(withSignal);
  const g = d3.scaleLinear().domain([0, 1]).range([6, W - 6]);
  const s = current?.signals ?? {};
  const color = STATUS_VAR[current?.state ?? "WAIT"];

  return (
    <div className={styles.meter} data-testid="stability-meter">
      <div className={styles.meterHead}>
        <span className="px muted">Semantic stability ŝ</span>
        <span>
          <span className={styles.meterValue} data-testid="stability-value">{score(sHat)}</span>
          <span className={`mono ${styles.meterTheta}`}> / θ {score(theta)}</span>
        </span>
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${TRACE_H + GAUGE_H}`} role="img"
        aria-label={`Stability ${score(sHat)} against threshold ${score(theta)}`}>
        {[0.25, 0.5, 0.75].map((v) => (
          <line key={v} x1={0} x2={W} y1={y(v)} y2={y(v)} stroke="var(--grid-line)" />
        ))}
        {stepTheta ? <path d={stepTheta} fill="none" stroke="var(--text-muted)" strokeDasharray="3 3" strokeWidth={1} /> : null}
        {path ? <motion.path d={path} fill="none" stroke="var(--text-primary)" strokeWidth={1.4}
          initial={false} animate={{ d: path }} transition={transition.normal} /> : null}
        {withSignal.map((d, i) =>
          d.state === "RETRIEVE" || d.state === "REFINE" ? (
            <g key={d.eventId}>
              <line x1={x(i)} x2={x(i)} y1={2} y2={TRACE_H - 2} stroke={STATUS_VAR[d.state]} strokeWidth={1} opacity={0.8} />
              <circle cx={x(i)} cy={y(d.signals.s_hat ?? 0)} r={2.6} fill={STATUS_VAR[d.state]} />
            </g>
          ) : (
            <circle key={d.eventId} cx={x(i)} cy={y(d.signals.s_hat ?? 0)} r={1.6} fill="var(--text-secondary)" />
          ),
        )}
        {/* gauge: ────●────│──── */}
        <g transform={`translate(0 ${TRACE_H + 6})`}>
          <line x1={6} x2={W - 6} y1={6} y2={6} stroke="var(--border-strong)" />
          {d3.range(0, 1.01, 0.1).map((t) => (
            <line key={t} x1={g(t)} x2={g(t)} y1={3} y2={9} stroke="var(--border-strong)" strokeWidth={t === 0.5 ? 1 : 0.6} />
          ))}
          {theta !== undefined ? (
            <motion.line initial={false} animate={{ x1: g(theta), x2: g(theta) }} transition={transition.normal}
              y1={-1} y2={13} stroke="var(--text-primary)" strokeWidth={1.4} />
          ) : null}
          {sHat !== undefined ? (
            <motion.circle initial={false} animate={{ cx: g(sHat) }} transition={transition.normal} cy={6} r={4.5}
              fill={color} stroke="var(--surface-raised)" strokeWidth={1.5} />
          ) : null}
        </g>
      </svg>
      <div className={styles.bars}>
        <MicroBar name="Complete" value={s.kappa} hint="κ completeness: syntactic closure, content, request cue" />
        <MicroBar name="Clarity" value={s.chi} hint="χ intent confidence: 1 − normalised entropy of probe scores" />
        <MicroBar name="Slots" value={s.sigma} hint="σ entity/slot stability since the previous chunk" />
        <MicroBar name="Results" value={s.rho} hint="ρ result stability: RBO of consecutive probe rankings (information gain = 1 − ρ)" />
        <MicroBar name="Novelty" value={s.nu} hint="ν novelty: 1 − similarity to queries already retrieved this session" />
      </div>
    </div>
  );
}

export const SemanticStabilityMeter = memo(SemanticStabilityMeterImpl);
