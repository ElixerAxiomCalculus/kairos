import { memo } from "react";
import type { RetrievalDecision } from "@/types/domain";
import { STATUS_VAR } from "@/tokens/motion";
import { humanReason, pad, score } from "@/utils/format";
import styles from "./controller.module.css";

const TRIGGER_SHORT: Record<string, string> = { EARLY: "EARLY", END_OF_UTTERANCE: "END", REFINE: "REFINE", DELTA: "DELTA" };

const GLOW: Record<string, string> = {
  RETRIEVE: "var(--glow-retrieve)",
  REFINE: "var(--glow-refine)",
  SUPPRESS: "var(--glow-suppress)",
  WAIT: "transparent",
};

/**
 * The iconic element: always visible, one word in machine type. A state change re-keys the word,
 * border trace and radial pulse, which replay short CSS animations (compositor-driven, so they never
 * stall mid-frame) — deliberate, never a full-screen flash.
 */
function RetrievalControllerImpl({ decision, queryVersion }: { decision: RetrievalDecision | null; queryVersion: number }) {
  const state = decision?.state ?? "WAIT";
  const color = STATUS_VAR[state];
  const s = decision?.signals ?? {};
  const active = decision !== null && state !== "WAIT";

  return (
    <div className={styles.controller} data-testid="retrieval-controller" data-state={state}
      style={{ borderColor: state === "WAIT" ? undefined : color }} aria-live="polite">
      {active ? (
        <>
          <div key={`pulse-${state}`} className={styles.pulse} style={{ background: `radial-gradient(circle, ${GLOW[state]}, transparent 70%)` }} aria-hidden />
          <span key={`trace-${state}`} className={styles.trace} style={{ borderColor: color }} aria-hidden />
        </>
      ) : null}
      <div className={`${styles.ctlHead} px`}>
        <span>CONTROLLER / {pad(queryVersion)}</span>
        <span>{decision?.trigger && decision.trigger !== "NONE" ? TRIGGER_SHORT[decision.trigger] : ""}</span>
      </div>
      <div className={styles.state}>
        <span key={state} className={styles.stateWord} style={{ color: state === "WAIT" ? "var(--text-secondary)" : color }}
          data-testid="controller-state">
          {state}
        </span>
      </div>
      <div className={styles.readout}>
        <span>CONFIDENCE</span><span>{score(decision?.confidence ?? s.s_hat)}</span>
        <span>THRESHOLD</span><span>{score(s.theta)}</span>
        <span>STABILITY ρ</span><span>{score(s.rho)}</span>
        <span>TURN</span><span>{decision?.turnType?.replace("_", " ") ?? "—"}</span>
      </div>
      <div className={styles.reasons} aria-label="Decision reasons">
        {(decision?.reasons ?? ["AWAITING_SPEECH"]).map((r) => (
          <span key={r} className={styles.reason}>{r === "AWAITING_SPEECH" ? "awaiting speech" : humanReason(r)}</span>
        ))}
      </div>
    </div>
  );
}

export const RetrievalController = memo(RetrievalControllerImpl);
