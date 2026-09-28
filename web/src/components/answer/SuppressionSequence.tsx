import { motion, useReducedMotion } from "framer-motion";
import type { SuppressionInfo } from "@/types/domain";
import { humanReason } from "@/utils/format";
import styles from "./answer.module.css";

const LABEL: Record<string, string> = {
  VECTOR_SEARCH: "Vector search",
  BM25: "Sparse search",
  RERANKER: "Reranker",
  FUSION: "Fusion",
};

/** SUPPRESS as a compact sequence: each pipeline stage shown as skipped, then evidence reuse. */
export function SuppressionSequence({ info }: { info: SuppressionInfo }) {
  const reduced = useReducedMotion();
  const steps = [...info.skipped.map((s) => ({ k: LABEL[s] ?? s, state: "SKIPPED", reuse: false })),
    { k: "Existing evidence", state: `REUSED · ${info.evidenceReused}`, reuse: true }];
  return (
    <div className={styles.banner} data-kind="suppress" data-testid="suppression-sequence">
      <div className={styles.bannerCell}>
        <span className={`px ${styles.k}`}>Controller</span>
        <span className={styles.v} style={{ color: "var(--status-suppress)" }}>SUPPRESS</span>
      </div>
      <div className={styles.bannerCell}>
        <span className={`px ${styles.k}`}>Reason</span>
        <span className={styles.v}>{humanReason(info.reason).toUpperCase()}</span>
      </div>
      <div className={styles.bannerCell}>
        <span className={`px ${styles.k}`}>Reused answer</span>
        <span className={styles.v}>{info.reusedAnswerVersion ? `V${String(info.reusedAnswerVersion).padStart(2, "0")}` : "—"}</span>
      </div>
      <div className={styles.skipList} aria-label={humanReason(info.reason)}>
        {steps.map((s, i) => (
          <motion.span key={s.k} className={`${styles.skip} ${s.reuse ? styles.reuse : ""}`}
            initial={reduced ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: reduced ? 0 : 0.12 * i, duration: 0.2 }}>
            <span className="px muted">{s.k}</span>
            <span className={`px ${"state"}`} style={{ color: s.reuse ? "var(--status-pass)" : "var(--status-suppress)" }}>{s.state}</span>
          </motion.span>
        ))}
      </div>
    </div>
  );
}
