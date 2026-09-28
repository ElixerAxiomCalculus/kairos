import { motion } from "framer-motion";
import type { AnswerVersion, Counterfactual, DeltaPlan } from "@/types/domain";
import { transition } from "@/tokens/motion";
import { int, ratio } from "@/utils/format";
import styles from "./answer.module.css";

/**
 * Makes selective retrieval legible: which constraint, how many claims were invalidated, and how the
 * delta compares with a full restart. Savings are shown only when the backend reported both numbers.
 */
export function DeltaBanner({ version, delta }: { version: AnswerVersion; delta: DeltaPlan | null }) {
  const cf = version.counterfactual as Counterfactual | undefined;
  const hasCf = cf && typeof cf.full_restart_retrievals === "number" && typeof cf.delta_retrievals === "number";
  const plan = delta && delta.utteranceId === version.utteranceId ? delta : null;
  const invalidated = plan ? plan.claimsInvalidated : version.diff.revised.length + version.diff.retracted.length;
  const total = plan ? plan.claimsTotal : undefined;
  return (
    <motion.div className={styles.banner} data-kind="refine" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
      transition={transition.normal} data-testid="delta-banner">
      <div className={styles.bannerCell}>
        <span className={`px ${styles.k}`}>Late constraint</span>
        <span className={styles.v}>{(version.constraint?.terms ?? []).join(" ").toUpperCase() || "—"}</span>
      </div>
      <div className={styles.bannerCell}>
        <span className={`px ${styles.k}`}>Claims invalidated</span>
        <span className={styles.v} data-testid="claims-invalidated">{invalidated}{total !== undefined ? ` / ${total}` : ""}</span>
      </div>
      <div className={styles.bannerCell}>
        <span className={`px ${styles.k}`}>Carried verbatim</span>
        <span className={styles.v}>{version.diff.kept.length}</span>
      </div>
      {hasCf ? (
        <>
          <div className={styles.bannerCell}>
            <span className={`px ${styles.k}`}>Delta retrieval</span>
            <span className={styles.v} data-testid="delta-retrievals">{int(cf.delta_retrievals)} ops</span>
          </div>
          <div className={styles.bannerCell}>
            <span className={`px ${styles.k}`}>Full restart</span>
            <span className={styles.v}>{int(cf.full_restart_retrievals)} ops</span>
          </div>
          <div className={styles.bannerCell}>
            <span className={`px ${styles.k}`}>Saved</span>
            <span className={styles.v}>{ratio(cf.retrieval_savings_ratio)} · ctx {int(cf.full_restart_context_tokens_est - cf.delta_context_tokens_est)} tok est</span>
          </div>
        </>
      ) : null}
    </motion.div>
  );
}
