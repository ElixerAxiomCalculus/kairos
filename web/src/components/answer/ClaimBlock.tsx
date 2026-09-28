import { motion } from "framer-motion";
import { memo } from "react";
import { useUi } from "@/stores/uiStore";
import type { Claim } from "@/types/domain";
import { transition } from "@/tokens/motion";
import { CitationAnchor } from "./CitationAnchor";
import styles from "./answer.module.css";

/** One verified claim. Boundaries appear on hover; click reveals claim → evidence → citation lineage. */
function ClaimBlockImpl({ claim, showDiff }: { claim: Claim; showDiff: boolean }) {
  const select = useUi((s) => s.select);
  const setHover = useUi((s) => s.setHover);
  const selected = useUi((s) => (s.selection?.kind === "claim" && s.selection.id === claim.id) ||
    (s.selection?.kind === "chunk" && s.selection.claimId === claim.id));
  const status = showDiff ? claim.status : "NEW";
  return (
    <motion.span
      className={styles.claim}
      data-status={status}
      data-verdict={claim.supportStatus}
      data-selected={selected}
      data-testid="claim"
      data-claim-id={claim.id}
      role="button"
      tabIndex={0}
      aria-label={`Claim ${claim.id}, ${claim.supportStatus.toLowerCase()}`}
      onClick={() => select({ kind: "claim", id: claim.id })}
      onKeyDown={(e) => e.key === "Enter" && select({ kind: "claim", id: claim.id })}
      onMouseEnter={() => setHover({ kind: "claim", id: claim.id })}
      onMouseLeave={() => setHover(null)}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={transition.normal}
      title={`${claim.id} · ${claim.supportStatus} · ${claim.evidenceIds.length} supporting chunk${claim.evidenceIds.length === 1 ? "" : "s"}`}
    >
      {claim.text}
      {claim.evidenceIds.map((e) => (
        <CitationAnchor key={e} chunkId={e} claimId={claim.id} support={claim.support} verdict={claim.supportStatus} />
      ))}{" "}
    </motion.span>
  );
}

export const ClaimBlock = memo(ClaimBlockImpl);
