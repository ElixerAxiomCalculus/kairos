import { memo } from "react";
import { StatusPill } from "@/components/common/StatusPill";
import type { AnswerVersion, GroundingResult } from "@/types/domain";

/** Compact firewall verdict pinned to the answer, so grounding is visible in every focus mode. */
function GroundingStripImpl({ version, grounding }: { version: AnswerVersion; grounding: GroundingResult[] }) {
  if (version.kind === "PRESENTATION") {
    return <StatusPill state="PASS" label="TRANSFORM FIREWALL · NO NEW FACTS" />;
  }
  const g = grounding.find((x) => x.answerVersion === version.id);
  if (!g) return <StatusPill state="WAIT" label="GROUNDING —" />;
  const state = g.status === "SUPPORTED" ? "SUPPORTED"
    : g.status === "CONFLICTING" ? "CONFLICTING"
      : g.status === "PARTIAL" ? "PARTIAL"
        : g.status === "INSUFFICIENT_EVIDENCE" ? "WAIT" : "FAIL";
  const label = g.status.replace(/_/g, " ");
  return (
    <span data-testid="grounding-strip">
      <StatusPill state={state}
        label={`${label} · ${g.emitted}/${g.checked} VERIFIED · ${g.fabricatedIdsBlocked} FABRICATED`} />
    </span>
  );
}

export const GroundingStrip = memo(GroundingStripImpl);
