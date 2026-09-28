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
  const pass = g.status === "PASS";
  return (
    <span data-testid="grounding-strip">
      <StatusPill state={pass ? "PASS" : "FAIL"}
        label={`GROUNDING ${pass ? "PASS" : "BLOCKED"} · ${g.emitted}/${g.checked} · ${g.fabricatedIdsBlocked} FABRICATED`} />
    </span>
  );
}

export const GroundingStrip = memo(GroundingStripImpl);
