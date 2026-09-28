import { memo } from "react";
import { Tip } from "@/components/common/Tip";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import { citationLabel, score } from "@/utils/format";
import styles from "./answer.module.css";

/** A citation is an object: hover previews provenance, click opens the evidence inspector and lights the graph path. */
function CitationAnchorImpl({ chunkId, claimId, support, verdict }: { chunkId: string; claimId: string; support: number; verdict: string }) {
  const evidence = useStream((s) => s.view.evidence[chunkId]);
  const select = useUi((s) => s.select);
  const setHover = useUi((s) => s.setHover);
  const selection = useUi((s) => s.selection);
  const active = selection?.kind === "chunk" && selection.id === chunkId;
  return (
    <Tip
      content={
        <div>
          <div className="px muted" style={{ marginBottom: 4 }}>{chunkId}</div>
          <div style={{ fontWeight: 500 }}>{evidence?.docTitle ?? "Unknown document"}</div>
          <div className="secondary">§{evidence?.section} {evidence?.sectionPath}</div>
          <div className="mono" style={{ marginTop: 6 }}>rerank {score(evidence?.rerankScore)} · support {score(support)} · {verdict}</div>
        </div>
      }
    >
      <button
        type="button"
        className={styles.cite}
        data-active={active}
        data-verdict={verdict}
        data-testid="citation-anchor"
        aria-label={`Citation ${chunkId}, ${evidence?.docTitle ?? ""}`}
        onMouseEnter={() => setHover({ kind: "chunk", id: chunkId, claimId })}
        onMouseLeave={() => setHover(null)}
        onFocus={() => setHover({ kind: "chunk", id: chunkId, claimId })}
        onBlur={() => setHover(null)}
        onClick={(e) => {
          e.stopPropagation();
          select({ kind: "chunk", id: chunkId, claimId });
        }}
      >
        {citationLabel(chunkId, evidence)}
      </button>
    </Tip>
  );
}

export const CitationAnchor = memo(CitationAnchorImpl);
