import { motion } from "framer-motion";
import { memo } from "react";
import { StatusPill } from "@/components/common/StatusPill";
import type { TranscriptChunk as Chunk } from "@/types/domain";
import { transition } from "@/tokens/motion";
import { clock, score } from "@/utils/format";
import styles from "./transcript.module.css";

interface Props {
  chunk: Chunk;
  latest: boolean;
  old: boolean;
  lead?: number | null;
  showCursor: boolean;
}

function TranscriptChunkImpl({ chunk, latest, old, showCursor }: Props) {
  return (
    <motion.div
      className={styles.chunk}
      data-latest={latest}
      data-old={old}
      data-testid="transcript-chunk"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: old ? 0.62 : 1, y: 0 }}
      transition={transition.normal}
    >
      <span className={styles.chunkTime}>{clock(chunk.timestamp)}<br />#{String(chunk.sequence).padStart(2, "0")}</span>
      <div>
        <span className={styles.chunkText}>
          {chunk.text}
          {chunk.retrievalFired && chunk.decision ? (
            <span className={styles.fired} data-testid="retrieval-marker">
              <StatusPill state={chunk.decision} label={chunk.decision === "REFINE" ? "⟳ REFINE" : "⚡ RETRIEVE"} />
            </span>
          ) : null}
          {showCursor ? <span className={`${styles.cursor} blink`} aria-hidden /> : null}
        </span>
        {chunk.stability !== undefined ? (
          <div className={styles.chunkMeta}>
            <span>ŝ {score(chunk.stability)}</span>
            <span>σ {score(chunk.entityStability)}</span>
            <span>ν {score(chunk.novelty)}</span>
            <span>κ {score(chunk.semanticCompleteness)}</span>
          </div>
        ) : null}
      </div>
    </motion.div>
  );
}

export const TranscriptChunk = memo(TranscriptChunkImpl);
