import { useEffect, useMemo, useRef } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { Panel } from "@/components/common/Panel";
import { StatusPill } from "@/components/common/StatusPill";
import { useStream } from "@/stores/streamStore";
import { SemanticStabilityMeter } from "./SemanticStabilityMeter";
import { TranscriptChunk } from "./TranscriptChunk";
import { TranscriptInput } from "./TranscriptInput";
import styles from "./transcript.module.css";

const TURN_LABEL: Record<string, string> = {
  NEW_INFO: "NEW INFORMATION",
  CONSTRAINT: "LATE CONSTRAINT",
  TRANSFORM: "PRESENTATION ONLY",
  ELABORATE: "ELABORATE",
};
const TURN_STATE: Record<string, string> = { NEW_INFO: "RETRIEVE", CONSTRAINT: "REFINE", TRANSFORM: "SUPPRESS", ELABORATE: "WAIT" };

export function LiveTranscript({ className }: { className?: string }) {
  const utterances = useStream((s) => s.view.utterances);
  const order = useStream((s) => s.view.utteranceOrder);
  const chunks = useStream((s) => s.view.chunks);
  const decisions = useStream((s) => s.view.decisions);
  const controller = useStream((s) => s.view.controller);
  const mode = useStream((s) => s.mode);
  const scrollRef = useRef<HTMLDivElement>(null);

  const lastUtt = order.at(-1);
  const uttDecisions = useMemo(() => decisions.filter((d) => d.utteranceId === lastUtt), [decisions, lastUtt]);
  const chunkCount = Object.keys(chunks).length;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [chunkCount]);

  const latestChunkId = lastUtt ? utterances[lastUtt]?.chunkIds.at(-1) : undefined;

  return (
    <Panel label="LIVE TRANSCRIPT" meta={lastUtt ? `${order.length} utterance${order.length === 1 ? "" : "s"}` : undefined}
      className={className} noPadding testId="panel-transcript">
      <div className={styles.column}>
        <SemanticStabilityMeter decisions={uttDecisions} current={controller} />
        <div className={styles.stream} ref={scrollRef} aria-live="polite">
          {order.length === 0 ? (
            <EmptyState label="Listening for query stream" animate>
              Speech arrives as timestamped chunks. Each one is scored before anything is retrieved.
            </EmptyState>
          ) : (
            order.map((uid) => {
              const u = utterances[uid];
              return (
                <div key={uid} className={styles.utterance} data-testid="utterance">
                  <div className={styles.uttHead}>
                    <span className="px">{uid.split(":").pop()?.toUpperCase()}</span>
                    {u.turnType ? <StatusPill state={TURN_STATE[u.turnType]} label={TURN_LABEL[u.turnType]} /> : null}
                  </div>
                  {u.chunkIds.map((cid) => (
                    <TranscriptChunk key={cid} chunk={chunks[cid]} latest={cid === latestChunkId}
                      old={uid !== lastUtt} showCursor={cid === latestChunkId && !u.closed} />
                  ))}
                </div>
              );
            })
          )}
        </div>
        {mode === "LIVE" ? <TranscriptInput /> : null}
      </div>
    </Panel>
  );
}
