import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { memo, useEffect, useMemo, useRef } from "react";
import { Tip } from "@/components/common/Tip";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import type { WireEvent } from "@/schemas/events";
import { STATUS_VAR } from "@/tokens/motion";
import { clock, ms } from "@/utils/format";
import { DemoControls } from "./DemoControls";
import styles from "./timeline.module.css";

const LANES = [
  { key: "INGEST", types: ["TranscriptChunkReceived", "QueryFrameUpdated", "TurnClassified", "SessionStarted", "SessionEnded"] },
  { key: "CONTROL", types: ["ProbeExecuted", "PrefetchDecision", "PrefetchQueued", "RetrievalDecision", "SubQueriesGenerated", "RetrievalSuppressed"] },
  { key: "RETRIEVAL", types: ["PrefetchStarted", "PrefetchCompleted", "PrefetchSuperseded", "PrefetchDiscarded", "RetrievalStarted", "RetrievalCompleted", "EvidenceFused", "DeltaPlanComputed"] },
  { key: "GROUNDING", types: ["EvidenceValidationStarted", "EvidenceValidated", "EvidenceInvalidated", "EvidenceCommitted", "ClaimsGenerated", "GroundingVerified", "TransformVerified", "TransformUnavailable"] },
  { key: "ANSWER", types: ["AnswerVersionCreated", "TelemetryEmitted"] },
] as const;

function laneOf(t: string): number {
  return Math.max(0, LANES.findIndex((l) => (l.types as readonly string[]).includes(t)));
}

function colorOf(e: WireEvent): string {
  const p = e.payload as Record<string, unknown>;
  switch (e.event_type) {
    case "RetrievalDecision":
      return STATUS_VAR[String(p.decision)] ?? "var(--status-wait)";
    case "RetrievalStarted":
    case "RetrievalCompleted":
      return e.retrieval_trigger === "DELTA" || e.retrieval_trigger === "REFINE" ? "var(--status-refine)" : "var(--status-retrieve)";
    case "PrefetchDecision":
    case "PrefetchQueued":
    case "PrefetchStarted":
    case "PrefetchCompleted":
    case "EvidenceValidationStarted":
      return "var(--status-wait)";
    case "EvidenceValidated":
    case "EvidenceCommitted":
      return "var(--status-pass)";
    case "PrefetchSuperseded":
    case "PrefetchDiscarded":
    case "EvidenceInvalidated":
      return "var(--status-refine)";
    case "DeltaPlanComputed":
      return "var(--status-refine)";
    case "RetrievalSuppressed":
    case "TransformVerified":
      return "var(--status-suppress)";
    case "GroundingVerified":
      return (p.summary as Record<string, unknown>)?.status === "SUPPORTED"
        ? "var(--status-pass)" : "var(--status-wait)";
    case "AnswerVersionCreated":
      return "var(--text-primary)";
    default:
      return "var(--text-muted)";
  }
}

function labelOf(e: WireEvent): string | null {
  const p = e.payload as Record<string, unknown>;
  if (e.event_type === "RetrievalDecision" && p.decision !== "WAIT") return String(p.decision);
  if (e.event_type === "AnswerVersionCreated") return `V${String(e.answer_version).padStart(2, "0")}`;
  if (e.event_type === "RetrievalSuppressed") return "SKIPPED";
  if (e.event_type === "DeltaPlanComputed") return "DELTA PLAN";
  if (e.event_type === "PrefetchQueued") return "PREPARING";
  if (e.event_type === "PrefetchCompleted") return "PROVISIONAL";
  if (e.event_type === "EvidenceValidationStarted") return "VALIDATING";
  if (e.event_type === "EvidenceValidated") return "VERIFIED";
  if (e.event_type === "EvidenceInvalidated") return "INVALIDATED";
  if (e.event_type === "EvidenceCommitted") return "COMMITTED";
  if (e.event_type === "SubQueriesGenerated") return `${(p.active as string[] | undefined)?.join(" ") ?? ""}`;
  return null;
}

function EventTimelineImpl() {
  const events = useStream((s) => s.events);
  const cursor = useStream((s) => s.cursor);
  const clockT = useStream((s) => s.clock);
  const playing = useStream((s) => s.playing);
  const togglePlay = useStream((s) => s.togglePlay);
  const restart = useStream((s) => s.restart);
  const stepEvent = useStream((s) => s.stepEvent);
  const nextChunk = useStream((s) => s.nextChunk);
  const jumpToEvent = useStream((s) => s.jumpToEvent);
  const mode = useStream((s) => s.mode);
  const presentation = useUi((s) => s.presentation);
  const select = useUi((s) => s.select);
  const selection = useUi((s) => s.selection);
  const trackRef = useRef<HTMLDivElement>(null);

  const pps = presentation ? 110 : 84; // pixels per session second
  const duration = Math.max(events.at(-1)?.timestamp ?? 0, clockT, 4) + 1;
  const width = duration * pps + 120;

  // de-overlap events that happen within milliseconds of each other in the same lane
  const placed = useMemo(() => {
    const lastX: number[] = LANES.map(() => -100);
    const lastLabelX: number[] = LANES.map(() => -1000);
    return events.map((e, i) => {
      const lane = laneOf(e.event_type);
      let x = 12 + e.timestamp * pps;
      if (x - lastX[lane] < 11) x = lastX[lane] + 11;
      lastX[lane] = x;
      let label = labelOf(e);
      if (label) {
        if (x - lastLabelX[lane] < label.length * 6.5 + 14) label = null;
        else lastLabelX[lane] = x;
      }
      return { e, i, lane, x, label };
    });
  }, [events, pps]);

  const playheadX = 12 + clockT * pps;
  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    if (playheadX < el.scrollLeft + 40 || playheadX > el.scrollLeft + el.clientWidth - 80) {
      el.scrollTo({ left: Math.max(0, playheadX - el.clientWidth * 0.35), behavior: "smooth" });
    }
  }, [playheadX]);

  return (
    <footer className={styles.timeline} aria-label="Event timeline" data-testid="event-timeline">
      <div className={styles.controls}>
        <span className="px muted">EVENT BUS</span>
        <Tip content="Restart"><button className="btn btn-icon" onClick={restart} aria-label="Restart replay" disabled={mode === "LIVE"}><RotateCcw size={14} /></button></Tip>
        <Tip content="Previous event (←)"><button className="btn btn-icon" onClick={() => stepEvent(-1)} aria-label="Previous event"><ChevronLeft size={15} /></button></Tip>
        <Tip content="Play / pause (space)">
          <button className="btn btn-icon" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"} data-testid="play-toggle" disabled={mode === "LIVE"}>
            {playing ? <Pause size={14} /> : <Play size={14} />}
          </button>
        </Tip>
        <Tip content="Next event (→)"><button className="btn btn-icon" onClick={() => stepEvent(1)} aria-label="Next event" data-testid="step-event"><ChevronRight size={15} /></button></Tip>
        <Tip content="Next transcript chunk"><button className="btn" onClick={nextChunk} aria-label="Next transcript chunk" data-testid="next-chunk"><SkipForward size={14} /><span className="px">CHUNK</span></button></Tip>
        <DemoControls />
        <span className={styles.spacer} />
        <span className={styles.readout} data-testid="timeline-readout">
          {clock(clockT)} · {cursor}/{events.length} events
        </span>
      </div>
      <div className={styles.lanes}>
        <div className={styles.laneLabels} style={{ gridTemplateRows: `repeat(${LANES.length}, 1fr)` }}>
          {LANES.map((l) => <span key={l.key} className="px">{l.key}</span>)}
        </div>
        <div className={styles.track} ref={trackRef}>
          <div className={styles.inner} style={{ width }}>
            {LANES.map((l, i) => (
              <div key={l.key} className={styles.laneLine} style={{ top: `${((i + 1) / LANES.length) * 100}%` }} />
            ))}
            {placed.map(({ e, i, lane, x, label }) => {
              const past = i < cursor;
              const top = `${((lane + 0.5) / LANES.length) * 100}%`;
              return (
                <div key={e.event_id}>
                  <button
                    className={styles.evt}
                    style={{ left: x, top, background: colorOf(e), opacity: past ? 1 : 0.22 }}
                    data-selected={selection?.kind === "event" && selection.id === e.event_id}
                    data-testid="timeline-event"
                    data-event-type={e.event_type}
                    title={`${e.event_id} ${e.event_type} +${ms((e.timestamp - (events[0]?.timestamp ?? 0)) * 1000, 0)}${e.sub_query_id ? ` ${e.sub_query_id}` : ""}${e.latency_ms !== null && e.latency_ms !== undefined ? ` · ${ms(e.latency_ms)}` : ""}`}
                    aria-label={`${e.event_type} at ${clock(e.timestamp)}`}
                    onClick={() => {
                      jumpToEvent(e.event_id);
                      select({ kind: "event", id: e.event_id });
                    }}
                  />
                  {label && past ? <span className={styles.evtLabel} style={{ left: x, top }}>{label}</span> : null}
                </div>
              );
            })}
            <div className={styles.playhead} style={{ left: playheadX }} aria-hidden />
          </div>
        </div>
      </div>
    </footer>
  );
}

export const EventTimeline = memo(EventTimelineImpl);
