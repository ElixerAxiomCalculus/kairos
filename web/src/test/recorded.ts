/**
 * Story/test helpers built on RECORDED engine traces (web/src/mocks/fixtures). States are produced by
 * folding real events up to a point in the session, so stories show what the engine actually emitted.
 */
import conflictEvents from "@/mocks/fixtures/conflict-demo.events.json";
import puneEvents from "@/mocks/fixtures/pune-demo.events.json";
import reimbursementEvents from "@/mocks/fixtures/reimbursement-demo.events.json";
import { foldEvents, type EngineView } from "@/features/streaming/reducer";
import { parseWireEvent, type WireEvent } from "@/schemas/events";
import { useStream } from "@/stores/streamStore";

function parse(raw: unknown[]): WireEvent[] {
  return raw.flatMap((r) => {
    const p = parseWireEvent(r);
    return p.ok ? [p.event] : [];
  });
}

export const RECORDINGS = {
  pune: parse(puneEvents as unknown[]),
  reimbursement: parse(reimbursementEvents as unknown[]),
  conflict: parse(conflictEvents as unknown[]),
};

export type RecordingKey = keyof typeof RECORDINGS;

/** Index just after the first event matching the predicate (or the full length). */
export function cursorAfter(events: WireEvent[], pred: (e: WireEvent) => boolean, nth = 1): number {
  let seen = 0;
  for (let i = 0; i < events.length; i++) {
    if (pred(events[i])) {
      seen += 1;
      if (seen === nth) return i + 1;
    }
  }
  return events.length;
}

export const at = {
  decision: (state: string, nth = 1) => (e: WireEvent) =>
    e.event_type === "RetrievalDecision" && (e.payload as { decision?: string }).decision === state && nth > 0,
  version: (v: number) => (e: WireEvent) => e.event_type === "AnswerVersionCreated" && e.answer_version === v,
  chunk: (seq: number, utt = "u1") => (e: WireEvent) =>
    e.event_type === "TranscriptChunkReceived" && e.utterance_id === utt && (e.payload as { seq?: number }).seq === seq,
};

export function viewAt(key: RecordingKey, pred: (e: WireEvent) => boolean, nth = 1): EngineView {
  const events = RECORDINGS[key];
  return foldEvents(events, cursorAfter(events, pred, nth));
}

/** Put the stream store into the state of a recording at a given point (for store-connected components). */
export function loadStoreAt(key: RecordingKey, pred: (e: WireEvent) => boolean, nth = 1): void {
  const events = RECORDINGS[key];
  const cursor = cursorAfter(events, pred, nth);
  const view = foldEvents(events, cursor);
  useStream.setState({
    mode: "MOCK",
    connection: "MOCK",
    scriptId: key,
    events,
    cursor,
    view,
    clock: cursor ? events[cursor - 1].timestamp : 0,
    playing: false,
    following: false,
  });
}
