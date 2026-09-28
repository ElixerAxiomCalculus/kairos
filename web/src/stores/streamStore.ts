import { create } from "zustand";
import { applyEvent, foldEvents, initialView, type EngineView } from "@/features/streaming/reducer";
import type { WireEvent } from "@/schemas/events";
import type { ConnectionState } from "@/types/domain";

/**
 * Event log + cursor. `view` is always fold(events[0..cursor]). Moving the cursor forward applies events
 * incrementally; moving it backward refolds from the start (cheap for demo-sized sessions).
 */
export interface StreamState {
  mode: "MOCK" | "LIVE";
  connection: ConnectionState;
  scriptId: string | null;
  events: WireEvent[];
  cursor: number;
  view: EngineView;
  clock: number;
  playing: boolean;
  speed: number;
  following: boolean;
  errors: { at: number; message: string }[];

  loadRecording: (events: WireEvent[], scriptId: string) => void;
  append: (event: WireEvent) => void;
  setCursor: (n: number) => void;
  setClock: (t: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  restart: () => void;
  setSpeed: (s: number) => void;
  stepEvent: (dir: 1 | -1) => void;
  nextChunk: () => void;
  jumpToVersion: (version: number) => void;
  jumpToEvent: (eventId: string) => void;
  jumpToFirst: (eventType: string) => void;
  setConnection: (c: ConnectionState) => void;
  setMode: (m: "MOCK" | "LIVE") => void;
  pushError: (message: string) => void;
  reset: () => void;
}

function withCursor(state: StreamState, n: number): Partial<StreamState> {
  const target = Math.max(0, Math.min(n, state.events.length));
  if (target === state.cursor) return {};
  let view: EngineView;
  if (target > state.cursor) {
    view = state.view;
    for (let i = state.cursor; i < target; i++) view = applyEvent(view, state.events[i]);
  } else {
    view = foldEvents(state.events, target);
  }
  return { cursor: target, view };
}

function cursorForClock(events: WireEvent[], t: number): number {
  let n = 0;
  while (n < events.length && events[n].timestamp <= t + 1e-9) n++;
  return n;
}

export const useStream = create<StreamState>((set, get) => ({
  mode: "MOCK",
  connection: "MOCK",
  scriptId: null,
  events: [],
  cursor: 0,
  view: initialView(),
  clock: 0,
  playing: false,
  speed: 1,
  following: true,
  errors: [],

  loadRecording: (events, scriptId) =>
    set({ events, scriptId, cursor: 0, view: initialView(), clock: 0, playing: false, following: false }),

  append: (event) =>
    set((s) => {
      const events = [...s.events, event];
      if (!s.following) return { events };
      const next = { ...s, events };
      return { events, ...withCursor(next, events.length), clock: event.timestamp };
    }),

  setCursor: (n) =>
    set((s) => {
      const patch = withCursor(s, n);
      const cursor = patch.cursor ?? s.cursor;
      const clock = cursor > 0 ? s.events[cursor - 1].timestamp : 0;
      return { ...patch, clock, following: s.mode === "LIVE" && cursor === s.events.length };
    }),

  setClock: (t) =>
    set((s) => {
      const n = cursorForClock(s.events, t);
      const done = n >= s.events.length && s.mode === "MOCK";
      return { ...withCursor(s, n), clock: t, playing: done ? false : s.playing };
    }),

  play: () =>
    set((s) => {
      if (s.mode === "MOCK" && s.cursor >= s.events.length) {
        return { ...withCursor(s, 0), clock: 0, playing: true };
      }
      return { playing: true };
    }),
  pause: () => set({ playing: false }),
  togglePlay: () => (get().playing ? get().pause() : get().play()),
  restart: () => set((s) => ({ ...withCursor(s, 0), clock: 0, playing: true, following: false })),
  setSpeed: (speed) => set({ speed }),

  stepEvent: (dir) => {
    const s = get();
    s.pause();
    s.setCursor(s.cursor + dir);
  },

  nextChunk: () => {
    const s = get();
    s.pause();
    const idx = s.events.findIndex((e, i) => i >= s.cursor && e.event_type === "TranscriptChunkReceived");
    if (idx === -1) return s.setCursor(s.events.length);
    // include the whole processing burst that follows the chunk
    let end = idx + 1;
    while (end < s.events.length && s.events[end].event_type !== "TranscriptChunkReceived") end++;
    s.setCursor(end);
  },

  jumpToVersion: (version) => {
    const s = get();
    s.pause();
    const idx = s.events.findIndex((e) => e.event_type === "AnswerVersionCreated" && e.answer_version === version);
    if (idx >= 0) s.setCursor(idx + 1);
  },

  jumpToEvent: (eventId) => {
    const s = get();
    s.pause();
    const idx = s.events.findIndex((e) => e.event_id === eventId);
    if (idx >= 0) s.setCursor(idx + 1);
  },

  jumpToFirst: (eventType) => {
    const s = get();
    s.pause();
    const idx = s.events.findIndex((e) => e.event_type === eventType);
    if (idx >= 0) s.setCursor(idx + 1);
  },

  setConnection: (connection) => set({ connection }),
  setMode: (mode) => set({ mode, following: mode === "LIVE" }),
  pushError: (message) => set((s) => ({ errors: [...s.errors.slice(-19), { at: Date.now(), message }] })),
  reset: () =>
    set((s) => ({ events: [], cursor: 0, view: initialView(), clock: 0, playing: false, errors: [], following: s.mode === "LIVE" })),
}));

export const selectView = (s: StreamState) => s.view;
