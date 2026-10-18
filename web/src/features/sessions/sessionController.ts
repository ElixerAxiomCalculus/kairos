import { DEFAULT_SCRIPT } from "@/mocks";
import { apiUrl } from "@/services/apiBase";
import { MockTransport } from "@/services/transport/MockTransport";
import type { StreamingTransport } from "@/services/transport/StreamingTransport";
import { WebSocketTransport } from "@/services/transport/WebSocketTransport";
import { useStream } from "@/stores/streamStore";

/**
 * Owns the active transport and wires it into the stream store. UI components call these functions;
 * they never touch sockets or fixtures themselves.
 */
let transport: StreamingTransport | null = null;
let utteranceSeq = 0;
let backendAvailable: boolean | null = null;
let autoStarted = false;
let activeVoice: { utteranceId: string; revision: number; finalized: boolean } | null = null;

/** First entry into the workspace: load the default recording once (never re-triggered by resets). */
export function ensureStarted(autoplay: boolean): void {
  if (autoStarted) return;
  autoStarted = true;
  if (useStream.getState().events.length === 0 && useStream.getState().mode === "MOCK") void startMock(undefined, autoplay);
}

export async function probeBackend(): Promise<boolean> {
  if (backendAvailable !== null) return backendAvailable;
  try {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), 1500);
    const res = await fetch(apiUrl("/api/health"), { signal: ctrl.signal });
    window.clearTimeout(timer);
    backendAvailable = res.ok && (await res.json()).status === "ok";
  } catch {
    backendAvailable = false;
  }
  return backendAvailable;
}

function handlers() {
  const s = useStream.getState();
  return {
    onEvent: s.append,
    onStatus: s.setConnection,
    onError: s.pushError,
    onSessionReset: s.reset,
  };
}

/** Load a recorded engine trace for deterministic replay (works with no backend). */
export async function startMock(scriptId = DEFAULT_SCRIPT, autoplay = true): Promise<void> {
  autoStarted = true;
  transport?.disconnect();
  activeVoice = null;
  const store = useStream.getState();
  store.setMode("MOCK");
  store.reset();
  const mock = new MockTransport((events, id) => {
    useStream.getState().loadRecording(events, id);
    if (autoplay) useStream.getState().play();
  });
  transport = mock;
  await mock.connect(handlers());
  await mock.load(scriptId);
}

/** Connect to the live backend. Resolves false (and stays in mock mode) if the backend is unreachable. */
export async function startLive(): Promise<boolean> {
  if (!(await probeBackend())) {
    useStream.getState().pushError("Backend unreachable at /api — staying in MOCK mode (recorded sessions).");
    return false;
  }
  autoStarted = true;
  transport?.disconnect();
  activeVoice = null;
  const store = useStream.getState();
  // mode first: an empty MOCK store must never be observable while switching to LIVE
  store.setMode("LIVE");
  store.reset();
  const ws = new WebSocketTransport();
  transport = ws;
  try {
    await ws.connect(handlers());
    return true;
  } catch (err) {
    store.setConnection("OFFLINE");
    store.pushError(`Backend disconnected: ${(err as Error).message}`);
    return false;
  }
}

export function replayOnBackend(scriptId: string, speed = 1): void {
  transport?.requestReplay(scriptId, speed);
}

/** Stream typed text word-group by word-group, like an ASR would. */
export async function streamText(text: string, msPerChunk = 380): Promise<void> {
  if (!transport || transport.kind !== "websocket") {
    useStream.getState().pushError("Typing a query needs the live backend (python -m kairos serve).");
    return;
  }
  if (activeVoice) {
    useStream.getState().pushError("Stop the active microphone turn before sending typed text.");
    return;
  }
  utteranceSeq += 1;
  const uid = `typed-${utteranceSeq}`;
  const words = text.trim().split(/\s+/).filter(Boolean);
  const groups: string[] = [];
  for (let i = 0; i < words.length; i += 3) groups.push(words.slice(i, i + 3).join(" "));
  for (let i = 0; i < groups.length; i++) {
    transport.sendTranscriptChunk({ utterance_id: uid, text: groups[i], final: i === groups.length - 1 });
    if (i < groups.length - 1) await new Promise((r) => window.setTimeout(r, msPerChunk));
  }
}

export function sendSpeechChunk(utteranceId: string, text: string, final: boolean): void {
  transport?.sendTranscriptChunk({ utterance_id: utteranceId, text, final });
}

export function beginVoiceTurn(): string | null {
  const store = useStream.getState();
  if (!transport || transport.kind !== "websocket" || store.connection !== "LIVE") {
    store.pushError("Microphone input needs a live backend connection.");
    return null;
  }
  if (activeVoice && !activeVoice.finalized) return null;
  const utteranceId = nextUtteranceId("mic");
  activeVoice = { utteranceId, revision: 0, finalized: false };
  return utteranceId;
}

export function sendVoiceSnapshot(utteranceId: string, text: string, asrFinal: boolean,
                                  endOfTurn = false): boolean {
  if (!activeVoice || activeVoice.utteranceId !== utteranceId || activeVoice.finalized) return false;
  activeVoice.revision += 1;
  if (endOfTurn) activeVoice.finalized = true;
  transport?.sendTranscriptSnapshot({
    utterance_id: utteranceId,
    revision: activeVoice.revision,
    text,
    asr_final: asrFinal,
    end_of_turn: endOfTurn,
    source: "microphone",
  });
  if (endOfTurn) activeVoice = null;
  return true;
}

export function abandonVoiceTurn(utteranceId: string): void {
  if (activeVoice?.utteranceId === utteranceId) activeVoice = null;
}

export function nextUtteranceId(prefix = "mic"): string {
  utteranceSeq += 1;
  return `${prefix}-${utteranceSeq}`;
}

export async function endSession(): Promise<void> {
  transport?.endSession();
  transport?.disconnect();
  transport = null;
  activeVoice = null;
  const store = useStream.getState();
  const mode = store.mode;
  store.reset();
  store.setConnection(mode === "LIVE" ? "OFFLINE" : "MOCK");
}

export function currentTransportKind(): "websocket" | "mock" | null {
  return transport?.kind ?? null;
}
