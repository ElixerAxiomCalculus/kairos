import { wsUrl } from "@/services/apiBase";
import { parseWireEvent, ServerMessageSchema } from "@/schemas/events";
import type { CorpusStatus, SessionConfig } from "@/types/domain";
import type {
  OutgoingChunk,
  StreamingTransport,
  TranscriptSnapshotOutgoing,
  TransportHandlers,
} from "./StreamingTransport";

const MAX_BACKOFF_MS = 8000;

/** Live transport to the Kairos backend (/ws). Reconnects with exponential backoff and resumes the session. */
export class WebSocketTransport implements StreamingTransport {
  readonly kind = "websocket" as const;
  private socket: WebSocket | null = null;
  private handlers: TransportHandlers | null = null;
  private sessionId: string | null = null;
  private resumeToken: string | null = null;
  private attempt = 0;
  private closedByUser = false;
  private queue: string[] = [];
  private pendingSnapshots = new Map<string, string>();
  private seenEvents = new Set<string>();
  private lastEventId: string | null = null;

  constructor(private readonly url: string = WebSocketTransport.defaultUrl()) {}

  static defaultUrl(): string {
    return wsUrl();
  }

  connect(handlers: TransportHandlers): Promise<void> {
    this.handlers = handlers;
    this.closedByUser = false;
    return new Promise((resolve, reject) => {
      const params = new URLSearchParams();
      if (this.sessionId) params.set("session_id", this.sessionId);
      if (this.resumeToken) params.set("resume_token", this.resumeToken);
      if (this.lastEventId) params.set("after_event_id", this.lastEventId);
      const url = params.size ? `${this.url}?${params.toString()}` : this.url;
      const socket = new WebSocket(url);
      this.socket = socket;
      let opened = false;
      socket.onopen = () => {
        opened = true;
        this.attempt = 0;
        handlers.onStatus("LIVE");
        resolve();
      };
      socket.onmessage = (msg) => this.handleMessage(msg.data);
      socket.onerror = () => {
        if (!opened) reject(new Error("Backend WebSocket unreachable"));
      };
      socket.onclose = () => {
        this.socket = null;
        if (this.closedByUser) {
          handlers.onStatus("OFFLINE");
          return;
        }
        if (!opened && this.attempt === 0) return; // initial failure: caller decides (falls back to MOCK)
        this.scheduleReconnect();
      };
    });
  }

  private scheduleReconnect(): void {
    if (!this.handlers) return;
    this.handlers.onStatus("RECONNECTING");
    const delay = Math.min(MAX_BACKOFF_MS, 500 * 2 ** this.attempt++);
    window.setTimeout(() => {
      if (this.closedByUser || !this.handlers) return;
      this.connect(this.handlers).catch(() => this.scheduleReconnect());
    }, delay);
  }

  private handleMessage(data: unknown): void {
    const handlers = this.handlers;
    if (!handlers || typeof data !== "string") return;
    let json: unknown;
    try {
      json = JSON.parse(data);
    } catch {
      handlers.onError("Malformed message from backend (invalid JSON)");
      return;
    }
    const msg = ServerMessageSchema.safeParse(json);
    if (!msg.success) {
      handlers.onError("Malformed message from backend (unknown shape)");
      return;
    }
    if (msg.data.type === "hello") {
      const previousSession = this.sessionId;
      this.sessionId = msg.data.session_id;
      this.resumeToken = msg.data.resume_token;
      if (previousSession && previousSession !== msg.data.session_id) {
        this.seenEvents.clear();
        this.lastEventId = null;
        this.pendingSnapshots.clear();
        handlers.onSessionReset?.();
        handlers.onError("The previous live session expired; a new session was started.");
      }
      handlers.onHello?.({
        sessionId: msg.data.session_id,
        corpus: msg.data.corpus as unknown as CorpusStatus,
        config: msg.data.config as unknown as SessionConfig,
      });
      if (this.socket?.readyState === WebSocket.OPEN) {
        for (const queued of this.queue.splice(0)) this.socket.send(queued);
        for (const pending of this.pendingSnapshots.values()) this.socket.send(pending);
      }
    } else if (msg.data.type === "event") {
      const parsed = parseWireEvent(msg.data.event);
      if (parsed.ok) {
        if (!this.seenEvents.has(parsed.event.event_id)) {
          this.seenEvents.add(parsed.event.event_id);
          this.lastEventId = parsed.event.event_id;
          handlers.onEvent(parsed.event);
        }
      } else handlers.onError(`Malformed event dropped: ${parsed.error}`);
    } else if (msg.data.type === "transcript_ack") {
      for (const key of this.pendingSnapshots.keys()) {
        const separator = key.lastIndexOf(":");
        const utteranceId = key.slice(0, separator);
        const revision = key.slice(separator + 1);
        if (utteranceId === msg.data.utterance_id && Number(revision) <= msg.data.revision) {
          this.pendingSnapshots.delete(key);
        }
      }
      handlers.onTranscriptAck?.(msg.data);
    } else {
      handlers.onError(msg.data.message);
    }
  }

  private send(payload: Record<string, unknown>): void {
    const text = JSON.stringify(payload);
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(text);
    else this.queue.push(text);
  }

  disconnect(): void {
    this.closedByUser = true;
    this.socket?.close();
    this.socket = null;
  }

  sendTranscriptChunk(chunk: OutgoingChunk): void {
    this.send({ type: "chunk", ...chunk });
  }

  sendTranscriptSnapshot(snapshot: TranscriptSnapshotOutgoing): void {
    const text = JSON.stringify({ type: "transcript_snapshot", ...snapshot });
    this.pendingSnapshots.set(`${snapshot.utterance_id}:${snapshot.revision}`, text);
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(text);
  }

  endUtterance(utteranceId: string): void {
    this.send({ type: "utterance_end", utterance_id: utteranceId });
  }

  requestReplay(scriptId: string, speed = 1): void {
    this.send({ type: "replay", script_id: scriptId, speed });
  }

  endSession(): void {
    this.send({ type: "end_session" });
    this.sessionId = null;
    this.resumeToken = null;
    this.lastEventId = null;
    this.seenEvents.clear();
    this.pendingSnapshots.clear();
  }
}
