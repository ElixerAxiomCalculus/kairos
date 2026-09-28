import { parseWireEvent, ServerMessageSchema } from "@/schemas/events";
import type { CorpusStatus, SessionConfig } from "@/types/domain";
import type { OutgoingChunk, StreamingTransport, TransportHandlers } from "./StreamingTransport";

const MAX_BACKOFF_MS = 8000;

/** Live transport to the Kairos backend (/ws). Reconnects with exponential backoff and resumes the session. */
export class WebSocketTransport implements StreamingTransport {
  readonly kind = "websocket" as const;
  private socket: WebSocket | null = null;
  private handlers: TransportHandlers | null = null;
  private sessionId: string | null = null;
  private attempt = 0;
  private closedByUser = false;
  private queue: string[] = [];

  constructor(private readonly url: string = WebSocketTransport.defaultUrl()) {}

  static defaultUrl(): string {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}/ws`;
  }

  connect(handlers: TransportHandlers): Promise<void> {
    this.handlers = handlers;
    this.closedByUser = false;
    return new Promise((resolve, reject) => {
      const url = this.sessionId ? `${this.url}?session_id=${encodeURIComponent(this.sessionId)}` : this.url;
      const socket = new WebSocket(url);
      this.socket = socket;
      let opened = false;
      socket.onopen = () => {
        opened = true;
        this.attempt = 0;
        handlers.onStatus("LIVE");
        for (const msg of this.queue.splice(0)) socket.send(msg);
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
      this.sessionId = msg.data.session_id;
      handlers.onHello?.({
        sessionId: msg.data.session_id,
        corpus: msg.data.corpus as unknown as CorpusStatus,
        config: msg.data.config as unknown as SessionConfig,
      });
    } else if (msg.data.type === "event") {
      const parsed = parseWireEvent(msg.data.event);
      if (parsed.ok) handlers.onEvent(parsed.event);
      else handlers.onError(`Malformed event dropped: ${parsed.error}`);
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

  endUtterance(utteranceId: string): void {
    this.send({ type: "utterance_end", utterance_id: utteranceId });
  }

  requestReplay(scriptId: string, speed = 1): void {
    this.send({ type: "replay", script_id: scriptId, speed });
  }

  endSession(): void {
    this.send({ type: "end_session" });
    this.sessionId = null;
  }
}
