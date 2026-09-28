import type { WireEvent } from "@/schemas/events";
import type { ConnectionState, CorpusStatus, SessionConfig } from "@/types/domain";

export interface HelloInfo {
  sessionId: string;
  corpus: CorpusStatus;
  config: SessionConfig;
}

export interface TransportHandlers {
  onEvent: (event: WireEvent) => void;
  onHello?: (info: HelloInfo) => void;
  onStatus: (status: ConnectionState) => void;
  onError: (message: string) => void;
}

export interface OutgoingChunk {
  utterance_id: string;
  text: string;
  final?: boolean;
  t0?: number;
  t1?: number;
}

/** Visualization code never talks to a socket directly — only to this interface. */
export interface StreamingTransport {
  readonly kind: "websocket" | "mock";
  connect(handlers: TransportHandlers): Promise<void>;
  disconnect(): void;
  sendTranscriptChunk(chunk: OutgoingChunk): void;
  endUtterance(utteranceId: string): void;
  requestReplay(scriptId: string, speed?: number): void;
  endSession(): void;
}
