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
  onTranscriptAck?: (ack: TranscriptAck) => void;
  onSessionReset?: () => void;
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

export interface TranscriptSnapshotOutgoing {
  utterance_id: string;
  revision: number;
  text: string;
  asr_final: boolean;
  end_of_turn: boolean;
  source: "microphone";
}

export interface TranscriptAck {
  utterance_id: string;
  revision: number;
  status: "accepted" | "duplicate" | "stale" | "finalized";
  committed_text: string;
  provisional_text: string;
  reconciled: boolean;
  finalized: boolean;
}

/** Visualization code never talks to a socket directly — only to this interface. */
export interface StreamingTransport {
  readonly kind: "websocket" | "mock";
  connect(handlers: TransportHandlers): Promise<void>;
  disconnect(): void;
  sendTranscriptChunk(chunk: OutgoingChunk): void;
  sendTranscriptSnapshot(snapshot: TranscriptSnapshotOutgoing): void;
  endUtterance(utteranceId: string): void;
  requestReplay(scriptId: string, speed?: number): void;
  endSession(): void;
}
