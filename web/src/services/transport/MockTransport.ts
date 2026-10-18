import { loadFixture } from "@/mocks";
import { parseWireEvent, type WireEvent } from "@/schemas/events";
import type { OutgoingChunk, StreamingTransport, TranscriptSnapshotOutgoing, TransportHandlers } from "./StreamingTransport";

/**
 * Deterministic mock transport. It does not invent events: it serves traces recorded from the real engine
 * (`python -m kairos record-demo`). Playback timing is handled by the stream store's clock, so pause,
 * step and speed controls behave identically for recordings and live sessions.
 */
export class MockTransport implements StreamingTransport {
  readonly kind = "mock" as const;
  private handlers: TransportHandlers | null = null;

  constructor(private readonly onRecording: (events: WireEvent[], scriptId: string) => void) {}

  async connect(handlers: TransportHandlers): Promise<void> {
    this.handlers = handlers;
    handlers.onStatus("MOCK");
  }

  disconnect(): void {
    this.handlers = null;
  }

  async load(scriptId: string): Promise<void> {
    const raw = await loadFixture(scriptId);
    const events: WireEvent[] = [];
    for (const item of raw) {
      const parsed = parseWireEvent(item);
      if (parsed.ok) events.push(parsed.event);
      else this.handlers?.onError(`Fixture event dropped: ${parsed.error}`);
    }
    this.onRecording(events, scriptId);
  }

  sendTranscriptChunk(_chunk: OutgoingChunk): void {
    this.handlers?.onError("Mock mode replays recorded sessions; connect the backend to stream your own speech.");
  }

  sendTranscriptSnapshot(_snapshot: TranscriptSnapshotOutgoing): void {
    this.handlers?.onError("Microphone input needs the live backend.");
  }

  endUtterance(_utteranceId: string): void {
    /* recordings already contain utterance boundaries */
  }

  requestReplay(scriptId: string): void {
    void this.load(scriptId);
  }

  endSession(): void {
    this.handlers?.onStatus("OFFLINE");
  }
}
