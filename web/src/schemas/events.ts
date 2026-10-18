import { z } from "zod";

/** Wire schema of a backend event (kairos/schemas/events.py). Payload is validated per consumer. */
export const EventTypeSchema = z.enum([
  "SessionStarted",
  "TranscriptReconciled",
  "TranscriptChunkReceived",
  "QueryFrameUpdated",
  "TurnClassified",
  "ProbeExecuted",
  "RetrievalDecision",
  "SubQueriesGenerated",
  "RetrievalStarted",
  "RetrievalCompleted",
  "EvidenceFused",
  "DeltaPlanComputed",
  "ClaimsGenerated",
  "GroundingVerified",
  "RetrievalSuppressed",
  "TransformVerified",
  "TransformUnavailable",
  "AnswerVersionCreated",
  "TelemetryEmitted",
  "SessionEnded",
]);

export const TriggerSchema = z.enum(["EARLY", "END_OF_UTTERANCE", "REFINE", "DELTA", "NONE"]);

export const WireEventSchema = z.object({
  event_id: z.string(),
  parent_event_ids: z.array(z.string()).default([]),
  timestamp: z.number(),
  wall_time: z.string(),
  session_id: z.string(),
  event_type: EventTypeSchema,
  utterance_id: z.string().nullable().optional(),
  query_version: z.number().nullable().optional(),
  answer_version: z.number().nullable().optional(),
  retrieval_trigger: TriggerSchema.nullable().optional(),
  sub_query_id: z.string().nullable().optional(),
  document_ids: z.array(z.string()).default([]),
  chunk_ids: z.array(z.string()).default([]),
  latency_ms: z.number().nullable().optional(),
  token_cost: z.record(z.number()).default({}),
  confidence: z.number().nullable().optional(),
  citations: z.array(z.string()).default([]),
  payload: z.record(z.unknown()).default({}),
});

export type WireEvent = z.infer<typeof WireEventSchema>;
export type EventType = z.infer<typeof EventTypeSchema>;

export const ServerMessageSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("hello"),
    session_id: z.string(),
    resumed: z.boolean().default(false),
    corpus: z.record(z.unknown()),
    config: z.record(z.unknown()),
  }),
  z.object({ type: z.literal("event"), event: z.unknown() }),
  z.object({
    type: z.literal("transcript_ack"),
    utterance_id: z.string(),
    revision: z.number().int(),
    status: z.enum(["accepted", "duplicate", "stale", "finalized"]),
    committed_text: z.string(),
    provisional_text: z.string(),
    reconciled: z.boolean(),
    finalized: z.boolean(),
  }),
  z.object({ type: z.literal("error"), message: z.string() }),
]);

export type ServerMessage = z.infer<typeof ServerMessageSchema>;

export interface ParseFailure {
  ok: false;
  error: string;
  raw: unknown;
}

export function parseWireEvent(raw: unknown): { ok: true; event: WireEvent } | ParseFailure {
  const result = WireEventSchema.safeParse(raw);
  if (result.success) return { ok: true, event: result.data };
  return { ok: false, error: result.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; "), raw };
}
