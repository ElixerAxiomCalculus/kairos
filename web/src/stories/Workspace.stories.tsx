import type { Meta, StoryObj } from "@storybook/react";
import type { ReactNode } from "react";
import { GroundedAnswer } from "@/components/answer/GroundedAnswer";
import { EvidenceGraph } from "@/components/evidence/EvidenceGraph";
import { IntentGraph } from "@/components/intent/IntentGraph";
import { TelemetryGrid } from "@/components/telemetry/TelemetryGrid";
import { EventTimeline } from "@/components/timeline/EventTimeline";
import { LiveTranscript } from "@/components/transcript/LiveTranscript";
import { at, loadStoreAt, viewAt, type RecordingKey } from "@/test/recorded";
import type { WireEvent } from "@/schemas/events";

/** Store-connected components rendered at real moments of recorded sessions. */
const meta: Meta = { title: "Workspace" };
export default meta;

function Frame({ w = 640, h = 520, children }: { w?: number; h?: number; children: ReactNode }) {
  return <div style={{ width: w, height: h, display: "grid" }}>{children}</div>;
}

function atMoment(key: RecordingKey, pred: (e: WireEvent) => boolean, nth = 1) {
  return (Story: () => ReactNode) => {
    loadStoreAt(key, pred, nth);
    return <Story />;
  };
}

export const TranscriptStreaming: StoryObj = {
  name: "Transcript / Streaming",
  decorators: [atMoment("pune", at.chunk(4))],
  render: () => <Frame w={340} h={640}><LiveTranscript /></Frame>,
};

const intents = (key: RecordingKey, v: number) => {
  const view = viewAt(key, at.version(v));
  return { intents: view.intentOrder.map((id) => view.intents[id]), qv: view.queryVersions.at(-1)?.version ?? 0 };
};

export const IntentGraphSingle: StoryObj = {
  name: "IntentGraph / Single",
  render: () => {
    const { intents: list, qv } = intents("reimbursement", 1);
    return <Frame w={520} h={200}><div style={{ position: "relative" }}><IntentGraph intents={list} queryVersion={qv} active={{}} recent={{}} delta={null} /></div></Frame>;
  },
};

export const IntentGraphMulti: StoryObj = {
  name: "IntentGraph / MultiIntent",
  render: () => {
    const { intents: list, qv } = intents("pune", 1);
    return <Frame w={520} h={220}><div style={{ position: "relative" }}><IntentGraph intents={list} queryVersion={qv} active={{}} recent={{}} delta={null} /></div></Frame>;
  },
};

export const AnswerV1: StoryObj = {
  name: "Answer / V1",
  decorators: [atMoment("pune", at.version(1))],
  render: () => <Frame w={720} h={620}><GroundedAnswer /></Frame>,
};

export const AnswerDeltaUpdate: StoryObj = {
  name: "Answer / DeltaUpdate",
  decorators: [atMoment("pune", at.version(2))],
  render: () => <Frame w={720} h={680}><GroundedAnswer /></Frame>,
};

export const EvidenceGraphSupported: StoryObj = {
  name: "EvidenceGraph / Supported",
  decorators: [atMoment("pune", at.version(2))],
  render: () => <Frame w={900} h={520}><EvidenceGraph /></Frame>,
};

export const EvidenceGraphConflict: StoryObj = {
  name: "EvidenceGraph / Conflict",
  decorators: [atMoment("conflict", at.version(1))],
  render: () => <Frame w={900} h={420}><EvidenceGraph /></Frame>,
};

export const TimelineActive: StoryObj = {
  name: "Timeline / Active",
  decorators: [atMoment("pune", at.version(2))],
  render: () => <div style={{ position: "relative", height: 200 }}><EventTimeline /></div>,
};

export const TelemetryLive: StoryObj = {
  name: "Telemetry / Live",
  decorators: [atMoment("pune", at.version(2))],
  render: () => <Frame w={420} h={560}><TelemetryGrid /></Frame>,
};
