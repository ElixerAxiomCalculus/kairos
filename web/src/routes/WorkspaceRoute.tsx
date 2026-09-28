import * as Tabs from "@radix-ui/react-tabs";
import { useEffect } from "react";
import { GroundedAnswer } from "@/components/answer/GroundedAnswer";
import { ControlCenter } from "@/components/controller/ControlCenter";
import { EvidenceGraph } from "@/components/evidence/EvidenceGraph";
import { EvidenceInspector } from "@/components/evidence/EvidenceInspector";
import { TelemetryGrid } from "@/components/telemetry/TelemetryGrid";
import { EventInspector } from "@/components/timeline/EventInspector";
import { EventTimeline } from "@/components/timeline/EventTimeline";
import { LiveTranscript } from "@/components/transcript/LiveTranscript";
import { ensureStarted } from "@/features/sessions/sessionController";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { usePlaybackClock } from "@/hooks/usePlaybackClock";
import { useStream } from "@/stores/streamStore";
import { useUi, type FocusMode } from "@/stores/uiStore";
import type { Stage } from "@/types/domain";

const DEMO_FOCUS: Record<Stage, FocusMode> = {
  idle: "systems",
  listening: "systems",
  retrieving: "systems",
  answer: "answer",
  refine: "evidence",
  suppress: "answer",
};

const HIDDEN: Record<FocusMode, string[]> = {
  split: [],
  answer: ["evidence"],
  evidence: ["transcript", "telemetry"],
  systems: ["evidence"],
};

/** The live workspace: asymmetric bento, focus modes, demo auto-framing, mobile tabs. */
export default function WorkspaceRoute() {
  usePlaybackClock();
  const focus = useUi((s) => s.focus);
  const setFocus = useUi((s) => s.setFocus);
  const demoMode = useUi((s) => s.demoMode);
  const showTelemetry = useUi((s) => s.showTelemetry);
  const stage = useStream((s) => s.view.stage);

  useEffect(() => {
    ensureStarted(new URLSearchParams(window.location.search).get("autoplay") !== "0");
  }, []);

  useEffect(() => {
    if (demoMode) setFocus(DEMO_FOCUS[stage]);
  }, [demoMode, stage, setFocus]);

  const narrow = useMediaQuery("(max-width: 860px)");
  const hide = (area: string) => (HIDDEN[focus].includes(area) || (area === "telemetry" && !showTelemetry) ? "hidden-area" : "");

  return (
    <>
      {!narrow ? (
        <main className="workspace" data-focus={focus} data-testid="workspace" data-stage={stage}>
          <LiveTranscript className={`area-transcript ${hide("transcript")}`} />
          <ControlCenter className={`area-control ${hide("control")}`} />
          <TelemetryGrid className={`area-telemetry ${hide("telemetry")}`} />
          <GroundedAnswer className={`area-answer ${hide("answer")}`} />
          <EvidenceGraph className={`area-evidence ${hide("evidence")}`} />
        </main>
      ) : (
      <Tabs.Root className="mobile-tabs" defaultValue="live">
        <Tabs.List style={{ display: "flex", gap: 4 }} aria-label="Workspace sections">
          {["live", "answer", "evidence", "telemetry"].map((t) => (
            <Tabs.Trigger key={t} value={t} className="btn px" style={{ flex: 1, justifyContent: "center" }}>{t.toUpperCase()}</Tabs.Trigger>
          ))}
        </Tabs.List>
        <Tabs.Content value="live" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <LiveTranscript className="" />
          <ControlCenter className="" />
        </Tabs.Content>
        <Tabs.Content value="answer"><GroundedAnswer className="" /></Tabs.Content>
        <Tabs.Content value="evidence" style={{ height: 480 }}><EvidenceGraph className="" /></Tabs.Content>
        <Tabs.Content value="telemetry"><TelemetryGrid className="" /></Tabs.Content>
      </Tabs.Root>
      )}
      <EventTimeline />
      <EvidenceInspector />
      <EventInspector />
    </>
  );
}
