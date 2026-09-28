import * as Dialog from "@radix-ui/react-dialog";
import { useState } from "react";
import { endSession, replayOnBackend, startLive, startMock } from "@/features/sessions/sessionController";
import { DEFAULT_SCRIPT, SCRIPTS } from "@/mocks";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import styles from "./timeline.module.css";

const SPEEDS = [0.5, 1, 1.5];

/** Replay/demo controls: script, speed, jumps to answer versions and to suppression, mode, end session. */
export function DemoControls() {
  const speed = useStream((s) => s.speed);
  const setSpeed = useStream((s) => s.setSpeed);
  const events = useStream((s) => s.events);
  const jumpToVersion = useStream((s) => s.jumpToVersion);
  const jumpToFirst = useStream((s) => s.jumpToFirst);
  const scriptId = useStream((s) => s.scriptId);
  const mode = useStream((s) => s.mode);
  const endOpen = useUi((s) => s.endSessionOpen);
  const setEndOpen = useUi((s) => s.setEndSessionOpen);
  const setViewVersion = useUi((s) => s.setViewVersion);
  const [liveScript, setLiveScript] = useState(DEFAULT_SCRIPT);

  const versions = events.filter((e) => e.event_type === "AnswerVersionCreated").map((e) => e.answer_version ?? 0);
  const hasSuppression = events.some((e) => e.event_type === "RetrievalSuppressed");

  return (
    <>
      <div className={styles.speed} role="group" aria-label="Playback speed">
        {SPEEDS.map((s) => (
          <button key={s} className="btn px" aria-pressed={speed === s} onClick={() => setSpeed(s)} disabled={mode === "LIVE"}>{s}×</button>
        ))}
      </div>
      {versions.map((v) => (
        <button key={v} className="btn px" onClick={() => { setViewVersion(null); jumpToVersion(v); }} data-testid={`jump-v${v}`}>
          V{String(v).padStart(2, "0")}
        </button>
      ))}
      {hasSuppression ? (
        <button className="btn px" onClick={() => jumpToFirst("RetrievalSuppressed")} data-testid="jump-suppress">SUPPRESS</button>
      ) : null}
      {mode === "MOCK" ? (
        <select className={styles.scripts} value={scriptId ?? ""} aria-label="Recorded session"
          onChange={(e) => void startMock(e.target.value)} data-testid="script-select">
          {SCRIPTS.map((s) => <option key={s.id} value={s.id}>{s.id}</option>)}
        </select>
      ) : (
        <>
          <select className={styles.scripts} value={liveScript} aria-label="Script to stream to backend" onChange={(e) => setLiveScript(e.target.value)}
            data-testid="live-script-select">
            {SCRIPTS.map((s) => <option key={s.id} value={s.id}>{s.id}</option>)}
          </select>
          <button className="btn px" onClick={() => replayOnBackend(liveScript, 1)}>STREAM TO ENGINE</button>
        </>
      )}
      <button className="btn px" onClick={() => void (mode === "MOCK" ? startLive() : startMock())} data-testid="mode-toggle">
        {mode === "MOCK" ? "GO LIVE" : "USE RECORDING"}
      </button>
      <Dialog.Root open={endOpen} onOpenChange={setEndOpen}>
        <Dialog.Trigger className="btn px">END SESSION</Dialog.Trigger>
        <Dialog.Portal>
          <Dialog.Overlay style={{ position: "fixed", inset: 0, background: "var(--surface-overlay)", zIndex: 55 }} />
          <Dialog.Content className="panel" style={{ position: "fixed", top: "34%", left: "50%", transform: "translateX(-50%)",
            width: "min(440px, calc(100vw - 32px))", zIndex: 56, padding: 24, gap: 16, boxShadow: "var(--shadow-overlay)" }}>
            <Dialog.Title className="px" style={{ margin: 0, fontWeight: 400 }}>END SESSION</Dialog.Title>
            <Dialog.Description className="secondary" style={{ margin: 0 }}>
              Ending this session clears ephemeral query, evidence, and answer state.
            </Dialog.Description>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <Dialog.Close className="btn">Keep session</Dialog.Close>
              <button className="btn btn-primary" onClick={() => { void endSession(); setEndOpen(false); }}>End session</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}
