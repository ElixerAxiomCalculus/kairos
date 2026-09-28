import * as Dialog from "@radix-ui/react-dialog";
import { useMemo, useState } from "react";
import { endSession, startLive, startMock } from "@/features/sessions/sessionController";
import { SCRIPTS } from "@/mocks";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import { navigate } from "./router";
import styles from "./shell.module.css";

interface Command {
  id: string;
  group: string;
  label: string;
  hint?: string;
  run: () => void;
}

/** Ctrl/⌘ + K. Keyboard-first: type to filter, ↑/↓ to move, Enter to run. */
export function CommandPalette() {
  const open = useUi((s) => s.paletteOpen);
  const setOpen = useUi((s) => s.setPaletteOpen);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const events = useStream((s) => s.events);
  const versions = useMemo(
    () => events.filter((e) => e.event_type === "AnswerVersionCreated").map((e) => e.answer_version ?? 0),
    [events],
  );

  const commands = useMemo<Command[]>(() => {
    const ui = useUi.getState();
    const stream = useStream.getState;
    const go = (fn: () => void) => () => {
      fn();
      setOpen(false);
    };
    return [
      { id: "new", group: "Session", label: "New session (live backend)", run: go(() => { navigate("/live"); void startLive(); }) },
      ...SCRIPTS.map((s) => ({ id: `replay-${s.id}`, group: "Session", label: `Replay demo: ${s.id}`, hint: s.title,
        run: go(() => { navigate("/live"); void startMock(s.id); }) })),
      { id: "clear", group: "Session", label: "Clear session (end, purge ephemeral state)", run: go(() => void endSession()) },
      { id: "pause", group: "Stream", label: "Pause event stream", hint: "Space", run: go(() => stream().pause()) },
      { id: "resume", group: "Stream", label: "Resume event stream", hint: "Space", run: go(() => stream().play()) },
      ...versions.map((v) => ({ id: `v${v}`, group: "Answer", label: `Go to Answer V${String(v).padStart(2, "0")}`,
        run: go(() => { useUi.getState().setViewVersion(null); stream().jumpToVersion(v); }) })),
      { id: "focus-answer", group: "View", label: "Answer focus", hint: "1", run: go(() => ui.setFocus("answer")) },
      { id: "focus-evidence", group: "View", label: "Toggle Evidence Graph (evidence focus)", hint: "2 / G", run: go(() => ui.setFocus("evidence")) },
      { id: "focus-systems", group: "View", label: "Toggle Intent Graph (systems focus)", hint: "3 / I", run: go(() => ui.setFocus("systems")) },
      { id: "focus-split", group: "View", label: "Split view", hint: "0", run: go(() => ui.setFocus("split")) },
      { id: "demo", group: "View", label: "Toggle demo mode", hint: "D", run: go(() => ui.toggleDemo()) },
      { id: "present", group: "View", label: "Toggle presentation mode", hint: "P", run: go(() => ui.togglePresentation()) },
      { id: "theme", group: "View", label: "Toggle dark / light", run: go(() => ui.toggleTheme()) },
      { id: "telemetry", group: "Navigate", label: "Open telemetry (live workspace)", run: go(() => navigate("/live")) },
      { id: "bench", group: "Navigate", label: "Open benchmark mode", hint: "B", run: go(() => navigate("/benchmark")) },
      { id: "exp", group: "Navigate", label: "Open experiments", run: go(() => navigate("/experiments")) },
      { id: "arch", group: "Navigate", label: "Open architecture", run: go(() => navigate("/architecture")) },
    ];
  }, [versions, setOpen]);

  const filtered = commands.filter((c) => `${c.group} ${c.label} ${c.hint ?? ""}`.toLowerCase().includes(q.toLowerCase()));
  const clamp = (i: number) => Math.max(0, Math.min(filtered.length - 1, i));

  return (
    <Dialog.Root open={open} onOpenChange={(o) => { setOpen(o); setQ(""); setActive(0); }}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.paletteOverlay} />
        <Dialog.Content className={styles.palette} aria-describedby={undefined} data-testid="command-palette"
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => clamp(a + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => clamp(a - 1)); }
            if (e.key === "Enter") { e.preventDefault(); filtered[active]?.run(); }
          }}>
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <input className={styles.paletteInput} autoFocus placeholder="Type a command…" value={q}
            onChange={(e) => { setQ(e.target.value); setActive(0); }} aria-label="Search commands" />
          <ul className={styles.paletteList} role="listbox">
            {filtered.map((c, i) => (
              <li key={c.id}>
                {i === 0 || filtered[i - 1].group !== c.group ? <div className={`${styles.group} px`}>{c.group}</div> : null}
                <button className={styles.paletteItem} data-active={i === active} role="option" aria-selected={i === active}
                  onMouseEnter={() => setActive(i)} onClick={c.run}>
                  <span>{c.label}</span>
                  {c.hint ? <span className="kbd">{c.hint}</span> : null}
                </button>
              </li>
            ))}
            {!filtered.length ? <li className="px muted" style={{ padding: 12 }}>NO MATCHING COMMAND</li> : null}
          </ul>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
