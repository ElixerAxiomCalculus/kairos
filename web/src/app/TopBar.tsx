import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { Command, Monitor, Moon, Presentation, Sun } from "lucide-react";
import { ConnectionStatus } from "@/components/common/ConnectionStatus";
import { CorpusStatus } from "@/components/common/CorpusStatus";
import { Tip } from "@/components/common/Tip";
import { BranchLine } from "@/components/common/BranchLine";
import { useStream } from "@/stores/streamStore";
import { useUi, type FocusMode } from "@/stores/uiStore";
import { BRAND } from "./brand";
import { navigate, useRoute, type Route } from "./router";
import styles from "./shell.module.css";

const NAV: { to: Route; label: string }[] = [
  { to: "/live", label: "LIVE" },
  { to: "/benchmark", label: "BENCHMARK" },
  { to: "/experiments", label: "EXPERIMENTS" },
  { to: "/architecture", label: "ARCHITECTURE" },
];

const FOCUS: { v: FocusMode; label: string; key: string }[] = [
  { v: "split", label: "SPLIT", key: "0" },
  { v: "answer", label: "ANSWER", key: "1" },
  { v: "evidence", label: "EVIDENCE", key: "2" },
  { v: "systems", label: "SYSTEMS", key: "3" },
];

export function TopBar() {
  const route = useRoute();
  const session = useStream((s) => s.view.session);
  const scriptId = useStream((s) => s.scriptId);
  const mode = useStream((s) => s.mode);
  const { focus, setFocus, demoMode, toggleDemo, presentation, togglePresentation, theme, toggleTheme, setPaletteOpen } = useUi();

  return (
    <header className={styles.topbar}>
      <button className="px" onClick={() => navigate("/")} aria-label={`${BRAND.name} home`}
        style={{ display: "flex", alignItems: "center", gap: 10, background: "none", border: 0, cursor: "pointer", color: "var(--text-primary)" }}>
        <span style={{ fontSize: 15 }}>{BRAND.mark}</span>
        <span className={`muted ${styles.hideMid}`} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <BranchLine width={28} height={12} /> {BRAND.tagline.toUpperCase()}
        </span>
      </button>
      <nav className={styles.nav} aria-label="Primary">
        {NAV.map((n) => (
          <button key={n.to} className={`${styles.navItem} px`} aria-current={route === n.to ? "page" : undefined}
            onClick={() => navigate(n.to)}>{n.label}</button>
        ))}
      </nav>
      <div className={styles.session}>
        {route === "/live" ? (
          <span className="px" data-testid="session-label">
            SESSION {session.id ? session.id.toUpperCase() : "—"} / {session.status}
            {mode === "MOCK" && scriptId ? <span className={`muted ${styles.hideMid}`}> · REC {scriptId.toUpperCase()}</span> : null}
          </span>
        ) : null}
      </div>
      <div className={styles.right}>
        {route === "/live" ? (
          <ToggleGroup.Root type="single" value={focus} onValueChange={(v) => v && setFocus(v as FocusMode)}
            className={`${styles.focus} ${styles.hideNarrow}`} aria-label="Focus mode">
            {FOCUS.map((f) => (
              <Tip key={f.v} content={`${f.label} focus (${f.key})`} side="bottom">
                <ToggleGroup.Item value={f.v} className={`${styles.focusItem} px`} data-testid={`focus-${f.v}`}>{f.label}</ToggleGroup.Item>
              </Tip>
            ))}
          </ToggleGroup.Root>
        ) : null}
        {route === "/live" ? (
          <Tip content="Demo mode: auto-frames the active stage (D)" side="bottom">
            <button className="btn btn-icon" aria-pressed={demoMode} onClick={toggleDemo} aria-label="Demo mode" data-testid="demo-toggle"><Monitor size={15} /></button>
          </Tip>
        ) : null}
        <Tip content="Presentation mode: larger type for projectors (P)" side="bottom">
          <button className="btn btn-icon" aria-pressed={presentation} onClick={togglePresentation} aria-label="Presentation mode" data-testid="presentation-toggle"><Presentation size={15} /></button>
        </Tip>
        <ConnectionStatus />
        <CorpusStatus />
        <Tip content={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} side="bottom">
          <button className="btn btn-icon" onClick={toggleTheme} aria-label="Toggle theme" data-testid="theme-toggle">
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </Tip>
        <Tip content="Command palette (Ctrl/⌘ K)" side="bottom">
          <button className="btn btn-icon" onClick={() => setPaletteOpen(true)} aria-label="Open command palette"><Command size={15} /></button>
        </Tip>
      </div>
    </header>
  );
}
