import { useStream } from "@/stores/streamStore";
import type { ConnectionState } from "@/types/domain";
import { Tip } from "./Tip";
import styles from "./common.module.css";

const COLORS: Record<ConnectionState, string> = {
  LIVE: "var(--status-pass)",
  RECONNECTING: "var(--status-suppress)",
  REPLAY: "var(--status-retrieve)",
  OFFLINE: "var(--status-fail)",
  MOCK: "var(--status-wait)",
};

const HINT: Record<ConnectionState, string> = {
  LIVE: "Connected to the Kairos backend over WebSocket.",
  RECONNECTING: "Connection lost — retrying with backoff. Session state resumes on reconnect.",
  REPLAY: "Replaying a recorded session.",
  OFFLINE: "Backend disconnected.",
  MOCK: "Replaying traces recorded from the real engine (no backend needed).",
};

export function ConnectionStatus() {
  const connection = useStream((s) => s.connection);
  const playing = useStream((s) => s.playing);
  const shown: ConnectionState = connection === "MOCK" && playing ? "REPLAY" : connection;
  return (
    <Tip content={HINT[shown]} side="bottom">
      <span className={styles.status} data-testid="connection-status" tabIndex={0}>
        <span className={`dot ${shown === "RECONNECTING" ? "blink" : ""}`} style={{ background: COLORS[shown] }} />
        <span className="px">{shown}</span>
      </span>
    </Tip>
  );
}
