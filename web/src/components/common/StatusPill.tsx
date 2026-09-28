import { STATUS_VAR } from "@/tokens/motion";
import styles from "./common.module.css";

/** State is always text + a colour bar — never colour alone. */
export function StatusPill({ state, label, title }: { state: string; label?: string; title?: string }) {
  const color = STATUS_VAR[state] ?? "var(--text-muted)";
  return (
    <span className={`${styles.pill} px`} title={title} data-state={state}>
      <span className={styles.bar} style={{ background: color }} aria-hidden />
      {label ?? state.replace(/_/g, " ")}
    </span>
  );
}
