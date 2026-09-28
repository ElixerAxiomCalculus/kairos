import type { ReactNode } from "react";
import styles from "./common.module.css";

interface PanelProps {
  label: string;
  meta?: ReactNode;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
  accent?: string;
  testId?: string;
  noPadding?: boolean;
}

/** Bento cell. The pixel label is the machine name of the region; meta is short live context. */
export function Panel({ label, meta, actions, className, bodyClassName, children, accent, testId, noPadding }: PanelProps) {
  return (
    <section
      className={`panel ${className ?? ""}`}
      aria-label={label}
      data-testid={testId}
      style={accent ? { boxShadow: `inset 0 2px 0 ${accent}` } : undefined}
    >
      <header className="panel-head">
        <div className={styles.panelTitle}>
          <span className={`px ${styles.label}`}>{label}</span>
          {meta ? <span className={`mono ${styles.meta}`}>{meta}</span> : null}
        </div>
        {actions ? <div className={styles.actions}>{actions}</div> : null}
      </header>
      <div className={`${noPadding ? "" : "panel-body"} ${bodyClassName ?? ""}`} style={noPadding ? { flex: 1, minHeight: 0 } : undefined}>
        {children}
      </div>
    </section>
  );
}
