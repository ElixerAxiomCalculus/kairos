import type { ReactNode } from "react";
import { BranchLine } from "./BranchLine";

/** Contextual empty state in machine type — says what will appear here and when. */
export function EmptyState({ label, children, animate = false }: { label: string; children?: ReactNode; animate?: boolean }) {
  return (
    <div className="empty" role="status">
      <span className="muted">
        <BranchLine width={96} animate={animate} />
      </span>
      <span className="px">{label}</span>
      {children ? <span className="secondary" style={{ fontSize: 13, maxWidth: 320 }}>{children}</span> : null}
    </div>
  );
}
