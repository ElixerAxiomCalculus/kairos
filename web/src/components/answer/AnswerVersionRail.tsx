import { Fragment, memo } from "react";
import { STATUS_VAR } from "@/tokens/motion";
import type { AnswerVersion } from "@/types/domain";
import { pad } from "@/utils/format";
import styles from "./answer.module.css";

function suffix(v: AnswerVersion): string {
  if (v.kind === "PRESENTATION") return "FORMAT ONLY";
  if (v.trigger === "CONSTRAINT") return `+ ${(v.constraint?.terms ?? []).join(" ").toUpperCase() || "CONSTRAINT"}`;
  if (v.trigger === "ELABORATE") return "+ DETAIL";
  return "";
}

/** V1 → V2 → V3. Semantic changes and presentation changes are visually distinct. */
function AnswerVersionRailImpl({ versions, current, onSelect }: { versions: AnswerVersion[]; current: number | null; onSelect: (v: number) => void }) {
  if (!versions.length) return null;
  return (
    <nav className={styles.rail} aria-label="Answer versions" data-testid="version-rail">
      {versions.map((v, i) => (
        <Fragment key={v.id}>
          {i > 0 ? <span className={styles.railLink} aria-hidden /> : null}
          <button type="button" className={`${styles.railItem} px`} aria-current={v.id === current}
            onClick={() => onSelect(v.id)} data-testid={`version-${v.id}`}>
            <span className="dot" style={{ background: v.kind === "PRESENTATION" ? STATUS_VAR.SUPPRESS : v.trigger === "CONSTRAINT" ? STATUS_VAR.REFINE : STATUS_VAR.RETRIEVE }} />
            {i === 0 ? "ANSWER " : ""}V{pad(v.id)} {suffix(v)}
          </button>
        </Fragment>
      ))}
    </nav>
  );
}

export const AnswerVersionRail = memo(AnswerVersionRailImpl);
