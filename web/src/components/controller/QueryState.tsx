import { AnimatePresence, motion } from "framer-motion";
import { memo, useMemo } from "react";
import type { QueryVersion } from "@/types/domain";
import { transition } from "@/tokens/motion";
import { pad } from "@/utils/format";
import styles from "./controller.module.css";

function words(s: string): string[] {
  return s.toLowerCase().split(/[\s·,;]+/).filter(Boolean);
}

/** The evolving query: normalized text, detected slots, and what the latest chunk added (text diff). */
function QueryStateImpl({ current, previous }: { current: QueryVersion | null; previous: QueryVersion | null }) {
  const added = useMemo(() => {
    if (!current) return [];
    if (!previous || previous.utteranceId !== current.utteranceId) return [];
    const prev = new Map<string, number>();
    for (const w of words(previous.normalized)) prev.set(w, (prev.get(w) ?? 0) + 1);
    const out: string[] = [];
    for (const w of words(current.normalized)) {
      const n = prev.get(w) ?? 0;
      if (n > 0) prev.set(w, n - 1);
      else out.push(w);
    }
    return out;
  }, [current, previous]);

  return (
    <div className={styles.query} data-testid="query-state">
      <div className={`${styles.queryHead} px`}>
        <span>QUERY / V{pad(current?.version ?? 0)}</span>
        <span>{current ? `${current.candidates.length} candidate need${current.candidates.length === 1 ? "" : "s"}` : ""}</span>
      </div>
      {current && current.normalized ? (
        <>
          <div className={styles.queryText} title={current.normalized}>
            {current.candidates.map((c, i) => (
              <span key={c.key}>
                {i > 0 ? <span className={styles.sep}>·</span> : null}
                <span style={{ opacity: c.closed ? 1 : 0.72 }}>{c.text}</span>
              </span>
            ))}
          </div>
          {added.length ? (
            <div className={styles.added} aria-label="Added in this version">
              {added.map((w, i) => <span key={`${w}-${i}`}>+ {w}</span>)}
            </div>
          ) : null}
          {current.repairs.map((r) => (
            <div key={`${r.from}-${r.to}`} className={styles.repair}>SELF-REPAIR  {r.from} → {r.to}</div>
          ))}
          <div className={styles.slots}>
            <AnimatePresence initial={false}>
              {current.slots.map((s) => (
                <motion.span key={s.key} className="chip px" layout initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={transition.fast}
                  title={`${s.kind} slot${s.revised_from ? ` (revised from ${s.revised_from})` : ""}`}>
                  {s.value}{s.kind === "number" ? ` ${s.key.split(":")[1] === "value" ? "" : s.key.split(":")[1]}` : ""}
                </motion.span>
              ))}
            </AnimatePresence>
          </div>
        </>
      ) : (
        <div className="px muted">NO QUERY FORMED YET</div>
      )}
    </div>
  );
}

export const QueryState = memo(QueryStateImpl);
