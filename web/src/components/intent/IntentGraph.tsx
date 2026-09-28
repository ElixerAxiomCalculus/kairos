import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { memo, useMemo } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { layoutIntents } from "@/graphs/intent/layout";
import { useElementSize } from "@/hooks/useElementSize";
import { useUi } from "@/stores/uiStore";
import type { DeltaPlan, Intent, RetrievalTrigger } from "@/types/domain";
import { ms, pad } from "@/utils/format";

interface Props {
  intents: Intent[];
  queryVersion: number;
  active: Record<string, { trigger: RetrievalTrigger; startedAt: number }>;
  recent: Record<string, number>; // intent id -> retrieval count, drives pulses
  delta: DeltaPlan | null;
}

const STATUS_COLOR: Record<string, string> = {
  PENDING: "var(--status-wait)",
  RETRIEVING: "var(--status-retrieve)",
  RETRIEVED: "var(--text-secondary)",
  REFINING: "var(--status-refine)",
  SUPPRESSED: "var(--status-suppress)",
};

/** Query → intents. Parallel retrievals animate simultaneously along each branch. */
function IntentGraphImpl({ intents, queryVersion, active, recent, delta }: Props) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const reduced = useReducedMotion();
  const hover = useUi((s) => s.hover);
  const setHover = useUi((s) => s.setHover);
  const select = useUi((s) => s.select);
  const layout = useMemo(() => layoutIntents(intents, size.width, size.height), [intents, size.width, size.height]);
  const affected = new Set(delta?.affected.map((a) => a.sub_query_id) ?? []);
  const hoverId = hover?.kind === "intent" ? hover.id : null;

  return (
    <div ref={ref} style={{ position: "absolute", inset: 0 }} data-testid="intent-graph">
      {intents.length === 0 ? (
        <EmptyState label="Intent graph will form as speech stabilizes" />
      ) : size.width > 0 ? (
        <svg width={size.width} height={size.height} role="img" aria-label={`${intents.length} intents detected`}>
          <text x={10} y={14} className="px" fill="var(--text-muted)" fontSize={9}>{intents.length} ACTIVE INTENT{intents.length === 1 ? "" : "S"}</text>
          {layout.nodes.map((nd) => {
            const status = active[nd.intent.id] ? (active[nd.intent.id].trigger === "DELTA" || active[nd.intent.id].trigger === "REFINE" ? "REFINING" : "RETRIEVING") : nd.intent.retrievalStatus;
            const dim = hoverId !== null && hoverId !== nd.intent.id;
            const isDelta = delta !== null && affected.has(nd.intent.id);
            const stroke = isDelta ? "var(--status-refine)" : "var(--edge)";
            return (
              <g key={`l-${nd.intent.id}`} opacity={dim ? 0.25 : 1}>
                <path d={nd.link} fill="none" stroke={stroke} strokeWidth={isDelta ? 1.6 : 1.1} pathLength={1}
                  className={reduced ? undefined : "draw-in"} />
                <AnimatePresence>
                  {!reduced && recent[nd.intent.id] ? (
                    <motion.circle key={`p-${nd.intent.id}-${recent[nd.intent.id]}`} r={3.2}
                      fill={STATUS_COLOR[status === "RETRIEVED" ? (nd.intent.lastTrigger === "DELTA" || nd.intent.lastTrigger === "REFINE" ? "REFINING" : "RETRIEVING") : status]}
                      initial={{ offsetDistance: "0%", opacity: 1 }} animate={{ offsetDistance: "100%", opacity: [1, 1, 0] }}
                      transition={{ duration: 0.9, ease: "easeInOut" }} style={{ offsetPath: `path("${nd.link}")` }} />
                  ) : null}
                </AnimatePresence>
              </g>
            );
          })}
          {layout.deps.map((d) => (
            <path key={`dep-${d.from.intent.id}-${d.to.intent.id}`} d={d.path} fill="none" stroke="var(--edge-dim)" strokeDasharray="2 3" />
          ))}
          <g>
            <circle cx={layout.root.x} cy={layout.root.y} r={layout.root.r} fill="var(--surface-raised)" stroke="var(--text-primary)" strokeWidth={1.2} />
            <text x={layout.root.x} y={layout.root.y - 2} textAnchor="middle" fontSize={9} className="px" fill="var(--text-muted)">QUERY</text>
            <text x={layout.root.x} y={layout.root.y + 10} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--text-primary)">V{pad(queryVersion)}</text>
          </g>
          {layout.nodes.map((nd, i) => {
            const it = nd.intent;
            const running = active[it.id];
            const status = running ? (running.trigger === "DELTA" || running.trigger === "REFINE" ? "REFINING" : "RETRIEVING") : it.retrievalStatus;
            const color = STATUS_COLOR[status];
            const isDelta = delta !== null && affected.has(it.id);
            const dim = hoverId !== null && hoverId !== it.id;
            return (
              <g key={it.id} className="fade-in" style={{ cursor: "pointer", opacity: dim ? 0.3 : 1,
                transition: "opacity var(--dur-normal)", animationDelay: reduced ? undefined : `${i * 60}ms` }}
                onMouseEnter={() => setHover({ kind: "intent", id: it.id })} onMouseLeave={() => setHover(null)}
                onClick={() => select({ kind: "intent", id: it.id })} role="button" tabIndex={0} aria-label={`Intent ${it.id}: ${it.label}`}
                onKeyDown={(e) => e.key === "Enter" && select({ kind: "intent", id: it.id })} data-testid="intent-node">
                <rect x={nd.x} y={nd.y} width={nd.w} height={nd.h} rx={6} fill="var(--surface-raised)"
                  stroke={isDelta ? "var(--status-refine)" : "var(--border-strong)"} />
                <rect x={nd.x} y={nd.y + 6} width={2.5} height={nd.h - 12} rx={1} fill={color} />
                <text x={nd.x + 12} y={nd.h < 30 ? nd.y + nd.h / 2 + 3.5 : nd.y + 16} fontSize={10} className="px" fill="var(--text-primary)">
                  {it.id} {it.label.toUpperCase().slice(0, Math.floor(nd.w / 7.4) - 4)}
                </text>
                {nd.h >= 44 ? (
                  <text x={nd.x + 12} y={nd.y + 32} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                    {status} · {it.documentIds.length} doc · {ms(it.lastLatencyMs)}{it.dependencies.length ? ` · ↳${it.dependencies.join(",")}` : ""}
                  </text>
                ) : null}
                {nd.h >= 56 ? (
                  <text x={nd.x + 12} y={nd.y + 47} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                    rel {it.confidence !== undefined ? it.confidence.toFixed(2) : "—"} · ×{it.retrievalCount}
                    {Object.keys(it.facetConstraints).length ? ` · ${Object.values(it.facetConstraints).join(",")}` : ""}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}

export const IntentGraph = memo(IntentGraphImpl);
