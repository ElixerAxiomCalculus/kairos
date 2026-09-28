import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { useReducedMotion } from "framer-motion";
import { memo, useMemo } from "react";
import { EmptyState } from "@/components/common/EmptyState";
import { Panel } from "@/components/common/Panel";
import { buildEvidenceGraph, connected, filterModel, type GEdge } from "@/graphs/evidence/model";
import { arcPath, layoutEvidence, linkPath, type PositionedNode } from "@/graphs/evidence/layout";
import { useElementSize } from "@/hooks/useElementSize";
import { useStream } from "@/stores/streamStore";
import { useUi, type GraphFilter, type Selection } from "@/stores/uiStore";
import { STATUS_VAR } from "@/tokens/motion";
import styles from "./evidence.module.css";

const FILTERS: GraphFilter[] = ["ALL", "ACTIVE", "CONFLICTS", "UPDATED", "UNSUPPORTED"];

function selectionNode(s: Selection): string | null {
  if (!s) return null;
  if (s.kind === "claim") return `CL:${s.id}`;
  if (s.kind === "chunk") return `CH:${s.id}`;
  if (s.kind === "intent") return `SQ:${s.id}`;
  return null;
}

function edgeStyle(e: GEdge): { stroke: string; dash?: string; width: number } {
  switch (e.kind) {
    case "RETRIEVED":
      return { stroke: e.trigger === "DELTA" ? "var(--status-refine)" : "var(--edge)", width: 1 };
    case "SUPPORTS":
      return { stroke: e.status === "CONFLICTING" ? "var(--status-conflict)" : "var(--edge)", width: 1.2 };
    case "DERIVED_FROM":
      return { stroke: "var(--status-refine)", dash: "3 3", width: 1 };
    case "CONTRADICTS":
      return { stroke: "var(--status-conflict)", dash: "4 3", width: 1.4 };
    case "DUPLICATES":
      return { stroke: "var(--edge-dim)", dash: "1 3", width: 1 };
    default:
      return { stroke: "var(--edge-dim)", width: 1 };
  }
}

function NodeShape({ n, color }: { n: PositionedNode; color: string }) {
  switch (n.kind) {
    case "query":
      return <circle className={styles.shape} r={13} fill="var(--surface-raised)" stroke="var(--text-primary)" strokeWidth={1.3} />;
    case "intent":
      return <rect className={styles.shape} x={-18} y={-9} width={36} height={18} rx={5} fill="var(--surface-raised)" stroke={color} strokeWidth={1.2} />;
    case "chunk":
      // document block with a folded corner
      return <path className={styles.shape} d="M-11 -13 H5 L11 -7 V13 H-11 Z M5 -13 V-7 H11" fill="var(--surface-raised)" stroke={color} strokeWidth={1.1} />;
    case "claim":
      return <path className={styles.shape} d="M0 -9 L9 0 L0 9 L-9 0 Z" fill={n.status === "SUPERSEDED" ? "var(--surface-raised)" : color} stroke={color} strokeWidth={1.2} />;
    case "version":
      return <rect className={styles.shape} x={-20} y={-12} width={40} height={24} rx={3} fill="var(--surface-raised)" stroke={color} strokeWidth={1.4} />;
  }
}

function nodeColor(n: PositionedNode): string {
  if (n.delta === "affected" || n.delta === "new") return "var(--status-refine)";
  if (n.kind === "claim") {
    if (n.status === "CONFLICTING") return STATUS_VAR.CONFLICTING;
    if (n.status === "SUPERSEDED") return "var(--text-muted)";
    if (n.status === "REVISED" || n.status === "ADDED") return "var(--status-refine)";
    if (n.status === "TRANSFORMED") return "var(--status-suppress)";
    return "var(--status-pass)";
  }
  if (n.kind === "version") return n.status === "PRESENTATION" ? "var(--status-suppress)" : "var(--text-primary)";
  if (n.kind === "chunk") return n.status === "CITED" ? "var(--text-primary)" : "var(--text-muted)";
  return "var(--border-strong)";
}

function EvidenceGraphImpl({ className }: { className?: string }) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const reduced = useReducedMotion();
  const intents = useStream((s) => s.view.intents);
  const relationships = useStream((s) => s.view.relationships);
  const versions = useStream((s) => s.view.versions);
  const claims = useStream((s) => s.view.claims);
  const delta = useStream((s) => s.view.delta);
  const view = useStream((s) => s.view);
  const filter = useUi((s) => s.graphFilter);
  const setFilter = useUi((s) => s.setGraphFilter);
  const selection = useUi((s) => s.selection);
  const hover = useUi((s) => s.hover);
  const select = useUi((s) => s.select);
  const setHover = useUi((s) => s.setHover);

  // rebuild only when structural slices change (not on every telemetry tick)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const model = useMemo(() => buildEvidenceGraph(view), [intents, relationships, versions, claims, delta]);
  const pos = useMemo(() => layoutEvidence(model, size.width, size.height), [model, size.width, size.height]);
  const focusId = selectionNode(hover) ?? selectionNode(selection);
  const lit = useMemo(() => (focusId ? connected(model, focusId) : null), [model, focusId]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const kept = useMemo(() => filterModel(model, filter, view), [model, filter, versions]);
  const deltaActive = model.nodes.some((n) => n.delta === "affected");
  // label density: hide per-node labels in a column whose rows are closer than a label's height
  const crowded = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const n of model.nodes) counts[n.kind] = (counts[n.kind] ?? 0) + 1;
    const usable = Math.max(1, size.height - 80);
    return new Set(Object.entries(counts).filter(([, c]) => usable / c < 26).map(([k]) => k));
  }, [model.nodes, size.height]);

  const visible = (id: string) => (kept ? kept.has(id) : true);
  const opacityFor = (id: string, base = 1) => {
    if (!visible(id)) return 0.06;
    if (lit) return lit.has(id) ? 1 : 0.14;
    return base;
  };

  const onNode = (n: PositionedNode) => {
    if (n.kind === "claim") select({ kind: "claim", id: n.refId });
    else if (n.kind === "chunk") select({ kind: "chunk", id: n.refId });
    else if (n.kind === "intent") select({ kind: "intent", id: n.refId });
  };
  const hoverNode = (n: PositionedNode | null) => {
    if (!n) return setHover(null);
    if (n.kind === "claim") setHover({ kind: "claim", id: n.refId });
    else if (n.kind === "chunk") setHover({ kind: "chunk", id: n.refId });
    else if (n.kind === "intent") setHover({ kind: "intent", id: n.refId });
  };

  return (
    <Panel
      label="EVIDENCE GRAPH"
      meta={model.nodes.length ? `${model.nodes.length} nodes · ${model.edges.length} edges` : undefined}
      className={className}
      noPadding
      testId="panel-evidence"
      actions={
        <ToggleGroup.Root type="single" value={filter} onValueChange={(v) => v && setFilter(v as GraphFilter)}
          className={styles.filters} aria-label="Graph filter">
          {FILTERS.map((f) => (
            <ToggleGroup.Item key={f} value={f} className={`${styles.filter} px`} data-testid={`graph-filter-${f}`}>{f}</ToggleGroup.Item>
          ))}
        </ToggleGroup.Root>
      }
    >
      <div className={styles.wrap} ref={ref} data-testid="evidence-graph">
        {model.nodes.length === 0 ? (
          <EmptyState label="No evidence requested yet">Retrieved chunks, the claims they support and the answer versions that contain them will appear here.</EmptyState>
        ) : size.width > 0 ? (
          <>
            <div className={`${styles.columns} px`} aria-hidden>
              <span>QUERY</span><span>INTENT</span><span>EVIDENCE</span><span>CLAIM</span><span>ANSWER</span>
            </div>
            <svg className={styles.canvas} width={size.width} height={size.height} role="img"
              aria-label={`Evidence graph with ${model.nodes.length} nodes`}>
              <g>
                {model.edges.map((e) => {
                  const a = pos.get(e.source);
                  const b = pos.get(e.target);
                  if (!a || !b) return null;
                  const st = edgeStyle(e);
                  const d = e.kind === "CONTRADICTS" || e.kind === "DUPLICATES" || e.kind === "DERIVED_FROM"
                    ? arcPath(a, b)
                    : linkPath({ source: [a.x, a.y], target: [b.x, b.y] }) ?? "";
                  const dim = deltaActive && !lit && (a.delta === "unaffected" || b.delta === "unaffected") ? 0.3 : 1;
                  const op = Math.min(opacityFor(e.source, dim), opacityFor(e.target, dim));
                  return (
                    <g key={e.id}>
                      <path d={d} fill="none" stroke={st.stroke} strokeWidth={st.width}
                        {...(st.dash ? { strokeDasharray: st.dash } : { pathLength: 1, className: reduced ? undefined : "draw-in" })}
                        style={{ opacity: op, transition: "opacity var(--dur-normal)" }} data-edge={e.kind} />
                      {e.kind === "SUPPORTS" ? (
                        <rect x={(a.x + b.x) / 2 - 2.5} y={(a.y + b.y) / 2 - 2.5} width={5} height={5}
                          fill={e.status === "CONFLICTING" ? "var(--status-conflict)" : "var(--text-secondary)"} opacity={op} />
                      ) : null}
                    </g>
                  );
                })}
              </g>
              <g>
                {Array.from(pos.values()).map((n) => {
                  const color = nodeColor(n);
                  const dim = deltaActive && !lit && n.delta === "unaffected" ? 0.35 : 1;
                  return (
                    <g key={n.id} className="fade-in" style={{ transform: `translate(${n.x}px, ${n.y}px)`,
                      transition: reduced ? undefined : "transform var(--dur-graph) var(--ease-standard)" }}>
                    <g className={styles.node} style={{ opacity: opacityFor(n.id, dim), transition: "opacity var(--dur-normal)" }}
                      onClick={() => onNode(n)} onMouseEnter={() => hoverNode(n)} onMouseLeave={() => hoverNode(null)}
                      role="button" tabIndex={n.kind === "claim" || n.kind === "chunk" || n.kind === "intent" ? 0 : -1}
                      aria-label={`${n.kind} ${n.label}`} onKeyDown={(ev) => ev.key === "Enter" && onNode(n)}
                      data-testid={`graph-node-${n.kind}`} data-node-id={n.id}>
                      <NodeShape n={n} color={color} />
                      {!crowded.has(n.kind) || (lit && lit.has(n.id)) ? (
                        <text x={n.kind === "chunk" || n.kind === "claim" ? 15 : 0} y={n.kind === "chunk" || n.kind === "claim" ? 3.5 : 24}
                          textAnchor={n.kind === "chunk" || n.kind === "claim" ? "start" : "middle"} fontSize={9.5}
                          fontFamily="var(--font-mono)" fill="var(--text-secondary)">{n.label}</text>
                      ) : null}
                      <title>{`${n.label}${n.sub ? ` - ${n.sub}` : ""}`}</title>
                      {n.kind === "version" || n.kind === "intent" ? (
                        <text y={3.5} textAnchor="middle" fontSize={8.5} className="px" fill="var(--text-primary)">
                          {n.kind === "version" ? n.label : n.label.replace("I", "")}
                        </text>
                      ) : null}
                    </g>
                    </g>
                  );
                })}
              </g>
            </svg>
            <div className={`${styles.legend} px`} aria-hidden>
              <span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="none" stroke="var(--text-primary)" /></svg>QUERY</span>
              <span><svg width="12" height="10"><rect x="1" y="2" width="10" height="6" rx="2" fill="none" stroke="var(--border-strong)" /></svg>INTENT</span>
              <span><svg width="10" height="12"><path d="M1 1H6L9 4V11H1Z" fill="none" stroke="var(--text-primary)" /></svg>EVIDENCE</span>
              <span><svg width="10" height="10"><path d="M5 0L10 5L5 10L0 5Z" fill="var(--status-pass)" /></svg>CLAIM</span>
              <span><svg width="6" height="6"><rect width="6" height="6" fill="var(--text-secondary)" /></svg>CITATION</span>
              <span style={{ color: "var(--status-conflict)" }}>- - CONFLICT</span>
              <span style={{ color: "var(--status-refine)" }}>— DELTA</span>
            </div>
          </>
        ) : null}
      </div>
    </Panel>
  );
}

export const EvidenceGraph = memo(EvidenceGraphImpl);
