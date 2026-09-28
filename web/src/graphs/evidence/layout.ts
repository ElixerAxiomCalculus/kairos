import * as d3 from "d3";
import type { GNode, GraphModel, NodeKind } from "./model";

export interface PositionedNode extends GNode {
  x: number;
  y: number;
}

const COLUMNS: NodeKind[] = ["query", "intent", "chunk", "claim", "version"];

/**
 * Layered layout: one column per node kind, rows by stable domain order. Coordinates are a pure
 * function of (order, count, size), so adding a node never moves unrelated nodes' columns and
 * transitions stay calm (no force simulation, no jitter).
 */
export function layoutEvidence(model: GraphModel, width: number, height: number): Map<string, PositionedNode> {
  const padX = 34;
  const padY = 30;
  const x = d3.scalePoint<NodeKind>().domain(COLUMNS).range([padX, Math.max(padX + 1, width - padX)]);
  const out = new Map<string, PositionedNode>();
  for (const kind of COLUMNS) {
    const col = model.nodes.filter((n) => n.kind === kind).sort((a, b) => a.order - b.order);
    const y = d3.scalePoint<number>().domain(d3.range(col.length)).range([padY, height - padY - 18]).padding(col.length === 1 ? 0.5 : 0.25);
    col.forEach((n, i) => out.set(n.id, { ...n, x: x(kind) ?? 0, y: y(i) ?? height / 2 }));
  }
  return out;
}

export const linkPath = d3
  .linkHorizontal<{ source: [number, number]; target: [number, number] }, [number, number]>()
  .source((d) => d.source)
  .target((d) => d.target);

export function arcPath(a: PositionedNode, b: PositionedNode): string {
  const dx = 26 + Math.abs(a.y - b.y) * 0.15;
  return `M${a.x + 12} ${a.y} C${a.x + dx} ${a.y} ${b.x + dx} ${b.y} ${b.x + 12} ${b.y}`;
}
