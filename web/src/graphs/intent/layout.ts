import * as d3 from "d3";
import type { Intent } from "@/types/domain";

export interface IntentNodeLayout {
  intent: Intent;
  x: number;
  y: number;
  w: number;
  h: number;
  link: string;
}

export interface IntentLayout {
  root: { x: number; y: number; r: number };
  nodes: IntentNodeLayout[];
  deps: { from: IntentNodeLayout; to: IntentNodeLayout; path: string }[];
}

/**
 * Deterministic tree layout (no force simulation): query root on the left, intents stacked on the right
 * in creation order. Positions depend only on (index, count, size), so nodes never jitter.
 */
export function layoutIntents(intents: Intent[], width: number, height: number): IntentLayout {
  const padY = 14;
  const rootX = Math.max(46, width * 0.13);
  const root = { x: rootX, y: height / 2, r: 24 };
  const w = Math.min(260, Math.max(150, width * 0.56));
  const x = width - w - 12;
  const n = intents.length;
  const slot = n ? (height - padY * 2) / n : 0;
  const h = Math.max(20, Math.min(62, slot - 6));
  const link = d3.linkHorizontal<unknown, [number, number]>().source((d) => (d as { s: [number, number] }).s)
    .target((d) => (d as { t: [number, number] }).t);
  const nodes = intents.map((intent, i) => {
    const cy = padY + slot * i + slot / 2;
    return {
      intent,
      x,
      y: cy - h / 2,
      w,
      h,
      link: link({ s: [root.x + root.r, root.y], t: [x, cy] } as never) ?? "",
    };
  });
  const byId = new Map(nodes.map((nd) => [nd.intent.id, nd]));
  const deps: IntentLayout["deps"] = [];
  for (const nd of nodes) {
    for (const dep of nd.intent.dependencies) {
      const from = byId.get(dep);
      if (!from) continue;
      const fx = from.x - 6;
      const tx = nd.x - 6;
      const path = `M${fx} ${from.y + from.h / 2} C${fx - 26} ${from.y + from.h / 2} ${tx - 26} ${nd.y + nd.h / 2} ${tx} ${nd.y + nd.h / 2}`;
      deps.push({ from, to: nd, path });
    }
  }
  return { root, nodes, deps };
}
