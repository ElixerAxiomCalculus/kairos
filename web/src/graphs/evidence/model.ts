import type { EngineView } from "@/features/streaming/reducer";
import type { GraphFilter } from "@/stores/uiStore";

export type NodeKind = "query" | "intent" | "chunk" | "claim" | "version";
export type EdgeKind = "DECOMPOSES" | "RETRIEVED" | "SUPPORTS" | "CONTAINS" | "DERIVED_FROM" | "CONTRADICTS" | "DUPLICATES";

export interface GNode {
  id: string;
  kind: NodeKind;
  label: string;
  sub?: string;
  order: number;
  status?: string; // verdict / claim status / version kind
  delta?: "affected" | "unaffected" | "new";
  refId: string; // id in the domain model (claim id, chunk id...)
}

export interface GEdge {
  id: string;
  kind: EdgeKind;
  source: string;
  target: string;
  trigger?: string;
  status?: string;
}

export interface GraphModel {
  nodes: GNode[];
  edges: GEdge[];
}

/** Build the evidence graph from event-derived state. Pure; memoised by the caller on `view` slices. */
export function buildEvidenceGraph(view: EngineView): GraphModel {
  const nodes: GNode[] = [];
  const edges: GEdge[] = [];
  const seen = new Set<string>();
  const add = (n: GNode) => {
    if (!seen.has(n.id)) {
      seen.add(n.id);
      nodes.push(n);
    }
  };

  const latest = view.versions.at(-1);
  const delta = view.delta && latest && latest.utteranceId === view.delta.utteranceId ? view.delta : null;
  const affected = new Set(delta?.affected.map((a) => a.sub_query_id) ?? []);

  const utts = Array.from(new Set(view.intentOrder.map((id) => view.intents[id].utteranceId)));
  utts.forEach((u, i) => add({ id: `Q:${u}`, kind: "query", label: `Q${i + 1}`, sub: u.split(":").pop(), order: i, refId: u }));

  view.intentOrder.forEach((id, i) => {
    const it = view.intents[id];
    add({ id: `SQ:${id}`, kind: "intent", label: id, sub: it.label, order: i, refId: id,
      status: it.retrievalStatus, delta: delta ? (affected.has(id) ? "affected" : "unaffected") : undefined });
    edges.push({ id: `dec:${id}`, kind: "DECOMPOSES", source: `Q:${it.utteranceId}`, target: `SQ:${id}` });
  });

  // Evidence column: chunks any claim cites, plus each retrieval's top-ranked hits (rank <= 2).
  // Lower-ranked hits stay in the inspector and timeline; drawing all of them buries the lineage.
  const cited = new Set(view.versions.flatMap((v) => v.claims.flatMap((c) => c.evidenceIds)));
  const chunkOrder = new Map<string, number>();
  let co = 0;
  const lastRetrieved = new Map<string, string>(); // edge key -> trigger
  for (const r of view.relationships) {
    if (r.type === "RETRIEVED") {
      const cid = r.target.slice(3);
      if (!cited.has(cid) && (r.rank ?? 99) > 2) continue;
      if (!chunkOrder.has(cid)) chunkOrder.set(cid, co++);
      lastRetrieved.set(`${r.source}|${r.target}`, r.trigger ?? "");
    }
  }
  for (const [cid, order] of chunkOrder) {
    const ev = view.evidence[cid];
    const isDeltaChunk = Array.from(lastRetrieved.entries()).some(([k, t]) => k.endsWith(`CH:${cid}`) && t === "DELTA");
    add({ id: `CH:${cid}`, kind: "chunk", label: cid, sub: ev?.sectionTitle, order, refId: cid,
      status: cited.has(cid) ? "CITED" : "RETRIEVED", delta: delta && isDeltaChunk ? "new" : undefined });
  }
  for (const [key, trigger] of lastRetrieved) {
    const [source, target] = key.split("|");
    edges.push({ id: `ret:${key}`, kind: "RETRIEVED", source, target, trigger });
  }

  const claimOrder: string[] = [];
  for (const v of view.versions) for (const c of v.claims) if (!claimOrder.includes(c.id)) claimOrder.push(c.id);
  const latestIds = new Set(latest?.claims.map((c) => c.id) ?? []);
  claimOrder.forEach((cid, i) => {
    const c = view.claims[cid];
    if (!c) return;
    add({ id: `CL:${cid}`, kind: "claim", label: cid, sub: c.text, order: i, refId: cid,
      status: latestIds.has(cid) ? (c.supportStatus === "CONFLICTING" ? "CONFLICTING" : c.status) : "SUPERSEDED",
      delta: delta ? (affected.has(c.subQueryId) ? "affected" : "unaffected") : undefined });
    for (const e of c.evidenceIds) {
      if (seen.has(`CH:${e}`)) edges.push({ id: `sup:${e}:${cid}`, kind: "SUPPORTS", source: `CH:${e}`, target: `CL:${cid}`, status: c.supportStatus });
    }
    for (const d of c.derivedFrom) edges.push({ id: `der:${d}:${cid}`, kind: "DERIVED_FROM", source: `CL:${d}`, target: `CL:${cid}` });
  });

  view.versions.forEach((v, i) => {
    add({ id: `AV:${v.id}`, kind: "version", label: `V${String(v.id).padStart(2, "0")}`, sub: v.kind === "PRESENTATION" ? "format" : v.trigger.toLowerCase(),
      order: i, refId: String(v.id), status: v.kind });
    for (const c of v.claims) if (seen.has(`CL:${c.id}`)) edges.push({ id: `con:${c.id}:${v.id}`, kind: "CONTAINS", source: `CL:${c.id}`, target: `AV:${v.id}` });
  });

  for (const r of view.relationships) {
    if ((r.type === "CONTRADICTS" || r.type === "DUPLICATES") && seen.has(r.source) && seen.has(r.target)) {
      edges.push({ id: r.id, kind: r.type, source: r.source, target: r.target });
    }
  }
  return { nodes, edges };
}

/** Node ids reachable from `start` along graph edges in both directions (provenance path). */
export function connected(model: GraphModel, start: string): Set<string> {
  const out = new Set<string>([start]);
  const down = new Map<string, string[]>();
  const up = new Map<string, string[]>();
  for (const e of model.edges) {
    if (e.kind === "CONTRADICTS" || e.kind === "DUPLICATES") continue;
    (down.get(e.source) ?? down.set(e.source, []).get(e.source)!).push(e.target);
    (up.get(e.target) ?? up.set(e.target, []).get(e.target)!).push(e.source);
  }
  const walk = (dir: Map<string, string[]>, from: string) => {
    const stack = [from];
    while (stack.length) {
      const cur = stack.pop()!;
      for (const nxt of dir.get(cur) ?? []) {
        if (!out.has(nxt)) {
          out.add(nxt);
          stack.push(nxt);
        }
      }
    }
  };
  walk(down, start);
  walk(up, start);
  return out;
}

export function filterModel(model: GraphModel, filter: GraphFilter, view: EngineView): Set<string> | null {
  if (filter === "ALL") return null;
  const keep = new Set<string>();
  const latest = view.versions.at(-1);
  if (filter === "ACTIVE" && latest) {
    for (const id of connected(model, `AV:${latest.id}`)) keep.add(id);
  }
  if (filter === "CONFLICTS") {
    for (const e of model.edges) if (e.kind === "CONTRADICTS") [e.source, e.target].forEach((id) => connected(model, id).forEach((x) => keep.add(x)));
    for (const n of model.nodes) if (n.status === "CONFLICTING") connected(model, n.id).forEach((x) => keep.add(x));
  }
  if (filter === "UPDATED") {
    for (const n of model.nodes) if (n.delta === "affected" || n.delta === "new" || n.status === "REVISED" || n.status === "ADDED")
      connected(model, n.id).forEach((x) => keep.add(x));
  }
  if (filter === "UNSUPPORTED" && latest) {
    for (const u of latest.uncertainty) if (u.sub_query_id) keep.add(`SQ:${u.sub_query_id}`);
    for (const n of model.nodes) if (n.kind === "claim" && (n.status === "UNSUPPORTED" || n.status === "PARTIAL")) keep.add(n.id);
  }
  return keep;
}
