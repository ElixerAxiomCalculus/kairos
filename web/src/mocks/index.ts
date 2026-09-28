/**
 * Recorded engine traces (not hand-authored). Regenerate with:
 *   python -m kairos record-demo
 * Each file is the exact event stream the backend emitted for a demo script.
 */
import scriptIndex from "./fixtures/index.json";

export interface ScriptInfo {
  id: string;
  title: string;
  description: string;
  chunks: number;
  duration_s: number;
}

export const SCRIPTS: ScriptInfo[] = scriptIndex as ScriptInfo[];
export const DEFAULT_SCRIPT = "pune-demo";

const loaders: Record<string, () => Promise<{ default: unknown }>> = {
  "pune-demo": () => import("./fixtures/pune-demo.events.json"),
  "reimbursement-demo": () => import("./fixtures/reimbursement-demo.events.json"),
  "conflict-demo": () => import("./fixtures/conflict-demo.events.json"),
  "self-repair-demo": () => import("./fixtures/self-repair-demo.events.json"),
};

export async function loadFixture(scriptId: string): Promise<unknown[]> {
  const loader = loaders[scriptId];
  if (!loader) throw new Error(`Unknown recorded script: ${scriptId}`);
  const mod = await loader();
  return mod.default as unknown[];
}

export async function loadBenchmarkSnapshot(): Promise<unknown> {
  const mod = await import("./fixtures/benchmark.latest.json");
  return mod.default;
}
