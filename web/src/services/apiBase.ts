/**
 * Backend location. Empty (default) = same origin, which is how `python -m kairos serve` and
 * `npm run dev` (proxy) work. For split hosting (e.g. Netlify UI + Render API) set at build time:
 *   VITE_API_BASE=https://kairos-api.onrender.com
 */
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export function apiUrl(path: string): string {
  return `${BASE}${path}`;
}

export function wsUrl(): string {
  if (BASE) return `${BASE.replace(/^http/, "ws")}/ws`;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}
