import dayjs from "dayjs";
import type { EvidenceChunk } from "@/types/domain";

export const DASH = "—";

/** Session clock seconds -> 00:04.241 */
export function clock(t: number | undefined | null): string {
  if (t === undefined || t === null || Number.isNaN(t)) return DASH;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}

export function ms(v: number | undefined | null, digits = 1): string {
  if (v === undefined || v === null || Number.isNaN(v)) return DASH;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(v < 10 ? Math.max(digits, 2) : digits)} ms`;
}

export function ratio(v: number | undefined | null, digits = 0): string {
  if (v === undefined || v === null || Number.isNaN(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

export function score(v: number | undefined | null, digits = 2): string {
  if (v === undefined || v === null || Number.isNaN(v)) return DASH;
  return v.toFixed(digits).replace(/^0\./, ".");
}

export function int(v: number | undefined | null): string {
  if (v === undefined || v === null || Number.isNaN(v)) return DASH;
  return Math.round(v).toLocaleString("en-US");
}

export function pad(n: number, width = 2): string {
  return String(n).padStart(width, "0");
}

export function wall(iso: string | undefined): string {
  return iso ? dayjs(iso).format("HH:mm:ss") : DASH;
}

/** D03:C04 + section 3.2 -> "D03 · §3.2 · C04" */
export function citationLabel(chunkId: string, evidence?: EvidenceChunk): string {
  const [doc, chunk] = chunkId.split(":");
  const section = evidence?.section && !evidence.section.startsWith("s") ? ` · §${evidence.section}` : "";
  return `${doc}${section} · ${chunk ?? ""}`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

export function humanReason(code: string): string {
  const map: Record<string, string> = {
    STABLE_ABOVE_THRESHOLD: "stability ≥ θ",
    RESULTS_PLATEAU: "results plateaued",
    CLAUSE_CLOSED: "clause closed",
    INCOMPLETE_CLAUSE: "clause incomplete",
    BELOW_THRESHOLD: "stability < θ",
    RESULTS_STILL_MOVING: "results still moving",
    AMBIGUOUS_PROBE: "ambiguous probe",
    END_OF_UTTERANCE: "end of utterance",
    NO_CANDIDATE_NEED: "no information need yet",
    ALREADY_RETRIEVED: "already retrieved",
    CONSTRAINT_ABSORBED: "constraint absorbed",
    RESULTS_WOULD_CHANGE: "results would change",
    SLOT_REVISED: "slot revised",
    LATE_CONSTRAINT: "late constraint",
    CONSTRAINT_INCOMPLETE: "constraint incomplete",
    DELTA_ALREADY_APPLIED: "delta applied",
    PRESENTATION_TRANSFORMATION: "presentation only",
    SUFFICIENT_SESSION_EVIDENCE: "session evidence suffices",
    DUPLICATE_OF_PRIOR_RETRIEVAL: "duplicate of prior retrieval",
    AWAITING_END_OF_UTTERANCE: "awaiting end of utterance",
    BUDGET_EXHAUSTED: "retrieval budget exhausted",
    NO_CONTENT: "no content terms",
    SESSION_EVIDENCE_INSUFFICIENT: "session evidence insufficient",
  };
  return map[code] ?? code.toLowerCase().replace(/_/g, " ");
}
