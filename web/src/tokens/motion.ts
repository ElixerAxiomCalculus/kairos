/** Centralised motion constants (mirrors tokens.css). Framer Motion durations are in seconds. */
export const DUR = {
  fast: 0.14,
  normal: 0.24,
  graph: 0.45,
} as const;

export const EASE = {
  standard: [0.2, 0, 0, 1] as [number, number, number, number],
  emphasis: [0.3, 0, 0, 1.15] as [number, number, number, number],
};

export const transition = {
  fast: { duration: DUR.fast, ease: EASE.standard },
  normal: { duration: DUR.normal, ease: EASE.standard },
  graph: { duration: DUR.graph, ease: EASE.standard },
  layout: { type: "spring" as const, stiffness: 380, damping: 38, mass: 0.9 },
};

/** Minimum time a retrieval pulse stays visible so millisecond retrievals remain perceivable. */
export const PULSE_HOLD_MS = 900;

export const STATUS_VAR: Record<string, string> = {
  WAIT: "var(--status-wait)",
  RETRIEVE: "var(--status-retrieve)",
  SUPPRESS: "var(--status-suppress)",
  REFINE: "var(--status-refine)",
  PASS: "var(--status-pass)",
  FAIL: "var(--status-fail)",
  SUPPORTED: "var(--status-pass)",
  CONFLICTING: "var(--status-conflict)",
  PARTIAL: "var(--status-suppress)",
  UNSUPPORTED: "var(--status-fail)",
  INVALID_CITATION: "var(--status-fail)",
  INSUFFICIENT_EVIDENCE: "var(--status-wait)",
  INVALID_GROUNDING: "var(--status-fail)",
  EARLY: "var(--status-retrieve)",
  END_OF_UTTERANCE: "var(--status-retrieve)",
  DELTA: "var(--status-refine)",
  NONE: "var(--status-wait)",
};
