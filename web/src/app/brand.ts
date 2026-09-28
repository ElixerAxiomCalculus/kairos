/** Branding lives in one place so the product name can be swapped without touching components. */
export const BRAND = {
  name: "KAIROS",
  mark: "KAIROS/RAG",
  tagline: "Streaming Evidence Engine",
  eyebrow: "SAMSUNG PRISM / LIVE RAG",
  hero: ["Retrieval,", "before", "the sentence ends."],
  lede:
    "A streaming evidence engine that predicts retrieval, decomposes intent, refines answers by delta, and exposes every decision.",
  principles: [
    { key: "PREDICT", text: "Retrieve when the corpus's own ranking stops moving — not after a token count." },
    { key: "DECOMPOSE", text: "Split a question where its evidence splits." },
    { key: "GROUND", text: "No claim reaches the screen before it is verified against a cited chunk." },
    { key: "REFINE", text: "A late constraint recomputes only the claims that depend on it." },
  ],
} as const;
