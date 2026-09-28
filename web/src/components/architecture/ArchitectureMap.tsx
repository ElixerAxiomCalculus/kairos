import { useState } from "react";
import styles from "@/routes/routes.module.css";

interface Module {
  key: string;
  title: string;
  what: string;
  why: string;
  events: string[];
  code: string;
}

export const MODULES: Module[] = [
  { key: "TRANSCRIPT STREAM", title: "Transcript stream", what: "Timestamped chunks arrive from replay, typed text, or the browser microphone.", why: "Speech is incremental; the engine must act on prefixes, not finished questions.", events: ["TranscriptChunkReceived"], code: "kairos/engine.py" },
  { key: "QUERY STATE", title: "Query state", what: "Each prefix becomes a QueryFrame: clauses, candidate needs, slots, self-repairs ('Pune, sorry, Mumbai').", why: "Constraints ('for 30 people') attach to the need they modify instead of becoming queries.", events: ["QueryFrameUpdated", "TurnClassified"], code: "kairos/streaming/frame.py" },
  { key: "RETRIEVAL CONTROLLER", title: "Retrieval controller (PSRC)", what: "Signals ρ, χ, κ, σ, π, ν → calibrated ŝ; retrieve when ŝ ≥ θ = 1 − λ_L·H / (λ_C·C_r + λ_N·N).", why: "Retrieve when the corpus's own ranking stops moving, not after a token count. SUPPRESS presentation turns; REFINE when results would change.", events: ["ProbeExecuted", "RetrievalDecision", "RetrievalSuppressed"], code: "kairos/controller/policy.py" },
  { key: "INTENT DECOMPOSER", title: "Intent decomposer (EFIS)", what: "Two clauses are separate intents iff their evidence footprints diverge (RBO of parent sections < θ).", why: "One vector for three questions retrieves for the dominant one only.", events: ["SubQueriesGenerated"], code: "kairos/decomposer/efis.py" },
  { key: "HYBRID RETRIEVAL", title: "Hybrid retrieval", what: "BM25 + dense (exact inner product) → RRF → reranker → facet filter → dedup; parallel per intent.", why: "Sparse catches exact policy vocabulary; dense catches paraphrase; the reranker yields an absolute relevance used for abstention.", events: ["RetrievalStarted", "RetrievalCompleted", "EvidenceFused"], code: "kairos/retrieval/index.py" },
  { key: "EVIDENCE GRAPH", title: "Evidence graph", what: "SubQuery → Chunk → Claim → AnswerVersion with SUPPORTS, CONTRADICTS, DUPLICATES, DERIVED_FROM edges.", why: "It is the dependency structure the delta engine walks when a constraint arrives.", events: ["EvidenceFused", "DeltaPlanComputed"], code: "kairos/evidence/graph.py · kairos/delta/engine.py" },
  { key: "GROUNDING FIREWALL", title: "Grounding firewall", what: "Citation must be retrieved in-session with a matching SHA; quote verbatim; anchors present; support ≥ τ; numeric conflicts flagged.", why: "Fabricated IDs are rejected by construction; unsupported claims become explicit uncertainty.", events: ["GroundingVerified", "TransformVerified"], code: "kairos/grounding/firewall.py" },
  { key: "SYNTHESIS", title: "Synthesis", what: "Extractive by default (verbatim evidence sentences); Gemini/Ollama optional with closed evidence handles.", why: "Deterministic, reproducible claims; generative backends are held to the same firewall.", events: ["ClaimsGenerated"], code: "kairos/synthesis/" },
  { key: "ANSWER VERSIONING", title: "Answer versioning", what: "Every answer is a version with parent, diff (kept / revised / added / retracted) and counterfactual cost.", why: "Late constraints recompute only dependent claims; presentation turns re-render without retrieval.", events: ["AnswerVersionCreated", "TelemetryEmitted"], code: "kairos/engine.py" },
];

/** Clean interactive flow: click a module for what it does, why it exists, and the events it emits. */
export function ArchitectureMap() {
  const [active, setActive] = useState(MODULES[2].key);
  const m = MODULES.find((x) => x.key === active) ?? MODULES[0];
  return (
    <div className={styles.arch} data-testid="architecture-map">
      <div className={styles.flow}>
        {MODULES.map((mod, i) => (
          <div key={mod.key}>
            {i > 0 ? <div className={styles.connector} aria-hidden /> : null}
            <button className={styles.module} aria-pressed={active === mod.key} onClick={() => setActive(mod.key)}>
              <span className="mono muted">{String(i + 1).padStart(2, "0")}</span>
              <span className="px">{mod.key}</span>
              <span className="mono muted" style={{ fontSize: 11 }}>{mod.events.length} evt</span>
            </button>
          </div>
        ))}
      </div>
      <aside className={styles.detail} aria-live="polite">
        <span className="px muted">MODULE</span>
        <h3>{m.title}</h3>
        <p style={{ margin: 0 }}>{m.what}</p>
        <p className="secondary" style={{ margin: 0 }}><span className="px muted">WHY </span>{m.why}</p>
        <div><span className="px muted">EMITS </span><span className="mono">{m.events.join(" · ")}</span></div>
        <div><span className="px muted">CODE </span><span className="mono">{m.code}</span></div>
      </aside>
    </div>
  );
}
