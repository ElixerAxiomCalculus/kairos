# Kairos — Streaming Live RAG as incremental computation

Samsung PRISM GenAI Hackathon 3.0 · Theme 4 (Streaming Live RAG)

Kairos retrieves while you are still talking, splits your question where the evidence splits, and when you
add a constraint it rewrites only the sentences that depend on it. Every sentence is verified against the
corpus before it is shown, and every decision is an event you can trace.

## Demo videos

> [!NOTE]
> **Full Kairos tutorial on YouTube — coming soon.** The published YouTube link will be added here.

### Live voice demonstration

[▶ Watch the Kairos voice demo (MP4, 151 MB)](demos/Kairos_voice_demo.mp4)

This recording demonstrates live microphone input driving the incremental transcript, retrieval controller,
intent decomposition, evidence inspection, grounding state, and answer updates in the existing interface.

```
speech chunks ─▶ QueryFrame ─▶ Retrieval Controller ─▶ Intent Decomposer ─▶ Hybrid Retrieval (parallel)
                  (slots,        WAIT / RETRIEVE /        (evidence-footprint     BM25 + dense → RRF →
                   self-repair)  REFINE / SUPPRESS         segmentation)          rerank → facets → dedup
                                                                                        │
 answer versions ◀─ Grounding Firewall ◀─ Claim Synthesis ◀─ Evidence Graph ◀───────────┘
 V1 → V2 (delta) → V3 (presentation)         (extractive default; Gemini optional)
 every stage ─▶ event bus ─▶ JSONL trace · SQLite · WebSocket UI · lineage validator
```

## Quickstart

**One command (Docker):**

```bash
docker compose up --build          # → http://localhost:8000
```

**Local (Python 3.10+, Node 20+):**

```bash
pip install -r requirements.txt
npm --prefix web ci && npm --prefix web run build
python -m kairos serve             # → http://localhost:8000  (API + UI)
```

The UI also runs without the backend. It then replays traces recorded from the real engine (`MOCK`), which
is the safe mode for a projector demo.

### Live microphone demo

Build the UI, start `python -m kairos serve`, open `http://localhost:8000/live`, click **GO LIVE**, then
click the microphone button. Start/Stop defines one conversational turn; browser recognition segment
boundaries do not. While speaking, the UI labels the mutable browser hypothesis as **PROVISIONAL** and
renders only backend-committed transcript chunks in the event stream.

Chrome or Edge is recommended because Web Speech recognition support is browser-dependent. Recognition
may use a browser/vendor speech service; Kairos does not persist raw audio and does not claim transcription
is local. HTTPS and WSS are required when the demo is hosted remotely.

The microphone protocol sends cumulative, revisioned snapshots:

```json
{"type":"transcript_snapshot","utterance_id":"mic-1","revision":7,"text":"I need a workshop venue in Pune","asr_final":false,"end_of_turn":false,"source":"microphone"}
```

The server responds with `transcript_ack` (`accepted`, `duplicate`, `stale`, or `finalized`) and continues
to emit ordinary `{ "type": "event", "event": ... }` engine messages. Stop sends one final snapshot with
`end_of_turn: true`; it does not also send `utterance_end`. Reconnect uses
`/ws?session_id=<id>&resume_token=<capability>&after_event_id=<id>` and replays missed authoritative events.
The token is issued in the initial `hello`; trace and delete requests require the same capability. Legacy `chunk`,
`utterance_end`, `replay`, and `end_session` messages remain supported.

Transcript text is included in JSONL/SQLite telemetry by default. Set `telemetry.trace_content: false` in
configuration to redact text-bearing payload fields while retaining event IDs and causal lineage.

| Command | What it does |
|---|---|
| `python -m kairos serve` | API, WebSocket and UI on :8000 |
| `python -m kairos replay pune-demo` | Runs a demo script through the engine and prints the event trace and lineage coverage |
| `python -m kairos bench` | Dev benchmark (two repeats plus a reproducibility check) → `runs/bench/benchmark/latest.json` |
| `python -m kairos semantic-bench` | V2.1 intent, relevance, grounding, streaming, external-label, and ablation report → `runs/bench/semantic/latest.json` |
| `python -m kairos hardening-bench --max-documents 5000` | Pinned official-corpus scale, concurrency, isolation, and short-soak report → `runs/bench/hardening/latest.json` |
| `python -m kairos import-scifact-eval` | Imports official SciFact test qrels/queries as non-indexed evaluation labels |
| `python -m kairos ablate` | Ablations A, B, C and E → `runs/bench/experiments/latest.json` |
| `python -m kairos train-stability` | Fits the stability model on the training streams (self-supervised) |
| `python -m kairos tune-controller` | Label-free tuning of controller cost weights on the training streams |
| `python -m kairos record-demo` | Re-records the UI's mock fixtures from the real engine |
| `python -m kairos replay multi-domain-stress` | Runs the long mixed-domain/OOD speech regression through the real engine |
| `python -m kairos import-corpus --dataset squad --destination corpus/imported/squad --dry-run` | Validates a bounded external-corpus import without writing it |
| `python -m kairos reproduce` | train → bench → ablate, in one command (`docker compose --profile bench run --rm bench`) |

## V2.1 semantic reliability and engineering hardening

V2.1 adds auditable EFIS objective/constraint/topic signals, objective-aware evidence admission, strict
versioned SHA-256 citations, PSRC decision/calibration diagnostics, authenticated session resume/trace access,
and race-safe microphone finalization. It also expands the evaluation to 100 explicit EFIS pairs, 48 SQuAD
labels, the official SciFact qrels intersecting the bounded corpus, 13 grounding adversaries, and a 5,000-
document official-corpus scale run.

Semantic run `20261018T154338Z` measured relevant-evidence precision 0.8750, expanded SQuAD Recall@5
0.8125 over 48 labels, and grounding accuracy 0.8462 over 13 adversaries. EFIS accuracy improved from 0.7100
to 0.7400 and its false-merge rate fell from 0.2241 to 0.1207, but merge recall fell from 0.6190 to 0.5476
and F1 from 0.6420 to 0.6389. The single matched official SciFact query scores 1.0 and is explicitly too small
to generalize. Read the [V2.1 hardening report](docs/KAIROS_V2_1_REPORT.md), historical
[V2 report](docs/KAIROS_V2_REPORT.md), and [corpus provenance](docs/CORPUS_PROVENANCE.md).

### Current published metrics

Final V2.1 development run `20261018T154934Z` uses the same 26-session project-authored development set as
the preserved baseline. Targets and measurements remain separate on the `/benchmark` page.

| Gate | Metric | Target | V2.1 measured | Status |
|---|---|---:|---:|---|
| G1 | Reproducibility delta | = 0 | 0.0000 | PASS |
| G2 | Early retrieval rate | >= 0.80 | **0.6875** | **FAIL** |
| G3 | Intent recall | >= 0.70 | 0.9444 | PASS |
| G4 | Grounded claim rate | >= 0.85 | 1.0000 | PASS |
| G4 | Fabricated document IDs | = 0 | 0 | PASS |
| G5 | Delta retrieval ratio | < 1.00 | 0.7000 | PASS |
| G5 | Presentation-turn false retrieval | = 0 | 0.0000 | PASS |
| G6 | Telemetry lineage coverage | = 1.00 | 1.0000 | PASS |

Additional V2.1 development measurements: intent precision 0.8947, intent-count accuracy 0.8846,
refinement correctness 1.0000, evidence survival 0.7333, premature retrieval 0.2963, mean retrieval latency
1.830 ms, and post-utterance latency p50/p95 of 6.229/13.790 ms. These are development measurements on
the 37-chunk sample corpus, not official third-party benchmark scores.

The pinned SciFact scale study reached 5,000 documents / 11,732 chunks: 114.20 s index build, 1.66 GiB
traced Python peak, 69.7 ms search p50, 1,047.9 ms p95, and 3.77 single-session QPS. Eight concurrent demo
sessions retained isolation. These laptop measurements use only 20 timed searches per scale point; the index
is memory-only and the short soak is not a production endurance claim.

### RAG corpus datasets, licenses, and citations

| Corpus | Indexed subset | License | Required attribution |
|---|---:|---|---|
| Project sample corpus | 8 documents / 37 chunks | Project-authored fictional demo content | KAIROS project |
| [SciFact via BEIR](https://huggingface.co/datasets/BeIR/scifact) + [official qrels](https://huggingface.co/datasets/BeIR/scifact-qrels) | 12 documents / 29 chunks; 1 intersecting test query kept outside retrieval | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Wadden et al., 2020](https://aclanthology.org/2020.emnlp-main.609/) and [Thakur et al., 2021](https://openreview.net/forum?id=wCu6T5xFjeJ) |
| [SQuAD 1.1](https://huggingface.co/datasets/rajpurkar/squad) | original 8-context profile plus expanded 21-context profile; 48 Q&A labels kept outside retrieval | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Rajpurkar et al., 2016](https://aclanthology.org/D16-1264/) |

The exact revisions, response and document SHA-256 values, source-ID mappings, import commands, and
share-alike/usage cautions are recorded in [Corpus provenance](docs/CORPUS_PROVENANCE.md).

## What makes it different

| Problem in conversational RAG | Kairos mechanism | Code |
|---|---|---|
| Retrieval waits for the end of the turn | **PSRC controller.** Signals ρ (result stability, RBO of BM25 probes), χ (clarity), κ (completeness), σ (slot stability), π (continuation) and ν (novelty) feed a calibrated ŝ. It retrieves when ŝ ≥ θ = 1 − λ_L·H / (λ_C·C_r + λ_N·N) | `kairos/controller/policy.py` |
| One vector for three questions | **EFIS.** Two clauses are separate intents iff their evidence footprints diverge (RBO₀.₈ of parent sections < 0.4). Constraint clauses attach to the need they modify | `kairos/decomposer/efis.py` |
| Refinement restarts everything | **Claim-dependency delta.** A late constraint re-retrieves only sub-queries whose *cited* evidence depends on it. Each claim is then KEEP, REVISE, RETRACT or ADD; kept claims carry over verbatim | `kairos/delta/engine.py` |
| "Make it shorter" triggers retrieval | Turn classifier plus **SUPPRESS**. The answer is re-rendered from its claim set, and a transformation firewall proves no new facts, numbers or citations were added | `kairos/controller/turns.py`, `kairos/synthesis/transform.py` |
| Decorative citations | **Grounding firewall.** A citation must be retrieved in this session with a matching versioned SHA-256 content hash. The quote must be verbatim, numeric and name anchors must be present, and support ≥ τ. Numeric conflicts are surfaced, never silently resolved | `kairos/grounding/firewall.py` |
| Opaque behaviour | Every stage emits a parented event, and a lineage validator checks six invariants on every run | `kairos/telemetry/` |

Facets such as `standard | domestic | international` are **derived from the corpus**: sibling headings that
differ only by their leading modifier. V2's separate domain router uses a small configured, explainable
lexicon; it does not change how facets are discovered.

## Preserved pre-V2 measured results (dev set)

Run `20260928T201927Z` · `benchmark/datasets/dev` (26 sessions, 36 utterances) · extractive synthesis ·
2 repeats. These are **development-set measurements**. The dev set was authored by the same team that built
the system, and some behaviour was iterated on it (see *Limitations*). They are not official benchmark results.

| Gate | Metric | Target | Measured | Status |
|---|---|---|---|---|
| G1 | Reproducibility Δ (non-timing metrics, 2 runs) | = 0 | 0 | PASS |
| G2 | Early retrieval rate (eligible utterances) | ≥ 0.80 | **0.69** | **FAIL** |
| G3 | Intent identification (recall) | ≥ 0.70 | 0.97 | PASS |
| G4 | Grounded claim rate (66 claims, automatic judge) | ≥ 0.85 | 1.00 | PASS |
| G4 | Fabricated document IDs | = 0 | 0 | PASS |
| G5 | Delta retrieval ratio | < 1 | 0.87 | PASS |
| G5 | False retrieval rate (presentation turns) | = 0 | 0.00 | PASS |
| G6 | Telemetry lineage coverage | = 1 | 1.00 | PASS |

Other measurements from the same run:
- **Early retrieval:** premature retrieval rate 0.19, evidence survival 0.76, median lead 1.9 s before the speaker finished.
- **Intents:** precision 0.95, count accuracy 0.96, gold-section MRR 0.85.
- **Grounding and turns:** abstention on unanswerable questions 2/2, turn classification 1.00.
- **Refinement:** correctness 0.60 (3 of 5 constraint turns exactly right).
- **Latency:** mean retrieval 0.23 ms, post-utterance latency p50 3.3 ms / p95 7.2 ms. These are CPU, extractive-mode timings on a 37-chunk corpus.

**Ablations** (run `20260928T204603Z`, same dev set):

| Experiment | Result |
|---|---|
| A · controller | PSRC 0.69 early rate at 1.81 retrievals per utterance (premature 0.19). A fixed 4-token trigger reaches **0.94** but at 2.35 retrievals per utterance (premature 0.26). Fixed 8-token: 0.63. End-of-utterance: 0. PSRC is **not** Pareto-dominant on early rate; its advantage is fewer and less premature retrievals. |
| B · retrieval | Hybrid, dense-only and sparse-only are indistinguishable on this small corpus (MRR 0.85). Removing the reranker drops MRR to 0.81 and intent recall to 0.92. A larger corpus is needed to separate the retrievers. |
| C · refinement | Delta: 7 re-retrievals and ≈1.8k context tokens vs full restart: 9 and ≈2.4k. Refinement correctness 0.60 vs 0.40. With the deterministic extractive synthesizer a full restart does not churn text, so the saving is retrieval and context. |
| E · decomposition | EFIS intent recall 0.97 vs 0.69 for a single whole-utterance query. |
| D · graph + firewall vs flat context | **Not run.** It needs a generative backend to be meaningful (extractive claims are verbatim quotes). |

Every number above is in `runs/bench/`; the UI's Benchmark and Experiments pages display these files and show
their provenance. Targets are never displayed as measurements.

## Limitations (honest list)

- **G2 is not met on dev** (0.69 < 0.80). Short two-chunk questions often only become determinable in their last chunk.
- **Tuning disclosure:** the controller was tuned label-free on the training streams, not on dev. The decomposer, delta rules and sufficiency check were iterated while inspecting dev failures.
- **EFIS tradeoff:** on 100 project-authored pairs, V2.1 reduces false merges but increases false splits; merge F1 is 0.6389. The set is not independently annotated.
- **External-label limits:** expanded SQuAD Recall@5 is 0.8125 (48 labels). Only one official SciFact test query intersects the bounded 12-document corpus, so its 1.0 result is not a general benchmark score.
- **Historical delta result.** The pre-V2 implementation over-triggered when one document mixed general and constraint-specific sections (for example S26), scoring 0.60 on the original run. V2's scope guard reaches 1.00 exact refinement correctness on the same development cases, but that set was used during iteration and is not an independent test.
- **Automatic grounding judge.** It is lexical (verbatim quote, token recall and numeric anchors) and independent of the firewall's IDF-weighted score. `runs/bench/benchmark/audit_sheet.csv` is prepared for a human audit, which has **not** been done yet.
- **Extractive by default.** Every claim is a quoted sentence. That makes grounding strong by construction but prose less fluent; generative synthesis (Gemini) is optional and held to the same firewall.
- **Scale ceiling.** The default corpus remains 8 documents / 37 chunks. The official 5,000-document stress run reaches 1.66 GiB traced peak and 1.05 s retrieval p95; the index is memory-only.
- **Grounding semantics.** Strict content hashes, retrieved-ID checks and quotes do not solve entailment: the lexical firewall still accepted negation and modality adversaries.
- **Security audit.** Production npm dependencies audit clean. Seven dev-tool advisories remain because remediation requires breaking Cypress/Storybook upgrades; see the V2.1 report.
- **Browser speech recognition.** Availability, accuracy, and network behavior depend on the browser and
  speech provider. Automated tests simulate the browser API; a real microphone smoke test is still required
  on the machine used for the presentation.

## Generative synthesis with Gemini (optional, free tier)

Put `GEMINI_API_KEY=...` in `kairos/.env` (git-ignored, and never copied into the Docker image), then:

```bash
python -m kairos --config configs/gemini.yaml serve
```

- **Model:** `gemini-flash-latest` (the free-tier alias; the resolved model version is recorded).
- **Low latency settings:** low thinking level and JSON-schema output.
- **Closed evidence handles:** the model sees `E1..En`, never document IDs.
- **Retries:** 429/503 errors are retried; if Gemini stays unavailable, synthesis falls back to extractive and that is labelled in telemetry.
- **Caching:** every response is cached under `runs/llm_cache/`. Set `synthesis.llm_mode: replay` to re-run from the cache only.

Measured on this machine, Gemini free-tier calls took 6–20 s, so the live demo uses extractive synthesis.

## Repository

```
kairos/                 engine (Python)
  streaming/frame.py    QueryFrame: clauses, candidate needs, slots, self-repair
  controller/           PSRC policy, turn classifier, self-supervised training, label-free tuning
  decomposer/efis.py    evidence-footprint intent segmentation
  retrieval/            BM25, dense (LSA default / sentence-transformers), hybrid index
  evidence/graph.py     typed evidence graph
  synthesis/            extractive, Gemini/Ollama (closed handles, record/replay), transforms
  grounding/firewall.py claim verification, conflicts, repair
  delta/engine.py       claim-dependency delta maintenance
  telemetry/            event bus, JSONL/SQLite/WebSocket sinks, lineage validator
  engine.py             per-session orchestrator (ephemeral state)
  api/server.py         FastAPI REST + WebSocket, serves web/dist
benchmark/              datasets (dev, train), runner, ablations
corpus/sample/          synthetic policy corpus (fictional organisation)
demo/scripts/           timestamped demo transcripts
configs/                default.yaml (all tunables), gemini.yaml
models/stability.json   trained stability model
tests/                  pytest (engine, firewall, isolation, leakage guard, API/WebSocket)
web/                    React + TypeScript UI (see web/README.md)
```

## Tests

```bash
python -m pytest -q                                   # 41 backend tests
cd web && npm run build && npx vite preview --port 4173 &
npx cypress run                                       # 10 UI acceptance tests (mock mode)
npx cypress run --spec cypress/e2e/live.cy.ts --config baseUrl=http://127.0.0.1:8000 --env LIVE=1   # against the live backend
```

On Windows inside VS Code, run Cypress with `ELECTRON_RUN_AS_NODE` unset.

## Demo script (≈4 minutes)

Open `/live`, press **Presentation** (`P`) and **Demo** (`D`), then use `pune-demo` and press play (or step with `→`/**CHUNK**).

1. **"I need to book a…"** The controller stays on **WAIT**: incomplete clause, ambiguous probe, ŝ well below θ.
2. **"…workshop venue in Pune"** The controller switches to **RETRIEVE** mid-sentence, marked on the transcript.
3. **"…for 30 people,"** The controller issues **REFINE**, because the constraint would change the results.
4. **"…cancellation policy and whether catering is available."** Three intents appear and retrieve in parallel. **V1** follows, with 6/6 claims verified and citations you can click into the Evidence Inspector.
5. **"Actually, this is for an international event."** Delta update:
   - only 2 of 3 intents re-retrieve;
   - the delta banner shows 2/6 claims invalidated and 4 carried verbatim;
   - retrieval ops are compared with a full restart;
   - press `2` for evidence focus to see the affected branches in violet.
6. **"Make that shorter."** The controller shows **SUPPRESS**. Vector search, reranker and fusion are skipped; **V3** is a presentation change checked by the transformation firewall.
7. Click any timeline event to trace lineage back to the transcript. `conflict-demo` shows contradictory sources surfaced side by side.
