# Local run guide

Prerequisites: Python 3.10+, Node 20+, and (optionally) Docker Desktop.

## 1. Backend (engine + API)

```bash
pip install -r requirements.txt
python -m kairos serve                  # http://localhost:8000  (API docs: /docs)
```

Quick checks without the UI:

```bash
python -m kairos replay pune-demo       # print a demo's event trace
python -m kairos replay multi-domain-stress  # mixed-domain and insufficient-evidence regression
python -m pytest -q                     # backend tests
python -m kairos bench                  # dev benchmark → runs/bench/benchmark/latest.json
python -m kairos semantic-bench         # V2.1 semantics, external labels, grounding, ablations
python -m kairos hardening-bench --max-documents 5000  # official-corpus scale/concurrency/short soak
```

PSRC V2.2 research and frozen-selection workflow:

```bash
python -m kairos controller-data build
python -m kairos controller-data validate
python -m kairos controller-train
python -m kairos controller-calibrate
python -m kairos controller-experiments
python -m kairos controller-oracle
python -m kairos controller-compare
python -m kairos controller-select
python -m kairos controller-heldout
python -m kairos controller-report
```

The complete experiment command writes `runs/bench/controller_v2_2/latest.json` plus the model comparison,
oracle, cost, selection, held-out, and 12-plot artifact set. Original PSRC remains the default unless a later
candidate passes all frozen promotion filters.

IC-COS V2.3 research workflow:

```bash
python -m kairos controller-v2-3-data build
python -m kairos controller-v2-3-data validate
python -m kairos controller-v2-3-train
python -m kairos controller-v2-3-experiments
python -m kairos controller-v2-3-report
```

The experiment writes `runs/bench/controller_v2_3/latest.json` and the frozen selection bundle. IC-COS did
not pass the promotion gates in run `20261019T051008Z`; it remains disabled by default. The validated evidence
cache is an independently selected execution optimization. See the V2.3 research and engineering reports for
the corrected intent denominators, latency definitions, and final-set limitations.

### Corpus profiles and reproducible imports

The default `sample` profile stays small. The repository also contains bounded, attributed `science`
(SciFact), `general` (SQuAD contexts), and `multi_domain_demo` profiles:

```bash
python -m kairos --config configs/science.yaml serve
python -m kairos --config configs/general.yaml serve
python -m kairos --config configs/multi-domain.yaml serve
```

Validate or reproduce imports with pinned revisions and checksums from
[`docs/CORPUS_PROVENANCE.md`](docs/CORPUS_PROVENANCE.md):

```bash
python -m kairos import-corpus --dataset scifact --destination corpus/imported/scifact --validate-only
python -m kairos import-corpus --dataset squad --destination corpus/imported/squad --dry-run
python -m kairos import-scifact-eval --destination corpus/imported/scifact
```

Imports use the Hugging Face Dataset Viewer API. Automated adapter tests instead use tiny local fixtures and
never require network access. SQuAD questions and answers are written to `evaluation-labels.jsonl`, not the
retrievable Markdown corpus.

## 2. Frontend

**Served by the backend** (one port):

```bash
npm --prefix web ci
npm --prefix web run build              # then open http://localhost:8000
```

**Dev server with hot reload** (backend must be running on :8000):

```bash
cd web
npm run dev                             # http://localhost:5173 (proxies /api and /ws to :8000)
```

With no backend running, the UI replays recorded sessions (status shows `MOCK`). Click **GO LIVE** in the
bottom bar to switch to the live engine.

For the microphone path, use Chrome or Edge, open `/live`, click **GO LIVE**, allow microphone access, and
use the microphone button as an explicit Start/Stop boundary. Ask the Pune workshop question, then start a
second turn with “Actually, this is for an international event,” and a third with “Make that shorter.” The
controller should show early RETRIEVE, then REFINE, then SUPPRESS, producing V1/V2/V3.

The application never stores raw audio. Browser speech recognition may send audio to the browser vendor.
Transcript payloads are persisted in trace telemetry unless `telemetry.trace_content` is set to `false`.

### Operator microphone smoke test

Run this on the exact presentation machine because automated tests cannot validate real audio hardware:

1. Serve the production build over HTTPS/WSS (localhost HTTP is acceptable for local browser testing) and set `KAIROS_CORS_ORIGINS` to the exact UI origin.
2. In current Chrome and Edge, allow the intended microphone, click **GO LIVE**, start once, speak a partial sentence, pause, continue, then stop once.
3. Confirm provisional text changes without duplicate committed chunks; after Stop, exactly one `end_of_turn` snapshot is acknowledged and no late recognizer callback changes the turn.
4. Start a second turn immediately and verify it has a new utterance ID and no text from the first turn.
5. Deny microphone permission, disconnect the device, background/restore the tab, and interrupt network access; each state must surface an error or reconnect without finalizing stale text.
6. Resume the same session and confirm the capability token is supplied, missed events replay once, and a wrong token is rejected.
7. Inspect telemetry for transcript content before recording a public demo; set `telemetry.trace_content: false` when text retention is not acceptable.

Hardware/vendor-ASR and hosted reverse-proxy validation are currently **PENDING**.

## 3. Containers

```bash
docker compose up --build                          # http://localhost:8000
docker compose --profile bench run --rm bench      # train → benchmark → ablations into ./runs
```

## 4. Optional: Gemini synthesis

Add `GEMINI_API_KEY=...` to `kairos/.env`, then:

```bash
python -m kairos --config configs/gemini.yaml serve
```

## 5. UI tests

```bash
cd web
npm run build
npx vite preview --port 4173            # keep running in a second terminal
npx cypress run                         # recorded-demo acceptance tests
npx cypress run --spec cypress/e2e/live.cy.ts --config baseUrl=http://127.0.0.1:8000 --env LIVE=1
npm run storybook                       # component stories on :6006
```

On Windows inside VS Code, run Cypress with `ELECTRON_RUN_AS_NODE` unset
(PowerShell: `Remove-Item Env:ELECTRON_RUN_AS_NODE`).
