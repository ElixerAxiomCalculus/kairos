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
python -m kairos semantic-bench         # V2 held-out/retrieval/grounding/scale report
```

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
