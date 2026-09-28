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
python -m pytest -q                     # backend tests
python -m kairos bench                  # dev benchmark → runs/bench/benchmark/latest.json
```

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
npm run storybook                       # component stories on :6006
```

On Windows inside VS Code, run Cypress with `ELECTRON_RUN_AS_NODE` unset
(PowerShell: `Remove-Item Env:ELECTRON_RUN_AS_NODE`).
