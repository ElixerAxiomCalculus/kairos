# Kairos UI

React 19 · TypeScript · Vite · Radix (behaviour only) · Framer Motion · D3 (intent and evidence graphs) ·
Recharts (sparklines and benchmark charts) · Zustand · Zod · React Hook Form · TanStack Table · Day.js ·
Storybook · Cypress.

Design language: *Monochrome Computational Editorialism*.

| Role | Typeface | Used for |
|---|---|---|
| Machine state | Silkscreen (pixel) | Controller states, labels, versions |
| Human language | Cormorant Garamond | Headings and the query |
| Interaction | Inter | Answers and controls |
| Data | IBM Plex Mono | IDs, scores and timings |

The UI is about 90% monochrome. Colour carries state: WAIT is graphite, RETRIEVE cyan, SUPPRESS amber,
REFINE violet, PASS green and FAIL red. All design values are CSS variables in `src/tokens/tokens.css`.

```bash
npm ci
npm run dev          # :5173, proxies /api and /ws to python -m kairos serve on :8000
npm run build        # dist/ is served by the Python server
npm run storybook    # component states from recorded engine traces
```

## How data flows

```
StreamingTransport (WebSocketTransport | MockTransport)
      └─▶ streamStore.events (append-only log) ──cursor──▶ view = fold(events[0..cursor])
                                                           └─▶ components select slices
```

- The whole UI state is a pure fold over backend events (`features/streaming/reducer.ts`). That makes pause, step back (←), jump to V1/V2/V3 and replay deterministic, and it means the UI cannot show anything no event reported.
- `MockTransport` serves traces **recorded from the real engine** (`src/mocks/fixtures`, regenerated with `python -m kairos record-demo`). It fabricates nothing, and the connection status shows `MOCK` or `REPLAY`.
- Components never touch sockets: `features/sessions/sessionController.ts` owns the transport.

## Workspace

| Area | Content |
|---|---|
| Left | Live transcript, with per-chunk ŝ/σ/ν/κ, ⚡ markers where retrieval fired, and the semantic-stability instrument (ŝ trace against θ, plus a gauge) |
| Centre | Controller (WAIT / RETRIEVE / SUPPRESS / REFINE), evolving query with a diff, intent graph, then the grounded answer: claim spans, citation anchors, version rail, delta banner and suppression sequence |
| Right | Telemetry cards with sparklines, the grounding firewall and the evidence graph (filters ALL / ACTIVE / CONFLICTS / UPDATED / UNSUPPORTED) |
| Bottom | Event bus swim-lanes, with replay controls and the event inspector (schema and raw JSON) |

Focus modes: `0` split, `1` answer, `2` evidence, `3` systems. Other shortcuts: `D` demo (auto-framing), `P` presentation, `Space` play/pause, `←`/`→` step, `Ctrl/⌘+K` command palette.

Routes: `/` (landing), `/live`, `/benchmark` (target vs measured, with provenance), `/experiments` (ablations; not-run experiments show empty slots), `/architecture`.

## Tests

- `cypress/e2e/demo.cy.ts` covers the acceptance flow over the recorded demo: WAIT → RETRIEVE → 3 intents → evidence → V1 → REFINE → V2 → SUPPRESS with no new retrieval → V3, plus the citation inspector, version navigation, conflict, event inspector, palette, benchmark, focus modes and theme.
- `cypress/e2e/live.cy.ts` runs the same flow against the real backend over WebSocket (`--env LIVE=1`).
- `cypress/e2e/storybook.cy.ts` smoke-tests stories (`--env STORYBOOK=1`, serve `storybook-static` on :6007).
