# KAIROS V2 engineering report

## Outcome

KAIROS V2 keeps the existing streaming architecture and frontend while making intent construction, evidence admission, abstention, attribution, and claim verification explicit. The default path remains deterministic and extractive; no LLM, vector database, cloud router, or new speech model was introduced.

Baseline revision: `a5659797423ecdea762d2e93591590186de5aa59`. The immutable pre-improvement snapshot is `runs/baselines/v2-pre-improvement/baseline.json`. Final development and semantic run `20261018T095654Z` is published in `runs/bench/benchmark/latest.json` and `runs/bench/semantic/latest.json`; each report retains its exact run ID internally.

## What changed

### QueryFrame V2

`kairos/streaming/frame_v2.py` adds deterministic request-onset and topic-boundary handling, candidate reconciliation, typed/scoped constraints, and explicit repair history. Candidate IDs no longer depend only on ordinal position. Each candidate records its normalized text, source span/chunk range, request cue, topic ID, constraints, completion state, revisions, and confidence. A configured place gazetteer supplements corpus-derived terms. Numeric parsing preserves surface forms while normalizing lakh/crore, INR, bedrooms, percentages, ranges, and per-month units; unclear financial roles remain ambiguous.

The legacy builder remains available only for the QueryFrame ablation. `kairos/streaming/frame.py` exports the V2 public interface so callers do not fork into a parallel runtime.

### EFIS and relevance

EFIS now treats an evidence footprint as valid only when retrieval passes relevance checks. Merge decisions combine footprint validity, lexical/semantic similarity, constraint compatibility, and explicit topic transitions. A coincidental overlap among irrelevant top-ranked documents can no longer justify merging unrelated questions.

Retrieval exposes candidate evidence separately from accepted evidence and returns `RELEVANT_SUFFICIENT`, `RELEVANT_PARTIAL`, `NO_RELEVANT_EVIDENCE`, or `OUT_OF_DOMAIN`. Gates consider weighted query coverage, reranker score, focus terms, domain compatibility, named-place agreement, and negation scope. Candidate hits remain inspectable, but only accepted hits can support synthesis. The evidence graph labels retrieved chunks as evidence and creates no artificial claim nodes for a zero-claim answer.

### Corpus routing and ingestion

The loader supports multiple corpus directories and rejects duplicate document IDs. Domain routing is configured and explainable, while facets continue to be inferred from sibling corpus headings. Reproducible SciFact and SQuAD adapters preserve source revisions, mappings, checksums, and licenses. SQuAD questions/answers are evaluation labels, never retrievable text. See `docs/CORPUS_PROVENANCE.md`.

### PSRC and index efficiency

Repeated BM25 probe results are cached by normalized probe state. On the 14-chunk stress stream this reduced actual BM25 probe computations from 49 to 20 with 29 cache hits, while retrieval count stayed 11. The final single run measured 99.878 ms without the cache versus 92.963 ms with it, but that small one-run difference is too noisy to claim as a general wall-clock speedup. Dense deduplication now uses sparse CSR storage rather than materializing a dense pairwise representation.

### Grounding and answers

The firewall verifies session provenance, document/chunk identity, evidence SHA-256, normalized exact quotation, numeric anchors, and named anchors. Status is now `SUPPORTED`, `CONFLICTING`, `PARTIAL`, `INSUFFICIENT_EVIDENCE`, or `INVALID_GROUNDING`; zero verified claims out of zero is insufficient evidence, not a pass. Extractive synthesis prefers sentences directly aligned with the intent focus and preserves partial success across independent intents.

### CDDM

Delta selection uses intent-owned constraints, claim dependencies, focus terms, and negation scope. The concrete S10/S26 over-selection failures are covered by tests. Unaffected claims are carried verbatim while affected scopes alone are re-retrieved.

### Live transport regression protection

The microphone path retains cumulative revisioned snapshots, a mutable ASR tail, duplicate/stale/finalized acknowledgements, explicit end-of-turn finalization, reconnection replay, and legacy message compatibility. Automated tests exercise corrections and ordering without requiring microphone hardware.

## Measured results

### Preserved pre-V2 baseline

The saved baseline records 28 passing backend tests and all four original demo replays. Development benchmark run `20261018T092600Z` measured early retrieval 0.6875, intent recall 0.9722, grounded-claim rate 1.0, and delta retrieval ratio 0.8667. G2 remained below its 0.80 target. The sample index contained 8 documents / 37 chunks; the traced build used 70,752,554 peak bytes and 3.695 seconds wall time under that measurement method.

### Semantic benchmark

Run `20261018T095654Z` uses 8 engineering cases containing 15 annotated intents. This set was authored alongside the upgrade and is not an independent public benchmark.

| Area | Metric | Result |
|---|---|---:|
| QueryFrame | intent precision / recall / boundary F1 | 1.000 / 1.000 / 1.000 |
| QueryFrame | count / topic transition / constraint ownership accuracy | 1.000 / 1.000 / 1.000 |
| QueryFrame | legacy → V2 intent F1 | 0.7857 → 1.000 |
| Retrieval | Recall@5 / MRR / nDCG@5 (4 positive queries) | 1.000 / 1.000 / 1.000 |
| Retrieval | relevant evidence precision | 0.5789 |
| Retrieval | OOD rejection / irrelevant acceptance (4 negative queries) | 1.000 / 0.000 |
| Retrieval | domain routing accuracy | 0.875 |
| External labels | SQuAD Recall@5 / MRR (12 questions) | 0.8333 / 0.7500 |
| Grounding | strengthened accuracy / false acceptance | 1.000 / 0.000 (5 adversarial cases) |
| CDDM dev | original → improved exact refinement correctness | 0.800 → 1.000 |
| CDDM dev | original → improved delta retrieval ratio | 0.7667 → 0.7000 |

The intent-recall Wilson 95% interval is `[0.7961, 1.0]`; the wide interval reflects the tiny test set. The OOD-rejection interval is `[0.5101, 1.0]` for the same reason.

The required ablations are stored in the report. Two deserve explicit qualification:

- Revised EFIS scored 2/3 on its very small ablation while legacy EFIS scored 3/3. The stricter validity gate correctly rejected an irrelevant footprint but failed to merge one cancellation-policy paraphrase at the current thresholds. This remains a tuning/research target.
- Domain-aware and single-mixed-corpus retrieval tied on the four positive queries. The bounded corpus is too small to establish a routing advantage.

### Bounded scale measurements

| Profile | Documents / chunks | Build ms | Python peak bytes | Retrieval p50 / p95 ms | Single-thread QPS |
|---|---:|---:|---:|---:|---:|
| sample | 8 / 37 | 63.313 | 1,346,047 | 1.399 / 1.960 | 669.56 |
| science | 12 / 29 | 121.301 | 2,510,487 | 1.688 / 2.241 | 560.40 |
| general | 8 / 16 | 55.974 | 795,204 | 1.103 / 1.657 | 809.10 |
| multi_domain_demo | 28 / 82 | 225.263 | 10,127,211 | 1.142 / 2.725 | 677.75 |

These are local, small-corpus measurements, not production capacity claims. The requested progressive 100/500/1,000/5,000-document study and multi-session concurrency saturation study were not run because only bounded licensed subsets were imported.

## Final validation

- Python compilation: passed for `kairos` and `benchmark`.
- Backend: 41/41 pytest tests passed; one third-party `pytz` deprecation warning remains.
- Frontend: TypeScript check and production Vite build passed.
- Browser acceptance: 10/10 recorded-demo tests, 3/3 live FastAPI/WebSocket and simulated-microphone tests, and 6/6 Storybook component renders passed.
- Engine replay: `pune-demo`, `reimbursement-demo`, `self-repair-demo`, `conflict-demo`, and `multi-domain-stress` all completed with telemetry lineage coverage 1.000.
- Conflict regression: both the 72-hour and 24-hour sourced claims remain visible rather than being silently resolved.
- Development benchmark: every configured gate passed except the unchanged G2 early-retrieval target (0.6875 measured versus 0.80 target). Intent recall was 0.9444, grounded-claim rate 1.0, delta retrieval ratio 0.7000, false presentation-turn retrieval 0, and reproducibility delta 0.

## Hackathon demonstration

Build and start the application:

```bash
npm --prefix web run build
python -m kairos serve
```

Then open `/live` and use the existing presentation/demo controls.

1. Positive multi-intent: replay `pune-demo`. Venue, cancellation, and catering intents produce cited extractive claims from the sample corpus.
2. Late constraint: continue the same demo with “Actually, this is for an international event.” Only dependent scopes are re-retrieved and the answer advances from V1 to V2.
3. Suppression: say “Make that shorter.” The controller emits `SUPPRESS`; retrieval is skipped and the transformation firewall checks that no new fact or citation appeared.
4. Deliberate insufficiency: replay `multi-domain-stress`. Housing listings, personalized medicine, nearest-hospital, and live travel requests expose rejected candidate evidence and yield `INSUFFICIENT_EVIDENCE` with zero fabricated claims.
5. External-corpus positive checks: run `python -m kairos semantic-bench`; the report includes a SciFact question and 12 SQuAD evaluation-label queries. Use `configs/science.yaml`, `configs/general.yaml`, or `configs/multi-domain.yaml` to serve a selected profile.

All five recorded scripts use the real `SessionEngine`; UI fixtures are regenerated from those engine traces rather than hand-authored success states.

## Changed-file inventory

Backend and evaluation:

- Query construction: `kairos/streaming/frame_v2.py`, compatibility exports in `frame.py`
- Retrieval/decomposition: `kairos/retrieval/index.py`, `kairos/decomposer/efis.py`, controller policy
- Grounding/synthesis/delta: firewall, schemas, extractive synthesis, delta engine
- Orchestration/telemetry: session engine, API, session store, event bus
- Corpus: multi-path ingestion and `kairos/corpus/adapters/`
- Evaluation: `benchmark/semantic_runner.py`, held-out engineering cases, offline dataset fixtures, V2 tests, stress demo, benchmark reports
- Configuration/documentation: corpus profiles, ablation switches, provenance and this report

Minimal frontend changes:

- Existing Grounding Strip and Firewall cards now render the new truthful statuses.
- Existing telemetry/timeline types accept evidence admission and status fields.
- Existing mock fixtures were re-recorded, including the stress demo.
- Existing microphone coordinator/transport work was retained and regression-tested; no visual redesign was made.

## Limitations and remaining work

- The 8-case engineering set is small and was authored during this upgrade. It is useful regression coverage, not an unbiased real-world test.
- Relevant-evidence precision is only 0.5789 because accepted top-k results can include same-domain but weakly useful passages. Admission and per-intent diversity need further calibration.
- External SQuAD Recall@5 is 0.8333 and MRR 0.75 on 12 labels; two queries miss the expected context in the first five results.
- The EFIS ablation regression described above must be resolved on a larger independently annotated paraphrase/topic-shift set before claiming universal improvement.
- SciFact qrels were not imported, so the science demo is a provenance/retrieval check rather than an official SciFact score.
- Cache work reduced BM25 computations but did not reduce wall time in the single short run.
- Scale testing stops at 82 chunks. No disk-backed index, 5,000-document benchmark, concurrent load test, or long-duration memory soak was performed.
- Grounding is lexical and extractive by default. It validates exact evidence, quotes, anchors, and numbers, but does not prove full semantic entailment.
- A human citation/answer audit has not been completed.
- Automated browser tests simulate recognition. A real microphone, browser permission, vendor-ASR correction, network interruption, and hosted WSS smoke test are still required on presentation hardware.
- Static corpora cannot answer live property availability, interest rates, nearby hospitals, flight prices, weather, or personalized medical questions. The correct behavior is abstention.
- FiQA, HotpotQA, and RAGBench remain unimplemented pending licensing, contamination, and resource review.
- The frontend dependency audit should be reviewed separately; dependency vulnerabilities are not semantic-engine correctness.
