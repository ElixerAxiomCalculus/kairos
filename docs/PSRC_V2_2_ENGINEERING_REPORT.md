# PSRC V2.2 Engineering Report

## Executive summary

V2.2 adds a provenance-controlled grouped dataset, policy-independent prefix extraction, extended and separately
calibrated survival models, cold-start constraint risk, guarded intent bypass, a validated session-local evidence
cache, real search accounting, P0-P12/CS0-CS4 ablations, a stronger oracle, and 12 plots. Strict selection retains
original PSRC. The best isolated engineering result is validated evidence reuse: identical development policy and
grounding outcomes with 65 rather than 77 actual full searches. It remains disabled by default because G2 still
fails and evaluation is small.

Final run: `20261019T012057Z`. The run records commit `25bdc65dd26f2c39b1ba40c090fadae8197e5eb4` plus dirty
working-tree state because this report describes unpublished local work.

## Changed implementation

- `benchmark/controller/dataset_v2_2.py`: build, provenance, group splits, manifests, leakage validation.
- `benchmark/controller/modeling_v2_2.py`: every-prefix features, survival targets, M0-M5 training, calibration.
- `benchmark/controller/v2_2.py`: baseline capture, P0-P12/CS0-CS4 runs, Pareto/selection, held-out, 12 plots.
- `benchmark/controller/metrics.py`: deterministic utterance-group bootstrap intervals.
- `benchmark/controller/oracle.py`: earliest-opportunity fields and explicit failure attribution.
- `kairos/controller/evidence_cache.py`: provenance fingerprints, bounded states, expiry, invalidation, locking.
- `kairos/controller/policy.py`: variable feature models, missing-history signal, calibrated loading, guarded
  cold-start/intent decisions, and reusable shadow results.
- `kairos/controller/cold_start.py`: preserved CS0, constraint risk, probability/confidence distinction.
- `kairos/engine.py`: cache-aware execution, actual-search markers, and complete decision provenance while
  preserving graph/grounding paths.
- `kairos/__main__.py`, `configs/default.yaml`: working commands and disabled-by-default controls.
- `tests/test_controller_v2_2.py`: data, fingerprint, isolation, concurrency, expiry, model, and risk tests.
- `docs/PSRC_V2_2_RESEARCH.md`, this report, corpus provenance, and README.

Generated assets are under `benchmark/datasets/controller_v2_2/`, `models/controller_v2_2/`, and
`runs/bench/controller_v2_2/`.

## Dataset and model results

The dataset has 117 streams, 389 prefixes, 90 families, and train/calibration/development/held-out/external
counts of 54/9/36/8/10. Exact duplicate, near-duplicate cross-split, family leakage, timestamp, prefix-order,
provenance, and manifest-hash checks pass.

Training uses 128 non-final prefix examples in 36 groups. Calibration uses 24 examples in four disjoint groups;
development probability evaluation uses 50. M4 reduced Brier/ECE/log loss from historical
`0.2293/0.2774/0.6554` to `0.1121/0.1012/0.3736`. Isotonic and cold-start calibration were not fitted because
independent calibration support was insufficient.

## Runtime and cache behavior

The cache is allocated per `SessionEngine`; cross-session reuse is impossible. A fingerprint changes with corpus
hash/profile, index/relevance settings, objective, normalized terms, typed/facet constraints, or retrieval mode.
Entries expire, memory is bounded, and access is locked.

A hit returns a copied `SearchResult`, then follows the ordinary sub-query update, retrieval event, evidence
graph, contradiction detection, synthesis, citation membership, checksum, quote, and claim-support path.
`search_executed: false` and the fingerprint are recorded. Provisional, stale, expired, source-incompatible, or
constraint-incompatible entries cannot be reused.

| Arm | Full searches | Shadow | Reused full | Duplicates avoided |
|---|---:|---:|---:|---:|
| P0 original | 77 | 19 | 0 | 0 |
| P6 reuse | 65 | 19 | 12 | 12 |
| P3 constraint cold-start | 87 | 26 | 0 | 0 |
| CS3 constraint cold-start + reuse | 70 | 19 | 15 | 20 |

These are actual `HybridIndex.search` executions plus actual shadow searches, not frontend event counts.

## Online selection

P3 improved useful early from 11/16 to 12/16 and improved premature risk versus CS0, but survival remained
`0.7097 < 0.7333` and work was `87 > 77`. P2's calibrated probabilities did not align with the historical policy
threshold and produced zero early retrievals. P5 made nine guarded bypass decisions without aggregate gain. P6
reduced work without changing behavior. P11/P12 were too conservative.

Only P0 passed all selection filters. `controller.policy_variant: baseline` remains configured. Calibration,
cold start, guarded intent bypass, evidence reuse, utility, and latency adaptation remain false by default.

## Frozen held-out result

P0 measured raw early 7/8, gold-useful early 4/8, premature 0, survival `0.8545`, intent recall `1.0`, grounded
claims `1.0`, fabricated document IDs `0`, telemetry `1.0`, and 26 full searches. The group-bootstrap interval
for useful early is `[0.125, 0.875]`. No tuning followed this result.

## Latency and overhead

The held-out run's mean measured retrieval stages were sparse `0.1953 ms`, dense `0.4246 ms`, fusion `0.0374
ms`, rerank `0.4046 ms`, and dedup `8.2041 ms`; retrieval p50/p95 was `8.750/16.502 ms`. These laptop figures
exclude microphone/ASR and WebSocket delivery. Transcript time is synthetic. The 1.2 s generative prior is a
scenario parameter, not measured extractive latency. Real microphone validation remains pending.

## Artifacts

The final run contains baseline copies/hashes, environment, configuration, dataset/split/model/corpus hashes,
metrics, per-utterance/intent/prefix results, decision traces, cost breakdown, calibration, selection, held-out
results, oracle JSON/CSV, modeling rows, and 12 SVG plots. Prior V2.1 artifacts were not overwritten.

## Commands

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
python -m pytest -q
python -m kairos bench --repeats 2
python -m kairos semantic-bench
python -m kairos replay pune-demo
python -m kairos replay reimbursement-demo
python -m kairos replay self-repair-demo
python -m kairos replay conflict-demo
python -m kairos replay multi-domain-stress
npm --prefix web run typecheck
npm --prefix web run build
```

## Final validation

- Dataset build/validation: PASS, 117 streams / 389 prefixes / 90 families, manifest hash matched.
- Controller experiment: PASS, all P0-P12 and CS0-CS4 arms completed; frozen selection retained P0.
- Publishable artifact audit: 30 JSON files parsed, 12 SVG plots present, repository-relative model paths.
- Backend: `58 passed` with one third-party `pytz` deprecation warning.
- Frontend: TypeScript typecheck and Vite production build PASS.
- Official development benchmark: reproducibility delta `0.0`; G2 remains the only failed gate at `0.6875`.
- Semantic benchmark: PASS; the known published limitations remain visible rather than being promoted away.
- Recorded demos: all five replayed with telemetry coverage `1.0`.
- Hygiene: `git diff --check` found no whitespace errors; no credential-pattern match was found. Expected Windows
  line-ending conversion notices are non-failing.

## Unresolved issues

Known limits are the four-family calibration split, written-question synthetic timing, no real
microphone/provider data, small shared-topic holdout, no independent human label audit, and no formal
calibrated-risk guarantee. Improved probability scores are not presented as proof of a deployable controller.
