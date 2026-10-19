# KAIROS V2.4 Baseline and V2.3 Method Audit

Status: corrections implemented and tested. Audited reference: `a7ee6642e7975105721a97438162b3dcbbb536be`.

## Reproduced baseline

The clean audited checkout used Python 3.13.5, Node 22.17.1, npm 10.9.2, and Windows 11. Before V2.4 production edits, `python -m pytest -q` passed 67 tests with one third-party deprecation warning. Benchmark run `20261019T070859Z` retained historical G2 at `11/16 = 0.6875` (FAIL against 0.80), G3 at `0.9444` (PASS), grounding coverage at `1.0`, zero fabricated citation IDs, and telemetry coverage at `1.0`. Semantic run `20261019T070917Z` measured Recall@5 `1.0` on the small in-project retrieval set and expanded-SQuAD Recall@5 `0.8125`.

All five canonical replays passed: Pune 110 events, reimbursement 68, self-repair 37, conflict 25, and multi-domain stress 111. Frontend typecheck passed. The baseline and artifact hashes are preserved in the V2.4 run bundle rather than rewriting V2.3 artifacts.

## Confirmed V2.3 anomalies and corrections

| Defect | Historical location | V2.4 correction | Measurement implication |
|---|---|---|---|
| Positive action value treated like strict evidence safety | `modeling_v2_3.py`, `metrics_v2_3.py` | Separate `beneficial_stop`, `PROXY_STRICT_USEFUL`, and nullable `VERIFIED_STRICT_USEFUL` | Calibration of benefit cannot be claimed as safety calibration |
| Terminal wait value initialized to zero | `counterfactual.py` | Terminal continuation includes final retrieval/verification utility and cost | Waiting until EOU is no longer assigned an unjustified zero value |
| Threshold risk used independent prefix rows | `modeling_v2_3.py` | One canonical chronological, first-action-per-lineage selector | Correlated prefixes no longer inflate selected-action samples |
| Zero selected actions reported as zero empirical risk | `modeling_v2_3.py` | Risk is `null`, coverage is zero, promotion is false | Fail-closed is not mislabeled validated-safe |
| A8 called “full integration” | `experiments_v2_3.py` | Stage A/B/C terminology is explicit | Offline predicates are not end-to-end measurements |
| One search per lineage reported as actual work | `metrics_v2_3.py` | Historical figures renamed `estimated_*`; V2.4 instruments every `HybridIndex.search` | V2.4 cost tables are execution counts, not assumptions |
| Proxy strict-usefulness treated as verified | `features_v2_3.py` | Proxy and verified fields are distinct; verified is unavailable unless executed | Unknown verification is never coerced to success or failure |
| Empty labels collapsed into negatives | `intent_alignment.py` | Explicit missing/undefined/unsupported/no-evidence/no-reference/incompatible states | Unknown-label coverage remains auditable |

Unit coverage is in `tests/test_caper_v2_4.py`; historical V2.3 JSON and reports remain unchanged. V2.3 results retain their original semantics and are not silently recalculated.

## Stage vocabulary

- Stage A: offline evidence/policy simulation.
- Stage B: real `SessionEngine` replay with actual retrieval and grounding behavior.
- Stage C: live or realistically scheduled streaming integration. V2.4 uses accelerated scheduled prefixes and is not physical microphone evidence.

The publishable V2.4 Stage-B run is `20261019T081648Z`. Physical microphone validation is `PENDING`.
