# PSRC V2.3 IC-COS Engineering Report

## Outcome

V2.3 implements shared train/serve features, intent-conditioned labels, deterministic and real-engine
counterfactual branches, finite-horizon stopping targets, calibrated classifier/value models, strict artifact
validation, guarded runtime integration, A0–A8 evaluation, and corrected metrics. The measured evidence did not
support deployment: `controller.iccos.enabled` remains `false`; IC-COS is not the active default.

The implementation started from publication commit `6a13a8ff951bd561cef8d0be751220c3c4398589`. The V2.2
experiment itself recorded execution revision `25bdc65dd26f2c39b1ba40c090fadae8197e5eb4` while the worktree was
dirty. That historical provenance is preserved. `6a13a8…` is the later publication revision, not a retroactive
claim that the earlier run executed on it. The V2.3 commit is the commit containing this report; before commit,
the measured report correctly records a potentially dirty worktree.

## Changed architecture and files

- `kairos/controller/features.py`: canonical online schema and extractor.
- `kairos/controller/stopping_model.py`: strict IC-COS artifact loader/scorer.
- `kairos/controller/policy.py`: disabled-by-default IC-COS inference, explicit rejection/fallback reasons, and
  evaluator-only branch intervention.
- `kairos/controller/evidence_cache.py`: side-effect-free reuse eligibility observation.
- `kairos/engine.py`: candidate-local query/slot/evidence-section state.
- `benchmark/controller/dataset_v2_3.py`: append-only grouped dataset and frozen final manifest.
- `intent_alignment.py`, `features_v2_3.py`, `counterfactual.py`, `modeling_v2_3.py`, `metrics_v2_3.py`,
  `experiments_v2_3.py`, and `audit_v2_3.py`: research/evaluation pipeline.
- `models/controller_v2_3/iccos_v1.json`: frozen linear model artifact.
- API and `/benchmark`: live V2.3 report endpoint/table; V2.2 remains backward compatible.
- focused tests in `tests/test_controller_v2_3*.py`.

## Feature schema and parity

Schema `kairos-iccos-online-v1` enumerates names, types, mathematical definitions, ranges, missing semantics,
availability, cost, history use, and first-observation safety. It repairs first-observation rho, cheap relevance,
section/chunk identity, informativeness, domain transform, candidate age, and slot stability mismatches. History
availability is modeled separately. The extractor runs before target-only full retrieval, which prevents hidden
full-search features.

Training replay and SessionEngine use the same code. The deterministic parity test compares all canonical values
for identical prefixes and candidate keys at tolerance `1e-4`. Model training was executed only after this test
passed.

## Dataset, intent alignment, and artifacts

`benchmark/datasets/controller_v2_3/` retains train, calibration, development, external, and the historical data,
renames old held-out data `historical_frozen`, and adds `final`. Whole question families remain in one split.
QueryFrame stable key is primary lineage, with a unique-topic fallback. Final evidence is candidate-specific.
All six final-alignment statuses are serialized. The final split was frozen before selection access, but its 12
questions are project-authored and use the sample corpus; `independent_authorship: false` is explicit.

The model artifact records:

- model/feature schema versions and feature-schema SHA-256;
- required feature names and complete linear weights;
- dataset plus train/calibration/development hashes;
- Platt calibration and selected threshold;
- objective, validation, risk, and group/sample counts.

Unknown schema, bad hash, mismatched weight keys, absent model file, and missing live features cause explicit
`ModelCompatibilityError`. Runtime records `FALLBACK_BASELINE:<reason>` rather than silently substituting zero.

## Counterfactual replay and deterministic validation

Stage A evaluates every current-prefix action only after features are frozen, computes separate outcome
components, and applies backward induction over non-final candidate trajectories. Stage B creates two fresh real
SessionEngines from identical inputs, forces exactly one candidate observation to `RETRIEVE` or `WAIT`, then
lets the same baseline continuation execute. It counts primary and shadow full searches and preserves the corpus,
configuration, EFIS, evidence graph, grounding, CDDM, telemetry, and event order. Three Stage-B pairs are stored
in run `20261019T051008Z`.

The hook is an in-memory evaluator method; production configuration cannot activate it. No engine lock or mutable
cache is shared across branches.

## Training, calibration, and risk selection

Commands:

```bash
python -m kairos controller-v2-3-data build
python -m kairos controller-v2-3-data validate
python -m kairos controller-v2-3-train
python -m kairos controller-v2-3-experiments
python -m kairos controller-v2-3-report
```

The trainer fitted a class-balanced logistic classifier (`C=0.5`) and ridge `DeltaQ` regressor (`alpha=1`) on
136 rows/36 train groups. Platt calibration used 24 rows/four separate groups. Selection used 56 rows/27
development groups. Development classifier Brier/ECE/log loss were `0.1534/0.1419/0.4787`; regression MAE was
`0.239157` and sign accuracy `0.803571`. Threshold `0.85` met the empirical-risk cap only with zero selected
lineages. The risk result is not a distribution-free guarantee.

## Policy integration and event compatibility

IC-COS is evaluated after existing no-content, incomplete-clause, domain, evidence, constraint, uncertainty, and
budget guards. It uses its own probability and predicted action advantage and never changes or consumes `s_hat`
or `theta`. Every rejection emits a named reason. End-of-utterance fallback, QueryFrame V2, EFIS, retrieval,
relevance admission, evidence graph, grounding firewall, CDDM, transcript protocol, sessions, and existing event
types remain intact. Canonical signals are additive payload fields, so old clients remain compatible.

## Evidence-cache behavior

Cache reuse remains an execution optimization. `is_reusable` observes eligibility without incrementing hit/miss
counters or moving LRU state. A hit still flows through graph, admission, grounding, and verification. Historical
P6 preserved policy and correctness metrics while reducing full searches `77 → 65`; it is selected independently
from the failed policy-promotion gate. All cost reports distinguish primary, final, shadow, speculative,
refinement, and cache reuse work.

## Benchmark and selection

Run `20261019T051008Z` measured A0–A8 plus fixed-4, fixed-8, EOU, and the V2.2 cold comparison. A3/A4/A5 each
reached 24/32 raw early, 21/32 gold-useful, and 9/32 strict-safe, but 15/24 selected actions were not strict-useful.
A6/A7/A8 retained 0/32 at the risk-selected threshold. A8 therefore failed the required useful-improvement and
retention criteria. Baseline P0 remains selected and IC-COS stays experimental.

Selection was frozen before final access. The actual baseline SessionEngine on the new final split measured
13/16 raw, 4/16 gold-useful, and 0/16 strict-safe. On the historical frozen comparison it measured 8/10 raw,
5/10 gold-useful, and 3/10 strict-safe. Profile evaluation covered `sample`, `science`, `general_expanded`, and
`multi_domain_demo`; A8 selected no actions on any profile. That is a non-retention failure, not generalized safety.

## Metric and runtime corrections

Historical G2 (`11/16`) remains unchanged. V2.3 reports raw, gold-useful, strict-useful, and strict-safe
candidate-lineage numerators/denominators separately. The former `intent_level_useful_early_retrieval_rate` is
documented as a legacy utterance-level alias. Accepted chunk RBO and section Jaccard are separate.

Historical `first_useful_evidence_latency_ms` was search execution time. V2.3 uses:

- `retrieval_execution_latency_ms`: backend execution duration;
- `first_useful_evidence_elapsed_ms`: utterance start to usable accepted evidence;
- `lead_before_end_ms`: evidence ready time relative to final speech.

Offline replay leaves execution latency unmeasured instead of substituting transcript timing. Hardware/browser
microphone validation is pending and is not replaced with simulated recognition.

## Historical V2.2 Anomalies Resolved

| Anomaly | Original behavior | Why inconsistent | Correction and tests | Effect on metrics |
|---|---|---|---|---|
| commit provenance | dirty run recorded `25bdc65…`; later publication was `6a13a8…` | execution and publication revisions were conflated | append-only clarification and run metadata | no historical number changed |
| “policy-independent” | phrase implied independent labels/features | only prefix coverage ignored PSRC decisions | wording now says policy-independent prefix coverage | no historical number changed |
| train/serve features | seven definitions differed | learned score did not receive its training semantics | canonical extractor plus parity test | V2.3 values are not silently substituted into V2.2 |
| empty final evidence | 29/3/22 non-final rows in train/calibration/development | undefined reference was treated as ordinary negative | all 54 rows audited: 13 missing annotation, 20 OOD, 21 rejected after admission | historical models unchanged; V2.3 statuses explicit |
| “stability” terminology | ranking, accepted evidence, gold, and usefulness blurred | different outcomes shared one word | separate fields/tests | corrected denominators differ from legacy values |
| intent alias | utterance useful rate copied into intent field | no real intent denominator | stable candidate-lineage denominator | old value remains marked legacy alias |
| latency alias | retrieval duration called first-useful latency | execution time is not elapsed evidence time | three explicit timing names | historical timings retained with correction note |
| oracle vs G2 | criteria appeared interchangeable | G2 is utterance timing; strict oracle adds intent evidence/correctness | four earliest-oracle fields; final excluded | G2 stays 11/16; no retroactive gate rewrite |
| held-out wording | prior set called untouched | it informed published research | renamed historical frozen comparison set | retained only for longitudinal comparison |
| P6 selection | cache rejected with policy G2 | cache did not alter policy timing | independent behavior-preserving cache gate | cache qualifies; policy does not |
| calibrated default | README implied calibration was live | selected baseline is uncalibrated | configuration/docs/tests distinguish variants | no deployment change |

The row-level audit is `runs/bench/controller_v2_3/20261019T051008Z/v2_2_empty_final_evidence_audit.json`.

## Testing and validation

Final validation record (2026-10-19):

- Python: 67 tests passed; one third-party `pytz` deprecation warning.
- Standard benchmark: completed as run `20261019T051609Z`; the historical G2 early-retrieval gate remains the
  only failing gate at `0.6875`, while every other gate passes.
- Semantic benchmark: completed as run `20261019T051614Z`.
- Canonical replays: Pune demo `110/110`, reimbursement `68/68`, self-repair `37/37`, conflict `25/25`, and
  multi-domain `111/111` expected events matched.
- Frontend typecheck and production build: passed.
- Cypress: all three specs completed; the 10 runnable browser tests passed, while three live-backend/microphone
  tests and six Storybook tests remained intentionally opt-in/pending.
- A physical microphone capture remains a manual verification item because CI cannot grant or assess real-device
  audio permissions.
- Documentation consistency is checked by `docs_consistency_audit` against README, both reports, benchmark JSON,
  model metadata, and disabled configuration. Narrative changes remain reviewed, not auto-overwritten.

## Known limitations

Small independent-group counts, correlated prefixes, synthetic written-question timing, project-authored final
data, incomplete independent alignment review, a hand-specified Stage-A utility, and no real microphone timing
prevent production or theoretical claims. IC-COS is an implemented research path with a negative promotion result.
