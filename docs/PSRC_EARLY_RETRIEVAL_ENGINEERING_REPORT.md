# PSRC Early-Retrieval Engineering Report

## A. Executive summary

KAIROS now contains reproducible PSRC opportunity analysis, seven experimental method families, grouped
calibration diagnostics, explicit retrieval-work accounting, version-bound speculative caching, selection rules,
tests, CLI commands, and run artifacts. The only variant that increased measured development useful-early
retrieval was cold start (`0.6875`→`0.7500`), but it also worsened premature retrieval (`0.2963`→`0.3030`),
evidence survival (`0.7333`→`0.6970`), and full work (`77`→`91`). Baseline PSRC therefore remains production
default. G2 remains failed at `0.6875 < 0.80`.

## B. Repository and environment

- Baseline commit: `25bdc65dd26f2c39b1ba40c090fadae8197e5eb4`
- Baseline snapshot: `runs/bench/controller/20261018T192517Z-baseline/`
- Final run: `20261018T194035Z`
- Python: 3.13 on Windows
- Corpus profile: `sample`
- Stability model: 42 examples, positive rate 0.4286, five features
- Development split: `benchmark/datasets/dev/sessions.json`
- Timing: benchmark-assigned transcript timing; `perf_counter` backend timings

Exact configuration, hashes, platform data, decisions, and result plots are in the final run directory.

## C. Baseline reproduction

Before editing, `python -m pytest -q` passed 44 tests. `python -m kairos bench --repeats 2` reproduced G2
`0.6875`, intent recall `0.9444`, grounded claim rate `1.0`, false retrieval `0`, telemetry `1.0`, and
reproducibility delta `0`. The semantic controller comparison reproduced premature retrieval `0.2963`, evidence
survival `0.7333`, and retrievals per information need `1.9615`.

## D. Oracle opportunity analysis

The final oracle uses accepted gold-relevant evidence plus RBO survival, not ranking stability alone. It emits 92
rows, 43 eligible prefixes across 19 utterances, and 17 missed rows. S06 is confirmed as a first-prefix missed
opportunity; S08 and the dog request in S23 are explicitly excluded as unanswerable/irrelevant. Files:

- `runs/bench/controller/oracle_opportunities.json`
- `runs/bench/controller/oracle_opportunities.csv`

## E. Individual methods

- **E1 cold start:** normalized IDF, rank separation, relevance coverage, domain compatibility, constraints,
  objective clarity, and ASR/closure state. Implemented online behind `controller.cold_start.enabled`.
- **E2 intent completion:** candidate-level readiness independent of utterance finalization. Implemented online
  behind `controller.intent_completion.enabled`.
- **E3 calibration:** identity runtime fallback plus grouped logistic, extended logistic, Platt, and isotonic
  diagnostics. None deployed because there is no untouched calibration/held-out streaming split.
- **E4 speculation:** version-bound provisional cache, staleness and exactly-once promotion. Evaluated with actual
  offline searches; deliberately not connected to claims.
- **E5 utility:** latency benefit minus expected refinement and marginal retrieval cost under a risk gate.
- **E6 ASR:** fixed-tail, agreement, adaptive-tail, provisional-probe, and hybrid pure policies on revision traces.
- **E7 scheduling:** deterministic marginal-value priority; not deployed without adequate starvation labels.

## F. Combined methods

C1 combined cold start and intent completion; C5 added utility. Intent completion did not change the cold-start
result. Utility dominated the decision and reduced useful coverage. No full Cartesian sweep was performed.

## G. Final selection

Correctness and telemetry filters were applied first, then premature-risk and survival constraints, then useful
early retrieval, cost/Pareto comparison, and complexity. Cold-start arms failed risk/survival. Utility arms were
feasible but substantially worse on the primary metric. Equivalent arms lost on complexity. `B0_baseline` remains
selected; `controller.policy_variant: baseline` and every experiment flag remains disabled by default. The frozen
selected policy was then run once on the eight-case held-out streaming split: raw early `0.875`, useful early
`0.500`, premature `0.000`, survival `0.8545`, and intent recall/grounding/telemetry `1.0`. No threshold changed
after this result was observed.

## H. Quantitative comparison

| Variant | Early | Useful early | Premature | Survival | Searches incl. shadow |
|---|---:|---:|---:|---:|---:|
| B0 baseline | 0.6875 | 0.6875 | 0.2963 | 0.7333 | 77 |
| B1 cold | 0.7500 | 0.7500 | 0.3030 | 0.6970 | 91 |
| B2 intent | 0.6875 | 0.6875 | 0.2963 | 0.7333 | 77 |
| B5 utility | 0.3125 | 0.1875 | 0.1667 | 0.8667 | 47 |
| C1 cold + intent | 0.7500 | 0.7500 | 0.3030 | 0.6970 | 91 |
| C5 cold + intent + utility | 0.3125 | 0.1875 | 0.1667 | 0.8667 | 47 |

All online variants preserved intent recall `0.9444`, grounded claim rate `1.0`, false presentation retrieval
`0`, and telemetry coverage `1.0`. Confidence intervals and all secondary fields are in `metrics.json`.

## I. Cost and latency

Baseline work consists of 95 real BM25 probe computations, 22 probe-cache hits, 58 emitted completed full
retrievals, and 19 shadow searches (77 full searches including shadow). Cold start raises shadow work to 28 and
total full work to 91. The speculative cache executed 30 full searches, reused 6, discarded 24, and required 24
additional final searches. These operations are not inferred solely from frontend events.

## J. Calibration

Existing Brier/ECE/log loss are `0.2038/0.2331/0.5996`. Grouped extended logistic reaches
`0.1510/0.1380/0.4517`; Platt has the lowest ECE at `0.0807`. These are correlated development diagnostics and
do not establish calibration guarantees. No formal LEC guarantee applies because exchangeability and an untouched
calibration set were not established.

## K. Regression and negative-result analysis

Cold start captures one additional eligible utterance but triggers more refinement/shadow work and lower evidence
survival. Intent readiness makes no aggregate difference. Utility waits too aggressively. Speculation’s 20% reuse
rate is insufficient. The baseline’s first-observation defect is real, but this dataset does not support a safe
replacement.

## L. Frontend compatibility

No frontend layout, route, protocol, or action vocabulary was changed. New explanation fields are carried in the
existing `RetrievalDecision` payload (`policy_variant`, reasons, and numeric signals). Existing WAIT, RETRIEVE,
REFINE, HOLD, and SUPPRESS behavior is preserved.

## M. Tests and commands

```bash
python -m pytest -q
python -m kairos controller-experiments
python -m kairos controller-oracle
python -m kairos controller-compare
python -m kairos controller-select
python -m kairos bench --repeats 2
python -m kairos semantic-bench
npm --prefix web run typecheck
npm --prefix web run build
```

The expanded backend suite passes 51 tests.

### Final checks

- `python -m pytest -q`: 51 passed, one third-party `pytz` deprecation warning.
- `python -m kairos bench --repeats 2`: run `20261018T193633Z`; all gates pass except the disclosed G2
  (`0.6875 < 0.80`); reproducibility delta is 0.
- `python -m kairos semantic-bench`: run `20261018T193641Z`; controller baselines and semantic metrics reproduce.
- All five required replays pass telemetry validation at 1.000: `pune-demo` 110/110,
  `reimbursement-demo` 68/68, `self-repair-demo` 37/37, `conflict-demo` 25/25, and
  `multi-domain-stress` 111/111 invariant instances.
- Controller CLI and artifact validation: eight required files and eight SVG plots present; JSON parses.
- `npm --prefix web run typecheck`: pass.
- `npm --prefix web run build`: pass, 3,138 modules transformed.

## N. Remaining limitations

- Small development denominator (16 eligible utterances) and held-out denominator (8); both use the same bounded
  sample corpus and related topics, so neither establishes external generalization.
- No real microphone/hardware validation.
- No asynchronous speculative work is allowed into production claims.
- No human audit of the gold opportunity labels or claims.
- No formal risk-control guarantee.

## O. Research attribution

KAIROS adapts streaming tool-use, active retrieval, RBO, calibration, selective risk, incremental ASR, and
exploration/exploitation ideas; it does not reproduce the cited papers’ complete architectures or guarantees.
Complete citations and methodological distinctions are in `docs/PSRC_EARLY_RETRIEVAL_RESEARCH.md` and the final
section of `README.md`.
