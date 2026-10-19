# PSRC V2.2 Research Report

## 1. Objective and outcome

This iteration tested whether policy-independent evidence-survival learning, constraint-aware cold start,
intent readiness, and validated evidence reuse can improve gold-useful early retrieval without increasing
premature retrieval or actual full-search work. The experiment did **not** justify replacing the original PSRC.
The frozen selection remains `P0_original_psrc`; all V2.2 policy features remain disabled by default.

One execution optimization was promising but was not promoted: `P6_evidence_reuse` preserved every measured
baseline policy and grounding outcome while reducing development full searches from 77 to 65 (12 searches,
15.6%). It remains configurable pending broader independent validation.

Final run: `runs/bench/controller_v2_2/20261019T012057Z/`. These are engineering measurements on small,
mixed-source streams, not production-generalization results.

## 2. Prior findings and selection rules

The prior study found baseline useful early retrieval `0.6875`, premature retrieval `0.2963`, evidence survival
`0.7333`, and 77 full searches including shadow work. Heuristic cold start increased useful early retrieval to
`0.7500`, but raised premature retrieval to `0.3030`, reduced survival to `0.6970`, and raised work to 91.
V2.2 froze selection as correctness, unchanged G2 (`>= 0.80`), premature risk, survival, actual full-search cost,
gold-useful early retrieval, and simplicity.

## 3. Dataset and provenance

`benchmark/datasets/controller_v2_2/` contains train, calibration, development, held-out, external, source,
annotation, and manifest artifacts. The builder produced 117 utterances, 389 cumulative prefixes, and 90
question families:

| Split | Utterances | Role |
|---|---:|---|
| Train | 54 | Model fitting only |
| Calibration | 9 | Post-calibration only |
| Development | 36 | Policy comparison/selection |
| Held-out | 8 | One frozen post-selection evaluation |
| External | 10 | Independent-source analysis, not policy tuning |

Sources are the project-authored training/development streams, the separately authored project holdout, 48
non-indexed SQuAD question/document labels, and one intersecting official SciFact qrel. SQuAD/SciFact groups are
split by source document or original query family; all prefixes from one utterance remain in one split. Exact
duplicate, near-duplicate cross-split, group leakage, timestamp, prefix-order, provenance, and manifest-hash
checks pass.

External QA questions are written questions segmented deterministically for incremental experiments. Their
timings are explicitly synthetic; they are not represented as microphone or natural-speech recordings. Source
corpora and evaluation labels remain separate. SQuAD and the local BEIR/SciFact packaging report CC BY-SA 4.0;
exact revisions, hashes, attribution, and limitations are in `docs/CORPUS_PROVENANCE.md` and the dataset source
manifest. The target of several hundred independent utterances was not reached, limiting every conclusion.

## 4. Useful-early metric and oracle

Historical G2 is unchanged. The separate **Useful Early Retrieval Rate** denominator is the set of `NEW_INFO`
utterances whose annotated eligible chunk precedes utterance end. A success requires a pre-end retrieval for the
correct need with accepted evidence overlapping independently defined gold sections/documents. The reranker's own
admission is not the sole ground truth.

The strengthened oracle reports earliest relevant, constraint-ready, and gold-useful prefixes; first actual and
first useful retrieval; finalized query; survival; cost; and failure attribution. It contains 92 prefix rows, 43
eligible rows, and 17 missed eligible decisions. Attribution counts are: `LOW_STABILITY_ESTIMATE` 17,
`LOW_EVIDENCE_RELEVANCE` 27, `NO_SAFE_EARLY_OPPORTUNITY` 26, `UNNECESSARY_HARD_GATE` 17,
`INCOMPLETE_CONSTRAINTS` 3, and `INSUFFICIENT_CORPUS` 2. The offline oracle may examine final text; deployed
policies never receive oracle fields.

## 5. Policy-independent survival labels

Every eligible prefix is retrieved with a frozen index, whether or not PSRC would act. Candidate text, accepted
prefix evidence, accepted final evidence, gold evidence, and lineage are recorded. With gold evidence, the target
is:

\[
s_t=0.45\,\mathrm{RBO}(E_t,E_f)+0.35\,J(S_t,S_f)+0.20\,G_t
\]

`RBO` compares accepted chunk rankings, `J` is accepted-section Jaccard, and `G_t` records independent gold
overlap. Without gold, the first two components are renormalized to `0.56` and `0.44`. The binary target is
`1[s_t >= 0.70]`. Ranking overlap and gold usefulness are retained separately. A first observation has a
missing-history indicator and neutral model input; it is not called observed stable or unstable. No deployed
feature uses future transcript content.

## 6. Features, training, and calibration

M1 retains `rho`, `chi`, `kappa`, `sigma`, and continuation. Extended models add first-observation status,
cold-start confidence, query informativeness, rank separation, objective/intent readiness, constraint
readiness/revision risk, evidence relevance, domain availability, normalized prefix length, candidate age,
slot change, and accepted-evidence overlap. All are available at decision time.

Training uses 128 non-final prefixes in 36 families; calibration uses 24 observations in four separate families;
development probability evaluation uses 50 observations.

| Model | Brier | ECE (10 equal-width bins) | Log loss |
|---|---:|---:|---:|
| M0 historical | 0.2293 | 0.2774 | 0.6554 |
| M1 retrained historical | 0.1868 | 0.2281 | 0.5587 |
| M2 extended logistic | 0.1390 | 0.2019 | 0.4247 |
| M3 regularized extended | 0.1540 | 0.2296 | 0.4780 |
| M4 separate-set Platt calibration | **0.1121** | **0.1012** | **0.3736** |

M5 isotonic was skipped because four independent calibration families are insufficient. The first-observation
calibration subset had only one target class, so no calibrated cold-start probability was fitted; V2.2 exposes
the score as heuristic confidence, not probability. M4 improved probability quality but was too conservative
under the historical decision threshold: P2 made no early retrievals. Calibration alone did not improve control.

## 7. Cold-start experiments

`NO_PREVIOUS_RANKING` is distinct from observed instability. CS1 adds explicit critical-constraint risk and
retains evidence relevance/domain availability guards. Cache variants reuse only source-compatible validated
results.

| Variant | Useful early | Premature | Survival | Full searches | Reused |
|---|---:|---:|---:|---:|---:|
| CS0 original heuristic | 0.7500 | 0.3030 | 0.6970 | 91 | 0 |
| CS1 constraint-aware | 0.7500 | 0.2903 | 0.7097 | 87 | 0 |
| CS2 calibrated | 0.7500 | 0.2903 | 0.7097 | 87 | 0 |
| CS3 constraint-aware + reuse | 0.7500 | 0.2903 | 0.7097 | 70 | 15 |
| CS4 calibrated + reuse | 0.7500 | 0.2903 | 0.7097 | 70 | 15 |

CS2/CS4 used the declared identity fallback because defensible cold calibration was unavailable. CS3 removes the
old cost regression and improves premature risk, but still misses G2 and remains below baseline survival.

## 8. Intent readiness

Readiness remains separate from survival. P5 found 71 ready rows, six ready-but-stability-blocked rows, and made
nine guarded first-observation bypass decisions. Aggregate useful early, premature, survival, and cost remained
exactly baseline (`0.6875/0.2963/0.7333/77`). The prior zero-gain result was therefore a mix of gate interaction
and no additional utterance-level success. No critical-constraint, relevance, domain, budget, or revision guard
is bypassed.

## 9. Evidence reuse

The session-local fingerprint includes corpus profile/hash, index configuration, canonical objective, typed and
facet constraints, retrieval mode, and relevance-gate configuration. States are `PROVISIONAL`, `VALIDATED`,
`REUSABLE`, `STALE`, and `INVALIDATED`. Memory/lifetime are bounded and access is locked. Corpus or constraint
changes invalidate compatibility; only `REUSABLE` entries can avoid a search.

Hits still traverse normal evidence admission, graph membership, content-hash, quote, and claim-support checks.
The cache cannot promote provisional evidence or cross sessions. P6 preserved baseline decisions and grounding
while reducing actual searches `77 -> 65`; CS3 reduced historical cold-start work `91 -> 70`.

## 10. Risk, latency, and combined policies

Hard correctness constraints remain separate from soft cost. P11 utility and P12's measured-extractive 20 ms
latency prior each suppressed all early retrievals. This shows that the older 1.2 s hypothetical generative prior
materially changes early-retrieval value. Simulated latency is not reported as measured current performance.

| Policy | Useful early | Premature | Survival | Full searches | Result |
|---|---:|---:|---:|---:|---|
| P0 original PSRC | 0.6875 | 0.2963 | 0.7333 | 77 | retained |
| P1 retrained stability | 0.6250 | 0.3636 | 0.7091 | 70 | reject |
| P2 calibrated stability | 0.0000 | n/a | n/a | 45 | reject: coverage |
| P3 constraint cold-start | 0.7500 | 0.2903 | 0.7097 | 87 | reject: survival/cost/G2 |
| P4 calibrated cold-start | 0.7500 | 0.2903 | 0.7097 | 87 | calibration unavailable; reject |
| P5 guarded intent bypass | 0.6875 | 0.2963 | 0.7333 | 77 | no aggregate gain |
| P6 validated reuse | 0.6875 | 0.2963 | 0.7333 | 65 | promising cost result; not promoted |
| P7 calibrated + reuse | 0.0000 | n/a | n/a | 45 | reject: coverage |
| P8 calibrated cold + reuse | 0.5625 | 0.2941 | 0.6824 | 64 | reject: survival/G2 |
| P9 calibrated + intent | 0.0000 | n/a | n/a | 45 | reject: coverage |
| P10 cold + intent + reuse | 0.6875 | 0.2727 | 0.6909 | 68 | reject: survival/G2 |
| P11 risk utility | 0.0000 | n/a | n/a | 45 | reject: coverage |
| P12 measured latency | 0.0000 | n/a | n/a | 45 | reject: coverage |

Counts, utterance-group bootstrap intervals, latency stages, and the Pareto set are machine-readable. No
candidate both dominates baseline and satisfies unchanged G2, survival, and cost filters.

## 11. Frozen held-out evaluation

After selection, P0 was evaluated once on eight untouched project-held-out utterances: raw G2 early `7/8 =
0.875`, gold-useful early `4/8 = 0.500`, premature `0.000`, survival `0.8545`, intent recall `1.000`, grounded
claims `1.000`, telemetry `1.000`, and 26 full searches. The useful-early group-bootstrap interval is
`[0.125, 0.875]`. This small split shares topics and the sample corpus and does not demonstrate external
generalization. No post-selection tuning occurred.

## 12. Final selection, limitations, and reproducibility

Original PSRC remains production default. G2 remains failed at `0.6875 < 0.80`; no benchmark target changed.
Experimental components stay runnable behind flags. Limitations include dataset size, four calibration families,
synthetic timing, one intersecting SciFact qrel, no independent human re-annotation, no hardware microphone run,
and no formal selective-risk guarantee.

```bash
python -m kairos controller-data build
python -m kairos controller-data validate
python -m kairos controller-train
python -m kairos controller-calibrate
python -m kairos controller-oracle
python -m kairos controller-experiments
python -m kairos controller-compare
python -m kairos controller-select
python -m kairos controller-heldout
python -m kairos controller-report
```

## 13. Scientific references

The verified bibliography and method-versus-inspiration distinctions are retained at the end of `README.md`.
V2.2 uses RBO as defined by Webber, Moffat, and Zobel (2010), evaluates separate-set Platt-style calibration in
the calibration tradition summarized by Guo et al. (2017), adapts active/streaming retrieval motivation from
Jiang et al. (2023) and Arora et al. (2026), and treats Wang et al. (2026) only as motivation. KAIROS does not
claim LEC's exchangeability or finite-sample guarantee applies here.
