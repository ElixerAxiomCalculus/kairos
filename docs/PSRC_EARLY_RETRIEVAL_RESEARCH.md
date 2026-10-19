# PSRC Early-Retrieval Research

## Scope and research question

KAIROS makes a sequential decision while a transcript is still evolving: retrieve now and hide downstream
latency, or wait for a better-defined intent and avoid obsolete evidence. This study tested whether lightweight,
interpretable additions to the Predictive Semantic Retrieval Controller (PSRC) improve **useful early retrieval**
without weakening grounding, intent accuracy, telemetry, evidence survival, or retrieval cost.

The pre-change implementation at commit `25bdc65dd26f2c39b1ba40c090fadae8197e5eb4` was preserved in
`runs/bench/controller/20261018T192517Z-baseline/`. The final experiment is
`runs/bench/controller/20261018T194035Z/`; `runs/bench/controller/latest.json` points to it.

## Hypotheses and methods

### E0 — prefix oracle

Each development decision prefix was retrieved offline and compared with both final-prefix evidence and the
answerable gold sections. An oracle row is eligible only when the prefix uses no future text, has adequate
constraint readiness, retrieves accepted gold-relevant evidence, and that evidence has RBO survival of at least
0.70. The 92-row output contains 43 eligible prefix rows across 19 utterances and 17 missed decision rows.
This is an opportunity analysis, not a deployable policy or an estimate of production performance.

The requested case audit found:

- S06 has a genuine first-prefix opportunity for “What is the cancellation”.
- S05 becomes useful only after “served at company venues”; the single word “alcohol” is not accepted as an
  oracle opportunity despite retrieving a partly relevant section.
- S08 is unanswerable and has no gold-relevant accepted evidence.
- S13’s prefixes do not pass the current accepted-evidence/gold criterion.
- S23’s dog request is not useful; only the later metro candidate retrieves its gold section.

See `runs/bench/controller/oracle_opportunities.json` and `.csv`.

### E1 — cold-start evidence confidence

`kairos/controller/cold_start.py` replaces fabricated first-observation stability with a monotone blend of:

\[
C_{cold}=0.20I+0.15M+0.25R+0.10D+0.15K+0.10O+0.05A
\]

where the inputs are mean-normalized IDF informativeness, within-query rank separation, accepted-section
coverage proxy, domain compatibility, constraint readiness, objective clarity, and committed-prefix/closure
status. Query length is not itself rewarded. Unknown domains, dangling critical constraints, ambiguous typed
constraints, and weak evidence reduce confidence. Coefficients are explicit engineering priors rather than
coefficients fitted on development labels.

The method improved raw and useful early retrieval from 0.6875 to 0.7500, but increased premature retrieval
from 0.2963 to 0.3030, reduced evidence survival from 0.7333 to 0.6970, and increased measured full work from
77 to 91 searches (including shadow searches). It therefore failed the preregistered safety filters.

### E2 — intent-level completion

`kairos/controller/intent_readiness.py` distinguishes candidate completion, objective clarity, typed-constraint
readiness, continuation safety, and semantic independence. Anaphoric/dependent candidates are penalized. The
method can bypass first-observation ranking maturity only when the ordinary stability and completeness gates
also pass. On this development set it made no aggregate decision change, so it was not promoted.

### E3 — grouped survival calibration

The historical model has 42 training prefix examples. Its development diagnostic has 92 observations in 26
utterance groups. All comparison predictions were made out of group using `GroupKFold`; prefixes from one
utterance never crossed a fold.

| Model | Brier | ECE | Log loss |
|---|---:|---:|---:|
| Existing deployed model | 0.2038 | 0.2331 | 0.5996 |
| Retrained existing features | 0.1609 | 0.1373 | 0.4873 |
| Extended features (`rho`, `chi`, `kappa`, `sigma`, `pi`, `nu`) | **0.1510** | 0.1380 | **0.4517** |
| Platt on existing score | 0.1714 | **0.0807** | 0.5198 |
| Isotonic on existing score | 0.1545 | 0.0932 | 0.6745 |

These are grouped development diagnostics, not an untouched calibration split. The identity transform remains
deployed. Better Brier/ECE alone was not treated as sufficient evidence for policy promotion.

### E4 — speculative retrieval and reuse

`kairos/controller/speculative.py` implements version-bound provisional entries, duplicate rejection, stale
invalidation, corpus compatibility, term-compatibility checks, and exactly-once promotion. The offline execution
performed 30 actual full index searches: 6 were reusable and 24 were discarded and followed by another full
search (reuse rate 0.20). No provisional result was allowed into claims or answer versions. Because the measured
reuse rate does not justify the added work, speculative retrieval is not integrated into the production engine.

### E5 — risk-aware utility

The heuristic utility implemented in `kairos/controller/utility.py` is:

\[
U_R=pL-(1-p)C_{refine}-C_{extra}
\]

with an explicit maximum predicted-risk gate. It is an empirical heuristic, not a theorem. It reduced premature
retrieval to 0.1667, raised survival to 0.8667, and reduced full work to 47 searches, but collapsed raw early
coverage to 0.3125 and gold-useful early retrieval to 0.1875. It was not selected.

### E6 — adaptive ASR commitment

`kairos/controller/asr_policy.py` compares fixed-tail, agreement, adaptive-tail, provisional-probe, and hybrid
availability on recorded revision sequences containing substitutions such as “whole”→“hall” and
“Pune”→“Mumbai”. The pure functions preserve the distinction between provisional probing and authoritative
commitment. Hardware microphone validation remains pending; therefore no ASR strategy was promoted.

### E7 — intent scheduling

`kairos/controller/scheduling.py` implements a deterministic marginal-value priority from readiness, predicted
survival, relevance, novelty, prior retrieval, and remaining budget. There is not enough independently labelled
multi-intent starvation data to justify replacing existing EFIS scheduling. No bandit was trained or deployed.

Bayesian online change-point detection was not implemented, so Adams and MacKay (2007) is not attributed as a
method used by KAIROS.

## Controlled comparison

All online rows below ran through the real `SessionEngine` on `benchmark/datasets/dev`. The official G2
definition was preserved. “Full work” includes emitted full searches and non-emitting shadow searches.

| Variant | Raw early | Useful early | Premature | Survival | Full work | Result |
|---|---:|---:|---:|---:|---:|---|
| B0 baseline | 0.6875 | 0.6875 | 0.2963 | 0.7333 | 77 | selected/default |
| B1 cold start | 0.7500 | 0.7500 | 0.3030 | 0.6970 | 91 | rejected: risk + survival |
| B2 intent completion | 0.6875 | 0.6875 | 0.2963 | 0.7333 | 77 | no measurable gain |
| B3 calibrated identity | 0.6875 | 0.6875 | 0.2963 | 0.7333 | 77 | calibration not deployable |
| B5 risk utility | 0.3125 | 0.1875 | 0.1667 | 0.8667 | 47 | rejected: coverage |
| C1 cold + intent | 0.7500 | 0.7500 | 0.3030 | 0.6970 | 91 | rejected: risk + survival |
| C5 cold + intent + utility | 0.3125 | 0.1875 | 0.1667 | 0.8667 | 47 | rejected: coverage |

B4 speculative reuse, B6 ASR policy, and B7 scheduling were executed as bounded offline experiments and did not
alter baseline online-policy metrics. Detailed values and confidence intervals are in `metrics.json`. The
baseline useful-early Wilson 95% interval is `[0.4440, 0.8584]`; cold start is `[0.5050, 0.8982]`. The overlap and
small sample prevent a strong generalization claim.

## Selection rationale

The selection order was fixed as correctness, risk/cost, useful early retrieval, Pareto comparison, complexity,
then a frozen held-out evaluation. Cold start is the only method that increased useful early retrieval, but it
failed the premature-risk and evidence-survival filters and increased work by 18%. Utility was safer and cheaper
but lost too much useful coverage. Equivalent arms receive the complexity penalty. Consequently B0 baseline
remains the default and all research components stay behind disabled configuration flags.

After selection was frozen, the chosen baseline was evaluated once on the project-authored
`benchmark/datasets/heldout_streaming` split without threshold changes. Across eight eligible utterances it
measured raw early `0.875`, gold-useful early `0.500`, premature `0.000`, survival `0.8545`, intent recall `1.0`,
grounded claims `1.0`, telemetry `1.0`, 2.125 retrievals per need, and 26 full searches including shadow work.
The raw/useful gap reinforces the selection metric. The split uses disjoint wording but shares the bounded sample
corpus and topics, so it is not external generalization evidence. No post-selection tuning was performed and no
formal theoretical guarantee was proved.

## Reproducibility

```bash
python -m kairos controller-experiments
python -m kairos controller-oracle
python -m kairos controller-compare
python -m kairos controller-select
python -m kairos controller-report
```

The final run includes `metrics.json`, `per_utterance.json`, `per_intent.json`, `decision_traces.jsonl`,
`selection.json`, `configuration.json`, `environment.json`, and eight SVG plots.

## Limitations and future work

- Development has only 16 G2-eligible utterances and was used during prior engineering; the frozen holdout has
  only eight eligible utterances and shares its corpus/topics.
- Candidate-to-gold mapping in the oracle is section based; independent human annotation is still needed.
- Synthetic chunk timing is not microphone or ASR-arrival timing.
- The 42-example training set is too small for model promotion.
- Speculative cache evaluation is offline and intentionally cannot promote evidence into claims.
- ASR hardware/provider behavior remains untested.
- A future release should add provenance-controlled train/calibration/development/held-out streaming families,
  freeze thresholds, and run held-out evaluation once.

## Scientific references

1. Siddhant Arora et al. “Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage.”
   *ICML 2026*, PMLR 306:3633–3663. <https://proceedings.mlr.press/v306/arora26a.html>
2. William Webber, Alistair Moffat, and Justin Zobel. “A Similarity Measure for Indefinite Rankings.”
   *ACM TOIS* 28(4), 2010. <https://doi.org/10.1145/1852102.1852106>
3. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. “On Calibration of Modern Neural Networks.”
   *ICML 2017*, PMLR 70:1321–1330. <https://proceedings.mlr.press/v70/guo17a.html>
4. Zhengbao Jiang et al. “Active Retrieval Augmented Generation.” *EMNLP 2023*, pp. 7969–7992.
   <https://aclanthology.org/2023.emnlp-main.495/>
5. Roxana Petcu et al. “Query Decomposition for RAG: Balancing Exploration-Exploitation.” *EACL 2026*,
   pp. 6857–6871. <https://aclanthology.org/2026.eacl-long.322/>
6. Zhiyuan Wang et al. “LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in
   Selective Prediction and Routing Systems.” *ICML 2026*, PMLR 306:124814–124840.
   <https://proceedings.mlr.press/v306/wang26a.html>
7. Ethan Selfridge, Iker Arizmendi, Peter Heeman, and Jason Williams. “Stability and Accuracy in Incremental
   Speech Recognition.” *SIGDIAL 2011*, pp. 110–119. <https://aclanthology.org/W11-2014/>
