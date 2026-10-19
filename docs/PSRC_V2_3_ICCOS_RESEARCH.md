# PSRC V2.3 IC-COS Research Report

Status: measured engineering research, not a deployment claim. Frozen run: `20261019T051008Z`.
IC-COS is a KAIROS-specific method and **not a published method**.

## A. Problem statement

Historical PSRC predicts whether an early result will remain stable and compares that estimate with a latency/cost
threshold. Stability alone is not the action objective: an unchanged result can be irrelevant, violate a late
constraint, be unanswerable, or be ineligible for grounded synthesis. V2.3 therefore asks a counterfactual action
question for each intent: is `RETRIEVE NOW` better than `WAIT NEXT`, given the same future transcript, corpus,
continuation policy, and correctness gates?

## B. Repository audit

The audit reproduced V2.2 from commit `6a13a8ff951bd561cef8d0be751220c3c4398589`; run
`20261019T044812Z` retained P0. Historical training/serving mismatches were:

| Signal | Historical offline definition | Live definition | V2.3 correction |
|---|---|---|---|
| first rank overlap | fabricated `0.5` | leave-one-term-out probe | value `0`, plus `ranking_history_available=0` |
| relevance | full-search rerank score | cheap top-section term coverage | shared cheap probe coverage; full search is target-only |
| evidence overlap | accepted chunk ranking | probe section ranking | section-to-section identity only |
| informativeness | mean normalized IDF | mean IDF plus bounded known-term coverage | one shared implementation |
| domain availability | Boolean × confidence | `0`, `0.65`, or `0.65+0.35c` | one shared transform |
| candidate age | global prefix index | candidate observation count | stable-key candidate-local count |
| slot stability | whole-frame slot change | time equality at the current frame | owner-scoped value snapshots plus availability |

The V2.2 extractor covered every prefix independently of PSRC action selection; that is **policy-independent
prefix coverage**, not fully policy-independent labels or observations. Corpus, index, QueryFrame, admission,
and final-query construction still determine its values.

## C. Corrected feature definitions

`kairos/controller/features.py` owns schema `kairos-iccos-online-v1`. Its SHA-256 is stored in the model. For
current and prior cheap section rankings `R_t` and `R_(t-1)`, `probe_section_rbo = RBO_0.9(R_t,R_(t-1))` only
when a real prior ranking exists. Missing history is `0` plus an availability bit; it is not negative evidence.
Probe clarity is one minus normalized score entropy. Candidate slots are scoped by QueryFrame owner key;
stability is the unchanged fraction over comparable current/prior slot keys. Query informativeness is
`0.8 × mean-IDF/corpus-max-IDF + 0.2 × min(known_terms/2,1)`. Top-section relevance uses in-memory BM25 term
coverage and never triggers full retrieval. The schema also records completeness, continuation safety, rank
separation, objective/constraint readiness, online revision risk, domain availability, candidate age, intent
readiness/independence, prior-query cosine, remaining time, and prefix length.

Every schema item specifies type, definition, range, missing behavior, availability, computational cost,
pre-retrieval status, history dependence, and first-observation safety. Training replay and live inference call
the same extractor. `tests/test_controller_v2_3_features.py` sends identical prefixes through the replay and
SessionEngine paths and requires equality within `1e-4` before training is allowed.

## D. Intent-conditioned evidence targets

QueryFrame stable keys are primary lineage identifiers; a unique topic lineage is a declared fallback. EFIS
candidate assignment remains unchanged. Each final candidate is retrieved separately, rather than using a
whole-utterance union. Alignment status is one of:

- `SUPPORTED_GOLD_EVIDENCE`
- `RELEVANT_BUT_CHANGED_EVIDENCE`
- `NO_RELEVANT_EVIDENCE`
- `OUT_OF_DOMAIN`
- `UNDEFINED_ALIGNMENT`
- `MISSING_GOLD_ANNOTATION`

Chunk identity, section identity, and document identity are explicit. The target stores gold relevance,
constraint compatibility, accepted-chunk RBO, accepted-section Jaccard, answerability, retrieval cost, lead
before end, revision cost, grounding eligibility, and the composite strict-useful label separately. Ranking
overlap, accepted-evidence overlap, gold relevance, survival, useful early retrieval, and strict-safe early
retrieval are not interchangeable synonyms for “stability.”

## E. Counterfactual framework

For candidate lineage `i` at prefix `t`, the state is the canonical online vector `x_(i,t)`. Actions are
`RETRIEVE NOW` and `WAIT NEXT`. The horizon is the last non-final prefix; end-of-utterance is a fallback and is
excluded from early oracles. Stage A replays every prefix with a frozen index and deterministic continuation.
Its retrieval reward keeps the usefulness dimensions visible:

`U_t = 0.30G + 0.15C + 0.22S + 0.10A + 0.13E + 0.08L - 0.12R - 0.20V`.

Here `G,C,S,A,E,L,R,V` denote gold relevance, constraint compatibility, survival, answerability, grounding
eligibility, bounded latency value, retrieval cost, and revision risk. Backward induction computes
`Q_t(wait)=V_(t+1)-0.015`, `DeltaQ_t=Q_t(retrieve)-Q_t(wait)`, and `V_t=max(Q_t(retrieve),Q_t(wait))`.
Future outcomes supervise labels but never enter online features.

Stage B reconstructs state from the event stream and runs paired real `SessionEngine` branches with the same
continuation/corpus/config. Three development branches were validated. The intervention is in-memory evaluator
state, not a deployable config and not a shared mutable engine checkpoint.

## F. Model training

The V2.3 dataset is append-only over V2.2, with stable train/calibration/development groups, the old set renamed
`historical_frozen`, a separate external split, and a new frozen final split. There are 136/24/56 non-final rows
and 36/4/27 groups for train/calibration/development. A class-balanced L2 logistic model (`C=0.5`) predicts
beneficial stop; ridge regression (`alpha=1`) predicts `DeltaQ`. No transformer or LLM is required.

The artifact records schema versions/hashes, required features, dataset and split hashes, weights, calibration,
threshold, objective, and validation. Loading rejects unknown/missing features or schema drift. Missing live
features are errors, never silently zero-filled.

## G. Calibration

Platt calibration is fitted only on the four calibration families. On 56 development rows the calibrated
classifier measured Brier `0.1534`, ECE `0.1419`, and log loss `0.4787`. Ridge regression measured MAE
`0.239157` and `DeltaQ` sign accuracy `0.803571`. Four calibration families are too few for a broad calibration
claim; these are engineering diagnostics.

## H. Policy selection

Thresholds `0.35` through `0.95` were swept on development with maximum empirical selected-action risk `0.20`.
The selected threshold was `0.85`; together with positive predicted `DeltaQ` and correctness guards it retained
0/32 intent lineages. Uncalibrated arms retained more actions but had unacceptable unsafe rates. Since retention
collapsed, A8 failed the useful-improvement gate and IC-COS remains disabled. The baseline is still the runtime
default; no threshold or target was weakened.

Cache promotion is independent. Historical P6 matched P0 policy, grounding, citation, survival, and premature
metrics while reducing full searches from 77 to 65. It therefore qualifies as a behavior-preserving execution
optimization, not a new early-retrieval policy.

## I. Experiments

All corrected rates use 32 candidate lineages with a non-final opportunity.

| Arm | Raw early | Gold-useful | Strict-safe | Premature / selected | Chunk / section survival |
|---|---:|---:|---:|---:|---:|
| A0 historical PSRC | 11/16 (68.75%), legacy utterance G2 | 11/16 legacy useful alias | not retrospectively defined | 29.63% legacy | 73.33% legacy composite |
| A1 shared features | 12/32 (37.50%) | 9/32 (28.12%) | 0/32 | 12/12 | 0.5501 / 0.5306 |
| A2 intent labels | 22/32 (68.75%) | 17/32 (53.12%) | 7/32 (21.88%) | 15/22 | 0.4536 / 0.4409 |
| A3 classifier | 24/32 (75.00%) | 21/32 (65.62%) | 9/32 (28.12%) | 15/24 | 0.5197 / 0.4962 |
| A4 value regression | 24/32 (75.00%) | 21/32 (65.62%) | 9/32 (28.12%) | 15/24 | 0.5197 / 0.4962 |
| A5 calibrated | 24/32 (75.00%) | 21/32 (65.62%) | 9/32 (28.12%) | 15/24 | 0.5197 / 0.4962 |
| A6 risk threshold | 0/32 | 0/32 | 0/32 | 0/0, undefined | not applicable |
| A7 cache | 0/32 | 0/32 | 0/32 | 0/0, undefined | not applicable |
| A8 full integration | 0/32 | 0/32 | 0/32 | 0/0, undefined | not applicable |

Fixed-4 measured 12/32 raw early and 0/32 strict-safe; fixed-8 measured 1/32 and 0/32; EOU has no non-final
action by definition. The V2.2 cold-start arm is retained only with its historical definitions.

## J. Evaluation

Selection was written to `selection_freeze.json` before final data was evaluated. The selected live baseline on
the 12-question project-authored final set produced 13/16 raw early, 4/16 gold-useful, and 0/16 strict-safe,
with Wilson 95% upper bound `0.1936` for strict-safe rate. This is a negative result. The final set was excluded
from V2.3 fitting and selection, but it shares the sample corpus and project authorship; it is not independent
external evidence.

The old eight-stream set is a **historical frozen comparison set**, not untouched V2.3 data. Under corrected
candidate-lineage metrics, live baseline yielded 8/10 raw, 5/10 gold-useful, and 3/10 strict-safe.

Historical official G2 remains 11/16 utterances. It is preserved, but it is not the corrected strict oracle.
The V2.2 “intent-level” metric was a legacy alias of its utterance calculation. V2.3 oracles separately record
earliest gold, constraint-ready, useful, and strict-safe prefixes, excluding final.

## K. Cost analysis

Offline ablations count primary, final fallback, shadow, speculative, and refinement searches explicitly; A1–A8
each require 32 primary/fallback searches over the 32 development lineages in the simplified replay. Stage-B
real-engine branches record `RetrievalCompleted.search_executed` plus shadow searches. Cache validation measured
12 fewer full searches (77 to 65) while retaining behavior. Probe work is not mislabeled as a full search.

Latency vocabulary is corrected: `retrieval_execution_latency_ms` is backend search duration;
`first_useful_evidence_elapsed_ms` is utterance-start to usable evidence; `lead_before_end_ms` is evidence-ready
lead over final speech. Offline replay does not fabricate backend execution latency. Synthetic speech time and
measured backend time remain separate.

## L. Error analysis

The V2.2 empty-final audit covers every non-final affected row: train 29, calibration 3, development 22. Causes
were missing gold annotation 13, out-of-domain 20, and no relevant evidence after admission 21. Historical
metrics were not rewritten. On V2.3 development, A3/A4/A5 improve gold-useful coverage but select 15 unsafe
actions among 24, mostly because relevance alone does not ensure constraint readiness, survival, and grounding
eligibility. A6 eliminates observed risk only by eliminating coverage.

## M. Generalization

The frozen A8 risk policy selected 0 actions on sample (0/32), general-expanded (0/9), science (0/1), and
multi-domain-demo (0/7). Those zeros demonstrate non-retention, not robust safety. Science has one labeled query,
and no profile supports a generalization claim.

## N. Limitations

- Few independent calibration groups and correlated prefixes.
- Most streams are written questions with simulated 0.3-second/word timing.
- Final questions are project-authored and reuse the sample corpus.
- Intent alignment lacks completed independent human review.
- Stage-A utility weights are engineering choices, not a causal or theoretical guarantee.
- Paired deterministic replay is not logged-bandit off-policy evaluation.
- Real microphone/provider timing remains pending because no microphone/browser hardware was available here.

## O. Research references

Directly implemented KAIROS mechanisms are distinguished from foundations and inspiration:

- Arora et al. (2026), “Stream RAG,” ICML, PMLR 306:3633–3663: [PMLR](https://proceedings.mlr.press/v306/arora26a.html).
- Webber, Moffat, and Zobel (2010), “A Similarity Measure for Indefinite Rankings,” *TOIS* 28(4): [DOI](https://doi.org/10.1145/1852102.1852106).
- Guo et al. (2017), “On Calibration of Modern Neural Networks,” ICML, PMLR 70:1321–1330: [PMLR](https://proceedings.mlr.press/v70/guo17a.html).
- Jiang et al. (2023), “Active Retrieval Augmented Generation,” EMNLP, 7969–7992: [ACL Anthology](https://aclanthology.org/2023.emnlp-main.495/).
- Petcu et al. (2026), “Query Decomposition for RAG: Balancing Exploration-Exploitation,” EACL, 6857–6871: [ACL Anthology](https://aclanthology.org/2026.eacl-long.322/).
- Wang et al. (2026), “LEC,” ICML, PMLR 306:124814–124840: [PMLR](https://proceedings.mlr.press/v306/wang26a.html).
- Selfridge et al. (2011), “Stability and Accuracy in Incremental Speech Recognition,” SIGDIAL, 110–119: [ACL Anthology](https://aclanthology.org/W11-2014/).
- Bellman (1957), *Dynamic Programming*, Princeton University Press: [bibliographic record](https://books.google.com/books?id=ZzoS0QEACAAJ).
- Sutton and Barto (2018), *Reinforcement Learning: An Introduction*, second edition, MIT Press: [publisher](https://mitpress.mit.edu/9780262352703/reinforcement-learning/).
- Dudík, Langford, and Li (2011), “Doubly Robust Policy Evaluation and Learning,” ICML, 1097–1104: [paper](https://icml.cc/2011/papers/554_icmlpaper.pdf).
- Swaminathan and Joachims (2015), “Counterfactual Risk Minimization,” ICML, PMLR 37:814–823: [PMLR](https://proceedings.mlr.press/v37/swaminathan15.html).

These sources do not describe the exact IC-COS algorithm and their guarantees are not claimed here.
