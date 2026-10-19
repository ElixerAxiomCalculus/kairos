# KAIROS V2.4 CAPER Research Report

Status: measured research implementation; **not promoted**. Frozen run: `20261019T081648Z`.

## Problem and hypothesis

PSRC coupled “is this request stable enough?” with retrieval. IC-COS improved the decision formulation but its risk-selected policy took no early actions. CAPER—**Continuous Anticipatory Prefetch and Evidence Reconciliation**—separates the expected value of starting bounded work from permission to use evidence in a claim.

For current state `x`, CAPER starts a provisional search when

`E[useful preparation | x] - E[compute + revision + queue cost | x] > 0`,

subject to domain, cheap-probe, content, concurrency, and queue gates. This is a deterministic engineering rule in V2.4; no learned model was trained. Commitment is a separate Boolean conjunction over current session/intent identity, compatible query revision, critical constraints, corpus version, source checksum, expiry, current relevance, and subsequent Grounding Firewall success.

## Architecture

Lane A continuously observes QueryFrame/EFIS intent candidates, uses the cheap BM25 probe, schedules bounded `asyncio` tasks, and produces provisional results. Lane B revalidates those results against current intent and constraints. Only a `VALIDATED` record can replace a primary search; only evidence used by a claim that passes grounding can become `COMMITTED`.

Lifecycle states are `ABSENT → PROBE_READY → PREFETCH_QUEUED → PREFETCH_RUNNING → PROVISIONAL → VALIDATION_PENDING → VALIDATED → COMMITTED`, with explicit `STALE`, `INVALIDATED`, `FAILED`, and `EXPIRED` paths. Blocking searches are not falsely described as canceled: queued work can be canceled before execution; running work finishes and is discarded if superseded.

CAPER retains QueryFrame V2, EFIS, hybrid retrieval, relevance admission, Grounding Firewall, CDDM, session isolation, and extractive synthesis. Provisional work cannot publish a claim.

## Corrected methodology

V2.4 separates action benefit from evidence safety; gives terminal waiting the value/cost of final retrieval; applies one chronological first-action rule per intent lineage; reports zero-action risk as undefined; distinguishes offline Stage A from real-engine Stage B; and counts actual `HybridIndex.search` calls. Proxy strict-usefulness and verified strict-usefulness are separate nullable labels. See `V2_4_BASELINE_AUDIT.md`.

## Development ablation

| Arm | Verified useful before final | Prepared before final | Unused completed preparation | Actual searches | Cache reuses |
|---|---:|---:|---:|---:|---:|
| C0 PSRC baseline | 12/25 (48%) | 0/25 (0%) | undefined | 60 | 0 |
| C1 validated cache | 12/25 (48%) | 0/25 (0%) | undefined | 48 | 12 |
| C2 prefetch/no reuse | 13/25 (52%) | 10/25 (40%) | 11/11 (100%) | 71 | 0 |
| C4 strict reuse | 13/25 (52%) | 10/25 (40%) | 8/11 (72.73%) | 67 | 4 |
| C8 deterministic CAPER | 13/25 (52%) | 10/25 (40%) | 8/11 (72.73%) | 67 | 4 |
| C9 eager bounded | 16/25 (64%) | 18/25 (72%) | 18/22 (81.82%) | 76 | 6 |
| C10 end of utterance | 0/25 (0%) | 0/25 (0%) | undefined | 47 | 0 |

C8's verified-early Wilson 95% interval is `[0.3350, 0.6997]`; C9's is `[0.4452, 0.7975]`. No arm reached the historical G2 engineering target of 0.80 under this new metric. C7 learned prefetch is `NOT_RUN`: deterministic CAPER had not met the cost/waste acceptance criteria needed to justify training complexity.

All measured development arms published zero invalid citations. C8 had `0/49` invalid citations. Actual C8 searches were 20 primary, 11 provisional, 11 refinement, and 25 final fallback searches.

## Frozen final check

Selection was written before final outcomes were loaded. On the project-authored final split, C8 and C0 both produced verified useful evidence before final on `2/11 = 18.18%`, with 33 versus 27 searches. C8 left `9/11 = 81.82%` completed preparations unused and produced `0/9` invalid citations. This split is not independent and shows no final improvement.

## Error analysis and release decision

The implementation demonstrates that validated evidence can be ready early and safely reused, but selection remains inefficient. C8 improved the grounding-linked development outcome by only one stream and tied baseline on the limited final split. Many early searches either returned no admitted evidence, became irrelevant to the final intent, or were not consumed by final query decomposition. Eager preparation increases coverage but worsens waste. The correct next research step is better low-cost selection/reconciliation, not weakening validation.

Decision: `RETAIN_AS_EXPERIMENTAL`. Historical PSRC remains the default. Reasons: deterministic CAPER gained only 4 percentage points and failed the ≥10-point meaningful-improvement gate, failed the ≤50% wasted-work budget, has no independent final set, did not reach the G2-equivalent 0.80 target, and lacks physical microphone validation.

## Foundations and attribution

Tavus [Griffin](https://www.tavus.io/griffin) is an industry architectural inspiration for continuous/full-duplex processing; KAIROS neither reproduces nor claims knowledge of its proprietary implementation. Arora et al., [Stream RAG](https://proceedings.mlr.press/v306/arora26a.html), motivates tool work during speech. Jiang et al., [Active Retrieval Augmented Generation](https://aclanthology.org/2023.emnlp-main.495/), motivates explicit retrieval timing. Webber et al., [Rank-Biased Overlap](https://doi.org/10.1145/1852102.1852106), supports evolving-rank comparison. Guo et al., [calibration](https://proceedings.mlr.press/v70/guo17a.html), motivates target-matched calibration. Selfridge et al., [incremental ASR stability](https://aclanthology.org/W11-2014/), motivates revision-aware processing. CAPER and all reported results are original KAIROS engineering, not results from those sources.
