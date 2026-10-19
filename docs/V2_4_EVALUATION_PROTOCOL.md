# KAIROS V2.4 Evaluation Protocol

## Objective and populations

The primary outcome is **verified useful evidence available before utterance finalization**, not speculative search starts. An eligible stream has gold evidence and at least one non-final prefix. A success requires either an early CAPER `EvidenceValidated` event or an early relevance-admitted `RetrievalCompleted` event, gold-document alignment, and evidence that is later cited by a claim emitted by the Grounding Firewall. This post-hoc evaluation rule gives baseline and CAPER the same safety criterion; future grounding is never an online policy feature.

Development selection uses 36 frozen V2.3 streams; 25 are eligible for the primary outcome. The post-freeze final set contains 12 project-authored streams; 11 are eligible. Question families are the uncertainty unit. The final set reuses the sample corpus and is not independently authored, so it is limited validation, not generalization evidence.

## Timing and stages

Stage B calls the real `SessionEngine`, `HybridIndex`, relevance admission, synthesis, Grounding Firewall, and CDDM. Prefix timestamps are deterministic synthetic written-question timing. A 10 ms scheduling opportunity lets background tasks run between chunks; this is accelerated scheduled replay, not natural speech latency. Physical microphone testing is `PENDING`.

## Arms

- C0: historical PSRC, no CAPER or evidence cache.
- C1: validated historical cache only.
- C2: deterministic prefetch, no commit reuse.
- C3: provisional cache, no commit reuse.
- C4: strict validation plus reuse.
- C5: query-version reconciliation enabled.
- C6: bounded async with one active worker and two queued tasks.
- C7: learned prefetch (`NOT_RUN`; deterministic evidence was required first).
- C8: full deterministic CAPER.
- C9: eager bounded comparator.
- C10: end-of-utterance controller.

All CAPER runtime arms retain strict source, session, query, constraint, relevance, expiry, and grounding checks. Ablations never disable the claim-safety boundary.

## Preregistered development gates

Before reading final outcomes, `selection_freeze.json` fixed these gates:

- verified-early rate must improve on C0 by at least 10 absolute percentage points;
- invalid citations must equal zero;
- actual searches must be at most 1.5× C0;
- unused completed preparations must be at most 50%;
- independent final evidence must exist for a deployment claim.

The deterministic C8 candidate improved the primary development rate from `12/25` to `13/25`, used 67 versus 60 searches, and had zero invalid citations across 49 emitted claims. Its 4-point gain failed the 10-point meaningful-improvement gate; it also failed the waste gate (`8/11 = 72.73%`) and the independent-final gate. The limited final split tied baseline at `2/11`. The frozen decision is `RETAIN_AS_EXPERIMENTAL`.

## Metrics and uncertainty

Every rate records numerator, denominator, unit, eligibility, timing source, grounding status, split, Wilson interval, and measurement source. A 2,000-sample deterministic question-family cluster bootstrap reports uncertainty for the primary outcome. Search counts come only from the `HybridIndex.search` execution registry and are split into primary, provisional, refinement, and final fallback work. Cache reuse is derived from `RetrievalCompleted.cache_status`; cache hits do not masquerade as search executions.

## Reproduction

```bash
python -m kairos controller-v2-4-data validate
python -m kairos controller-v2-4-evaluate
python -m kairos controller-v2-4-compare
python -m kairos controller-v2-4-select
python -m kairos controller-v2-4-final
python -m kairos controller-v2-4-security-audit
```

The complete frozen bundle is `runs/bench/controller_v2_4/20261019T081648Z/`.
