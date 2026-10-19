# KAIROS V2.4 CAPER Engineering Report

Status: implementation complete; runtime feature remains disabled by default. Frozen measurement run: `20261019T081648Z`.

## Implementation

`kairos/controller/caper.py` adds the session-local evidence state machine, deterministic/eager preparation decisions, semantic query and constraint fingerprints, bounded async scheduling, supersession, validation, consumption, and grounded commitment. `evidence_cache.py` now distinguishes provisional, validated, invalidated, and expired entries and requires matching session, intent, query version, constraint fingerprint, corpus version, and source hashes for promotion.

`HybridIndex.search` registers every real execution with a unique ID, type, session, intent, query version, start/end time, failure, and discard status. Cache hits do not increment the search counter. `SessionEngine` observes CAPER after cheap probes, optionally waits a bounded interval at finalization, reuses only validated evidence, and emits `EvidenceCommitted` only after `GroundingVerified`.

Concurrency is bounded by a per-session semaphore (default 2), per-intent active lineage (default 1), queue depth (default 4), debounce (80 ms), expiry (30 s), and final wait (250 ms). A queued task may be truly canceled. Python thread work already inside a blocking search is allowed to finish and is recorded as discarded when stale; the system never claims that work was canceled.

## Events and UI

New mirrored Python/TypeScript events are `PrefetchDecision`, `PrefetchQueued`, `PrefetchStarted`, `PrefetchCompleted`, `PrefetchSuperseded`, `PrefetchDiscarded`, `EvidenceValidationStarted`, `EvidenceValidated`, `EvidenceInvalidated`, and `EvidenceCommitted`. The timeline distinguishes preparing, provisional, validating, verified, invalidated, and committed states. `/api/controller-v2-4/latest` feeds the existing `/benchmark` page with measured CAPER results and the negative release decision.

## Security/API migration

WebSocket resume tokens moved from query strings to the first message and rotate after successful resume. Clients using URL tokens must migrate. Session trace deletion is available at `DELETE /api/sessions/{session_id}/data` with `X-Kairos-Session-Token`. Trace-content persistence is off by default and nested redaction is recursive. See `V2_4_SECURITY_PRIVACY_AUDIT.md`.

## Configuration and fallback

```yaml
controller:
  caper:
    enabled: false
    policy: deterministic
    max_active_prefetches_per_session: 2
    max_active_prefetches_per_intent: 1
    max_queued_prefetches_per_session: 4
    provisional_cache_enabled: true
    validated_reuse_enabled: true
    strict_commit_validation: true
```

CAPER is not silently labeled active when disabled. Historical PSRC remains the default. Learned prefetch has no model artifact and is `NOT_RUN`.

## Commands

```bash
python -m kairos controller-v2-4-data validate
python -m kairos controller-v2-4-prefetch-bench
python -m kairos controller-v2-4-cache-bench
python -m kairos controller-v2-4-security-audit
python -m kairos controller-v2-4-compare
python -m kairos controller-v2-4-train
python -m kairos controller-v2-4-select
python -m kairos controller-v2-4-final
python -m kairos controller-v2-4-report
```

## Verification and limitations

The final backend suite passed `77` tests with one third-party `pytz` deprecation warning. The V2.4-specific suite covers lifecycle ordering, critical city revision invalidation, actual search accounting, cache isolation, corrected risk semantics, terminal fallback, recursive redaction, token rotation/rejection, persistent deletion, and documentation/artifact consistency. All five canonical replays retained telemetry coverage `1.0` (110/68/37/25/111 invariant instances). Final benchmark run `20261019T074646Z` preserved G2 `0.6875` (FAIL) and passed all other configured gates. Semantic run `20261019T074649Z` preserved expanded-SQuAD Recall@5 `0.8125`, intent metrics `1.0` on the eight-case semantic set, and relevant-evidence precision `0.875`. Frontend typecheck and production build passed.

Cypress is `NOT_RUN_ENVIRONMENT_BLOCKED`: Cypress 14.5.4 could not start on this Windows 10.0.26200 environment because its cached executable rejected the required `--smoke-test`/`--ping` options. The combined wrapper also depends on absent `wmic.exe`. This is not reported as a browser-test pass.

Known limitations are high unused-preparation cost, small related datasets, synthetic timing, no learned model, no new independent final collection, and pending physical microphone testing. Accordingly the release decision is `RETAIN_AS_EXPERIMENTAL`; correctness mechanisms ship behind a disabled flag.
