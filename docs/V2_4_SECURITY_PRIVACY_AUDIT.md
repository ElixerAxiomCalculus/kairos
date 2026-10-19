# KAIROS V2.4 Security and Privacy Audit

Status: automated controls pass; this is not a formal penetration test.

## Changes

- Persistent trace content is default-deny: `telemetry.trace_content: false`.
- Redaction recursively covers strings inside arbitrary nested objects and arrays. Operational enums and integrity hashes may remain; transcript-like strings are replaced with SHA-256 markers.
- Session resume capabilities are no longer placed in WebSocket URLs. A resumed socket sends the token in its first message, under a timeout, and successful resume rotates the token.
- Old and incorrect tokens are rejected. Cross-session access still requires the exact session capability.
- `DELETE /api/sessions/{session_id}/data` performs authorized deletion through the in-memory bus and configured JSONL/SQLite sinks.
- Ending a session and deleting persisted data are distinct operations and are documented as such.

## Automated audit

The frozen run scanned 591 current text files and the complete Git patch history with a deterministic secret-pattern scanner. It found one deliberately synthetic OpenAI-shaped token in `tests/test_caper_v2_4.py`, classified it as synthetic, and serialized no matched value. No finding required review. The report status is `PASS`.

The scanner checks private-key headers and common AWS, GitHub, OpenAI, Google, and JWT token shapes. It records only pattern class, source location, classification, and `value_disclosed: false`. Regex scanning cannot prove absence of secrets. Gitleaks was not installed or run, so the built-in scan is a reproducible minimum, not an industry certification.

## Tests

V2.4 tests cover nested redaction, synthetic-secret detection, wrong and rotated session tokens, first-message WebSocket resume, authorized persistent deletion, and session-scoped provisional cache promotion. Existing API/session tests cover trace authorization and isolation.

## Remaining operator responsibilities

- Terminate TLS and secure proxy logs in deployment.
- Protect filesystem permissions for trace and SQLite locations.
- Set restrictive CORS origins; wildcard origins are unsuitable for exposed deployments.
- Establish retention/backup deletion policy. The API deletes active configured sinks but cannot erase external backups or copied logs.
- Run a dedicated secret scanner and dependency/vulnerability process in CI.
- Do not enable transcript persistence without a lawful retention basis and user disclosure.
