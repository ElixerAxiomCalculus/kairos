# Dependency audit — 2026-10-18

- `npm-audit-production.json`: 0 production dependency vulnerabilities.
- `npm-audit-all.json`: 7 development-tool findings (5 moderate, 2 high) after compatible lockfile fixes. Remaining paths are through Cypress/Storybook tooling; npm proposes breaking major changes for full remediation, so they were not forced into this hardening pass.
- `pip-check.txt`: results from the shared user Python installation. The reported conflicts belong to unrelated globally installed packages and are not evidence that KAIROS's declared dependency set resolves incorrectly. Production deployment should use a clean virtual environment.

These files are point-in-time scanner results, not a guarantee that the dependency graph has no vulnerabilities.
