"""Documentation, artifact, configuration, and claim consistency audit for V2.4."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kairos.config import load_config, resolve_path


def docs_consistency_audit() -> dict[str, Any]:
    latest_path = resolve_path("runs/bench/controller_v2_4/latest.json")
    failures: list[dict[str, Any]] = []
    if not latest_path.exists():
        return {"status": "FAIL", "failures": [{"reason": "MISSING_V2_4_REPORT"}]}
    report = json.loads(latest_path.read_text(encoding="utf-8"))
    run_id = report["run_id"]
    run_dir = latest_path.parent / run_id
    c8 = report["development"]["C8_full_caper_deterministic"]["metrics"]
    required = {
        "README.md": ["KAIROS V2.4", run_id, "13/25", "8/11", "RETAIN AS EXPERIMENTAL",
                      "https://www.tavus.io/griffin", "Research Foundations and References"],
        "docs/KAIROS_V2_4_CAPER_RESEARCH.md": [run_id, "13/25", "8/11", "RETAIN_AS_EXPERIMENTAL"],
        "docs/KAIROS_V2_4_CAPER_ENGINEERING.md": [run_id, "enabled: false", "RETAIN_AS_EXPERIMENTAL"],
        "docs/V2_4_EVALUATION_PROTOCOL.md": [run_id, "accelerated scheduled replay", "not independently authored"],
        "docs/V2_4_SECURITY_PRIVACY_AUDIT.md": ["resume capabilities", "Regex scanning cannot prove"],
    }
    for relative, needles in required.items():
        path = resolve_path(relative)
        if not path.exists():
            failures.append({"file": relative, "reason": "MISSING_FILE"})
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append({"file": relative, "reason": "MISSING_REQUIRED_TEXT", "text": needle})
    if c8["verified_useful_evidence_before_final"]["numerator"] != 13:
        failures.append({"reason": "PUBLISHED_C8_NUMERATOR_DRIFT"})
    if c8["search_cost"]["actual_hybrid_search_executions"] != 67:
        failures.append({"reason": "PUBLISHED_C8_SEARCH_COUNT_DRIFT"})
    cfg = load_config()
    if bool(cfg["controller"]["caper"]["enabled"]):
        failures.append({"file": "configs/default.yaml", "reason": "UNPROMOTED_CAPER_ENABLED"})
    if report["active_runtime_policy"] != "historical PSRC; CAPER disabled":
        failures.append({"reason": "DEPLOYMENT_LABEL_DRIFT"})
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if report["selection"]["development_dataset_hash"] != manifest["split_file_sha256"]["development"]:
        failures.append({"reason": "DEVELOPMENT_DATASET_HASH_DRIFT"})
    for relative, expected in report.get("artifact_hashes", {}).items():
        path = run_dir / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            failures.append({"file": relative, "reason": "ARTIFACT_HASH_DRIFT"})
    latency = c8["retrieval_latency_ms"]
    if latency.get("unit") != "ms" or "measurement_source" not in latency:
        failures.append({"reason": "LATENCY_UNIT_OR_SOURCE_MISSING"})
    return {"status": "PASS" if not failures else "FAIL", "run_id": run_id, "failures": failures}
