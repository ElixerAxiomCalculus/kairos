"""Mandatory V2.2 anomaly audit and append-only documentation checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.controller.modeling_v2_2 import extract_split
from kairos.config import Config, resolve_path
from kairos.retrieval.index import RetrievalQuery


EXPECTED_EMPTY_NONFINAL = {"train": 29, "calibration": 3, "development": 22}


def audit_v22_empty_final_evidence(cfg: Config, output: Path) -> dict[str, Any]:
    rows = []
    contexts = {}
    for split, expected in EXPECTED_EMPTY_NONFINAL.items():
        extracted = extract_split(cfg, split)
        empty = [row for row in extracted if not row["is_final"] and not row["final_accepted_evidence"]]
        if len(empty) != expected:
            raise RuntimeError(f"V2.2 anomaly count drift for {split}: {len(empty)} != {expected}")
        for row in empty:
            profile = row["corpus_profile"]
            if profile not in contexts:
                from kairos.engine import EngineContext
                contexts[profile] = EngineContext(cfg.with_overrides({"corpus.profile": profile}))
            ctx = contexts[profile]
            query = RetrievalQuery.build(row["final_text"])
            domain, confidence, compatible = ctx.index.route_domain(query.dense_text)
            if not row.get("gold_sections") and not row.get("gold_documents"):
                cause = "MISSING_GOLD_ANNOTATION"
            elif not compatible:
                cause = "OUT_OF_DOMAIN"
            else:
                cause = "NO_RELEVANT_EVIDENCE_AFTER_ADMISSION"
            rows.append({
                "split": split,
                "stream_id": row["stream_id"],
                "candidate_id": row["candidate_id"],
                "prefix_index": row["prefix_index"],
                "final_text": row["final_text"],
                "gold_sections": row.get("gold_sections", []),
                "gold_documents": row.get("gold_documents", []),
                "routed_domain": domain,
                "domain_confidence": round(confidence, 6),
                "domain_compatible": compatible,
                "cause": cause,
            })
    result = {
        "status": "AUDITED",
        "scope": "historical V2.2 non-final feature rows with empty final accepted evidence",
        "expected_counts": EXPECTED_EMPTY_NONFINAL,
        "actual_counts": dict(Counter(row["split"] for row in rows)),
        "cause_counts": dict(Counter(row["cause"] for row in rows)),
        "rows": rows,
        "historical_metrics_rewritten": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def docs_consistency_audit() -> dict[str, Any]:
    required = {
        "README.md": ["KAIROS V2.3", "not the active default", "Research Foundations and References"],
        "docs/PSRC_V2_3_ICCOS_RESEARCH.md": ["IC-COS", "not a published method", "historical frozen comparison set"],
        "docs/PSRC_V2_3_ICCOS_ENGINEERING_REPORT.md": ["25bdc65", "6a13a8", "first_useful_evidence_elapsed_ms"],
    }
    failures = []
    for relative, needles in required.items():
        path = resolve_path(relative)
        if not path.exists():
            failures.append({"file": relative, "reason": "MISSING_FILE"})
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append({"file": relative, "reason": "MISSING_REQUIRED_TEXT", "text": needle})
        if relative == "README.md" and text.rfind("## Research Foundations and References") < max(text.rfind("## License"), text.rfind("## KAIROS V2.3")):
            failures.append({"file": relative, "reason": "REFERENCES_NOT_AT_END"})
    report_path = resolve_path("runs/bench/controller_v2_3/latest.json")
    model_path = resolve_path("models/controller_v2_3/iccos_v1.json")
    if report_path.exists() and model_path.exists():
        from kairos.config import load_config
        from kairos.controller.features import feature_schema_hash
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model = json.loads(model_path.read_text(encoding="utf-8"))
        cfg = load_config()
        documents = {
            relative: resolve_path(relative).read_text(encoding="utf-8")
            for relative in required if resolve_path(relative).exists()
        }
        for relative, document in documents.items():
            if report["run_id"] not in document:
                failures.append({"file": relative, "reason": "STALE_RUN_ID", "expected": report["run_id"]})
        a3 = report["variants"]["A3_iccos_classifier"]["metrics"]["strict_safe_early"]
        expected_fraction = f"{a3['numerator']}/{a3['denominator']}"
        if expected_fraction not in documents.get("README.md", ""):
            failures.append({"file": "README.md", "reason": "STALE_A3_METRIC", "expected": expected_fraction})
        if model["feature_schema_hash"] != feature_schema_hash():
            failures.append({"file": str(model_path), "reason": "FEATURE_SCHEMA_HASH_DRIFT"})
        if bool(cfg["controller"].get("iccos", {}).get("enabled", False)):
            failures.append({"file": "configs/default.yaml", "reason": "UNPROMOTED_POLICY_ENABLED"})
        if model["decision_threshold"] != report["modeling"]["artifact"]["decision_threshold"]:
            failures.append({"file": str(model_path), "reason": "MODEL_REPORT_THRESHOLD_DRIFT"})
    else:
        failures.append({"reason": "MISSING_REPORT_OR_MODEL_FOR_CONSISTENCY_AUDIT"})
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}
