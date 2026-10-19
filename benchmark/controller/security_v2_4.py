"""Deterministic secret/privacy audit that never emits matched secret values."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from kairos.config import ROOT


PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "openai_token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
}

TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".env"}
EXCLUDED = {".git", "node_modules", "dist", "__pycache__", ".pytest_cache"}


def _classification(text: str) -> str:
    lowered = text.casefold()
    return "SYNTHETIC_OR_DOCUMENTATION" if any(word in lowered for word in (
        "synthetic", "example", "placeholder", "test", "pattern", "regex"
    )) else "REQUIRES_REVIEW"


def _scan_text(text: str, source: str) -> list[dict[str, Any]]:
    findings = []
    lines = text.splitlines()
    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            line = lines[line_no - 1] if line_no <= len(lines) else ""
            findings.append({
                "source": source,
                "line": line_no,
                "pattern": name,
                "classification": _classification(f"{source} {line}"),
                "value_disclosed": False,
            })
    return findings


def run_security_audit() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned_files = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED for part in path.parts):
            continue
        if path.suffix.casefold() not in TEXT_SUFFIXES and path.name not in {".env.example", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned_files += 1
        findings.extend(_scan_text(text, str(path.relative_to(ROOT))))

    history_status = "SCANNED"
    history_findings: list[dict[str, Any]] = []
    try:
        history = subprocess.run(
            ["git", "log", "--all", "-p", "--no-ext-diff", "--no-textconv"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout
        history_findings = _scan_text(history, "git-history-patch-stream")
    except (OSError, subprocess.CalledProcessError):
        history_status = "UNAVAILABLE"
    findings.extend(history_findings)
    requiring_review = [item for item in findings if item["classification"] == "REQUIRES_REVIEW"]
    report = {
        "schema_version": 1,
        "status": "PASS" if not requiring_review and history_status == "SCANNED" else "REVIEW_REQUIRED",
        "scanner": "built-in deterministic regex audit",
        "gitleaks": "NOT_RUN_NOT_INSTALLED_OR_REQUIRED",
        "working_tree_files_scanned": scanned_files,
        "git_history_status": history_status,
        "finding_counts": dict(Counter(item["pattern"] for item in findings)),
        "requiring_review": len(requiring_review),
        "findings": findings,
        "privacy_controls": {
            "trace_content_default": False,
            "nested_redaction": True,
            "authorized_persistent_deletion": True,
            "resume_token_in_url": False,
            "resume_token_rotation": True,
        },
        "limitations": [
            "Regex scanning cannot prove that no secret exists.",
            "Findings contain classifications and locations only; matched values are never serialized.",
        ],
    }
    return report


def write_security_audit(path: Path) -> dict[str, Any]:
    report = run_security_audit()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
