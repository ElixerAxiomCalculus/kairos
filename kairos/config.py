"""Configuration loading. YAML files are merged over configs/default.yaml."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Config(dict):
    """Nested dict with attribute-style section access: cfg.retrieval["top_k"]."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - programmer error
            raise AttributeError(name) from exc

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied, e.g. {"retrieval.mode": "dense"}."""
        data = copy.deepcopy(dict(self))
        for dotted, value in overrides.items():
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data)


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_env() -> None:
    """Load KEY=VALUE lines from kairos/.env and ./.env into os.environ (existing variables win).
    Values are never logged or included in telemetry."""
    for env_path in (ROOT / "kairos" / ".env", ROOT / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    load_env()
    data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path:
        extra = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = _deep_merge(data, extra)
    cfg = Config(data)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path
