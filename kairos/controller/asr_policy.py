"""Pure functions for replayable ASR prefix-commitment experiments."""

from __future__ import annotations


def common_prefix_length(left: list[str], right: list[str]) -> int:
    size = min(len(left), len(right))
    idx = 0
    while idx < size and left[idx] == right[idx]:
        idx += 1
    return idx


def committed_length(previous: str, current: str, *, strategy: str, mutable_tail: int = 2,
                     recent_revision_rate: float = 0.0) -> int:
    prev_tokens, current_tokens = previous.split(), current.split()
    agreed = common_prefix_length(prev_tokens, current_tokens)
    if strategy == "fixed_tail":
        tail = mutable_tail
    elif strategy == "agreement":
        tail = 0
    elif strategy == "adaptive_tail":
        tail = mutable_tail + (1 if recent_revision_rate > 0.25 else -1 if recent_revision_rate == 0 else 0)
        tail = max(0, tail)
    elif strategy in {"provisional_probe", "hybrid"}:
        return len(current_tokens)
    else:
        raise ValueError(f"unknown ASR strategy: {strategy}")
    return max(0, agreed - tail)
