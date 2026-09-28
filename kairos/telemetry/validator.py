"""Lineage validator: telemetry coverage is a checked property, not a hope.

Invariants (each instance counts toward coverage = satisfied / total):
  I1  every AnswerVersionCreated has a causal path back to a TranscriptChunkReceived
  I2  every RetrievalStarted has a matching RetrievalCompleted (same sub-query, parented)
  I3  every claim in an answer version was checked (GroundingVerified or TransformVerified)
  I4  every citation in an answer version resolves to a chunk from a RetrievalCompleted in the session
  I5  every RetrievalSuppressed names the answer version / sub-query whose evidence it reused
  I6  every event carries the required envelope fields and a known parent (except roots)
"""

from __future__ import annotations

from collections.abc import Iterable

from kairos.schemas.events import Event

ROOT_TYPES = {"SessionStarted", "TranscriptChunkReceived"}


def validate(events: Iterable[Event | dict]) -> dict:
    evs = [e if isinstance(e, dict) else e.model_dump() for e in events]
    by_id = {e["event_id"]: e for e in evs}
    checks: dict[str, list[bool]] = {f"I{i}": [] for i in range(1, 7)}
    failures: list[dict] = []

    def reaches_chunk(eid: str) -> bool:
        seen, stack = set(), [eid]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in by_id:
                continue
            seen.add(cur)
            if by_id[cur]["event_type"] == "TranscriptChunkReceived":
                return True
            stack.extend(by_id[cur]["parent_event_ids"])
        return False

    retrieved_chunks = {c for e in evs if e["event_type"] == "RetrievalCompleted" for c in e["chunk_ids"]}
    verified_claims: set[str] = set()
    for e in evs:
        if e["event_type"] == "GroundingVerified":
            verified_claims |= {r["claim_id"] for r in e["payload"].get("results", [])}
        if e["event_type"] == "TransformVerified":
            verified_claims |= set(e["payload"].get("claims", []))

    completed_parents = {p for e in evs if e["event_type"] == "RetrievalCompleted" for p in e["parent_event_ids"]}

    for e in evs:
        et = e["event_type"]
        ok6 = bool(e["session_id"] and e["event_type"] and e["timestamp"] is not None)
        if et not in ROOT_TYPES and et != "SessionEnded":
            ok6 = ok6 and bool(e["parent_event_ids"]) and all(p in by_id for p in e["parent_event_ids"])
        checks["I6"].append(ok6)
        if not ok6:
            failures.append({"invariant": "I6", "event_id": e["event_id"], "event_type": et})

        if et == "AnswerVersionCreated":
            ok = reaches_chunk(e["event_id"])
            checks["I1"].append(ok)
            if not ok:
                failures.append({"invariant": "I1", "event_id": e["event_id"]})
            for claim in e["payload"].get("claims", []):
                ok3 = claim["claim_id"] in verified_claims
                checks["I3"].append(ok3)
                if not ok3:
                    failures.append({"invariant": "I3", "claim_id": claim["claim_id"]})
                for cid in claim.get("evidence", []):
                    ok4 = cid in retrieved_chunks
                    checks["I4"].append(ok4)
                    if not ok4:
                        failures.append({"invariant": "I4", "claim_id": claim["claim_id"], "chunk_id": cid})
        elif et == "RetrievalStarted":
            ok = e["event_id"] in completed_parents
            checks["I2"].append(ok)
            if not ok:
                failures.append({"invariant": "I2", "event_id": e["event_id"]})
        elif et == "RetrievalSuppressed":
            p = e["payload"]
            ok = p.get("reused_answer_version") is not None or p.get("reused_sub_query") is not None
            checks["I5"].append(ok)
            if not ok:
                failures.append({"invariant": "I5", "event_id": e["event_id"]})

    total = sum(len(v) for v in checks.values())
    passed = sum(sum(v) for v in checks.values())
    return {
        "coverage": round(passed / total, 6) if total else 1.0,
        "instances": total,
        "passed": passed,
        "per_invariant": {k: {"checked": len(v), "passed": sum(v)} for k, v in checks.items()},
        "failures": failures[:50],
    }
