from __future__ import annotations

import asyncio

from benchmark.controller.features_v2_3 import feature_parity_differences, replay_feature_rows
from kairos.controller.features import FEATURE_NAMES, FEATURE_SCHEMA, feature_schema_hash
from kairos.engine import ChunkIn, EngineContext, SessionEngine


def _stream():
    return {
        "stream_id": "parity-1",
        "session_id": "parity-1",
        "utterance_id": "u1",
        "question_family": "parity-1",
        "source_dataset": "test",
        "source_question_id": "parity-1",
        "corpus_profile": "sample",
        "split": "test",
        "prefixes": [
            "tell me the cancellation",
            "tell me the cancellation policy",
            "tell me the cancellation policy for venues?",
        ],
        "prefix_timestamps": [1.0, 2.0, 3.0],
        "gold_sections": ["Venue Selection Guide > Cancellation"],
    }


def test_feature_schema_is_explicit_and_stable():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert len(feature_schema_hash()) == 64
    assert all(spec.dtype and spec.definition and spec.value_range and spec.missing_value for spec in FEATURE_SCHEMA)
    history_features = {spec.name for spec in FEATURE_SCHEMA if spec.uses_history}
    assert {"probe_section_rbo", "slot_stability"} <= history_features
    assert {"ranking_history_available", "slot_history_available"} <= set(FEATURE_NAMES)


def test_first_observation_missing_history_is_not_fabricated(ctx):
    row = replay_feature_rows(ctx, _stream())[0]
    assert row["features"]["ranking_history_available"] == 0.0
    assert row["features"]["probe_section_rbo"] == 0.0
    assert row["features"]["slot_history_available"] == 0.0
    assert row["features"]["slot_stability"] == 0.0


def test_training_replay_matches_live_inference(cfg):
    parity_cfg = cfg.with_overrides({"controller.mode": "eou", "corpus.profile": "sample"})
    stream = _stream()
    training_rows = replay_feature_rows(EngineContext(parity_cfg), stream)

    async def execute():
        engine = SessionEngine(EngineContext(parity_cfg), "feature-parity")
        engine.start()
        chunks = ["tell me the cancellation", "policy", "for venues?"]
        start = 0.0
        for index, (text, end) in enumerate(zip(chunks, stream["prefix_timestamps"]), start=1):
            await engine.on_chunk(ChunkIn(
                utterance_id="u1", text=text, t0=start, t1=end,
                final=index == len(chunks),
            ))
            start = end
        return engine.events

    differences = feature_parity_differences(training_rows, asyncio.run(execute()))
    assert differences == []
