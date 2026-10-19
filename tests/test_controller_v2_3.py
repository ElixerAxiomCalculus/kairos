from __future__ import annotations

import copy
import json

import pytest

from benchmark.controller.counterfactual import label_optimal_stopping, oracle_summary
from benchmark.controller.audit_v2_3 import docs_consistency_audit
from benchmark.controller.intent_alignment import (
    AlignmentStatus,
    EvidenceIdentity,
    classify_final_alignment,
    survival,
)
from kairos.controller.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, feature_schema_hash
from kairos.controller.stopping_model import ModelCompatibilityError, StoppingModel


def _row(index, *, final=False, useful=0, time_saved=1.0, revision=0.0):
    return {
        "stream_id": "s", "candidate_id": "k", "prefix_index": index, "is_final": final,
        "targets": {
            "gold_relevance": useful, "constraint_compatibility": useful,
            "evidence_survival": useful, "answerability": useful,
            "grounding_eligibility": useful, "strict_useful": useful,
            "retrieval_cost": 1.0, "time_saved_s": time_saved, "revision_cost": revision,
        },
    }


def test_intent_alignment_statuses_and_survival_are_explicit():
    identity = EvidenceIdentity(("c1",), ("D1#1",), ("D1",))
    gold = [{"intent_id": "A", "gold_sections": ["D1#1"], "gold_documents": []}]
    assert classify_final_alignment(identity=identity, gold_intents=gold, domain_available=True,
                                    lineage_defined=True) == AlignmentStatus.SUPPORTED_GOLD_EVIDENCE
    assert classify_final_alignment(identity=None, gold_intents=gold, domain_available=True,
                                    lineage_defined=False) == AlignmentStatus.UNDEFINED_ALIGNMENT
    assert classify_final_alignment(identity=identity, gold_intents=[], domain_available=True,
                                    lineage_defined=True) == AlignmentStatus.MISSING_GOLD_ANNOTATION
    values = survival(identity, identity)
    assert values["accepted_chunk_rbo"] == 1.0
    assert values["accepted_section_jaccard"] == 1.0


def test_backward_induction_excludes_final_and_separates_actions():
    rows = [_row(1, useful=1, time_saved=2.0), _row(2, useful=0, revision=1.0), _row(3, final=True)]
    labeled = label_optimal_stopping(rows)
    assert [item["prefix_index"] for item in labeled] == [1, 2]
    assert all("q_retrieve" in item and "q_wait" in item and "delta_q" in item for item in labeled)
    summary = oracle_summary(rows)
    assert summary["terminal_rows_excluded"] == 1
    assert summary["earliest_useful_prefix"] == 1


def test_stopping_model_rejects_schema_drift(tmp_path):
    artifact = json.loads(open("models/controller_v2_3/iccos_v1.json", encoding="utf-8").read())
    artifact["feature_schema_hash"] = "bad"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ModelCompatibilityError, match="FEATURE_SCHEMA_HASH_MISMATCH"):
        StoppingModel.load(path)


def test_stopping_model_requires_every_live_feature():
    model = StoppingModel.load("models/controller_v2_3/iccos_v1.json")
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.pop(model.features[0])
    with pytest.raises(ModelCompatibilityError, match="MISSING_LIVE_FEATURES"):
        model.predict(features)


def test_model_artifact_has_reproducibility_contract():
    model = StoppingModel.load("models/controller_v2_3/iccos_v1.json")
    artifact = model.artifact
    assert artifact["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert artifact["feature_schema_hash"] == feature_schema_hash()
    assert len(artifact["dataset_hash"]) == 64
    assert artifact["objective"]
    assert artifact["validation"]["future_feature_audit"] == "PASS"


def test_v2_3_documentation_and_artifacts_are_consistent():
    result = docs_consistency_audit()
    assert result["status"] == "PASS", result["failures"]
