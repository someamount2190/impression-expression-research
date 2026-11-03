import hashlib
import json

import pytest

from core.capability_broker import CapabilityBroker
from core.tool_registry import OPERATION_SPECS
from experiments.expression_growth_001.curriculum.build_curriculum import (
    build_curriculum, load_training_examples,
)
from experiments.expression_growth_001.symbolic_ir.typed_plan import parse_plan


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_builds_valid_curriculum_with_family_disjoint_sealed_100(tmp_path):
    manifest = build_curriculum(tmp_path, seed=42)
    train = read_jsonl(tmp_path / "train" / "examples.jsonl")
    validation = read_jsonl(tmp_path / "validation" / "examples.jsonl")
    sealed = read_jsonl(tmp_path / "sealed_evaluation" / "manifest.jsonl")

    assert len(train) == manifest["train_count"]
    assert len(validation) == manifest["validation_count"]
    assert len(sealed) == 100
    family_sets = [{item["family"] for item in split} for split in (train, validation, sealed)]
    assert not (family_sets[0] & family_sets[1])
    assert not (family_sets[0] & family_sets[2])
    assert not (family_sets[1] & family_sets[2])
    for split in (train, validation, sealed):
        for example in split:
            parse_plan(example["target_plan"])
            assert example["provenance"].startswith("synthetic_curriculum_template")
            for schema in example["available_capability_schemas"]:
                key = (schema["capability"], schema["operation"])
                assert key in OPERATION_SPECS
                assert schema["operation"] in CapabilityBroker.CAPABILITIES[schema["capability"]]
                live = OPERATION_SPECS[key]
                assert set(schema["parameters"]) == set(live.params)


def test_required_curriculum_case_types_are_present(tmp_path):
    manifest = build_curriculum(tmp_path)
    all_types = set()
    for split in ("train", "validation", "sealed_evaluation"):
        path = tmp_path / split / ("manifest.jsonl" if split == "sealed_evaluation" else "examples.jsonl")
        rows = read_jsonl(path)
        all_types.update(row["case_type"] for row in rows)
    assert {"positive_multistep", "unsupported_operation_abstention", "permission_abstention",
            "error_recovery", "dependency_and_observation", "observation_integrity",
            "session_continuity"} <= all_types
    assert manifest["historical_trace_source_used"] is False


def test_sealed_hash_freezes_exact_bytes_and_training_loader_omits_it(tmp_path):
    build_curriculum(tmp_path, seed=7)
    sealed_path = tmp_path / "sealed_evaluation" / "manifest.jsonl"
    digest = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / "hash_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sealed_evaluation_frozen"] is True
    assert digest == manifest["sealed_evaluation_manifest_sha256"]
    assert len(load_training_examples(tmp_path)) == 2
    train, validation = load_training_examples(tmp_path)
    assert all(row["split"] == "train" for row in train)
    assert all(row["split"] == "validation" for row in validation)


def test_builder_is_deterministic_for_identical_seed(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    m1 = build_curriculum(first, seed=123)
    m2 = build_curriculum(second, seed=123)
    assert m1["sealed_evaluation_manifest_sha256"] == m2["sealed_evaluation_manifest_sha256"]
    assert m1["train_sha256"] == m2["train_sha256"]
    assert m1["validation_sha256"] == m2["validation_sha256"]


def test_frozen_sealed_manifest_refuses_seed_change_or_tampering(tmp_path):
    build_curriculum(tmp_path, seed=20)
    with pytest.raises(RuntimeError, match="would change the frozen sealed evaluation"):
        build_curriculum(tmp_path, seed=21)
    sealed_path = tmp_path / "sealed_evaluation" / "manifest.jsonl"
    sealed_path.write_bytes(sealed_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        build_curriculum(tmp_path, seed=20)
