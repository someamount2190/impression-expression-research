from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.impression_002b.residency.manager import (
    PartitionResidencyManager, ResidencyError, ResidencyState, _hf_name,
)


def test_known_gguf_names_map_to_pinned_hf_names():
    assert _hf_name("blk.27.attn_norm.weight") == "model.layers.27.input_layernorm.weight"
    assert _hf_name("blk.27.attn_q.weight") == "model.layers.27.self_attn.q_proj.weight"
    assert _hf_name("blk.27.ffn_down.weight") == "model.layers.27.mlp.down_proj.weight"


def test_unknown_or_non_llama_name_is_rejected():
    with pytest.raises(ResidencyError):
        _hf_name("blk.27.unknown.weight")
    with pytest.raises(ResidencyError):
        _hf_name("made_up.tensor")


def test_manager_rejects_wrong_checkpoint_revision(tmp_path: Path):
    snap = tmp_path / "snapshot" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    registry = tmp_path / "experiments" / "impression_001" / "partitions" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}")
    report = registry.parents[1] / "reports" / "selective_training.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"checkpoint_provenance": {"revision": "different"}}))
    with pytest.raises(ResidencyError, match="expected immutable revision"):
        PartitionResidencyManager(registry_path=registry, snapshot_path=snap,
                                  expected_revision="expected", minimum_available_bytes=0)


def test_manager_can_be_constructed_for_real_registry_without_loading_weights():
    root = Path(__file__).resolve().parents[3]
    report = json.loads((root / "experiments/impression_001/reports/selective_training.json").read_text())
    manager = PartitionResidencyManager(
        registry_path=root / "experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json",
        snapshot_path=report["checkpoint_provenance"]["local_snapshot_path"],
        expected_revision=report["checkpoint_provenance"]["revision"], minimum_available_bytes=0)
    with pytest.raises(ResidencyError, match="Unknown registry partition"):
        manager.load("not-a-partition")
    assert manager.snapshot()["resident_partition_ids"] == []


def test_release_invalidates_execution_view_for_real_pinned_tensor():
    root = Path(__file__).resolve().parents[3]
    report = json.loads((root / "experiments/impression_001/reports/selective_training.json").read_text())
    manager = PartitionResidencyManager(
        registry_path=root / "experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json",
        snapshot_path=report["checkpoint_provenance"]["local_snapshot_path"],
        expected_revision=report["checkpoint_provenance"]["revision"], minimum_available_bytes=0)
    handle = manager.load("IMP_L27_ATTENTION_INPUT_NORM")
    with manager.executing("IMP_L27_ATTENTION_INPUT_NORM") as tensor_view:
        assert len(tensor_view) == 1
    manager.release("IMP_L27_ATTENTION_INPUT_NORM")
    assert handle == {}
    assert len(tensor_view) == 0
