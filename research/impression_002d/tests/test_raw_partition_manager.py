import json
from pathlib import Path

import psutil
import pytest

from experiments.impression_002b.residency.manager import ResidencyError
from experiments.impression_002d.worker.raw_partition_manager import RawPartitionResidencyManager

ROOT = Path(__file__).resolve().parents[3]
REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
REGISTRY = ROOT / "experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json"
SNAPSHOT = Path(json.loads((ROOT / "experiments/impression_002d/runtime_baseline.json").read_text())
                ["checkpoint"]["snapshot_path"])


def _manager() -> RawPartitionResidencyManager:
    reserve = int(psutil.virtual_memory().total * 0.20) + 64 * 1024**2
    return RawPartitionResidencyManager(registry_path=REGISTRY, snapshot_path=SNAPSHOT,
        expected_revision=REVISION, minimum_available_bytes=reserve, max_resident_partitions=1)


def test_raw_manager_load_release_hash_matches_002b_genuine_layer0():
    manager = _manager()
    block_id = "IMP_L00_ATTENTION_INPUT_NORM"
    weights = manager.load(block_id)
    receipt = json.loads((ROOT / "experiments/impression_002d/diagnostics/worker/genuine_layer0/child_receipt.json").read_text())
    expected = next(row for row in receipt["partitions"] if row["partition_id"] == block_id)
    assert manager.last_measurement["content_sha256"] == expected["content_sha256"]
    with manager.executing(block_id) as view:
        assert list(view) == ["model.layers.0.input_layernorm.weight"]
        assert str(view["model.layers.0.input_layernorm.weight"].dtype) == "torch.bfloat16"
    manager.release(block_id)
    weights.clear()
    assert manager.snapshot()["resident_partition_ids"] == []
    assert manager.snapshot()["records"][block_id]["state"] == "UNLOADED"


def test_raw_manager_rejects_shape_mismatch_before_materializing():
    manager = _manager()
    manager.registry["tensors"]["blk.0.attn_norm.weight"]["array_shape"] = [3, 1024]
    with pytest.raises(ResidencyError, match="shape mismatch"):
        manager.load("IMP_L00_ATTENTION_INPUT_NORM")
    assert manager.snapshot()["resident_partition_ids"] == []


def test_raw_manager_rejects_snapshot_outside_revision():
    with pytest.raises(ResidencyError, match="pinned revision"):
        RawPartitionResidencyManager(registry_path=REGISTRY, snapshot_path=SNAPSHOT.parent,
            expected_revision=REVISION, minimum_available_bytes=0)


def test_raw_manager_rejects_nested_directory_under_revision_named_ancestor():
    impostor = SNAPSHOT.parent / REVISION / "nested"
    with pytest.raises(ResidencyError, match="directory name does not equal pinned revision"):
        RawPartitionResidencyManager(registry_path=REGISTRY, snapshot_path=impostor,
            expected_revision=REVISION, minimum_available_bytes=0)


def test_exact_layer4_raw_probe_artifact_is_released_and_matches_prior_digest():
    artifact = ROOT / "experiments/impression_002d/diagnostics/worker/phase_e_raw_layer4_probe_run01/child_receipt.json"
    receipt = json.loads(artifact.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["job_type"] == "raw_partition_probe"
    assert receipt["final_resident_partition_ids"] == []
    assert receipt["raw_load_measurement"]["content_sha256"] == "1089a66f68616ad96aa04d2ef2fadcc9d5f564a0944a088430fae8104e7720d8"
    assert receipt["raw_load_measurement"]["tensor_metadata"][0]["content_sha256"] == "2ebe566272eb6afe8f449571f98f360cd57b418fadeb7c62b6cd30b40937957f"

