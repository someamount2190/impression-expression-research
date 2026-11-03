import json

import pytest

from experiments.impression_002d.worker.raw_partition_manager import validate_resident_cap
from experiments.impression_002e.streaming_training.train_and_verify import (
    EXPLICIT_HEADROOM_BYTES, SUFFIX_PARTITIONS, atomic_json, canonical_hash, fsync_file,
    replace_with_retry,
    estimate_training_memory, make_causal_objective, _source_run_identity,
)


def test_causal_objective_records_shift_labels_mask_and_reduction():
    objective = make_causal_objective([11, 22, 33, 44])
    assert objective["logit_positions"] == [0, 1, 2]
    assert objective["labels"] == [22, 33, 44]
    assert objective["label_mask"] == [1, 1, 1]
    assert objective["ignore_index"] == -100
    assert objective["valid_label_count"] == 3
    assert objective["reduction"] == "mean over non-ignored shifted labels"
    with pytest.raises(ValueError, match="at least two"):
        make_causal_objective([11])


def test_causal_objective_can_mask_prompt_and_train_only_completion_tokens():
    objective = make_causal_objective([10, 11, 12, 13, 14], label_mask=[0, 0, 1, 1])
    assert objective["labels"] == [11, 12, 13, 14]
    assert objective["label_mask"] == [0, 0, 1, 1]
    assert objective["valid_label_count"] == 2
    with pytest.raises(ValueError, match="label_mask"):
        make_causal_objective([10, 11, 12], label_mask=[0, 0])


def test_memory_estimator_uses_exact_suffix_workset_and_twenty_percent_reserve():
    sizes = {partition: (index + 1) * 1024 for index, partition in enumerate(SUFFIX_PARTITIONS)}
    result = estimate_training_memory(total_ram_bytes=8_000_000_000,
        available_ram_bytes=7_000_000_000, prefix_largest_partition_bytes=4_000_000,
        suffix_partition_bytes=sizes, sequence_length=16, hidden_size=64,
        intermediate_size=128, vocab_size=256)
    assert len(result["suffix_partition_bytes"]) == 11
    assert result["suffix_partition_bytes_total"] == sum(sizes.values())
    assert result["reserve_bytes"] == 1_600_000_000
    assert result["required_available_ram_bytes"] == (
        result["reserve_bytes"] + result["peak_working_set_bytes"] + EXPLICIT_HEADROOM_BYTES)
    assert result["max_prefix_resident_partitions"] == 1
    assert result["max_suffix_resident_partitions"] == 11
    assert result["feasible"] is True


def test_memory_estimator_rejects_incomplete_workset_and_low_memory():
    with pytest.raises(ValueError, match="exact suffix.*11"):
        estimate_training_memory(total_ram_bytes=1_000_000, available_ram_bytes=1_000_000,
            prefix_largest_partition_bytes=100, suffix_partition_bytes={"only": 1},
            sequence_length=16, hidden_size=64, intermediate_size=128, vocab_size=256)
    sizes = {partition: 1_000_000 for partition in SUFFIX_PARTITIONS}
    result = estimate_training_memory(total_ram_bytes=100_000_000, available_ram_bytes=1,
        prefix_largest_partition_bytes=10_000_000, suffix_partition_bytes=sizes,
        sequence_length=16, hidden_size=64, intermediate_size=128, vocab_size=256)
    assert result["feasible"] is False


def test_only_explicit_workset_opt_in_allows_multi_partition_residency():
    assert validate_resident_cap(1) == 1
    with pytest.raises(ValueError, match="allow_workset"):
        validate_resident_cap(11)
    assert validate_resident_cap(11, allow_workset=True) == 11
    with pytest.raises(ValueError, match="positive"):
        validate_resident_cap(0, allow_workset=True)


def test_run_identity_binds_source_cache_token_and_training_contract():
    samples = {"trace_edge_0035": {"token_ids": [1, 2]},
               "trace_schema_0034": {"token_ids": [3, 4]}}
    caches = {"trace_edge_0035": "a" * 64, "trace_schema_0034": "b" * 64}
    identity = _source_run_identity(source_metadata_sha256="c" * 64,
        revision="392a6f57223e7ccfe6ef4ebdb2ff101a42d57364", samples=samples,
        cache_identities=caches)
    digest = identity.pop("run_identity_sha256")
    assert canonical_hash(identity) == digest
    assert identity["source_metadata_sha256"] == "c" * 64
    assert identity["prefix_input_identity_sha256"] == caches
    identity["cache_identity_sha256"]["trace_edge_0035"] = "d" * 64
    assert canonical_hash(identity) != digest


def test_atomic_receipt_replacement_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "training_receipt.json"
    atomic_json(path, {"status": "preflight_passed", "last_completed_operation": "metadata_only_preflight"})
    first = json.loads(path.read_text())
    atomic_json(path, {**first, "status": "training_in_progress", "last_completed_operation": "prefix_done"})
    result = json.loads(path.read_text())
    assert result["status"] == "training_in_progress"
    assert result["last_completed_operation"] == "prefix_done"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["training_receipt.json"]


def test_adapter_file_fsync_uses_write_capable_descriptor(tmp_path):
    path = tmp_path / "adapter.safetensors.tmp"
    path.write_bytes(b"durable adapter payload")
    fsync_file(path)
    assert path.read_bytes() == b"durable adapter payload"


def test_atomic_replace_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    import experiments.impression_002e.streaming_training.train_and_verify as training

    source, destination = tmp_path / "new.tmp", tmp_path / "receipt.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    original = training.os.replace
    calls = {"count": 0}

    def transient_once(src, dst):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("simulated Windows sharing violation")
        return original(src, dst)

    monkeypatch.setattr(training.os, "replace", transient_once)
    replace_with_retry(source, destination, attempts=3)
    assert calls["count"] == 3
    assert destination.read_text(encoding="utf-8") == "new"
