"""Static controller/fallback and benchmark-schema tests; no model workloads."""
import copy

import pytest

from experiments.impression_002.controller.efficiency_controller import BlockCostProfile, ResourceEnvelope
from experiments.impression_002c.streamed_placement import (
    BLOCK_ID, QUALITY_TOLERANCE, scheduler_select, validate_benchmark_payload,
)


def _profile(config, placement, wall, quality, *, peak_vram=30_000_000):
    return BlockCostProfile(profile_id=f"{config}_r1", config_id=config, config_version=1,
        task_id="fixed_full_[BASE_MODEL]_forward_prompt_8tok_layer27_qproj_placement", block_id=BLOCK_ID,
        block_version=1, checkpoint_sha256="a" * 64, placement=placement, status="success",
        parameter_count=3072 * 3072, trainable_parameter_count=0,
        forward_seconds=wall, total_wall_seconds=wall, setup_seconds=0.0, setup_in_total=True,
        peak_rss_bytes=500_000_000, ram_available_before_bytes=10_000_000_000,
        ram_total_bytes=24_000_000_000, peak_vram_bytes=peak_vram if placement == "cuda" else 0,
        vram_available_before_bytes=3_000_000_000, transferred_bytes=1_000_000 if placement == "cuda" else 0,
        transfer_seconds=0.01 if placement == "cuda" else 0.0, transfer_in_total=True,
        validation_before=0.0, validation_after=quality, metric_lower_is_better=True,
        training_ids=("fixed_prompt",), heldout_ids=("fixed_prompt",), steps=1,
        extra={"minimum_system_ram_available_bytes": 8_000_000_000})


def test_scheduler_uses_measured_controller_and_can_retain_cpu():
    resources = ResourceEnvelope(ram_available_bytes=10_000_000_000, ram_reserve_bytes=4_000_000_000,
                                 vram_available_bytes=3_000_000_000)
    cpu = _profile("cpu_full_stream", "cpu", 100.0, 0.0)
    cuda = _profile("cuda_layer27_qproj", "cuda", 80.0, QUALITY_TOLERANCE / 2)
    selected = scheduler_select([cpu, cuda], resources=resources, cuda_allowed=True)
    assert selected["config_id"] == "cuda_layer27_qproj"
    slower = _profile("cuda_layer27_qproj", "cuda", 101.0, 0.0)
    assert scheduler_select([cpu, slower], resources=resources, cuda_allowed=True)["config_id"] == "cpu_full_stream"


def test_scheduler_cpu_fallback_on_preflight_rejection():
    result = scheduler_select([], resources=ResourceEnvelope(8_000_000_000, 4_000_000_000), cuda_allowed=False)
    assert result["config_id"] == "cpu_full_stream"
    assert result["basis"] == "cpu_fallback_preflight"


def _completed_payload():
    run = {"run_id": "cpu1", "placement": "cpu", "wall_seconds": 10.0,
        "partition_load_seconds_sum": 1.0, "partition_transfer_seconds_sum": 0.0,
        "partition_release_seconds_sum": 2.0, "total_transfer_bytes": 0,
        "peak_rss_bytes": 100, "minimum_available_ram_bytes": 9_000,
        "max_concurrent_registered_partitions": 1, "final_resident_partition_ids": [],
        "output_path": "cpu.npy", "output_sha256": "a" * 64,
        "partitions": [{"block_id": BLOCK_ID, "device": "cpu", "transfer_seconds": 0.0, "transferred_bytes": 0}]}
    gpu = copy.deepcopy(run)
    gpu.update(run_id="gpu1", placement="cuda", total_transfer_bytes=100)
    gpu["partitions"] = [{"block_id": BLOCK_ID, "device": "cuda", "transfer_seconds": .01, "transferred_bytes": 100}]
    return {"status": "completed", "checkpoint_revision": "r", "scope": "test", "preflight": {},
        "warmup_runs_separate": True, "timed_repeat_count_per_placement": 1, "attempts": [],
        "runs": {"cpu": [run], "cuda": [gpu]}, "profiles": [],
        "controller_selection": {"config_id": "cpu_full_stream"},
        "metrics": {"full_forward_gpu_used": True}, "level5_claim": False, "claim_limit": "test"}


def test_benchmark_schema_requires_full_cost_residency_and_transfer_fields():
    payload = _completed_payload()
    validate_benchmark_payload(payload)
    bad = copy.deepcopy(payload)
    bad["runs"]["cuda"][0]["partitions"][0]["transferred_bytes"] = 0
    with pytest.raises(ValueError, match="explicit transfers"):
        validate_benchmark_payload(bad)
    bad = copy.deepcopy(payload)
    bad["runs"]["cpu"][0]["max_concurrent_registered_partitions"] = 2
    with pytest.raises(ValueError, match="residency invariant"):
        validate_benchmark_payload(bad)
