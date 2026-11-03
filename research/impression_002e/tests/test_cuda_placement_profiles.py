import pytest

from experiments.impression_002.controller.efficiency_controller import AdaptiveEfficiencyController
from experiments.impression_002e.cuda.placement_profiles import profiles_from_probe, select_placement


def fake_probe(*, cpu_total=0.20, cuda_total=0.12, cuda_correct=True, release_ok=True):
    return {
        "status": "passed" if cuda_correct and release_ok else "quality_check_failed",
        "checkpoint_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364",
        "partition_id": "IMP_L27_ATTENTION_QUERY_PROJECTION",
        "record_id": "trace_edge_0035", "cache_identity_sha256": "a" * 64,
        "input_bf16_rounded_sha256": "b" * 64, "weight_raw_bf16_sha256": "c" * 64,
        "operation": "BF16-rounded operands -> FP32 F.linear -> BF16 cast",
        "release_ok": release_ok,
        "transfer_integrity": {"input_bf16_roundtrip_matches_source": True,
                                "weight_bf16_roundtrip_matches_source": True},
        "resources": {"ram_available_preflight_bytes": 12_000_000_000,
            "system_ram_available_after_release_bytes": 11_900_000_000,
            "ram_reserve_bytes": 4_000_000_000, "vram_free_preflight_bytes": 3_000_000_000},
        "partition_load": {"rss_after_bytes": 1_000_000_000},
        "preflight": {"cpu": {"partition_bytes_bf16": 20_000_000}},
        "input_shape": [1, 16, 3072],
        "timings": {"cpu_profile_total_seconds": cpu_total,
            "cuda_profile_total_seconds": cuda_total, "release_seconds": 0.01},
        "hardware": {"cpu": "test-cpu", "gpu": "NVIDIA GeForce RTX 2050"},
        "runtime_versions": {"torch": "2.5.1"},
        "cpu": {"status": "completed", "kernel_seconds": cpu_total - 0.02,
            "bf16_output_vs_fp64_quantization": {"actual_equals_nearest_even_rounded_reference": True,
                "count_exceeding_combined_rounding_and_accumulation_bound": 0,
                "max_abs_error_vs_fp64_reference": 0.001}},
        "cuda": {"status": "completed", "kernel_seconds": cuda_total - 0.04,
            "transfer_in_seconds": 0.02, "transfer_out_seconds": 0.01,
            "peak_allocated_vram_bytes": 200_000_000,
            "bf16_output_vs_fp64_quantization": {"actual_equals_nearest_even_rounded_reference": cuda_correct,
                "count_exceeding_combined_rounding_and_accumulation_bound": 0 if cuda_correct else 2,
                "max_abs_error_vs_fp64_reference": 0.001 if cuda_correct else 0.1}},
    }


def test_profile_conversion_produces_matched_cpu_and_cuda_profiles():
    profiles = profiles_from_probe(fake_probe(), checkpoint_sha256="d" * 64)
    assert len(profiles) == 2
    cpu, cuda = profiles
    assert cpu.placement == "cpu"
    assert cuda.placement == "cuda:0"
    assert cpu.objective_fingerprint == cuda.objective_fingerprint
    assert cpu.heldout_ids == cuda.heldout_ids == ("trace_edge_0035",)
    assert cpu.total_measured_cost_seconds == pytest.approx(0.20)
    assert cuda.total_measured_cost_seconds == pytest.approx(0.12)
    assert cuda.status == "success"


def test_controller_selects_cuda_only_when_correct_and_total_measured_cost_is_lower():
    faster = select_placement(fake_probe(cuda_total=0.10), checkpoint_sha256="d" * 64)
    assert faster["selected_placement"] == "cuda_qproj_fp32_bf16_operands"
    slower = select_placement(fake_probe(cuda_total=0.25), checkpoint_sha256="d" * 64)
    assert slower["selected_placement"] == "cpu_qproj_fp32_bf16_operands"
    incorrect = select_placement(fake_probe(cuda_total=0.05, cuda_correct=False), checkpoint_sha256="d" * 64)
    assert incorrect["selected_placement"] == "cpu_qproj_fp32_bf16_operands"

