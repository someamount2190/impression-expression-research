import numpy as np
import pytest

from experiments.impression_002e.numerical.compare_qproj import (
    bf16_rounding_bound_metrics, compare_arrays, pairwise_comparisons, prepare_operands,
)


def test_compare_arrays_reports_errors_and_largest_indices():
    ref = np.array([[0.0, 2.0], [4.0, -8.0]], dtype=np.float32)
    actual = np.array([[0.0, 2.25], [3.0, -8.0]], dtype=np.float32)
    result = compare_arrays(actual, ref, top_k=2)
    assert result["max_abs_error"] == 1.0
    assert result["mean_abs_error"] == pytest.approx(0.3125)
    assert result["rms_error"] == pytest.approx(np.sqrt(1.0625 / 4))
    assert result["mismatch_count"] == 2
    assert result["all_finite"] is True
    assert result["largest_mismatches"][0]["index"] == [1, 0]
    assert result["largest_mismatches"][0]["actual"] == 3.0


def test_compare_arrays_handles_nonfinite_and_shape_mismatch():
    result = compare_arrays(np.array([1.0, np.inf]), np.array([1.0, 2.0]))
    assert result["all_finite"] is False
    assert result["actual_nonfinite_count"] == 1
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays(np.zeros((1,)), np.zeros((2,)))


def test_fp32_accumulation_uses_identical_bf16_rounded_operands():
    torch = pytest.importorskip("torch")
    activation = torch.tensor([[0.12345, -0.9876]], dtype=torch.float32)
    weight = torch.tensor([[0.4567, 1.2345]], dtype=torch.bfloat16)
    bf16_input, bf16_weight, bf16_hashes = prepare_operands(
        torch, activation, weight, device="cpu", output_dtype=torch.bfloat16,
        bf16_operands_fp32_accumulation=False)
    fp32_input, fp32_weight, fp32_hashes = prepare_operands(
        torch, activation, weight, device="cpu", output_dtype=torch.float32,
        bf16_operands_fp32_accumulation=True)
    assert fp32_input.dtype == torch.float32
    assert fp32_weight.dtype == torch.float32
    assert torch.equal(fp32_input, bf16_input.float())
    assert torch.equal(fp32_weight, bf16_weight.float())
    assert bf16_hashes == fp32_hashes


def test_pairwise_comparison_includes_every_available_pair():
    arrays = {"cpu_bf16": np.array([1.0]), "cpu_fp32": np.array([1.25]),
              "cuda_fp32": np.array([1.5])}
    result = pairwise_comparisons(arrays)
    assert len(result) == 3
    assert result["cpu_bf16_vs_cpu_fp32"]["max_abs_error"] == pytest.approx(0.25)


def test_bf16_rounding_bound_is_value_scaled_and_includes_measured_accumulation():
    torch = pytest.importorskip("torch")
    reference = np.array([1.0004, -44.1249, 0.0], dtype=np.float32)
    actual = torch.from_numpy(reference).to(torch.bfloat16).float().numpy()
    passed = bf16_rounding_bound_metrics(actual, reference)
    assert passed["passed"] is True
    assert passed["outside_local_bound_count"] == 0
    failed = bf16_rounding_bound_metrics(actual + np.array([0.1, 0.5, 0.0]), reference)
    assert failed["passed"] is False
    assert failed["outside_local_bound_count"] > 0
    accumulation = bf16_rounding_bound_metrics(actual, reference,
        accumulation_error=np.array([2e-4, 1e-4, 0.0]))
    assert accumulation["max_allowed_error"] >= passed["max_allowed_error"]


def test_one_ulp_bound_handles_adjacent_bf16_rounding_at_a_halfway_value():
    torch = pytest.importorskip("torch")
    reference = np.array([-44.125137], dtype=np.float64)
    adjacent_bf16 = np.array([-44.0], dtype=np.float64)
    strict = bf16_rounding_bound_metrics(adjacent_bf16, reference, ulp_factor=0.5)
    backend = bf16_rounding_bound_metrics(adjacent_bf16, reference, ulp_factor=1.0)
    assert strict["passed"] is False
    assert backend["passed"] is True
    assert backend["max_abs_error_to_fp32_reference"] == pytest.approx(0.125137)

