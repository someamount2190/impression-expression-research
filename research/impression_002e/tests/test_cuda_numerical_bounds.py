import numpy as np
import pytest

from experiments.impression_002d.worker.phase_f_qproj_cuda import (
    bf16_half_ulp_bound, quantized_output_analysis, summarize_mismatches,
)


def test_local_bf16_half_ulp_bound_handles_binades_and_subnormal_floor():
    bound = bf16_half_ulp_bound(np.array([1.0, 2.0, 0.0, 2.0 ** -140]))
    assert bound[0] == pytest.approx(2.0 ** -8)
    assert bound[1] == pytest.approx(2.0 ** -7)
    assert bound[2] == pytest.approx(2.0 ** -134)
    assert bound[3] == pytest.approx(2.0 ** -134)


def test_quantized_output_checks_nearest_even_reference_and_bound_count():
    torch = pytest.importorskip("torch")
    ref = np.array([[1.0 + 2.0 ** -8, 2.0]], dtype=np.float64)
    rounded = torch.from_numpy(ref).to(torch.bfloat16).float().numpy()
    result = quantized_output_analysis(rounded, ref)
    assert result["actual_equals_nearest_even_rounded_reference"] is True
    assert result["count_exceeding_local_half_ulp_bound"] == 0
    wrong = rounded.copy()
    wrong[0, 1] = 2.03125
    failed = quantized_output_analysis(wrong, ref)
    assert failed["actual_equals_nearest_even_rounded_reference"] is False
    assert failed["count_exceeding_local_half_ulp_bound"] == 1
    assert failed["comparison_to_fp64"]["largest_mismatches"][0]["index"] == [0, 1]


def test_mismatch_summary_lists_actual_indices_and_values():
    result = summarize_mismatches(np.array([[1.0, 3.0]]), np.array([[1.0, 2.0]]), top_k=1)
    assert result["mismatch_count_exact"] == 1
    assert result["largest_mismatches"][0] == {
        "index": [0, 1], "actual": 3.0, "reference": 2.0, "abs_error": 1.0}
