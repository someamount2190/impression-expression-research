"""Static and artifact checks for the isolated native-runtime matrix."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "experiments/impression_002d/phase_b_matrix.py"
SPEC = importlib.util.spec_from_file_location("impression_002d_phase_b_matrix", MODULE_PATH)
MATRIX_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX_MODULE)


def test_matrix_has_all_isolated_cases_and_single_attempt_policy():
    expected = {"A", "B", "C", "D", "E1", "E2", "F1", "F2", "G", "H", "I", "J"}
    assert set(MATRIX_MODULE.MATRIX) == expected
    assert MATRIX_MODULE.native_status(3221225477) == "0xC0000005"
    assert MATRIX_MODULE.native_status(1) is None
    assert MATRIX_MODULE.MATRIX["H"]["risk_rank"] < MATRIX_MODULE.MATRIX["I"]["risk_rank"]
    assert MATRIX_MODULE.MATRIX["I"]["risk_rank"] < MATRIX_MODULE.MATRIX["J"]["risk_rank"]


def test_recorded_matrix_artifacts_have_receipts_and_no_retries():
    state_path = ROOT / "experiments/impression_002d/diagnostics/phase_b_matrix/matrix_state.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for test_id, summary in state["results"].items():
        assert test_id in MATRIX_MODULE.MATRIX
        assert summary.get("attempt_count", 1) <= (2 if test_id in {"A", "I"} else 1)
        if summary["status"] == "skipped_after_lower_risk_native_crash":
            assert summary["attempt_count"] == 0
            continue
        result_path = Path(summary.get("attempt_artifact", state_path.parent / test_id / "result.json"))
        record = json.loads(result_path.read_text(encoding="utf-8"))
        run_dir = result_path.parent
        assert record["attempt_count"] == 1
        assert record["returncode"] is not None
        if record["status"] in {"native_crash", "timed_out"}:
            receipt = json.loads((run_dir / "crash_receipt.json").read_text(encoding="utf-8"))
            assert receipt["test_id"] == test_id
            assert "pre_op_flushed" in receipt
    if "J" in state["results"]:
        j_summary = state["results"]["J"]
        assert j_summary["status"] == "completed_with_native_fault_warning"
        assert j_summary["operation_result"] == "passed_exit_0"
        receipt = json.loads((state_path.parent / "J/native_fault_diagnostic.json").read_text(encoding="utf-8"))
        assert receipt["classification"] == "completed_with_native_fault_warning"
        assert receipt["child_exit_code"] == 0
        assert receipt["no_retry"] is True
