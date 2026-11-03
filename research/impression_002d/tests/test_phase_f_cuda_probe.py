import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experiments/impression_002d/diagnostics/worker/phase_f_qproj_cuda_run01"


def test_phase_f_cuda_probe_artifact_schema_when_run_exists():
    supervisor_path = RUN / "supervisor_receipt.json"
    if not supervisor_path.exists():
        pytest.skip("single Phase F CUDA probe has not run yet")
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    child_path = RUN / "child_receipt.json"
    child = json.loads(child_path.read_text(encoding="utf-8"))
    if supervisor["status"] == "passed":
        assert supervisor["child_ntstatus_hex"] is None
        assert supervisor["child_returncode"] == 0
        assert supervisor["stderr"] == ""
        assert child["status"] == "passed"
        assert child["partition_id"] == "IMP_L27_ATTENTION_QUERY_PROJECTION"
        assert child["record_id"] == "trace_edge_0035"
        assert child["output_comparison"]["within_tolerance"] is True
        assert child["output_comparison"]["max_abs_diff"] <= 0.0078125
        assert child["transfer_measurements"]["transfer_in_seconds"] >= 0.0
        assert child["cuda_kernel_seconds"] >= 0.0
        assert child["transfer_measurements"]["transfer_out_seconds"] >= 0.0
        assert child["cuda_memory"]["peak_allocated_bytes"] > 0
        assert child["final_resident_partition_ids"] == []
        assert child["transformers_imported"] is False
        assert child["pandas_imported"] is False
        assert child["pyarrow_imported"] is False
    elif supervisor["status"] == "native_crash":
        assert supervisor["child_ntstatus_hex"] is not None
        assert child["last_attempted_operation"] is not None
    elif "output_comparison" in child:
        # The preserved single CUDA result is informative but fails the
        # predeclared max-error acceptance threshold; retain that distinction.
        assert supervisor["status"] == "failed_exit"
        assert child["status"] == "python_error"
        assert child["output_comparison"]["within_tolerance"] is False
        assert child["output_comparison"]["max_abs_diff"] == 0.25
        assert child["output_comparison"]["tolerance"] == 0.0078125
        assert child["final_resident_partition_ids"] == []
        assert child["partition_release"]["state"] == "UNLOADED"
        assert child["cuda_memory"]["peak_allocated_bytes"] > 0
        assert child["transfer_measurements"]["transfer_in_seconds"] >= 0.0
        assert child["cuda_kernel_seconds"] >= 0.0
        assert child["transfer_measurements"]["transfer_out_seconds"] >= 0.0
    else:
        assert supervisor["child_ntstatus_hex"] is None
        assert child["status"] in {"preflight_rejected", "python_error"}
        assert "preflight" in child
        assert child.get("final_resident_partition_ids", []) == []

