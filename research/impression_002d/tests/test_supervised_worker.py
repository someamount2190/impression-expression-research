"""Crash-recovery and metadata-boundary tests for the Phase C worker."""
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments/impression_002d/worker/supervised_worker.py"


def test_abnormal_child_is_recorded_and_parent_runs_benign_job(tmp_path):
    output = tmp_path / "supervisor-test"
    command = [sys.executable, str(WORKER), "--self-test", "--output-dir", str(output)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60, check=True)
    summary = json.loads(completed.stdout)
    assert summary["status"] == "passed"
    assert summary["parent_remained_alive"] is True
    assert summary["fault_child_status"] == "abnormal_exit_before_completion"
    assert summary["benign_child_status"] == "benign_worker_job_passed"

    recovery = json.loads((output / "recovery_test/supervisor_recovery_test.json").read_text(encoding="utf-8"))
    fault = recovery["abnormal_child"]
    assert fault["child_returncode"] == 86
    assert fault["child_receipt"]["last_attempted_operation"] == "fault_injection_exit"
    assert fault["child_receipt"]["imports_at_start"] == {"torch": False, "transformers": False}
    assert recovery["parent_pid_before"] == recovery["parent_pid_after"]
    benign = recovery["benign_child"]
    assert benign["status"] == "benign_worker_job_passed"
    assert benign["child_receipt"]["last_completed_operation"] == "benign_noop"


def test_genuine_worker_receipt_is_metadata_only_when_present():
    path = ROOT / "experiments/impression_002d/diagnostics/worker/genuine_layer0/supervisor_receipt.json"
    if not path.exists():
        return
    supervisor = json.loads(path.read_text(encoding="utf-8"))
    assert supervisor["status"] == "passed"
    child = supervisor["child_receipt"]
    assert child["status"] == "passed"
    assert child["checkpoint_revision"] == "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
    assert child["preflight"]["passed"] is True
    assert len(child["partitions"]) == 9
    assert child["final_resident_partition_ids"] == []
    for partition in child["partitions"]:
        assert partition["final_state"] == "UNLOADED"
        assert partition["content_sha256"]
        assert partition["tensor_metadata"]
        for tensor in partition["tensor_metadata"]:
            assert set(tensor) == {"name", "shape", "dtype", "bytes"}
            assert "data" not in tensor and "values" not in tensor
