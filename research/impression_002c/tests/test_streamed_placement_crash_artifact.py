"""Confirm the failed warmup is retained and no GPU child was started."""
import json
from pathlib import Path


def test_native_crash_stops_all_placement_workloads():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments/impression_002c/benchmarks/streamed_placement/comparison.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "stopped_on_warmup_failure"
    assert len(result["attempts"]) == 1
    attempt = result["attempts"][0]
    assert attempt["placement"] == "cpu"
    assert attempt["kind"] == "warmup"
    assert attempt["returncode"] == 3221225477
    assert attempt["ntstatus_hex"] == "0xC0000005"
    assert "pyarrow" in attempt["stderr"].lower()
    assert "torch\\storage.py" in attempt["stderr"].lower()
    assert result["attempts"][0].get("run") is None
    assert not any(row.get("placement") == "cuda" for row in result["attempts"])
    preflight = json.loads((path.parent / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["cpu"]["status"] == preflight["cuda"]["status"] == "feasible"
