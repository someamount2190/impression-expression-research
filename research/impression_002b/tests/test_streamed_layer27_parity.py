"""Focused regression over the reproducible official-vs-streamed block proof."""
import json
from pathlib import Path


def test_saved_layer27_streamed_parity_evidence():
    root = Path(__file__).resolve().parents[3]
    result_path = root / "experiments/impression_002b/benchmarks/layer27_streamed_parity.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["reference"]["class"].endswith("LlamaDecoderLayer")
    assert result["reference"]["transformers_version"] == "4.56.2"
    assert result["checkpoint_revision"] == "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
    assert result["parity"]["passed"] is True
    assert result["parity"]["max_abs_diff"] <= result["parity"]["max_abs_tolerance"]
    streamed = result["streamed"]
    assert streamed["max_concurrent_registered_partitions"] == 1
    assert streamed["manager_state"]["resident_partition_ids"] == []
    assert len(streamed["timings"]) == 9
    assert all(row["resident_partition_count_during_execution"] == 1 for row in streamed["timings"])
    assert {row["device"] for row in streamed["timings"]} == {"cpu"}
