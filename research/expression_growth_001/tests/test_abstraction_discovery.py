import json
from pathlib import Path

from experiments.expression_growth_001.discovery.mine_abstractions import (
    SOURCE, discover, verify_library,
)


def test_discovery_mines_trace_pattern_and_persists_verified_transfer(tmp_path):
    library = tmp_path / "library.json"
    result = discover(SOURCE, library)
    assert result["distinct_verified_success_rows"] == 3
    assert result["accepted_count"] == 1
    macro = next(item for item in result["proposals"] if item["status"] == "accepted")
    assert macro["primitive_dependencies"] == ["E002", "E501"]
    assert len(macro["discovery_trace_refs"]) == 2
    assert set(macro["validation"]["transfer_families"]) == {
        "resource_priority_scoring", "sensor_weighted_index"}
    assert macro["usage_statistics"]["successful_reuses"] == 2
    assert verify_library(library)["status"] == "passed"


def test_discovery_ignores_error_trace_rows(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"plan_id": "bad", "status": "EXECUTION_ERROR", "step_count": 2,
        "raw_text": {"steps": [{"op": "E501", "arguments": {"u": [1], "v": [2]}, "bind": "x"},
                                 {"op": "E002", "arguments": {"v0": "$x", "v1": 10}, "bind": "y"}]},
        "trace": [{"status": "SUCCESS", "result": 2}, {"status": "ERROR", "result": None}]}) + "\n",
        encoding="utf-8")
    result = discover(bad, tmp_path / "empty_library.json")
    assert result["distinct_verified_success_rows"] == 0
    assert result["candidate_count"] == 0
    assert result["accepted_count"] == 0
