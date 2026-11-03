"""Fast acceptance checks for the preserved full streamed forward artifact."""
import json
from pathlib import Path


def test_saved_full_forward_is_complete_and_bounded():
    root = Path(__file__).resolve().parents[3]
    run = root / "experiments/impression_002b"
    result = json.loads((run / "benchmarks/full_forward_cpu_streamed.json").read_text(encoding="utf-8"))
    ids = json.loads((run / "references/full_forward_input_ids.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["checkpoint_revision"] == "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
    execution = result["execution"]
    assert execution["layers_executed"] == 28
    assert execution["registered_weight_operations"] == 255
    assert execution["unique_registered_partitions"] == 254
    assert execution["max_concurrent_registered_partitions"] == 1
    assert execution["final_resident_partition_ids"] == []
    assert execution["full_model_loaded"] is False
    assert execution["tied_output_projection_reused_embedding"] is True
    assert execution["logits_finite"] is True
    assert execution["output_shape"] == [1, len(ids["token_ids"][0]), 128258]
    assert execution["minimum_available_system_ram_bytes"] > result["preflight"]["minimum_reserve_bytes"]
    assert result["layer27_parity"]["status"] == "passed"
    assert result["layer27_parity"]["max_abs_diff"] == 0.0
    assert result["complete_model_parity"]["status"] == "unresolved"
    assert len(result["partitions"]) == 255
    assert all(row["resident_partition_count_during_execution"] == 1 for row in result["partitions"])
