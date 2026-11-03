"""Fast artifact checks; these deliberately do not rerun GPU or training work."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "experiments/impression_002b"
REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"


def test_saved_cpu_cuda_q_projection_controller_evidence():
    result = json.loads((ROOT / "benchmarks/layer27_q_placement_comparison.json").read_text())
    assert result["status"] == "passed"
    assert result["checkpoint_revision"] == REVISION
    assert result["block_id"] == "IMP_L27_ATTENTION_QUERY_PROJECTION"
    assert result["preflight"]["status"] == "feasible"
    assert result["preflight"]["bf16_supported"] is True
    assert result["execution_scope"].startswith("single real [BASE_MODEL] layer-27 q_proj")

    cpu = result["repeats"]["cpu"]
    cuda = result["repeats"]["cuda"]
    assert len(cpu) == len(cuda) == 3
    assert all(row["weight_sha256"] == cpu[0]["weight_sha256"] for row in cpu + cuda)
    assert all(row["transferred_bytes"] > 0 and row["transfer_seconds"] > 0 for row in cuda)
    assert all(row["peak_vram_bytes"] > 0 for row in cuda)
    choice = result["controller"]
    assert choice["decision"] == "cuda_qproj_layer27"
    assert choice["max_quality_regression"] == 1 / 128
    assert choice["selection"]["evidence"]["repeat_count"] == 3
    assert choice["selection"]["evidence"]["median_total_measured_cost_seconds"] < choice["selection"]["evidence"]["baseline_median_total_seconds"]
    assert max(row["scaled_max_abs_output_error_vs_cpu"] for row in cuda) <= choice["max_quality_regression"]
    assert "only to this q_proj microbenchmark" in result["no_efficiency_claim"]


def test_saved_streamed_training_evidence_keeps_reload_unverified():
    run = ROOT / "checkpoints/streamed_lora_layer27_qproj_run02"
    metadata = json.loads((run / "adapter/adapter_metadata.json").read_text())
    failure = json.loads((run / "fresh_process_reload_failure.json").read_text())
    assert metadata["checkpoint_revision"] == REVISION
    assert metadata["target_module"] == "model.layers.27.self_attn.q_proj"
    assert metadata["rank"] == 4
    assert set(metadata["samples"]) == {"trace_edge_0035", "trace_schema_0034"}
    assert set(metadata["caches"]) == {"trace_edge_0035", "trace_schema_0034"}
    for sample in metadata["caches"].values():
        assert len(sample["cache_identity_sha256"]) == 64
        assert Path(sample["activation_path"]).is_file()
        assert Path(sample["manifest_path"]).is_file()

    adapter = run / "adapter/adapter.safetensors"
    assert adapter.is_file()
    expected_fingerprint = failure["independent_adapter_file_fingerprint"]
    assert expected_fingerprint["matches"] is True
    assert expected_fingerprint["loaded_sha256"] == metadata["post_adapter_fingerprint_sha256"]
    assert expected_fingerprint["metadata_sha256"] == metadata["post_adapter_fingerprint_sha256"]
    for record in ("trace_edge_0035", "trace_schema_0034"):
        for suffix in ("pre", "post"):
            logits = run / f"{record}_{suffix}_logits.npy"
            assert logits.is_file()
            values = np.load(logits, mmap_mode="r")
            assert values.shape[0] == 1 and values.shape[1] == 16
            assert np.isfinite(values).all()

    assert failure["status"] == "reload_not_verified"
    assert failure["verification_repair"]["process_exit_code"] == 1
    assert failure["limitations"]
    assert not (run / "training_report.json").exists()
    assert "gradient and optimizer timing/state metrics were not persisted" in " ".join(failure["limitations"])
    losses = failure["saved_logits_recomputed_causal_loss"]
    assert losses["trace_edge_0035"]["post"] < losses["trace_edge_0035"]["pre"]
    assert losses["trace_schema_0034"]["post"] > losses["trace_schema_0034"]["pre"]
