"""Artifact assertions for Phase C; no model reads occur in this test."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VER = ROOT / "experiments/impression_002c/training/verification_results"


def test_corrected_raw_norm_evidence_and_single_process_reload_parity():
    resolution = json.loads((VER / "phase_c_raw_norm_probe_resolution.json").read_text(encoding="utf-8"))
    initial = json.loads((VER / "phase_c_raw_norm_probe.json").read_text(encoding="utf-8"))
    assert initial["returncode"] == 1  # Preserve the original comparator bug evidence.
    assert resolution["initial_exit_was_comparator_failure"] is True
    assert resolution["reread_or_retry_performed"] is False
    assert resolution["status"] == "passed_after_aggregate_digest_correction"
    assert resolution["matched"] is True
    assert resolution["recomputed_manager_sha256"] == resolution["preserved_manager_sha256"]

    result = json.loads((VER / "phase_c_legacy_adapter_verification.json").read_text(encoding="utf-8"))
    assert result["returncode"] == 0
    assert result["ntstatus_hex"] is None
    evidence = result["result"]
    assert evidence["status"] == "passed"
    assert evidence["full_model_prefix_recomputed"] is False
    assert evidence["preflight"]["passed"] is True
    assert evidence["preflight"]["tensor_bytes"] == 989362176
    assert evidence["adapter_fingerprint_matches_metadata"] is True
    assert evidence["resident_partition_equivalent_count"] == 11
    assert set(evidence["records"]) == {"trace_edge_0035", "trace_schema_0034"}
    for record in evidence["records"].values():
        assert record["pre_logits"]["exact"] is True
        assert record["post_logits"]["exact"] is True
        assert record["pre_logits"]["max_abs_diff"] == 0.0
        assert record["post_logits"]["max_abs_diff"] == 0.0
    assert "no transformers" in evidence["verification_type"].lower()


def test_training_receipt_remains_separate_and_unverified():
    receipt_path = ROOT / "experiments/impression_002c/training/receipts/run02_legacy_training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verification"]["status"] == "unverified"
    assert receipt["verification"]["fresh_process_reload_verified"] is False
    assert Path(receipt["verification"]["evidence_path"]).is_file()
    actual_result = VER / "phase_c_legacy_adapter_verification.json"
    assert actual_result.is_file() and actual_result.resolve() != receipt_path.resolve()
