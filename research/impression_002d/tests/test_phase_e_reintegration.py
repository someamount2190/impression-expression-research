import json
from pathlib import Path

import pytest

from experiments.impression_002d.worker.phase_e_reintegration import validate_cache_identity


ROOT = Path(__file__).resolve().parents[3]
REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"


def test_saved_cache_identity_accepts_exact_record_and_rejects_token_change():
    manifest_path = ROOT / "experiments/impression_002b/checkpoints/activation_cache/trace_edge_0035-855a3c4946b7a2cd.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["cache_identity"]
    digest = manifest["cache_identity_sha256"]
    assert validate_cache_identity(manifest, record_id="trace_edge_0035", revision=REVISION,
        input_ids=identity["input_ids"], position_ids=identity["position_ids"],
        attention_mask=identity["attention_mask"], metadata_identity=digest) == digest
    changed_ids = list(identity["input_ids"])
    changed_ids[0] += 1
    with pytest.raises(ValueError, match="input_ids mismatch"):
        validate_cache_identity(manifest, record_id="trace_edge_0035", revision=REVISION,
            input_ids=changed_ids, position_ids=identity["position_ids"],
            attention_mask=identity["attention_mask"], metadata_identity=digest)


def test_phase_e_artifact_schema_when_run_exists():
    run_dir = ROOT / "experiments/impression_002d/diagnostics/worker/phase_e_adapter_reintegration_run01"
    receipt_path = run_dir / "child_receipt.json"
    if not receipt_path.exists():
        pytest.skip("bounded Phase E worker has not run yet")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    supervisor = json.loads((run_dir / "supervisor_receipt.json").read_text(encoding="utf-8"))
    assert receipt["job_type"] == "saved_adapter_reintegration"
    assert supervisor["status"] == "native_crash"
    assert supervisor["child_ntstatus_hex"] == "0xC0000005"
    assert "Windows fatal exception: access violation" in supervisor["stderr"]
    assert receipt["status"] == "starting"  # child died before final status could be written
    assert receipt["transformers_imported"] is False
    assert receipt["adapter_load"]["matches_metadata"] is True
    assert [row["record_id"] for row in receipt["records"]] == ["trace_edge_0035"]
    row = receipt["records"][0]
    assert row["cache_prefix_comparison"]["exact_after_bf16_conversion"]
    assert row["base_logits_comparison_to_saved_pre"]["exact"]
    assert row["adapter_logits_comparison_to_saved_post"]["exact"]
    assert row["base_logits_comparison_to_saved_pre"]["max_abs_diff"] == 0.0
    assert row["adapter_logits_comparison_to_saved_post"]["max_abs_diff"] == 0.0
    assert receipt["last_attempted_operation"].endswith("IMP_L04_ATTENTION_INPUT_NORM:attention_rms_norm")
    assert receipt["completed_operation_count"] == 303
    assert Path(receipt["telemetry_path"]).is_file()


def test_phase_e_raw_backend_full_reintegration_artifact():
    run_dir = ROOT / "experiments/impression_002d/diagnostics/worker/phase_e_adapter_reintegration_raw_run01"
    supervisor_path = run_dir / "supervisor_receipt.json"
    if not supervisor_path.exists():
        pytest.skip("raw backend Phase E worker has not run yet")
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    receipt = json.loads((run_dir / "child_receipt.json").read_text(encoding="utf-8"))
    assert supervisor["status"] == "passed"
    assert supervisor["child_returncode"] == 0
    assert supervisor["child_ntstatus_hex"] is None
    assert supervisor["stderr"] == ""
    assert receipt["status"] == "passed"
    assert receipt["completed_operation_count"] == 532
    assert receipt["final_resident_partition_ids"] == []
    assert receipt["adapter_load"]["fingerprint_matches_metadata"] is True
    assert receipt["transformers_imported"] is False
    assert receipt["pandas_imported"] is False
    assert receipt["pyarrow_imported"] is False
    assert {row["record_id"] for row in receipt["records"]} == {"trace_edge_0035", "trace_schema_0034"}
    for row in receipt["records"]:
        assert row["cache_prefix_comparison"]["exact_after_bf16_conversion"] is True
        assert row["base_logits_comparison_to_saved_pre"]["exact"] is True
        assert row["adapter_logits_comparison_to_saved_post"]["exact"] is True
        assert row["base_logits_comparison_to_saved_pre"]["max_abs_diff"] == 0.0
        assert row["adapter_logits_comparison_to_saved_post"]["max_abs_diff"] == 0.0
    assert Path(receipt["telemetry_path"]).is_file()

