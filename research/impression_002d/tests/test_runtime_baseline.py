"""Baseline schema test; the test does not import model dependencies."""
import json
from pathlib import Path


def test_native_runtime_baseline_required_fields_and_scope():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments/impression_002d/runtime_baseline.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    assert baseline["runtime"]["windows_os_identity"]["caption"] == "Microsoft Windows 11 Home Single Language"
    assert baseline["runtime"]["windows_version_build_compatibility"] == "10.0.26200"
    assert baseline["experiment"] == "IMPRESSION-002D"
    assert baseline["checkpoint"]["revision"] == "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
    assert baseline["checkpoint"]["config_sha256"]
    assert baseline["checkpoint"]["index_sha256"]
    assert baseline["checkpoint"]["index_tensor_count"] > 0
    assert baseline["adapter"]["file_sha256"]
    assert baseline["resources"]["system_ram"]["total_physical_bytes"] > 0
    assert baseline["resources"]["gpu"]["driver_version"]
    assert baseline["installed_distribution_versions"]["packages"]["torch"]
    assert baseline["runtime_import_provenance"]["direct_critical_path"]
    assert baseline["runtime_import_provenance"]["transformers_transitive_or_lazy_imports_observed_in_crash_stack"]
    assert baseline["safety_scope"]["model_tensor_materialized"] is False
    assert baseline["safety_scope"]["model_import_performed"] is False
    assert baseline["evidence_summary"]["002c_phase_d_e_crash_ntstatus"] == "0xC0000005"
    assert baseline["evidence_summary"]["002c_phase_d_e_gpu_child_started"] is False
