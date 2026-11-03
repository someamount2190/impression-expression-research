"""Artifact assertions only: never load model weights in the test process."""
import json
from pathlib import Path


def test_phase_a_numpy_and_torch_failure_diagnostics_are_preserved():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments/impression_002c/diagnostics/phase_a_safetensors_numpy_diagnostic.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["checkpoint_revision"] == "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
    assert artifact["runtime_dependency_versions"]["packages"]["safetensors"] == "0.6.2"

    adapter = artifact["adapter_numpy_result"]
    assert adapter["status"] == "passed"
    assert adapter["framework"] == "numpy"
    assert adapter["metadata_fingerprint_matches"] is True
    assert adapter["fingerprint_sha256"] == artifact["adapter_metadata_summary"]["post_adapter_fingerprint_sha256"]
    assert adapter["tensors"]["lora_A"] == {"shape": [4, 3072], "dtype": "float32", "nbytes": 49152}
    assert adapter["tensors"]["lora_B"] == {"shape": [3072, 4], "dtype": "float32", "nbytes": 49152}
    assert artifact["adapter_numpy_subprocess"]["returncode"] == 0

    norm = artifact["single_norm_numpy_subprocess"]
    assert artifact["single_norm_identity"]["shard_from_index"] is True
    assert artifact["single_norm_identity"]["logical_bytes"] == 6144
    assert norm["returncode"] == 1
    assert "TypeError: data type 'bfloat16' not understood" in norm["stderr"]
    assert norm["ntstatus_hex"] is None

    prior = artifact["prior_diagnostic"]
    assert prior["os_exception_code_hex"].lower() == "0xc0000005"
    assert "torch" in prior["stderr"] and "storage.py" in prior["stderr"]
    assert "safetensors.numpy" not in prior["stderr"]
