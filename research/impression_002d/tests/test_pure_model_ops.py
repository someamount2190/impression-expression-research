"""Synthetic CPU checks for the isolated pure-Torch Llama operations."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import torch

from experiments.impression_002d.worker import pure_llama_ops as ops


ROOT = Path(__file__).resolve().parents[3]


def test_rms_norm_matches_direct_definition():
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], dtype=torch.float32)
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(ops.rms_norm(x, weight, 1e-5), expected, rtol=1e-6, atol=1e-6)

    # Match the installed LlamaRMSNorm order: normalize in FP32, cast back to
    # activation dtype, then multiply by the parameter in that dtype.
    x_bf16 = x.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    official_formula = weight_bf16 * (x_bf16.float() * torch.rsqrt(x_bf16.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(x_bf16.dtype)
    source = Path(importlib.metadata.distribution("transformers").locate_file("transformers/models/llama/modeling_llama.py")).read_text(encoding="utf-8")
    assert "return self.weight * hidden_states.to(input_dtype)" in source
    assert torch.equal(ops.rms_norm(x_bf16, weight_bf16, 1e-5), official_formula)


def test_llama3_rope_and_rotation_preserve_zero_position():
    config = {"hidden_size": 8, "num_attention_heads": 2, "head_dim": 4, "rope_theta": 500000.0,
              "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}}
    inv = ops.llama3_inv_freq(config)
    assert inv.shape == (2,)
    assert torch.isfinite(inv).all() and torch.all(inv > 0)
    # Independent pure-Torch transcription of installed Transformers' Llama 3
    # frequency source; source is inspected as text, never imported.
    head_dim = config["head_dim"]
    base_inv = 1.0 / (config["rope_theta"] ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    rope_cfg = config["rope_scaling"]
    wavelengths = 2 * torch.pi / base_inv
    low_wavelength = rope_cfg["original_max_position_embeddings"] / rope_cfg["low_freq_factor"]
    high_wavelength = rope_cfg["original_max_position_embeddings"] / rope_cfg["high_freq_factor"]
    expected_inv = torch.where(wavelengths > low_wavelength, base_inv / rope_cfg["factor"], base_inv)
    smooth_factor = (rope_cfg["original_max_position_embeddings"] / wavelengths - rope_cfg["low_freq_factor"]) / (rope_cfg["high_freq_factor"] - rope_cfg["low_freq_factor"])
    smoothed = (1 - smooth_factor) * expected_inv / rope_cfg["factor"] + smooth_factor * expected_inv
    medium = ~(wavelengths < high_wavelength) * ~(wavelengths > low_wavelength)
    expected_inv = torch.where(medium, smoothed, expected_inv)
    rope_source = Path(importlib.metadata.distribution("transformers").locate_file("transformers/modeling_rope_utils.py")).read_text(encoding="utf-8")
    assert "def _compute_llama3_parameters" in rope_source
    assert "smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor" in rope_source
    torch.testing.assert_close(inv, expected_inv, rtol=0, atol=1e-7)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    cos, sin = ops.rotary_cos_sin(positions, inv, 4, torch.float32)
    q = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    k = q + 1
    q_rot, k_rot = ops.apply_rotary(q, k, cos, sin)
    torch.testing.assert_close(q_rot[:, :, :1], q[:, :, :1])
    torch.testing.assert_close(k_rot[:, :, :1], k[:, :, :1])
    assert q_rot.shape == q.shape and k_rot.shape == k.shape


def test_grouped_query_repeat_and_causal_attention():
    kv = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    repeated = ops.repeat_kv(kv, 2)
    assert repeated[:, :, :, 0].tolist() == [[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]]

    query = torch.ones((1, 2, 3, 2), dtype=torch.bfloat16)
    key = torch.ones_like(query)
    value = torch.tensor([[[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]]], dtype=torch.bfloat16)
    mask = ops.causal_additive_mask(1, 3, torch.bfloat16, "cpu")
    output = ops.causal_attention(query, key, value, mask, scaling=2 ** -0.5)
    changed_future = value.clone()
    changed_future[:, :, 2, :] = 100
    output_changed = ops.causal_attention(query, key, changed_future, mask, scaling=2 ** -0.5)
    assert output.shape == (1, 3, 2, 2)
    torch.testing.assert_close(output[:, :2], output_changed[:, :2], rtol=0, atol=0)


def test_swiglu_and_worker_path_has_no_transformers_import():
    gate = torch.tensor([-1.0, 0.0, 1.0])
    up = torch.tensor([2.0, 3.0, 4.0])
    torch.testing.assert_close(ops.swiglu(gate, up), torch.nn.functional.silu(gate) * up)
    source = (ROOT / "experiments/impression_002d/worker/phase_d_block.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*(?:from|import)\s+transformers\b", source, flags=re.MULTILINE) is None


def test_saved_layer0_worker_schema_and_output_hash_when_present():
    path = ROOT / "experiments/impression_002d/diagnostics/worker/phase_d_layer0/supervisor_receipt.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    child = result["child_receipt"]
    assert result["status"] == "passed"
    assert child["status"] == "passed"
    assert child["job_type"] == "pure_torch_layer0_block"
    assert child["transformers_imported"] is False
    assert child["comparison"]["status"] == "unavailable"
    output = child["output_artifact"]
    data = Path(output["path"]).read_bytes()
    assert len(data) == output["bytes"] == 1 * 8 * 3072 * 2
    assert hashlib.sha256(data).hexdigest() == output["sha256"]
    assert output["shape"] == [1, 8, 3072]
    assert child["final_resident_partition_ids"] == []


def test_full_forward_request_and_artifact_schema_when_present(tmp_path):
    from experiments.impression_002d.worker.supervised_worker import default_full_forward_request
    request = default_full_forward_request(tmp_path)
    assert request["job_type"] == "full_forward"
    assert request["partition_ids"][0] == "IMP_GLOBAL_EMBEDDING_TOKEN"
    assert len(request["partition_ids"]) == 254
    assert request["partition_ids"][-1] == "IMP_GLOBAL_LANGUAGE_MODEL_OUTPUT_NORM"
    assert len(request["input_ids"][0]) == 8
    assert request["absolute_tolerance"] == 0.0625
    source = (ROOT / "experiments/impression_002d/worker/phase_d_forward.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*(?:from|import)\s+transformers\b", source, flags=re.MULTILINE) is None

    path = ROOT / "experiments/impression_002d/diagnostics/worker/phase_d_full_forward/supervisor_receipt.json"
    if not path.exists():
        return
    supervisor = json.loads(path.read_text(encoding="utf-8"))
    child = supervisor["child_receipt"]
    assert supervisor["status"] == "passed"
    assert child["status"] == "passed"
    assert child["layers_executed"] == 28
    assert child["registered_weight_operations"] == 255
    assert child["unique_registered_partitions"] == 254
    assert child["max_concurrent_registered_partitions"] == 1
    assert child["final_resident_partition_ids"] == []
    assert child["transformers_imported"] is False
    assert child["pandas_imported"] is False
    assert child["pyarrow_imported"] is False
    assert child["comparisons"]["reference_source"].startswith("preserved IMPRESSION-002B")
    assert child["comparisons"]["logits"]["passed"] is True
    assert child["comparisons"]["layer27_hidden"]["passed"] is True
    for artifact_key in ("logits_artifact", "layer27_output_artifact"):
        artifact = child[artifact_key]
        data = Path(artifact["path"]).read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]

    pass2_path = ROOT / "experiments/impression_002d/diagnostics/worker/phase_d_full_forward_run02/supervisor_receipt.json"
    comparison_path = ROOT / "experiments/impression_002d/diagnostics/worker/phase_d_comparison/pass1_vs_pass2.json"
    if pass2_path.exists() and comparison_path.exists():
        pass2 = json.loads(pass2_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        assert pass2["status"] == "passed"
        assert pass2["child_returncode"] == 0
        assert pass2["child_ntstatus_hex"] is None
        assert "Windows fatal exception" not in pass2["stderr"]
        assert pass2["child_receipt"]["registered_weight_operations"] == 255
        assert pass2["child_receipt"]["max_concurrent_registered_partitions"] == 1
        assert comparison["status"] == "passed"
        for key in ("logits", "layer27_hidden"):
            assert comparison[key]["byte_identical"] is True
            assert comparison[key]["max_abs_diff"] == 0.0
            assert comparison[key]["mean_abs_diff"] == 0.0
            assert comparison[key]["rms_diff"] == 0.0
