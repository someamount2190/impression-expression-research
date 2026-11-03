from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize, quantize
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "models/[BASE_MODEL].gguf"
MERGED = ROOT / "experiments/impression_006/adapter_inference/gguf/[BASE_MODEL]3-impression004-adapter-merged.gguf"
PEFT = ROOT / "experiments/impression_006/adapter_inference/peft_adapter/adapter_model.safetensors"
OUT = ROOT / "experiments/impression_006/adapter_inference/tensor_mapping_verification.json"


def field(reader, key):
    return reader.fields[key].contents()


def permute_qk(weight: np.ndarray, heads: int) -> np.ndarray:
    return (weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:])
            .swapaxes(1, 2).reshape(weight.shape))


def main():
    base_before = hashlib.sha256(BASE.read_bytes()).hexdigest()
    source = GGUFReader(str(BASE))
    merged = GGUFReader(str(MERGED))
    src = {tensor.name: tensor for tensor in source.tensors}
    dst = {tensor.name: tensor for tensor in merged.tensors}
    key = "blk.27.attn_q.weight"
    companion = "blk.27.attn_k.weight"
    if key not in src or key not in dst or companion not in src or companion not in dst:
        raise RuntimeError("Expected target or untouched companion tensor missing")
    with safe_open(str(PEFT), framework="np") as tensors:
        a = tensors.get_tensor("base_model.model.model.layers.27.self_attn.q_proj.lora_A.weight").astype(np.float32)
        b = tensors.get_tensor("base_model.model.model.layers.27.self_attn.q_proj.lora_B.weight").astype(np.float32)
    q_heads = int(field(source, "llama.attention.head_count"))
    delta = permute_qk(b @ a * 2.0, q_heads)
    base_weight = dequantize(src[key].data, src[key].tensor_type).astype(np.float32)
    expected = quantize(base_weight + delta, GGMLQuantizationType.Q8_0)
    actual = dst[key].data
    if not np.array_equal(actual, expected):
        raise RuntimeError("Merged target tensor is not the exact expected Q8 quantization")
    actual_dequant = dequantize(actual, dst[key].tensor_type).astype(np.float32)
    error = actual_dequant - (base_weight + delta)
    same_companion_bytes = np.array_equal(src[companion].data, dst[companion].data)
    base_after = hashlib.sha256(BASE.read_bytes()).hexdigest()
    receipt = {
        "status": "passed" if same_companion_bytes and base_before == base_after else "failed",
        "base_gguf_sha256_before": base_before, "base_gguf_sha256_after": base_after,
        "merged_gguf_sha256": hashlib.sha256(MERGED.read_bytes()).hexdigest(),
        "adapter_source_sha256": hashlib.sha256((ROOT / "experiments/impression_004/partition_training_receipts/cuda_head_sparse02/adapter/adapter.safetensors").read_bytes()).hexdigest(),
        "target_mapping": key, "target_mapping_verified": True,
        "lora_shapes": {"A": list(a.shape), "B": list(b.shape)}, "alpha_over_rank": 2.0,
        "qk_rope_permutation_applied": True,
        "target_output_type": str(dst[key].tensor_type),
        "exact_expected_quantized_tensor_match": True,
        "target_delta_vs_q4_dequant_plus_delta": {
            "max_abs_error_from_q8_quantization": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error))),
        },
        "untouched_companion_tensor": companion, "companion_raw_bytes_preserved": same_companion_bytes,
        "full_logits_parity_between_merged_gguf_and_raw_bf16_hf": "not measured; Q4 base plus Q8 target changes numerical path",
    }
    OUT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "passed":
        raise RuntimeError("GGUF tensor mapping or base preservation failed")


if __name__ == "__main__":
    main()
