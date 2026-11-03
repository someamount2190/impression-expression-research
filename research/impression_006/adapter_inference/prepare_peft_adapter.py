"""Repackage the verified IMPRESSION safetensors for the repo GGUF merger."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/impression_004/partition_training_receipts/cuda_head_sparse02/adapter/adapter.safetensors"
SOURCE_META = SOURCE.with_name("adapter_metadata.json")
DEST = ROOT / "experiments/impression_006/adapter_inference/peft_adapter"
BASE_GGUF = ROOT / "models/[BASE_MODEL].gguf"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with safe_open(str(SOURCE), framework="np") as file:
        a, b = file.get_tensor("lora_A"), file.get_tensor("lora_B")
    if a.shape != (4, 3072) or b.shape != (3072, 4):
        raise RuntimeError(f"Unexpected IMPRESSION adapter shapes: A={a.shape}, B={b.shape}")
    save_file({
        "base_model.model.model.layers.27.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.27.self_attn.q_proj.lora_B.weight": b,
    }, str(DEST / "adapter_model.safetensors"))
    config = {
        "base_model_name_or_path": "[BASE_MODEL_REPOSITORY]",
        "revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364",
        "r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
        "target_modules": ["model.layers.27.self_attn.q_proj"],
        "bias": "none", "fan_in_fan_out": False,
        "use_dora": False, "use_rslora": False, "task_type": "CAUSAL_LM",
    }
    (DEST / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        "source_adapter": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_adapter_sha256": sha(SOURCE), "source_metadata_sha256": sha(SOURCE_META),
        "source_base_gguf_sha256": sha(BASE_GGUF),
        "source_keys": ["lora_A", "lora_B"],
        "converted_keys": [
            "base_model.model.model.layers.27.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.27.self_attn.q_proj.lora_B.weight",
        ],
        "rank": 4, "alpha": 8, "scale": 2.0,
        "target_module": "model.layers.27.self_attn.q_proj",
        "base_checkpoint_revision": "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364",
        "conversion_method": "scripts/merge_lora_to_gguf.py; Q4 target dequantized, scaled LoRA added with Q/K GGUF permutation, target requantized to Q8_0; unrelated tensors copied raw",
        "adapter_is_impression_trained": True,
    }
    (DEST.parent / "adapter_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "source_sha256": manifest["source_adapter_sha256"],
        "adapter_model_sha256": sha(DEST / "adapter_model.safetensors"), "target": manifest["target_module"]}, indent=2))

if __name__ == "__main__": main()
