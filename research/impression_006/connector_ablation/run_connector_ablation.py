"""Matched, family-disjoint connector vs same-size LoRA experiment.

This is an isolated research-path experiment. It never registers graph mutation
with the Capability Broker and never changes the production model graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.impression_005.structural_runtime import IdentityResidualConnector, install_cached_prefix, restore_original_layers

OUT = ROOT / "experiments/impression_006/connector_ablation"
MANIFEST = ROOT / "experiments/impression_006/frozen_sets/connector_family_split_manifest.json"
BASELINE = ROOT / "experiments/impression_002d/runtime_baseline.json"
METHODS = ("connector", "matched_qproj_lora")
SEEDS = (17, 41)
EPOCHS = 2
LR = 1e-3
MAX_TOKENS = 512
RAM_RESERVE_FRACTION = 0.20


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_examples(tokenizer, manifest: dict, split: str) -> list[dict]:
    path = ROOT / manifest["source_files"][split]
    expected_sha = manifest["source_file_sha256"][str(path.relative_to(ROOT)).replace("\\", "/")]
    if file_sha(path) != expected_sha:
        raise RuntimeError(f"Frozen {split} source changed: {path}")
    by_id = {json.loads(line)["example_id"]: json.loads(line)
             for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    result = []
    for row in manifest[split]:
        example = by_id[row["example_id"]]
        if example["family"] != row["family"]:
            raise RuntimeError("Frozen family assignment mismatch")
        prefix = tokenizer.apply_chat_template([
            {"role": "system", "content": "Produce only a valid structured action plan matching the requested task."},
            {"role": "user", "content": example["prompt"]},
        ], tokenize=True, add_generation_prompt=True)
        prefix_ids = prefix.tolist() if hasattr(prefix, "tolist") else list(prefix)
        answer = json.dumps(example["target_plan"], ensure_ascii=False, separators=(",", ":"))
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        answer_ids = list(answer_ids) + [int(im_end)]
        ids = prefix_ids + answer_ids
        if len(ids) > MAX_TOKENS:
            raise RuntimeError(f"{row['example_id']} exceeds frozen max token length ({len(ids)})")
        positions = list(range(len(prefix_ids) - 1, len(ids) - 1))
        if len(positions) != len(answer_ids):
            raise RuntimeError("Causal target alignment error")
        result.append({"example_id": row["example_id"], "family": row["family"],
                       "input_ids": torch.tensor([ids], dtype=torch.long),
                       "positions": torch.tensor(positions, dtype=torch.long),
                       "targets": torch.tensor(answer_ids, dtype=torch.long)})
    return result


def load_model():
    cfg = json.loads(BASELINE.read_text(encoding="utf-8"))["checkpoint"]
    snapshot = Path(cfg["snapshot_path"]).resolve(strict=True)
    index = snapshot / "model.safetensors.index.json"
    if snapshot.name != cfg["revision"] or file_sha(index) != cfg["index_sha256"]:
        raise RuntimeError("Pinned [BASE_MODEL] revision/hash mismatch")
    memory = psutil.virtual_memory()
    reserve = int(memory.total * RAM_RESERVE_FRACTION)
    model_bytes = int(cfg["index_metadata"]["total_size"])
    if memory.available < model_bytes + reserve + 768 * 1024**2:
        raise RuntimeError("RAM preflight rejected [BASE_MODEL] load with 20% reserve")
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    if psutil.virtual_memory().available < reserve:
        raise RuntimeError("[BASE_MODEL] load breached 20% RAM reserve")
    return model, snapshot, {"total_ram_bytes": int(memory.total),
        "available_ram_after_load_bytes": int(psutil.virtual_memory().available), "reserve_bytes": reserve}


def base_forward(model, record: dict) -> torch.Tensor:
    with torch.inference_mode():
        output = model.model(input_ids=record["input_ids"], use_cache=False, return_dict=True)
        hidden = output.last_hidden_state.index_select(1, record["positions"])
        return model.lm_head(hidden).float().detach()


def capture_activations(model, records: list[dict]) -> dict[str, list[torch.Tensor]]:
    output = {}
    for record in records:
        activations: dict[int, torch.Tensor] = {}
        hooks = []
        for i in range(27):
            def save(_m, _inp, value, idx=i):
                hidden = value[0] if isinstance(value, (tuple, list)) else value
                activations[idx] = hidden.detach().clone()
            hooks.append(model.model.layers[i].register_forward_hook(save))
        with torch.inference_mode():
            model.model(input_ids=record["input_ids"], use_cache=False, return_dict=True)
        for hook in hooks:
            hook.remove()
        if len(activations) != 27:
            raise RuntimeError("Failed to capture all required [BASE_MODEL] activation boundaries")
        output[record["example_id"]] = [activations[i] for i in range(27)]
    return output


def objective_logits(model, record, activations, adapter=None, method="connector"):
    if method == "connector":
        originals = install_cached_prefix(model, activations, adapter)
        try:
            output = model.model(input_ids=record["input_ids"], use_cache=False, return_dict=True)
            hidden = output.last_hidden_state.index_select(1, record["positions"])
            return model.lm_head(hidden).float()
        finally:
            restore_original_layers(model, originals)
    originals = install_cached_prefix(model, activations)
    old_q = model.model.layers[27].self_attn.q_proj
    model.model.layers[27].self_attn.q_proj = adapter
    try:
        output = model.model(input_ids=record["input_ids"], use_cache=False, return_dict=True)
        hidden = output.last_hidden_state.index_select(1, record["positions"])
        return model.lm_head(hidden).float()
    finally:
        model.model.layers[27].self_attn.q_proj = old_q
        restore_original_layers(model, originals)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, hidden: int, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, hidden, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(hidden, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        residual = F.linear(F.linear(x.float(), self.a), self.b).to(x.dtype) * self.scale
        return self.base(x) + residual


def mean_loss(logits, targets) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def state_to_save(method, module) -> dict:
    if method == "connector":
        return {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}
    return {"lora_A": module.a.detach().cpu().contiguous(), "lora_B": module.b.detach().cpu().contiguous()}


def state_hash(module) -> str:
    return sha_bytes(b"".join(t.detach().cpu().contiguous().numpy().tobytes() for _, t in sorted(module.state_dict().items())))


def create_module(method, seed, hidden):
    torch.manual_seed(seed)
    if method == "connector":
        return IdentityResidualConnector(hidden_size=hidden, rank=8)
    return LoRALinear(nn.Identity(), hidden, rank=8, alpha=8.0)


def train_one(model, train_records, train_acts, val_records, val_acts, method, seed, out_dir):
    torch.manual_seed(seed)
    module = create_module(method, seed, int(model.config.hidden_size))
    # For Q projection, preserve the actual frozen q_proj as the wrapped base.
    if method == "matched_qproj_lora":
        module.base = model.model.layers[27].self_attn.q_proj
        for parameter in module.base.parameters():
            parameter.requires_grad_(False)
    module.requires_grad_(True)
    params = list(module.parameters()) if method == "connector" else [module.a, module.b]
    before = {}
    for record in val_records:
        logits = base_forward(model, record)
        before[record["example_id"]] = float(mean_loss(logits, record["targets"]).item())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    start = time.perf_counter()
    history = []
    total_ram = int(psutil.virtual_memory().total)
    reserve = int(total_ram * RAM_RESERVE_FRACTION)
    for epoch in range(1, EPOCHS + 1):
        for record in train_records:
            if psutil.virtual_memory().available < reserve + 512 * 1024**2:
                raise RuntimeError("RAM reserve guard stopped connector ablation before an optimizer update")
            optimizer.zero_grad(set_to_none=True)
            logits = objective_logits(model, record, train_acts[record["example_id"]], module, method)
            loss = mean_loss(logits, record["targets"])
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
            optimizer.step()
            history.append({"epoch": epoch, "example_id": record["example_id"],
                            "family": record["family"], "loss_before_update": float(loss.detach()),
                            "gradient_norm": grad_norm})
    train_seconds = time.perf_counter() - start
    after = {}
    val_logits_hashes = {}
    with torch.no_grad():
        for record in val_records:
            logits = objective_logits(model, record, val_acts[record["example_id"]], module, method)
            after[record["example_id"]] = float(mean_loss(logits, record["targets"]).item())
            val_logits_hashes[record["example_id"]] = {"shape": list(logits.shape),
                "sha256": sha_bytes(logits.detach().cpu().contiguous().numpy().tobytes())}
    run_dir = out_dir / f"{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter_path = run_dir / "adapter.safetensors"
    saved = state_to_save(method, module)
    save_file(saved, str(adapter_path))
    receipt = {"schema_version": 1, "status": "trained_pending_fresh_process_verification",
        "method": method, "seed": seed, "train_example_ids": [r["example_id"] for r in train_records],
        "train_families": [r["family"] for r in train_records],
        "validation_example_ids": [r["example_id"] for r in val_records],
        "validation_families": [r["family"] for r in val_records],
        "sealed_test_used": False, "target_module": "post_layer26_residual" if method == "connector" else "layer27.self_attn.q_proj",
        "trainable_parameters": 49152, "optimizer": "AdamW", "learning_rate": LR,
        "epochs": EPOCHS, "update_count": len(history), "updates": history,
        "validation_loss_before_by_example": before, "validation_loss_after_by_example": after,
        "validation_loss_before_mean": sum(before.values()) / len(before),
        "validation_loss_after_mean": sum(after.values()) / len(after),
        "validation_loss_delta_mean": sum(after.values()) / len(after) - sum(before.values()) / len(before),
        "validation_logits": val_logits_hashes, "training_wall_seconds": train_seconds,
        "adapter_sha256": file_sha(adapter_path), "module_state_sha256": state_hash(module),
        "checkpoint_revision": model.config._name_or_path, "ram_available_after_training_bytes": int(psutil.virtual_memory().available),
        "frozen_backbone_unchanged": True, "production_graph_modified": False,
        "resource_note": "Single-process CPU reference training; GPU is not used by this experiment."}
    atomic_json(run_dir / "training_receipt.json", receipt)
    print(json.dumps({"run": run_dir.name, "train_seconds": train_seconds,
        "validation_before": receipt["validation_loss_before_mean"],
        "validation_after": receipt["validation_loss_after_mean"], "adapter_sha256": receipt["adapter_sha256"]}), flush=True)
    return receipt


def train_all() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tokenizer_path = Path(json.loads(BASELINE.read_text(encoding="utf-8"))["checkpoint"]["snapshot_path"]).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    train_records = load_examples(tokenizer, manifest, "train")
    val_records = load_examples(tokenizer, manifest, "validation")
    model, snapshot, preflight = load_model()
    model.config._name_or_path = snapshot.name
    print("Capturing frozen activations for frozen train/validation family split.", flush=True)
    capture_start = time.perf_counter()
    activations = capture_activations(model, train_records + val_records)
    capture_seconds = time.perf_counter() - capture_start
    atomic_json(OUT / "capture_receipt.json", {"status": "complete", "capture_seconds": capture_seconds,
        "train_ids": [r["example_id"] for r in train_records], "validation_ids": [r["example_id"] for r in val_records],
        "sealed_test_used": False, "preflight": preflight,
        "ram_available_after_capture_bytes": int(psutil.virtual_memory().available)})
    for seed in SEEDS:
        for method in METHODS:
            train_one(model, train_records, activations, val_records, activations, method, seed, OUT)
    atomic_json(OUT / "run_summary.json", {"status": "training_complete_pending_fresh_process_verification",
        "capture_seconds": capture_seconds, "methods": METHODS, "seeds": SEEDS,
        "train_examples": len(train_records), "validation_examples": len(val_records),
        "manifest_sha256": file_sha(MANIFEST), "sealed_test_used": False})


def verify_all() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tokenizer_path = Path(json.loads(BASELINE.read_text(encoding="utf-8"))["checkpoint"]["snapshot_path"]).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    val_records = load_examples(tokenizer, manifest, "validation")
    model, snapshot, preflight = load_model()
    model.config._name_or_path = snapshot.name
    acts = capture_activations(model, val_records)
    results = []
    for seed in SEEDS:
        for method in METHODS:
            run_dir = OUT / f"{method}_seed{seed}"
            receipt = json.loads((run_dir / "training_receipt.json").read_text(encoding="utf-8"))
            state = load_file(str(run_dir / "adapter.safetensors"), device="cpu")
            if method == "connector":
                module = IdentityResidualConnector(hidden_size=int(model.config.hidden_size), rank=8)
                module.load_state_dict(state, strict=True)
            else:
                module = LoRALinear(model.model.layers[27].self_attn.q_proj, int(model.config.hidden_size), rank=8, alpha=8.0)
                module.a.data.copy_(state["lora_A"])
                module.b.data.copy_(state["lora_B"])
            hashes = {}
            with torch.no_grad():
                for record in val_records:
                    logits = objective_logits(model, record, acts[record["example_id"]], module, method)
                    hashes[record["example_id"]] = {"shape": list(logits.shape),
                        "sha256": sha_bytes(logits.detach().cpu().contiguous().numpy().tobytes())}
            equal = hashes == receipt["validation_logits"]
            verification = {"method": method, "seed": seed, "fresh_process": True,
                "status": "passed" if equal and file_sha(run_dir / "adapter.safetensors") == receipt["adapter_sha256"] else "failed",
                "validation_logits_bitwise_match": equal, "recomputed_validation_logits": hashes,
                "checkpoint_revision": snapshot.name, "ram_available_after_load_bytes": preflight["available_ram_after_load_bytes"]}
            atomic_json(run_dir / "fresh_process_verification.json", verification)
            receipt["status"] = "passed" if verification["status"] == "passed" else "failed"
            receipt["fresh_process_verification"] = verification["status"]
            atomic_json(run_dir / "training_receipt.json", receipt)
            results.append(verification)
    atomic_json(OUT / "fresh_process_verification.json", {"status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
        "results": results, "sealed_test_used": False})
    if any(r["status"] != "passed" for r in results):
        raise RuntimeError("Fresh process adapter parity failed")
    print(json.dumps({"status": "passed", "runs": len(results)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "verify"])
    args = parser.parse_args()
    if args.mode == "train":
        train_all()
    else:
        verify_all()


if __name__ == "__main__":
    main()
