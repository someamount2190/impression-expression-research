"""Bounded, CPU-only LoRA training for one registered model partition.

The local GGUF is used as the authoritative structural inventory. Training uses
the pinned BF16 Safetensors publication of the same configured base model,
streamed by Transformers into FP16 CPU parameters. This script never
falls back to another model or objective and stops before work if its memory
reserve cannot be maintained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impression.registry import PartitionRegistry, discover_registry


TARGET_PARTITION = "IMP_L00_FEED_FORWARD_GATE_PROJECTION"
TARGET_MODULE = "model.layers.0.mlp.gate_proj"
MODEL_REPO = os.environ.get("BASE_MODEL_REPO", "[BASE_MODEL_REPO]")
MODEL_REVISION = os.environ.get("BASE_MODEL_REVISION", "[BASE_MODEL_REVISION]")
LICENSE_TAG = os.environ.get("BASE_MODEL_LICENSE_TAG", "[BASE_MODEL_LICENSE_TAG]")
LICENSE_URL = os.environ.get("BASE_MODEL_LICENSE_URL", "[BASE_MODEL_LICENSE_URL]")
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REVISION}"
GGUF_PATH = Path(os.environ.get("BASE_MODEL_GGUF", str(ROOT / "models" / "[BASE_MODEL].gguf")))
TRAIN_DATA = Path(os.environ.get("TRAINING_DATA", str(ROOT / "data" / "training_data.jsonl")))
RUN_DIR = ROOT / "experiments" / "impression_001"
REPORT_PATH = RUN_DIR / "reports" / "selective_training.json"
ADAPTER_DIR = ROOT / "checkpoints" / "impression_001" / "layer0_ffn_gate_lora"
EXPECTED_LOGITS_PATH = RUN_DIR / "checkpoints" / "post_train_logits.npy"
VERIFY_INPUT_PATH = RUN_DIR / "checkpoints" / "verify_input.json"

MIN_AVAILABLE_BEFORE_LOAD_BYTES = 10 * 1024**3
MIN_FREE_RAM_FRACTION = 0.20
MAX_TRAINING_STEPS = 3
MAX_SEQUENCE_LENGTH = 64
LORA_RANK = 4
LEARNING_RATE = 1e-4
MODEL_DTYPE_BYTES = 2
SAFE_TENSOR_SHA256 = {
    "model-00001-of-00002.safetensors": "30660d7b8dc6288690933fcad0609d250bd7cd8d88e633e5886c7f5a500236f2",
    "model-00002-of-00002.safetensors": "ea93d07c32c36d1385491bf45f13188e1e08647c066de01e33fc20a262448bf5",
}
SAFE_TENSOR_SIZES = {
    "model-00001-of-00002.safetensors": 4_965_811_384,
    "model-00002-of-00002.safetensors": 1_459_729_952,
}


class ResourcePreflightError(RuntimeError):
    """Raised when the run cannot preserve the required physical RAM reserve."""


@dataclass(frozen=True)
class SafetensorsMemoryEstimate:
    parameter_count: int
    model_resident_bytes: int
    largest_shard_bytes: int
    conservative_load_peak_bytes: int
    required_free_ram_bytes_after_load: int
    available_ram_bytes: int
    total_ram_bytes: int
    feasible: bool


@dataclass(frozen=True)
class TrainingExample:
    input_ids: Tuple[int, ...]
    labels: Tuple[int, ...]
    record_id: str


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def estimate_safetensors_memory(
    registry: PartitionRegistry,
    *,
    available_ram_bytes: int,
    total_ram_bytes: int,
    largest_shard_bytes: int = max(SAFE_TENSOR_SIZES.values()),
    model_bytes_per_parameter: int = MODEL_DTYPE_BYTES,
) -> SafetensorsMemoryEstimate:
    """Conservative, sharded HF load estimate; excludes training activations."""
    count = sum(t.parameter_count for t in registry.tensors.values())
    resident = count * model_bytes_per_parameter
    peak = resident + largest_shard_bytes
    reserve = int(total_ram_bytes * MIN_FREE_RAM_FRACTION)
    feasible = (
        available_ram_bytes >= MIN_AVAILABLE_BEFORE_LOAD_BYTES
        and available_ram_bytes >= peak + reserve
    )
    return SafetensorsMemoryEstimate(
        parameter_count=count,
        model_resident_bytes=resident,
        largest_shard_bytes=largest_shard_bytes,
        conservative_load_peak_bytes=peak,
        required_free_ram_bytes_after_load=reserve,
        available_ram_bytes=available_ram_bytes,
        total_ram_bytes=total_ram_bytes,
        feasible=feasible,
    )


def validate_exact_target(named_parameters: Iterable[Tuple[str, object]], target_module: str) -> str:
    expected = f"{target_module}.weight"
    matches = [name for name, _ in named_parameters if name == expected]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one target parameter {expected!r}; found {len(matches)}")
    return matches[0]


def windows_memory_status() -> Tuple[int, int]:
    """Return available and total physical memory in bytes using Windows API."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _peak_rss_bytes() -> int:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except Exception:
        return _rss_bytes()


def _verify_architecture(registry: PartitionRegistry, config: Any) -> dict:
    expected = {
        "num_hidden_layers": int(registry.metadata["llama.block_count"]),
        "hidden_size": int(registry.metadata["llama.embedding_length"]),
        "intermediate_size": int(registry.metadata["llama.feed_forward_length"]),
        "num_attention_heads": int(registry.metadata["llama.attention.head_count"]),
        "num_key_value_heads": int(registry.metadata["llama.attention.head_count_kv"]),
    }
    actual = {key: int(getattr(config, key)) for key in expected}
    if actual != expected:
        raise ValueError(f"HF checkpoint architecture does not match GGUF: expected {expected}, got {actual}")
    return actual


def _resolve_snapshot() -> Tuple[Path, dict]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise ValueError(f"Resolved HF revision mismatch: expected {MODEL_REVISION}, got {info.sha}")
    license_tag = (info.card_data or {}).get("license")
    if license_tag != LICENSE_TAG:
        raise ValueError(f"Unexpected model license tag: {license_tag!r}")

    remote = {item.rfilename: item for item in info.siblings}
    for filename, expected_hash in SAFE_TENSOR_SHA256.items():
        item = remote.get(filename)
        if item is None or item.size != SAFE_TENSOR_SIZES[filename] or not item.lfs:
            raise ValueError(f"Expected pinned Safetensors file metadata is missing for {filename}")
        if item.lfs.sha256 != expected_hash:
            raise ValueError(f"Pinned Safetensors hash mismatch in HF metadata for {filename}")

    local = Path(snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[
            "README.md", "LICENSE", "config.json", "generation_config.json",
            "model.safetensors.index.json", "model-*.safetensors",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        ],
    ))
    if local.name != MODEL_REVISION:
        raise ValueError(f"Snapshot cache path is not pinned to {MODEL_REVISION}: {local}")

    verified_files = {}
    for filename, expected_hash in SAFE_TENSOR_SHA256.items():
        path = local / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if path.stat().st_size != SAFE_TENSOR_SIZES[filename] or actual_hash != expected_hash:
            raise ValueError(f"Downloaded Safetensors file failed SHA-256 verification: {filename}")
        verified_files[filename] = {"size_bytes": path.stat().st_size, "sha256": actual_hash}
    return local, {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "local_snapshot_path": str(local),
        "license_tag": license_tag,
        "license_url": LICENSE_URL,
        "model_url": MODEL_URL,
        "verified_files": verified_files,
    }


def _load_model(snapshot: Path) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype="auto",
        device_map={"": "cpu"},
    )
    return model, tokenizer


def prepare_training_example(record: Mapping[str, Any], tokenizer: Any, max_length: int = MAX_SEQUENCE_LENGTH) -> TrainingExample:
    turns = record.get("turns")
    if not isinstance(turns, list):
        raise ValueError("Training record must contain a turns list")
    pair: Optional[Tuple[str, str]] = None
    for i, turn in enumerate(turns):
        if turn.get("role") != "assistant" or not isinstance(turn.get("content"), str):
            continue
        previous_user = next((turns[j] for j in range(i - 1, -1, -1) if turns[j].get("role") == "user"), None)
        if previous_user is not None and isinstance(previous_user.get("content"), str):
            pair = (previous_user["content"], turn["content"])
    if pair is None:
        raise ValueError(f"Record {record.get('id')} has no user/assistant pair")

    user_text, assistant_text = pair
    user_ids = tokenizer.encode(f"<|im_start|>user\n{user_text}<|im_end|>\n", add_special_tokens=False)
    assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    answer_ids = tokenizer.encode(f"{assistant_text}<|im_end|>", add_special_tokens=False)
    if not answer_ids or len(assistant_header) >= max_length:
        raise ValueError(f"Record {record.get('id')} has no usable assistant target")

    answer_budget = min(len(answer_ids), max_length - len(assistant_header) - 1)
    user_budget = max_length - len(assistant_header) - answer_budget
    user_part = user_ids[-user_budget:] if user_budget > 0 else []
    answer_part = answer_ids[:answer_budget]
    input_ids = [*user_part, *assistant_header, *answer_part]
    labels = [-100] * (len(user_part) + len(assistant_header)) + answer_part
    if len(input_ids) < 2 or not any(label != -100 for label in labels[1:]):
        raise ValueError(f"Record {record.get('id')} has no shifted assistant labels")
    return TrainingExample(tuple(input_ids), tuple(labels), str(record.get("id", "unknown")))


def _parameter_fingerprint(named_parameters: Iterable[Tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    seen = 0
    for name, parameter in named_parameters:
        if ".lora_A." in name or ".lora_B." in name or ".lora_embedding_" in name:
            continue
        normalized = name
        if normalized.startswith("base_model.model."):
            normalized = normalized[len("base_model.model."):]
        normalized = normalized.replace(".base_layer.", ".")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        raw = parameter.detach().contiguous().view(-1).view(__import__("torch").uint8).numpy()
        digest.update(memoryview(raw))
        seen += 1
    if not seen:
        raise ValueError("No frozen backbone parameters were fingerprinted")
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < MAX_TRAINING_STEPS + 1:
        raise ValueError("Need at least 4 independent tool-training records")
    return records


def _input_tensors(example: TrainingExample, torch: Any) -> Tuple[Any, Any]:
    return (
        torch.tensor([example.input_ids], dtype=torch.long, device="cpu"),
        torch.tensor([example.labels], dtype=torch.long, device="cpu"),
    )


def _logits_for(model: Any, example: TrainingExample, torch: Any) -> Any:
    input_ids, _ = _input_tensors(example, torch)
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits.detach().float().cpu().numpy()


def _validation_example_from_report(report: Mapping[str, Any], tokenizer: Any) -> TrainingExample:
    validation_id = str(report["validation_example"])
    rows = _load_records(TRAIN_DATA)
    record = next((row for row in rows if str(row.get("id")) == validation_id), None)
    if record is None:
        raise ValueError(f"Validation record is missing from the local dataset: {validation_id}")
    return prepare_training_example(record, tokenizer)


def run_train(*, max_steps: int = MAX_TRAINING_STEPS) -> dict:
    if max_steps < 1 or max_steps > MAX_TRAINING_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAX_TRAINING_STEPS}")
    if not GGUF_PATH.is_file() or not TRAIN_DATA.is_file():
        raise FileNotFoundError("The authoritative GGUF or local training data is unavailable")
    registry = discover_registry(GGUF_PATH)
    target = registry.partitions.get(TARGET_PARTITION)
    if target is None or len(target.tensor_names) != 1 or target.tensor_names[0] != "blk.0.ffn_gate.weight":
        raise ValueError(f"Registry target is missing or ambiguous: {TARGET_PARTITION}")

    import torch
    import numpy as np
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    # PEFT/Transformers may initialize CUDA bookkeeping during import even
    # without allocating a model there. Enforce CPU placement on the loaded
    # model below instead of treating import-side initialization as GPU work.
    available_before, total_ram = windows_memory_status()
    if available_before < MIN_AVAILABLE_BEFORE_LOAD_BYTES:
        raise ResourcePreflightError(f"Pre-load RAM {available_before} is below 10 GiB")
    estimate = estimate_safetensors_memory(
        registry,
        available_ram_bytes=available_before,
        total_ram_bytes=total_ram,
    )
    if not estimate.feasible:
        raise ResourcePreflightError(
            f"Pinned Safetensors load estimate {estimate.conservative_load_peak_bytes} plus the 20% RAM reserve "
            f"exceeds available RAM {available_before}"
        )

    disk_root = Path.home()
    required_download_bytes = sum(SAFE_TENSOR_SIZES.values())
    disk_free = shutil.disk_usage(disk_root).free
    if disk_free < required_download_bytes + 1024**3:
        raise ResourcePreflightError(
            f"Only {disk_free} bytes are free under {disk_root}; a checkpoint download would not leave 1 GiB headroom"
        )
    _atomic_json(RUN_DIR / "reports" / "safetensors_preflight.json", {
        "experiment": "IMPRESSION-001",
        "status": "ready_to_resolve_pinned_checkpoint",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "license_tag": LICENSE_TAG,
        "license_url": LICENSE_URL,
        "memory_estimate": asdict(estimate),
        "free_disk_bytes_before_download": disk_free,
        "download_bytes": required_download_bytes,
        "training_started": False,
        "device": "cpu",
    })

    snapshot, provenance = _resolve_snapshot()
    provenance["local_snapshot_path"] = str(snapshot)
    available_before_load, total_after_download = windows_memory_status()
    estimate = estimate_safetensors_memory(
        registry,
        available_ram_bytes=available_before_load,
        total_ram_bytes=total_after_download,
    )
    if not estimate.feasible:
        raise ResourcePreflightError(
            f"Post-download available RAM {available_before_load} no longer meets the conservative load estimate"
        )
    config_path = snapshot / "config.json"
    remote_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "num_hidden_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
    }
    for key, value in expected_config.items():
        if int(remote_config[key]) != value:
            raise ValueError(f"Pinned checkpoint config mismatch for {key}: {remote_config[key]} != {value}")

    import psutil
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter()
    model, tokenizer = _load_model(snapshot)
    load_seconds = time.perf_counter() - started
    if next(model.parameters()).device.type != "cpu" or any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Model parameters were not loaded exclusively on CPU")
    if str(next(model.parameters()).dtype) not in ("torch.float16", "torch.bfloat16"):
        raise RuntimeError(f"Unexpected model dtype: {next(model.parameters()).dtype}")
    architecture = _verify_architecture(registry, model.config)
    target_name = validate_exact_target(model.named_parameters(), TARGET_MODULE)
    target_parameter = dict(model.named_parameters())[target_name]
    if tuple(target_parameter.shape) != (8192, 3072):
        raise ValueError(f"Target module shape mismatch: {tuple(target_parameter.shape)}")

    available_after_load, _ = windows_memory_status()
    reserve_bytes = int(total_ram * MIN_FREE_RAM_FRACTION)
    if available_after_load < reserve_bytes:
        del model, tokenizer
        raise ResourcePreflightError(
            f"Post-load available RAM {available_after_load} is below 20% reserve {reserve_bytes}; training not started"
        )

    frozen_before = _parameter_fingerprint(model.named_parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[TARGET_MODULE],
    )
    peft_model = get_peft_model(model, lora_config)
    trainable = [(name, parameter) for name, parameter in peft_model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name or TARGET_MODULE not in name for name, _ in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {[name for name, _ in trainable[:10]]}")
    trainable_count = sum(parameter.numel() for _, parameter in trainable)

    rows = _load_records(TRAIN_DATA)
    val_record = rows[-1]
    train_records = [row for row in rows[:-1] if str(row.get("id")) != str(val_record.get("id"))][:max_steps]
    if len(train_records) < max_steps:
        raise ValueError("Could not select enough disjoint training examples")
    train_examples = [prepare_training_example(row, tokenizer) for row in train_records]
    validation_example = prepare_training_example(val_record, tokenizer)
    if {e.record_id for e in train_examples} & {validation_example.record_id}:
        raise ValueError("Training and validation IDs overlap")

    peft_model.eval()
    validation_inputs, labels = _input_tensors(validation_example, torch)
    with torch.no_grad():
        baseline_validation_loss = float(peft_model(
            input_ids=validation_inputs,
            labels=labels,
            use_cache=False,
        ).loss.item())
    baseline_logits = _logits_for(peft_model, validation_example, torch)
    initial_adapter = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=LEARNING_RATE)
    loss_history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    peft_model.train()
    for step, example in enumerate(train_examples, start=1):
        available_now, _ = windows_memory_status()
        if available_now < reserve_bytes:
            raise ResourcePreflightError(f"RAM reserve fell below 20% before step {step}; stopping")
        input_ids, step_labels = _input_tensors(example, torch)
        optimizer.zero_grad(set_to_none=True)
        loss = peft_model(input_ids=input_ids, labels=step_labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        value = float(loss.detach().item())
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in trainable):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], max_norm=1.0)
        optimizer.step()
        loss_history.append({"step": step, "train_loss": value})

    training_seconds = time.perf_counter() - training_started
    peft_model.eval()
    with torch.no_grad():
        post_train_inputs, post_train_labels = _input_tensors(validation_example, torch)
        validation_loss = float(peft_model(
            input_ids=post_train_inputs,
            labels=post_train_labels,
            use_cache=False,
        ).loss.item())
    updated_logits = _logits_for(peft_model, validation_example, torch)
    effective_output_change = float(np.max(np.abs(updated_logits - baseline_logits)))
    changed_adapter_parameters = sum(
        int(not torch.equal(initial_adapter[name], parameter.detach()))
        for name, parameter in trainable
    )
    adapter_update_l2 = math.sqrt(sum(
        float((parameter.detach().float() - initial_adapter[name].float()).pow(2).sum().item())
        for name, parameter in trainable
    ))
    if changed_adapter_parameters == 0 or adapter_update_l2 == 0.0:
        raise RuntimeError("Gradient steps did not measurably modify the selected adapter")
    with peft_model.disable_adapter():
        rolled_back_logits = _logits_for(peft_model, validation_example, torch)
    rollback_max_abs_diff = float(np.max(np.abs(rolled_back_logits - baseline_logits)))
    rollback_verified = bool(np.allclose(rolled_back_logits, baseline_logits, rtol=0.0, atol=1e-6))
    if not rollback_verified:
        raise RuntimeError(f"Disabling the adapter did not restore baseline outputs: max diff={rollback_max_abs_diff}")
    frozen_after = _parameter_fingerprint(peft_model.named_parameters())
    if frozen_before != frozen_after:
        raise RuntimeError("A frozen base parameter changed during selective training")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ADAPTER_DIR), safe_serialization=True)
    VERIFY_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_INPUT_PATH.write_text(json.dumps({
        "input_ids": list(validation_example.input_ids),
        "labels": list(validation_example.labels),
        "record_id": validation_example.record_id,
    }, indent=2), encoding="utf-8")
    expected_logits = _logits_for(peft_model, validation_example, torch)
    np.save(EXPECTED_LOGITS_PATH, expected_logits.astype(np.float32, copy=False))

    available_after_training, _ = windows_memory_status()
    report = {
        "experiment": "IMPRESSION-001",
        "phase": "selective_training",
        "status": "trained_pending_fresh_process_verification",
        "training_started": True,
        "model": MODEL_REPO,
        "registry_source_checkpoint": str(GGUF_PATH),
        "registry_source_sha256": registry.source_sha256,
        "checkpoint_provenance": provenance,
        "memory_estimate": asdict(estimate),
        "architecture_verified_against_gguf": architecture,
        "selected_partition": TARGET_PARTITION,
        "selected_module": target_name,
        "selected_base_parameter_count": int(target_parameter.numel()),
        "trainable_adapter_parameters": int(trainable_count),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameters_unchanged": True,
        "train_examples": [example.record_id for example in train_examples],
        "validation_example": validation_example.record_id,
        "validation_loss_before": baseline_validation_loss,
        "validation_loss_after": validation_loss,
        "validation_loss_improved": validation_loss < baseline_validation_loss,
        "adapter_parameters_changed": changed_adapter_parameters,
        "adapter_update_l2": adapter_update_l2,
        "validation_logits_max_abs_change": effective_output_change,
        "rollback_verified": rollback_verified,
        "rollback_logits_max_abs_diff": rollback_max_abs_diff,
        "training_steps": len(loss_history),
        "loss_history": loss_history,
        "training_wall_seconds": training_seconds,
        "load_wall_seconds": load_seconds,
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "rss_before_load_bytes": rss_before,
        "available_ram_before_load_bytes": available_before,
        "available_ram_after_load_bytes": available_after_load,
        "available_ram_after_training_bytes": available_after_training,
        "total_ram_bytes": total_ram,
        "minimum_free_ram_fraction": MIN_FREE_RAM_FRACTION,
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "adapter_path": str(ADAPTER_DIR),
        "fresh_process_reload_verified": False,
        "efficiency_gain_measured": False,
        "limitations": [
            "Only one layer-zero FFN gate projection's LoRA adapter was trained.",
            "Validation-loss change on one held-out local sample is not evidence of general capability improvement.",
            "No speed or memory efficiency gain was measured against a conventional full-model or PEFT baseline.",
        ],
    }
    _atomic_json(REPORT_PATH, report)
    return report


def run_fresh_process_reference() -> dict:
    """Create an output reference from a fresh base + saved-adapter process."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if not report.get("training_started") or not ADAPTER_DIR.is_dir():
        raise ValueError("No saved IMPRESSION adapter is available for fresh-process reference generation")
    import numpy as np
    import torch
    from peft import PeftModel

    snapshot = Path(report["checkpoint_provenance"].get("local_snapshot_path", ""))
    if not snapshot.exists():
        raise FileNotFoundError(f"Pinned model snapshot is unavailable: {snapshot}")
    available_before, total = windows_memory_status()
    if available_before < MIN_AVAILABLE_BEFORE_LOAD_BYTES:
        raise ResourcePreflightError("Fresh-process verification lacks the required 10 GiB pre-load RAM")
    base, tokenizer = _load_model(snapshot)
    available_after, _ = windows_memory_status()
    if available_after < int(total * MIN_FREE_RAM_FRACTION):
        del base, tokenizer
        raise ResourcePreflightError("Fresh-process verification cannot preserve the 20% free RAM reserve")
    # Keep reintegration on the same CPU device as training. PEFT 0.21 can
    # otherwise re-dispatch an hf_device_map model to CUDA via "auto".
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False, device_map={"": "cpu"})
    model.eval()
    example = _validation_example_from_report(report, tokenizer)
    input_ids, labels = _input_tensors(example, torch)
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        logits = output.logits.detach().float().cpu().numpy()
        validation_loss = float(output.loss.item())
    VERIFY_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFY_INPUT_PATH.write_text(json.dumps({
        "input_ids": list(example.input_ids),
        "labels": list(example.labels),
        "record_id": example.record_id,
    }, indent=2), encoding="utf-8")
    np.save(EXPECTED_LOGITS_PATH, logits.astype(np.float32, copy=False))
    logits_hash = hashlib.sha256(memoryview(np.ascontiguousarray(logits))).hexdigest()
    initial_training_parity = report.get("fresh_process_max_abs_logit_diff")
    if initial_training_parity is not None:
        report["training_process_reload_parity"] = {
            "verified": False,
            "initial_fresh_process_max_abs_logit_diff": float(initial_training_parity),
            "rtol": float(report.get("fresh_process_logit_rtol", 1e-4)),
            "atol": float(report.get("fresh_process_logit_atol", 1e-4)),
            "expected_logits_artifact_was_removed_after_failure": True,
            "reconstruction_status": "not_recoverable_from_training_process",
        }
    report.update({
        "fresh_process_reference_saved": True,
        "fresh_process_reference_source": "fresh process loaded configured pinned Safetensors base and saved PEFT adapter",
        "fresh_process_reference_validation_loss": validation_loss,
        "training_to_fresh_reference_validation_loss_delta": validation_loss - float(report["validation_loss_after"]),
        "fresh_process_reference_logits_sha256": logits_hash,
        "fresh_process_reference_available_ram_before_bytes": available_before,
        "fresh_process_reference_available_ram_after_load_bytes": available_after,
        "status": "fresh_process_reference_saved_pending_repeat_load_training_process_parity_failed",
    })
    _atomic_json(REPORT_PATH, report)
    return report


def run_fresh_process_verify() -> dict:
    """Reload in another process and compare to a saved fresh-process reference."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "fresh_process_reference_saved_pending_repeat_load_training_process_parity_failed":
        raise ValueError("No fresh-process adapter reference is pending verification")
    import numpy as np
    import torch
    from peft import PeftModel

    snapshot = Path(report["checkpoint_provenance"].get("local_snapshot_path", ""))
    if not snapshot.exists():
        raise FileNotFoundError(f"Pinned model snapshot is unavailable: {snapshot}")
    available_before, total = windows_memory_status()
    if available_before < MIN_AVAILABLE_BEFORE_LOAD_BYTES:
        raise ResourcePreflightError("Fresh-process verification lacks the required 10 GiB pre-load RAM")
    base, tokenizer = _load_model(snapshot)
    available_after, _ = windows_memory_status()
    if available_after < int(total * MIN_FREE_RAM_FRACTION):
        del base, tokenizer
        raise ResourcePreflightError("Fresh-process verification cannot preserve the 20% free RAM reserve")
    # Match the CPU-only training placement; do not let PEFT choose a different
    # device map during reload.
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False, device_map={"": "cpu"})
    model.eval()
    inputs = json.loads(VERIFY_INPUT_PATH.read_text(encoding="utf-8"))
    input_ids = torch.tensor([inputs["input_ids"]], dtype=torch.long, device="cpu")
    labels = torch.tensor([inputs["labels"]], dtype=torch.long, device="cpu")
    expected = np.load(EXPECTED_LOGITS_PATH, allow_pickle=False)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
        actual = output.logits.detach().float().cpu().numpy()
        validation_loss = float(output.loss.item())
    elapsed = time.perf_counter() - started
    delta = np.abs(actual - expected)
    max_abs = float(np.max(delta))
    mean_abs = float(np.mean(delta))
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    exact = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-4))
    report.update({
        "fresh_process_reload_verified": exact,
        "fresh_process_max_abs_logit_diff": max_abs,
        "fresh_process_mean_abs_logit_diff": mean_abs,
        "fresh_process_logits_rmse": rmse,
        "fresh_process_validation_loss": validation_loss,
        "fresh_process_reference_loss_delta": validation_loss - float(report["fresh_process_reference_validation_loss"]),
        "fresh_process_reload_wall_seconds": elapsed,
        "fresh_process_available_ram_before_bytes": available_before,
        "fresh_process_available_ram_after_load_bytes": available_after,
        "peak_process_rss_bytes": max(int(report.get("peak_process_rss_bytes", 0)), _peak_rss_bytes()),
        "fresh_process_logit_rtol": 1e-4,
        "fresh_process_logit_atol": 1e-4,
        "fresh_process_determinism_verified": exact,
        "status": (
            "fresh_load_determinism_verified_training_process_parity_failed"
            if exact else "fresh_load_determinism_failed_training_process_parity_failed"
        ),
    })
    _atomic_json(REPORT_PATH, report)
    return report


def _record_blocked(error: BaseException, status: str) -> dict:
    report = {
        "experiment": "IMPRESSION-001",
        "phase": "selective_training",
        "status": status,
        "training_started": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "license_tag": LICENSE_TAG,
        "selected_partition": TARGET_PARTITION,
        "efficiency_gain_measured": False,
    }
    _atomic_json(REPORT_PATH, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_TRAINING_STEPS)
    parser.add_argument("--verify-fresh-process", action="store_true")
    parser.add_argument("--write-fresh-reference", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_fresh_process:
            result = run_fresh_process_verify()
            print(json.dumps(result, indent=2))
            return 0 if result.get("fresh_process_reload_verified") else 3
        if args.write_fresh_reference:
            result = run_fresh_process_reference()
            print(json.dumps(result, indent=2))
            return 0
        result = run_train(max_steps=args.max_steps)
        print(json.dumps(result, indent=2))
        return 0
    except ResourcePreflightError as exc:
        print(json.dumps(_record_blocked(exc, "blocked_resource_preflight"), indent=2))
        return 2
    except (ImportError, OSError) as exc:
        print(json.dumps(_record_blocked(exc, "blocked_framework_or_checkpoint"), indent=2))
        return 2
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        print(json.dumps(_record_blocked(exc, "blocked_model_validation_or_training"), indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
