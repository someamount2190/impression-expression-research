"""Bounded, resumable frozen-prefix boundary capture for connector feasibility.

This captures only the layer-26 output needed by the connector and layer-27
LoRA objectives. Each example is persisted atomically so a later run can
resume without recapturing completed examples. It does not train or use test
examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.impression_006.connector_ablation.run_connector_ablation import (
    BASELINE, MANIFEST, file_sha, load_examples,
)


class _BoundaryReached(Exception):
    pass


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--example-id", required=True)
    args = ap.parse_args()
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"probe directory must be new and empty: {out}")

    manifest_hash = file_sha(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("sealed_test_used_for_training_or_hyperparameter_selection") is not False:
        raise RuntimeError("Frozen connector manifest test-use guard is missing/false")
    if args.example_id not in {x["example_id"] for x in manifest["train"]}:
        raise RuntimeError("Probe must use exactly one frozen connector train example")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    checkpoint = baseline["checkpoint"]
    snapshot = Path(checkpoint["snapshot_path"]).resolve(strict=True)
    index = snapshot / "model.safetensors.index.json"
    if snapshot.name != checkpoint["revision"] or file_sha(index) != checkpoint["index_sha256"]:
        raise RuntimeError("Pinned [BASE_MODEL] revision/hash mismatch")
    total = int(psutil.virtual_memory().total)
    reserve = int(total * 0.20)
    model_bytes = int(checkpoint["index_metadata"]["total_size"])
    if psutil.virtual_memory().available < model_bytes + reserve + 768 * 1024**2:
        raise RuntimeError("RAM preflight rejected model load with 20% reserve")

    record_source = Path(ROOT / manifest["source_files"]["train"])
    source_sha = file_sha(record_source)
    if source_sha != manifest["source_file_sha256"][str(record_source.relative_to(ROOT)).replace("\\", "/")]:
        raise RuntimeError("Frozen train source hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    record = next(x for x in load_examples(tokenizer, manifest, "train") if x["example_id"] == args.example_id)
    if psutil.virtual_memory().available < model_bytes + reserve:
        raise RuntimeError("RAM reserve preflight failed before [BASE_MODEL] load")
    torch.set_num_threads(1)
    available_before_load = int(psutil.virtual_memory().available)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    load_seconds = time.perf_counter() - load_started
    model.requires_grad_(False)
    if psutil.virtual_memory().available < reserve:
        raise RuntimeError("Model load breached 20% system RAM reserve")

    model_profile = {"checkpoint_revision": snapshot.name, "checkpoint_index_sha256": file_sha(index),
        "connector_manifest_sha256": manifest_hash, "train_source_sha256": source_sha,
        "example_id": args.example_id, "sequence_length": int(record["input_ids"].shape[1]),
        "torch_threads": torch.get_num_threads(), "total_ram_bytes": total,
        "available_ram_before_load_bytes": available_before_load,
        "reserve_bytes": reserve, "gpu_used": False}
    capture_started = time.perf_counter()
    state = {"status": "running", "phase": "load_and_capture", "profile": model_profile,
        "run_started_utc": time.time(), "output_path": str(out)}
    (out / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    captured: dict[str, torch.Tensor] = {}
    def stop_after_frozen_prefix(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured["layer26_output"] = hidden.detach().cpu().contiguous()
        raise _BoundaryReached()
    hook = model.model.layers[26].register_forward_hook(stop_after_frozen_prefix)
    status = "failed"
    error = None
    try:
        with torch.inference_mode():
            model.model(input_ids=record["input_ids"], use_cache=False, return_dict=True)
    except _BoundaryReached:
        status = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        hook.remove()
    capture_seconds = time.perf_counter() - capture_started
    if status != "complete" or "layer26_output" not in captured:
        raise RuntimeError(f"Boundary capture failed: {error or status}")
    activation = captured["layer26_output"]
    if activation.dtype != torch.bfloat16 or activation.shape[0] != 1 or activation.shape[1] != record["input_ids"].shape[1]:
        raise RuntimeError(f"Unexpected captured activation metadata: {activation.dtype} {tuple(activation.shape)}")
    artifact = out / f"{args.example_id}.safetensors"
    atomic_safetensors(artifact, {"layer26_output": activation})
    payload = activation.numpy().tobytes()
    receipt = {"status": "complete", "example_id": args.example_id,
        "split": "frozen_train", "sealed_test_used": False, "optimizer_updates": 0,
        "graph_modified": False, "model_load_seconds": load_seconds,
        "prefix_forward_seconds": capture_seconds, "total_seconds": load_seconds + capture_seconds,
        "activation_shape": list(activation.shape),
        "activation_dtype": str(activation.dtype), "activation_sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_sha256": file_sha(artifact), "artifact_bytes": artifact.stat().st_size,
        "ram_available_after_capture_bytes": int(psutil.virtual_memory().available),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
        "profile": model_profile}
    (out / "capture_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    state.update({"status": "complete", "phase": "done", "model_load_seconds": load_seconds,
        "prefix_forward_seconds": capture_seconds})
    (out / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
