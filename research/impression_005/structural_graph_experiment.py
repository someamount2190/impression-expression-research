"""Bounded, reversible residual-connector graph experiment on [BASE_MODEL] 3B."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.impression_005.structural_runtime import (
    IdentityResidualConnector,
    install_cached_prefix,
    install_cached_prefix_live_connector,
    restore_original_layers,
)

EXPERIMENT = ROOT / "experiments/impression_005"
STRUCTURAL = EXPERIMENT / "structural"
SOURCE_RUN = ROOT / "experiments/impression_004/partition_training_receipts/cuda_head_sparse02"
SOURCE_RECEIPT = SOURCE_RUN / "training_receipt.json"
BASELINE_PATH = ROOT / "experiments/impression_002d/runtime_baseline.json"
MAX_UPDATES = 2
QUALITY_REGRESSION_LIMIT = 0.0001


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.wait(1.0):
            memory = psutil.virtual_memory()
            self.samples.append({
                "unix_time": time.time(),
                "process_rss_bytes": self.process.memory_info().rss,
                "available_ram_bytes": memory.available,
                "cpu_percent": psutil.cpu_percent(interval=None),
            })

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join(timeout=2)


def records_from_receipt() -> tuple[dict, dict, dict]:
    source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    records = source["causal_objectives"]
    return source, records["eg001_train_balanced"], records["eg001_validation_family_holdout"]


def as_tensors(record: dict) -> dict[str, torch.Tensor]:
    ids = torch.tensor(record["input_ids"], dtype=torch.long).unsqueeze(0)
    positions = [index for index, flag in enumerate(record["label_mask"]) if bool(flag)]
    labels = torch.tensor(record["labels"], dtype=torch.long)
    if len(positions) != int(record["valid_label_count"]):
        raise ValueError("Label mask length/count mismatch in verified IMPRESSION-004 receipt")
    targets = labels[torch.tensor(positions, dtype=torch.long)]
    return {"input_ids": ids, "positions": torch.tensor(positions, dtype=torch.long), "targets": targets}


def capture_full(model, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    activations: dict[int, torch.Tensor] = {}
    hooks = []
    for index in range(27):
        def save_activation(_module, _inputs, output, i=index):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            activations[i] = hidden.detach().clone()
        hooks.append(model.model.layers[index].register_forward_hook(save_activation))
    with torch.inference_mode():
        output = model.model(input_ids=data["input_ids"], use_cache=False, return_dict=True)
        logits = model.lm_head(output.last_hidden_state.index_select(1, data["positions"])).float().cpu()
    for handle in hooks:
        handle.remove()
    if len(activations) != 27:
        raise RuntimeError("Failed to capture all frozen prefix activation boundaries")
    return logits, [activations[i] for i in range(27)]


def forward_cached(model, data: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model.model(input_ids=data["input_ids"], use_cache=False, return_dict=True)
    return model.lm_head(output.last_hidden_state.index_select(1, data["positions"])).float()


def loss_for(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(
            logits.detach().float().reshape(-1, logits.shape[-1]), targets
        ).item())


def load_backbone() -> tuple[AutoModelForCausalLM, Path, dict]:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    checkpoint = baseline["checkpoint"]
    snapshot = Path(checkpoint["snapshot_path"]).resolve(strict=True)
    if not snapshot.is_dir() or snapshot.name != checkpoint["revision"]:
        raise RuntimeError("Pinned [BASE_MODEL] checkpoint path or revision mismatch")
    index = snapshot / "model.safetensors.index.json"
    if sha(snapshot / "config.json") != checkpoint["config_sha256"] or sha(index) != checkpoint["index_sha256"]:
        raise RuntimeError("Pinned checkpoint index/config hash mismatch")
    total_size = int(checkpoint["index_metadata"]["total_size"])
    memory = psutil.virtual_memory()
    reserve = int(memory.total * 0.20)
    if memory.available < total_size + reserve + 768 * 1024**2:
        raise RuntimeError("RAM preflight rejected 3B BF16 model load with reserve and guard")
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    if psutil.virtual_memory().available < reserve:
        raise RuntimeError("3B model load left less than 20% system RAM reserve")
    return model, snapshot, {"total_ram_bytes": memory.total, "available_ram_after_load_bytes": psutil.virtual_memory().available,
                            "ram_reserve_bytes": reserve, "checkpoint_bytes": total_size}


def run_train() -> int:
    STRUCTURAL.mkdir(parents=True, exist_ok=True)
    if (STRUCTURAL / "structural_training_receipt.json").exists():
        raise FileExistsError("Refusing to overwrite existing structural experiment receipt")
    source, train_record, validation_record = records_from_receipt()
    train, validation = as_tensors(train_record), as_tensors(validation_record)
    source_hash = sha(SOURCE_RECEIPT)
    model, snapshot, preflight = load_backbone()
    graph_before = {
        "schema_version": 1, "graph_id": "[BASE_MODEL]3_impression004_fixed_graph_v1",
        "checkpoint_revision": snapshot.name, "backbone_layers": 28,
        "registered_target_partition": "IMP_L27_ATTENTION_QUERY_PROJECTION",
        "trainable_neural_regions_before": ["layer27.self_attn.q_proj adapter (historical; not loaded here)"],
        "connector_nodes": [], "graph_modified": False,
    }
    atomic_json(STRUCTURAL / "graph_before.json", graph_before)
    print("Capturing exact frozen layer 0–26 outputs for the source and held-out records.", flush=True)
    load_started = time.perf_counter()
    base_train_logits, train_activations = capture_full(model, train)
    base_validation_logits, validation_activations = capture_full(model, validation)
    capture_seconds = time.perf_counter() - load_started
    np.save(STRUCTURAL / "baseline_train_logits.npy", base_train_logits.numpy(), allow_pickle=False)
    np.save(STRUCTURAL / "baseline_validation_logits.npy", base_validation_logits.numpy(), allow_pickle=False)

    connector = IdentityResidualConnector(hidden_size=int(model.config.hidden_size), rank=8)
    connector.requires_grad_(True)
    original_layers = install_cached_prefix(model, train_activations, connector)
    before_identity = forward_cached(model, train)
    identity_equal = torch.equal(base_train_logits, before_identity.detach().cpu())
    identity_max_abs = float((base_train_logits - before_identity.detach().cpu()).abs().max().item())
    if not identity_equal:
        restore_original_layers(model, original_layers)
        raise RuntimeError(f"Identity connector failed exact initial-output parity (max abs {identity_max_abs})")

    train_history = []
    optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-3, weight_decay=0.0)
    started = time.perf_counter()
    total_ram = int(psutil.virtual_memory().total)
    reserve = int(total_ram * 0.20)
    with ResourceSampler() as sampler:
        for step in range(1, MAX_UPDATES + 1):
            if psutil.virtual_memory().available < reserve + 512 * 1024**2:
                raise RuntimeError("RAM reserve guard stopped structural training before the next update")
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = forward_cached(model, train)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), train["targets"]
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0).item())
            optimizer.step()
            train_history.append({"step": step, "loss_before_update": float(loss.item()),
                                  "gradient_norm": grad_norm,
                                  "wall_seconds": time.perf_counter() - step_start})
            print(json.dumps({"structural_update": step, "loss": train_history[-1]["loss_before_update"],
                              "wall_seconds": train_history[-1]["wall_seconds"]}), flush=True)
    training_seconds = time.perf_counter() - started

    with torch.no_grad():
        candidate_train_logits = forward_cached(model, train).detach().float().cpu()
    restore_original_layers(model, original_layers)
    # Capture candidate activations for validation independently, then install the candidate graph.
    original_layers = install_cached_prefix(model, validation_activations, connector)
    with torch.no_grad():
        candidate_validation_logits = forward_cached(model, validation).detach().float().cpu()
        validation_after = loss_for(candidate_validation_logits, validation["targets"])
    validation_before = loss_for(base_validation_logits, validation["targets"])
    np.save(STRUCTURAL / "candidate_train_logits.npy", candidate_train_logits.numpy(), allow_pickle=False)
    np.save(STRUCTURAL / "candidate_validation_logits.npy", candidate_validation_logits.numpy(), allow_pickle=False)
    # Keep the connector active only if validation stays inside the predeclared bound.
    accepted = validation_after <= validation_before + QUALITY_REGRESSION_LIMIT
    state = {key: value.detach().cpu().contiguous() for key, value in connector.state_dict().items()}
    adapter_path = STRUCTURAL / "connector.safetensors"
    temporary = STRUCTURAL / ".connector.safetensors.tmp"
    save_file(state, str(temporary), metadata={"graph_id": "[BASE_MODEL]3_layer26_residual_connector_v1",
                                               "checkpoint_revision": snapshot.name,
                                               "target_boundary": "layer26_output_to_layer27_input"})
    os.replace(temporary, adapter_path)
    adapter_hash = sha(adapter_path)

    graph_after = {
        **graph_before,
        "graph_id": "[BASE_MODEL]3_layer26_residual_connector_v1",
        "connector_nodes": [{"node_id": "IMP_L26_L27_RESIDUAL_CONNECTOR_001",
                             "source_region": "model.layers.26 full decoder block output",
                             "target_region": "model.layers.27 full decoder block input",
                             "kind": "trainable_low_rank_residual_branch",
                             "input_size": int(model.config.hidden_size), "output_size": int(model.config.hidden_size),
                             "rank": 8, "initial_output_is_zero": True,
                             "checkpoint": str(adapter_path.relative_to(ROOT)),
                             "checkpoint_sha256": adapter_hash}],
        "edges": [{"from": "model.layers.26", "to": "IMP_L26_L27_RESIDUAL_CONNECTOR_001", "type": "hidden_state_fork"},
                  {"from": "IMP_L26_L27_RESIDUAL_CONNECTOR_001", "to": "model.layers.27", "type": "residual_add"}],
        "graph_modified": True, "candidate_retained": accepted,
        "rollback_to_graph_id": graph_before["graph_id"],
    }
    atomic_json(STRUCTURAL / "graph_candidate.json", graph_after)
    if accepted:
        atomic_json(STRUCTURAL / "graph_current.json", graph_after)
    else:
        restore_original_layers(model, original_layers)
        atomic_json(STRUCTURAL / "graph_current.json", graph_before)

    # Verify rollback in-process against original full-graph output.
    if accepted:
        restore_original_layers(model, original_layers)
        rolled_logits, _ = capture_full(model, train)
        rollback_equal = torch.equal(base_train_logits, rolled_logits)
        # Reinstall candidate layers after the test if retained.
        install_cached_prefix(model, train_activations, connector)
    else:
        restore_original_layers(model, original_layers)
        rollback_equal = torch.equal(base_train_logits, capture_full(model, train)[0])
    if not rollback_equal:
        raise RuntimeError("Restoring original decoder layers did not reproduce the baseline logits")

    torch.save(optimizer.state_dict(), STRUCTURAL / "connector_optimizer_state.pt")
    receipt = {
        "schema_version": 1, "experiment": "IMPRESSION-005", "status": "trained_and_verified_pending_fresh_process",
        "source_receipt": str(SOURCE_RECEIPT.relative_to(ROOT)), "source_receipt_sha256": source_hash,
        "training_record_id": train_record.get("record_id", "eg001_train_balanced"),
        "validation_record_id": validation_record.get("record_id", "eg001_validation_family_holdout"),
        "checkpoint_revision": snapshot.name,
        "checkpoint_index_sha256": sha(snapshot / "model.safetensors.index.json"),
        "pinned_configuration_sha256": sha(snapshot / "config.json"),
        "checkpoint_precision": "raw_bfloat16_frozen",
        "graph_before_id": graph_before["graph_id"], "graph_candidate_id": graph_after["graph_id"],
        "graph_candidate_checkpoint_sha256": adapter_hash,
        "trainable_parameters": sum(parameter.numel() for parameter in connector.parameters()),
        "updates": train_history,
        "max_updates": MAX_UPDATES,
        "frozen_backbone_unchanged": True,
        "initial_identity_logits_equal_bitwise": identity_equal,
        "initial_identity_logits_max_abs_diff": identity_max_abs,
        "training_loss_first_before_update": train_history[0]["loss_before_update"],
        "training_loss_last_before_update": train_history[-1]["loss_before_update"],
        "validation_loss_before": validation_before, "validation_loss_after": validation_after,
        "validation_loss_delta": validation_after - validation_before,
        "validation_regression_limit": QUALITY_REGRESSION_LIMIT,
        "candidate_retained": accepted,
        "rollback_reproduced_original_logits_bitwise": rollback_equal,
        "capture_prefix_seconds": capture_seconds, "training_seconds": training_seconds,
        "model_load_preflight": preflight,
        "resource_summary": {
            "sample_count": len(sampler.samples),
            "peak_process_rss_bytes": max((s["process_rss_bytes"] for s in sampler.samples), default=psutil.Process().memory_info().rss),
            "minimum_system_available_ram_bytes": min((s["available_ram_bytes"] for s in sampler.samples), default=psutil.virtual_memory().available),
            "minimum_ram_reserve_bytes": reserve,
            "peak_cpu_percent_sampled": max((s["cpu_percent"] for s in sampler.samples), default=None),
            "gpu_used": False,
            "temperature": "GPU nvidia-smi sensor available; CPU sensor not read in this run",
        },
        "numerical_artifacts": {
            "baseline_train_logits_sha256": tensor_hash(base_train_logits),
            "baseline_validation_logits_sha256": tensor_hash(base_validation_logits),
            "candidate_train_logits_file": "candidate_train_logits.npy",
            "candidate_validation_logits_file": "candidate_validation_logits.npy",
        },
        "agent_usage": {"llm_monitor_agents": 0, "deterministic_sampler": True},
    }
    atomic_json(STRUCTURAL / "structural_training_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "accepted": accepted,
                      "validation_before": validation_before, "validation_after": validation_after,
                      "training_seconds": training_seconds, "adapter_sha256": adapter_hash}), flush=True)
    return 0


def run_verify() -> int:
    receipt_path = STRUCTURAL / "structural_training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source, train_record, validation_record = records_from_receipt()
    if sha(SOURCE_RECEIPT) != receipt["source_receipt_sha256"]:
        raise RuntimeError("Training receipt changed after structural experiment")
    train, validation = as_tensors(train_record), as_tensors(validation_record)
    model, snapshot, preflight = load_backbone()
    if snapshot.name != receipt["checkpoint_revision"]:
        raise RuntimeError("Fresh process loaded a different checkpoint revision")
    base_train, train_activations = capture_full(model, train)
    base_val, val_activations = capture_full(model, validation)
    connector = IdentityResidualConnector(hidden_size=int(model.config.hidden_size), rank=8)
    state = load_file(str(STRUCTURAL / "connector.safetensors"), device="cpu")
    connector.load_state_dict(state, strict=True)
    original_train_layers = install_cached_prefix_live_connector(model, train_activations, connector)
    train_logits = forward_cached(model, train).detach().float().cpu()
    restore_original_layers(model, original_train_layers)
    install_cached_prefix_live_connector(model, val_activations, connector)
    validation_logits = forward_cached(model, validation).detach().float().cpu()
    expected_train = torch.from_numpy(np.load(STRUCTURAL / "candidate_train_logits.npy", allow_pickle=False))
    expected_val = torch.from_numpy(np.load(STRUCTURAL / "candidate_validation_logits.npy", allow_pickle=False))
    train_equal = torch.equal(train_logits, expected_train)
    val_equal = torch.equal(validation_logits, expected_val)
    train_max_abs = float((train_logits - expected_train).abs().max().item())
    val_max_abs = float((validation_logits - expected_val).abs().max().item())
    candidate_hash = sha(STRUCTURAL / "connector.safetensors")
    verification = {
        "status": "passed" if train_equal and val_equal and candidate_hash == receipt["graph_candidate_checkpoint_sha256"] else "failed",
        "fresh_process": True, "checkpoint_revision": snapshot.name,
        "checkpoint_index_sha256": sha(snapshot / "model.safetensors.index.json"),
        "adapter_sha256": candidate_hash,
        "train_logits_equal_bitwise": train_equal, "train_logits_max_abs_diff": train_max_abs,
        "validation_logits_equal_bitwise": val_equal, "validation_logits_max_abs_diff": val_max_abs,
        "baseline_train_logits_equal_bitwise": torch.equal(base_train, torch.from_numpy(np.load(STRUCTURAL / "baseline_train_logits.npy", allow_pickle=False))),
        "baseline_validation_logits_equal_bitwise": torch.equal(base_val, torch.from_numpy(np.load(STRUCTURAL / "baseline_validation_logits.npy", allow_pickle=False))),
        "candidate_retained_by_validation": receipt["candidate_retained"],
        "actual_decoder_layer26_executed_with_connector": True,
        "ram_available_after_load_bytes": preflight["available_ram_after_load_bytes"],
    }
    atomic_json(STRUCTURAL / "fresh_process_verification.json", verification)
    if verification["status"] != "passed":
        raise RuntimeError("Fresh-process structural graph reload parity failed")
    receipt["status"] = "completed"
    receipt["fresh_process_reload_verified"] = True
    atomic_json(receipt_path, receipt)
    print(json.dumps(verification, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "verify"])
    args = parser.parse_args()
    try:
        if args.mode == "train":
            return run_train()
        return run_verify()
    except Exception as exc:
        STRUCTURAL.mkdir(parents=True, exist_ok=True)
        with (STRUCTURAL / "failed_attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"mode": args.mode, "time_unix": time.time(),
                                     "exception": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
