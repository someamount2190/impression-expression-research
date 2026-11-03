"""Bounded same-backbone generation comparison; this module never executes plans.

`prepare` freezes a ten-task balanced slice. `run` is an explicit, potentially
expensive command and requires a verified adapter. No inference runs on import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sealed_balanced_30.jsonl"
SUBSET = HERE / "sealed_balanced_10.jsonl"
MANIFEST = HERE / "generation_comparison_manifest.json"
REVISION = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
ARMS = {"A": "original_[BASE_MODEL]_same_runtime_schema",
        "B": "trained_adapter_same_runtime_schema",
        "C": "trained_adapter_same_runtime_schema_plus_accepted_macros"}
SYSTEM = (
    "Produce a version 1 typed symbolic plan as JSON only. Use only the supplied capability schemas and "
    "accepted EXPRESSION primitives. Never invent an operation. Never provide authoritative observations, "
    "receipts, or completion claims. Respect permissions; abstain or ask for trusted state when required. "
    "The plan object has version, objective, intent, variables, nodes, and optionally macros. "
    "A variable has name, type, and optional source=input. Nodes use id, kind, depends_on, and kind-specific "
    "fields. Action nodes use capability, operation, arguments and optional bind. A binding is a declared "
    "variable name or {name,field}; references are $variable. Return nodes use value. Observe nodes use "
    "source and bind. Assert/if nodes use condition. Never put runtime results in the plan."
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def write_new_or_identical(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to replace frozen artifact: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def prepare() -> dict[str, Any]:
    source_hash = sha_file(SOURCE)
    original = json.loads((HERE / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if source_hash != original["subset_sha256"]:
        raise RuntimeError("frozen 30-task source hash mismatch")
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = {}
    for row in sorted(rows, key=lambda row: row["example_id"]):
        chosen.setdefault(row["family"], row)
    selected = [chosen[family] for family in sorted(chosen)]
    text = "".join(canonical(row) + "\n" for row in selected)
    write_new_or_identical(SUBSET, text)
    manifest = {
        "schema_version": 1, "experiment": "EXPRESSION-GROWTH-001",
        "frozen_before_inference": True, "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": source_hash, "subset": str(SUBSET.relative_to(REPO)),
        "subset_sha256": sha_file(SUBSET), "subset_count": len(selected),
        "selection_rule": "first lexical example_id per family from frozen sealed_balanced_30",
        "task_ids": [row["example_id"] for row in selected],
        "families": [row["family"] for row in selected], "configurations": ARMS,
        "same_system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "target_plan_exposed_to_model": False, "authoritative_execution_enabled": False,
        "checkpoint_revision": REVISION, "device": "cpu", "dtype": "bfloat16",
        "attention_implementation": "eager", "generation_seed": 20261103,
        "do_sample": False, "max_new_tokens": 512, "per_call_max_seconds": 180,
        "maximum_runtime_seconds": 5400, "maximum_calls": 30,
        "arithmetic_limit": "HF eager BF16 generation is not parity-verified against the streamed FP32-accumulation path",
        "scope": "model generation and static validation only; task completion/recovery/transfer benefit remain untested",
    }
    write_new_or_identical(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def macro_descriptors(path: Path) -> list[dict[str, Any]]:
    from experiments.expression_growth_001.symbolic_ir.interpreter import MacroLibrary
    if not path.is_file():
        raise RuntimeError("C requires a persistent accepted macro library file")
    library = MacroLibrary.load(path)  # fail closed on the discovery-only format
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for entry in raw.get("macros", []):
        if entry.get("id") in library:
            result.append({"id": entry["id"], "version": entry["version"],
                           "inputs": entry["signature"]["inputs"], "output": entry["signature"]["output"]})
    if not result:
        raise RuntimeError("C requires at least one interpreter-loadable accepted macro")
    return result


def model_messages(row: dict[str, Any], macros: list[dict[str, Any]]) -> list[dict[str, str]]:
    # Explicit allow-list: target_plan, target_sha256 and all scoring fields are excluded.
    user = f"Capability schemas: {canonical(row['available_capability_schemas'])}\nObjective: {row['prompt']}"
    if macros:
        user += "\nAccepted macro declarations: " + canonical(macros)
        user += "\nMacro calls use kind=macro_call, macro_id, arguments, and optional bind. Declare only accepted macros."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def first_choice(raw: dict[str, Any]) -> tuple[Any, ...] | None:
    nodes = raw.get("nodes")
    if not isinstance(nodes, list):
        return None
    for node in nodes:
        if isinstance(node, dict) and node.get("kind") == "action":
            return ("action", node.get("capability"), node.get("operation"))
        if isinstance(node, dict) and node.get("kind") == "macro_call":
            return ("macro_call", node.get("macro_id"))
    return ("no_action", raw.get("intent"))


def score_output(text: str, row: dict[str, Any], macros: list[dict[str, Any]]) -> dict[str, Any]:
    from experiments.expression_growth_001.symbolic_ir.typed_plan import parse_plan
    metric = {"raw_json_object": False, "raw_schema_validity": False,
              "raw_first_action_signature_match": False, "dependency_static_validity": False,
              "hallucinated_receipt_key_count": 0, "accepted_macro_call_count": 0,
              "post_validation_action_validity": None, "task_completion": None,
              "error_recovery": None, "repeated_failure_loops": None, "macro_transfer": None,
              "tool_execution_latency_ms": None, "validation_error": None}
    try:
        raw = json.loads(text)  # deliberately no extraction, repair, normalization or fallback
        if not isinstance(raw, dict):
            raise ValueError("output is not a JSON object")
        metric["raw_json_object"] = True
        def count_keys(value: Any) -> int:
            banned = {"receipt", "receipts", "authoritative_receipt", "observation", "observation_data", "evidence", "verified_facts"}
            if isinstance(value, dict):
                return sum(int(str(k).lower() in banned) + count_keys(v) for k, v in value.items())
            return sum(map(count_keys, value)) if isinstance(value, list) else 0
        metric["hallucinated_receipt_key_count"] = count_keys(raw)
        metric["raw_first_action_signature_match"] = first_choice(raw) == first_choice(row["target_plan"])
        parse_plan(raw)
        accepted = {item["id"]: item for item in macros}
        for declaration in raw.get("macros", []):
            if declaration != accepted.get(declaration.get("id")):
                raise ValueError("model declared an unavailable or mismatched macro")
        calls = [node for node in raw["nodes"] if node.get("kind") == "macro_call"]
        if any(node.get("macro_id") not in accepted for node in calls):
            raise ValueError("model called an unavailable macro")
        metric["accepted_macro_call_count"] = len(calls)
        metric["raw_schema_validity"] = True
        metric["dependency_static_validity"] = True
    except (ValueError, TypeError, KeyError) as error:
        metric["validation_error"] = str(error)
    return metric


def run(args: argparse.Namespace) -> None:
    run_started = time.monotonic()
    # Validate all provenance and safety gates before importing/loading model weights.
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha_file(SUBSET) != manifest["subset_sha256"] or sha_file(SOURCE) != manifest["source_sha256"]:
        raise RuntimeError("frozen evaluation input hash mismatch")
    if hashlib.sha256(SYSTEM.encode()).hexdigest() != manifest["same_system_prompt_sha256"]:
        raise RuntimeError("frozen system prompt hash mismatch")
    baseline = json.loads((REPO / "experiments/impression_002d/runtime_baseline.json").read_text(encoding="utf-8"))
    snapshot = Path(baseline["checkpoint"]["snapshot_path"]).resolve(strict=True)
    if snapshot.name != REVISION or sha_file(snapshot / "config.json") != baseline["checkpoint"]["config_sha256"]:
        raise RuntimeError("pinned checkpoint configuration mismatch")
    if sha_file(snapshot / "model.safetensors.index.json") != baseline["checkpoint"]["index_sha256"]:
        raise RuntimeError("pinned checkpoint index mismatch")
    trained = set(args.arms) & {"B", "C"}
    adapter_meta = None
    if trained:
        receipt = json.loads((args.adapter_run / "fresh_process_verification.json").read_text(encoding="utf-8"))
        if receipt.get("status") != "passed" or receipt.get("checkpoint_revision") != REVISION:
            raise RuntimeError("trained arms require passed fresh-process adapter verification")
        adapter_meta = json.loads((args.adapter_run / "adapter/adapter_metadata.json").read_text(encoding="utf-8"))
        if adapter_meta.get("target_module") != "model.layers.27.self_attn.q_proj" or adapter_meta.get("checkpoint_revision") != REVISION:
            raise RuntimeError("adapter target/revision mismatch")
        if sha_file(args.adapter_run / "adapter/adapter.safetensors") != receipt["adapter_file_sha256"]:
            raise RuntimeError("adapter file hash differs from passed verification")
    macros = macro_descriptors(args.macro_library) if "C" in args.arms else []
    import psutil
    required = int(psutil.virtual_memory().total * 0.2) + baseline["checkpoint"]["index_metadata"]["total_size"] + 2 * 1024**3
    if psutil.virtual_memory().available < required:
        raise RuntimeError(f"generation RAM preflight needs {required} available bytes including 20% reserve")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = {"status": "starting", "requested_arms": args.arms, "manifest_sha256": sha_file(MANIFEST),
              "subset_sha256": sha_file(SUBSET), "checkpoint_snapshot": str(snapshot),
              "cpu_threads": args.cpu_threads,
              "adapter_sha256": sha_file(args.adapter_run / "adapter/adapter.safetensors") if trained else None,
              "macro_library_sha256": sha_file(args.macro_library) if macros else None,
              "execution_enabled": False, "runtime_completion_scored": False, "provider_usage": None}
    def save() -> None:
        (args.output_dir / "run_receipt.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    save()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(manifest["generation_seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    record["model_load_seconds"] = time.monotonic() - run_started
    if psutil.virtual_memory().available < int(psutil.virtual_memory().total * 0.2):
        record["status"] = "ram_reserve_rejected_after_model_load"
        save()
        raise RuntimeError("generation model load left less than the established 20% RAM reserve")
    active = {"adapter": False}
    hook = None
    if trained:
        from safetensors.torch import load_file
        weights = load_file(str(args.adapter_run / "adapter/adapter.safetensors"), device="cpu")
        a, b = weights["lora_A"].float(), weights["lora_B"].float()
        projection = model.model.layers[27].self_attn.q_proj
        if a.shape != (adapter_meta["rank"], projection.in_features) or b.shape != (projection.out_features, adapter_meta["rank"]):
            raise RuntimeError("adapter shape mismatch")
        def add_adapter(module, inputs, output):
            if not active["adapter"]:
                return output
            delta = torch.nn.functional.linear(torch.nn.functional.linear(inputs[0].float(), a), b)
            return output + (delta * float(adapter_meta["scale"])).to(output.dtype)
        hook = projection.register_forward_hook(add_adapter)
    rows = [json.loads(line) for line in SUBSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = run_started
    limit = min(args.max_runtime_seconds, manifest["maximum_runtime_seconds"])
    completed = []
    try:
        for row in rows:
            for arm in args.arms:
                remaining = limit - (time.monotonic() - started)
                if remaining <= 0:
                    record["status"] = "runtime_budget_exhausted"
                    return
                if psutil.virtual_memory().available < int(psutil.virtual_memory().total * 0.2):
                    record["status"] = "ram_reserve_stop_before_call"
                    return
                descriptors = macros if arm == "C" else []
                messages = model_messages(row, descriptors)
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                active["adapter"] = arm in {"B", "C"}
                call_start = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                        max_new_tokens=manifest["max_new_tokens"], max_time=min(remaining, manifest["per_call_max_seconds"]),
                        use_cache=True, pad_token_id=tokenizer.eos_token_id)
                elapsed = time.monotonic() - call_start
                generated = result[0, ids.shape[1]:].tolist()
                text = tokenizer.decode(generated, skip_special_tokens=True)
                output = {"arm": arm, "example_id": row["example_id"], "family": row["family"],
                          "messages": messages, "raw_output": text, "raw_generated_token_ids": generated,
                          "input_tokens": ids.shape[1], "output_tokens": len(generated),
                          "hit_token_limit": len(generated) >= manifest["max_new_tokens"],
                          "ended_with_eos": bool(generated and generated[-1] == tokenizer.eos_token_id),
                          "model_inference_latency_ms": elapsed * 1000,
                          "end_to_end_generation_latency_ms": elapsed * 1000,
                          "latency_scope": "generation call only; no tool execution",
                          "process_rss_after_call_bytes": psutil.Process().memory_info().rss,
                          "available_ram_after_call_bytes": psutil.virtual_memory().available,
                          "metrics": score_output(text, row, descriptors)}
                with (args.output_dir / "raw_outputs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(canonical(output) + "\n")
                    stream.flush()
                completed.append({"arm": arm, "example_id": row["example_id"]})
                record["completed_calls"] = completed
                save()
        record["status"] = "generation_complete_execution_untested"
    finally:
        record["elapsed_run_seconds"] = time.monotonic() - started
        record["completed_calls"] = completed
        record["model_load_time_included_in_runtime_budget"] = True
        save()
        if hook is not None:
            hook.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--adapter-run", type=Path, default=REPO / "experiments/expression_growth_001/impression_training/run02")
    parser.add_argument("--macro-library", type=Path, default=REPO / "experiments/expression_growth_001/learned_library/library.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generation_run01")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--max-runtime-seconds", type=int, default=5400)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="PyTorch CPU intra-op threads; keep identical across all requested arms")
    args = parser.parse_args()
    if (len(args.arms) != len(set(args.arms)) or args.max_runtime_seconds <= 0
            or args.cpu_threads <= 0 or args.cpu_threads > (os.cpu_count() or 1)):
        parser.error("arms must be unique, runtime positive, and cpu_threads within the logical CPU count")
    if args.mode == "prepare":
        print(json.dumps(prepare(), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
