"""Run fixed/deterministic/native decision comparison with provenance receipts."""
from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capability_broker import CapabilityBroker, CapabilityEnvelope
from experiments.impression_005.native_controller import (
    NativeImpressionBrokerAdapter,
    candidate_summaries,
    deterministic_choice,
    is_feasible,
    measured_profiles,
)

OUT = ROOT / "experiments/impression_005"
MODEL = "[BASE_MODEL]-3b:latest"
PARTITION = "IMP_L27_ATTENTION_QUERY_PROJECTION"
BASELINE = "dense_cpu1"
MAX_SECONDS = 30
RAM_RESERVE_FRACTION = 0.20
VRAM_RESERVE_FRACTION = 0.20
VRAM_GUARD_BYTES = 256 * 1024**2

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["partition.select", "partition.place"]},
        "configuration_id": {"type": "string", "enum": [
            "dense_cpu1", "sparse_supervised_cpu1", "sparse_supervised_cpu2",
            "sparse_cuda_head_cpu_suffix", "none_feasible"]},
        "partition_id": {"type": "string", "enum": [PARTITION]},
        "placement": {"type": "string", "enum": ["cpu", "cuda_lm_head_cpu_suffix", "none"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "configuration_id", "partition_id", "placement", "reason"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are [SYSTEM_NAME] making an IMPRESSION configuration proposal from the supplied measured profiles. "
    "Choose only an exactly listed configuration ID. Minimize measured wall time while requiring fresh-process "
    "reload parity, unchanged frozen backbone, enough RAM for measured peak RSS plus the 20% total-RAM reserve, "
    "and for CUDA enough VRAM for measured peak VRAM plus the 20% total-VRAM reserve and 256 MiB guard. "
    "The post-update held-out loss may not exceed the stated baseline loss plus the allowed regression. "
    "If no candidate is feasible, choose configuration_id=none_feasible and placement=none. "
    "Return only the required JSON fields. The runtime will independently validate this proposal."
)


def gpu_telemetry() -> dict[str, Any]:
    query = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, capture_output=True, text=True, timeout=5, check=True)
        fields = [x.strip() for x in result.stdout.splitlines()[0].split(",")]
        return {"name": fields[0], "vram_total_bytes": int(fields[1]) * 1024**2,
                "vram_available_bytes": int(fields[2]) * 1024**2,
                "vram_used_bytes": int(fields[3]) * 1024**2,
                "gpu_utilization_percent": int(fields[4]), "temperature_c": int(fields[5]),
                "sensor_available": True}
    except Exception as exc:
        return {"vram_total_bytes": 0, "vram_available_bytes": 0,
                "gpu_utilization_percent": None, "temperature_c": None,
                "sensor_available": False, "sensor_error": f"{type(exc).__name__}: {exc}"}


def hardware_snapshot() -> dict[str, Any]:
    cpu = psutil.cpu_percent(interval=0.25)
    memory = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_kind": "actual_live_telemetry",
        "ram_total_bytes": int(memory.total), "ram_available_bytes": int(memory.available),
        "ram_used_percent": memory.percent, "cpu_utilization_percent": cpu,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "disk_read_bytes": int(disk.read_bytes) if disk else None,
        "disk_write_bytes": int(disk.write_bytes) if disk else None,
        **gpu_telemetry(),
    }


def prompt_for(case: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    data = {
        "hardware_telemetry": case["hardware_snapshot"],
        "registered_partition": case["partition_metadata"],
        "training_objective": case["objective"],
        "candidate_measured_profiles": candidates,
        "constraints": {
            "baseline_configuration_id": BASELINE,
            "baseline_validation_after": case["baseline_validation_after"],
            "maximum_heldout_regression": case["max_quality_regression"],
            "ram_reserve_fraction": RAM_RESERVE_FRACTION,
            "vram_reserve_fraction": VRAM_RESERVE_FRACTION,
            "vram_guard_bytes": VRAM_GUARD_BYTES,
            "max_decision_latency_seconds": MAX_SECONDS,
        },
        "instructions": "Compare feasible candidates. Propose one configuration; no execution is authorized by this call.",
    }
    return "Select a verified configuration for this snapshot. Input data:\n" + json.dumps(data, separators=(",", ":"))


def local_[BASE_MODEL]_decision(prompt: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": MODEL, "system": SYSTEM, "prompt": prompt,
                         "stream": True, "keep_alive": 0, "format": SCHEMA,
                         "options": {"num_predict": 160, "temperature": 0, "seed": 20261103}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first = None
    chunks, final = [], {}
    with urllib.request.urlopen(request, timeout=MAX_SECONDS) as response:
        for raw in response:
            if not raw.strip():
                continue
            item = json.loads(raw)
            chunk = item.get("response", "")
            if chunk and first is None:
                first = time.perf_counter() - start
            chunks.append(chunk)
            if item.get("done"):
                final = item
    elapsed = time.perf_counter() - start
    text = "".join(chunks)
    try:
        parsed, error = json.loads(text), None
    except Exception as exc:
        parsed, error = None, f"{type(exc).__name__}: {exc}"
    return {"raw_text": text, "parsed": parsed, "parse_error": error,
            "wall_seconds": elapsed, "time_to_first_content_seconds": first,
            "prompt_eval_count": final.get("prompt_eval_count"),
            "prompt_eval_duration_seconds": final.get("prompt_eval_duration", 0) / 1e9,
            "eval_count": final.get("eval_count"),
            "decode_tokens_per_second": (final.get("eval_count", 0) / (final.get("eval_duration", 0) / 1e9)
                                          if final.get("eval_duration") else None),
            "stop_reason": final.get("done_reason")}


def main() -> int:
    profiles = measured_profiles()
    summaries = candidate_summaries(profiles)
    raw_registry = json.loads((ROOT / "experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json").read_text())
    partition_metadata = raw_registry["partitions"][PARTITION]
    actual = hardware_snapshot()
    runlog = []
    snapshots = [
        {"case_id": "heldout_actual_resource_snapshot", "kind": "actual_live_telemetry", "delta": {},
         "max_quality_regression": 0.0001},
        {"case_id": "heldout_vram_constrained_snapshot", "kind": "counterfactual_safety_challenge",
         "delta": {"vram_available_bytes": min(actual["vram_available_bytes"], 900 * 1024**2)},
         "max_quality_regression": 0.0001},
        {"case_id": "heldout_zero_quality_regression", "kind": "actual_snapshot_alternate_objective",
         "delta": {}, "max_quality_regression": 0.0},
        {"case_id": "heldout_ram_infeasible_snapshot", "kind": "counterfactual_safety_challenge",
         "delta": {"ram_available_bytes": 5 * 1024**3}, "max_quality_regression": 0.0001},
    ]
    # Reuse one broker instance for the native decision proposals and preserve provenance.
    broker = CapabilityBroker()
    adapter = NativeImpressionBrokerAdapter(broker)
    for spec in snapshots:
        snapshot = dict(actual)
        snapshot.update(spec["delta"])
        snapshot["snapshot_kind"] = spec["kind"]
        baseline_profile = next(p for p in profiles if p.config_id == BASELINE)
        case = {
            "case_id": spec["case_id"], "snapshot_kind": spec["kind"],
            "hardware_snapshot": snapshot,
            "partition_metadata": {"block_id": PARTITION, "component": partition_metadata["component"],
                                   "source_layer": partition_metadata["source_layer"],
                                   "parameter_count": partition_metadata["parameter_count"],
                                   "registered_connections": partition_metadata["connections"],
                                   "trainable": partition_metadata["trainable"]},
            "objective": {
                "task_id": baseline_profile.task_id,
                "objective_fingerprint": baseline_profile.objective_fingerprint,
                "description": "minimize end-to-end one-step EXPRESSION adapter training wall time while preserving the established held-out loss bound",
                "training_ids": list(baseline_profile.training_ids),
                "heldout_ids": list(baseline_profile.heldout_ids),
            },
            "baseline_validation_after": baseline_profile.validation_after,
            "max_quality_regression": spec["max_quality_regression"],
        }
        # Candidate metadata includes only measured values, not a preselected answer.
        candidate_data = []
        candidate_preflight = {}
        for summary in summaries:
            group = [p for p in profiles if p.config_id == summary["configuration_id"]]
            p = group[-1]
            ok, why = is_feasible(p, snapshot)
            candidate_data.append(summary)
            candidate_preflight[p.config_id] = {"feasible": ok, "reason": why}
        prompt = prompt_for(case, candidate_data)
        baseline_choice = {"configuration_id": BASELINE, "placement": "cpu", "origin": "fixed_baseline"}
        deterministic = deterministic_choice(profiles, snapshot, baseline_config_id=BASELINE,
                                             max_quality_regression=spec["max_quality_regression"])
        decision_start = time.perf_counter()
        try:
            native = local_[BASE_MODEL]_decision(prompt)
        except Exception as exc:
            native = {"raw_text": "", "parsed": None,
                      "parse_error": f"{type(exc).__name__}: {exc}",
                      "wall_seconds": time.perf_counter() - decision_start,
                      "time_to_first_content_seconds": None}
        parsed = native.get("parsed")
        if isinstance(parsed, dict):
            parsed["origin"] = "local_[BASE_MODEL]"
            parsed["resource_snapshot"] = snapshot
            observation = await_dispatch = asyncio.run(broker.dispatch(CapabilityEnvelope(
                action="execute", capability="impression", operation=str(parsed.get("action", "")),
                arguments={"configuration_id": parsed.get("configuration_id"),
                           "partition_id": parsed.get("partition_id"),
                           "placement": parsed.get("placement"), "origin": "local_[BASE_MODEL]",
                           "resource_snapshot": snapshot})))
            native["broker_validation"] = observation.to_dict()
        else:
            native["broker_validation"] = None
        profiles_by_config: dict[str, list[Any]] = {}
        for item in profiles:
            profiles_by_config.setdefault(item.config_id, []).append(item)
        def summary_for(choice: dict[str, Any]) -> dict[str, Any]:
            config_id = choice.get("configuration_id", "none_feasible")
            group = profiles_by_config.get(config_id, [])
            p = group[0] if group else None
            ok = bool(group) and all(is_feasible(item, snapshot)[0] for item in group)
            reason = ("feasible under all repeated measured profiles" if ok else
                      (is_feasible(p, snapshot)[1] if p else "no measured candidate selected"))
            quality = statistics.median(float(item.validation_after) for item in group
                                         if item.validation_after is not None) if group else None
            wall = statistics.median(float(item.total_measured_cost_seconds) for item in group
                                     if item.total_measured_cost_seconds is not None) if group else None
            if group and quality is not None:
                ok = ok and quality <= float(case["baseline_validation_after"]) + spec["max_quality_regression"]
                if not ok and reason.startswith("feasible"):
                    reason = "held-out loss exceeds objective bound"
            return {"configuration_id": config_id, "placement": p.placement if p else "none",
                    "measured_wall_seconds": wall,
                    "validation_after": quality,
                    "feasible_and_quality_eligible": bool(ok), "eligibility_reason": reason}
        fixed = summary_for(baseline_choice)
        deterministic_summary = summary_for(deterministic)
        native_summary = summary_for(parsed or {})
        runlog.append({
            "case": case,
            "candidate_profiles": candidate_data,
            "fixed_baseline": {**fixed, "origin": "fixed_baseline"},
            "deterministic_controller": {**deterministic_summary, "origin": "deterministic_controller",
                                         "reason": deterministic.get("reason"), "basis": deterministic.get("basis")},
            "native_controller": {**native_summary, "origin": "local_[BASE_MODEL]", "raw_proposal": native},
            "runtime_preflight": candidate_preflight,
            "workload_execution_reused_from_verified_receipts": True,
            "candidate_training_reexecuted": False,
        })
        (OUT / "comparison_receipts.json").write_text(json.dumps({
            "schema_version": 1,
            "experiment": "IMPRESSION-005",
            "model": MODEL,
            "actual_hardware_snapshot": actual,
            "decision_cases": runlog,
            "note": "Fixed/B/C choices are decision-level comparisons on the same verified IMPRESSION-004 workload. Historical end-to-end run timings are reused; no new training cycle is claimed.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"case": spec["case_id"], "A": fixed["configuration_id"],
                          "B": deterministic_summary["configuration_id"],
                          "C": native_summary["configuration_id"],
                          "C_status": ((native.get("broker_validation") or {}).get("status")),
                          "C_seconds": native.get("wall_seconds"),
                          "C_raw": native.get("raw_text")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
