"""Compare the IMPRESSION-005 decision prompt with evidence/order ablations.

No profiles or decisions are executed here. The broker independently validates
each local [BASE_MODEL] proposal against actual live hardware telemetry.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import psutil

from experiments.impression_005.native_controller import (
    NativeImpressionBrokerAdapter,
    candidate_summaries,
    measured_profiles,
)
from experiments.impression_005.run_native_decision_evaluation import (
    MODEL, PARTITION, SYSTEM, SCHEMA, local_[BASE_MODEL]_decision, prompt_for,
)
from core.capability_broker import CapabilityBroker, CapabilityEnvelope

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/impression_006/decision_diagnostics/prompt_ablation.json"
REPEATS = 2


def actual_snapshot() -> dict:
    memory = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    query = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"]
    try:
        fields = subprocess.run(query, capture_output=True, text=True, timeout=5, check=True).stdout.strip().split(",")
        gpu = {"name": fields[0].strip(), "vram_total_bytes": int(fields[1]) * 1024**2,
               "vram_available_bytes": int(fields[2]) * 1024**2, "vram_used_bytes": int(fields[3]) * 1024**2,
               "gpu_utilization_percent": int(fields[4]), "temperature_c": int(fields[5]), "sensor_available": True}
    except Exception as exc:
        gpu = {"sensor_available": False, "sensor_error": f"{type(exc).__name__}: {exc}",
               "vram_total_bytes": 0, "vram_available_bytes": 0}
    return {"captured_utc": datetime.now(timezone.utc).isoformat(), "snapshot_kind": "actual_live_telemetry",
            "ram_total_bytes": int(memory.total), "ram_available_bytes": int(memory.available),
            "ram_used_percent": memory.percent, "cpu_utilization_percent": psutil.cpu_percent(interval=0.25),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "disk_read_bytes": int(disk.read_bytes) if disk else None,
            "disk_write_bytes": int(disk.write_bytes) if disk else None, **gpu}


def run_model(prompt: str) -> dict:
    request = urllib.request.Request("http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": MODEL, "system": SYSTEM, "prompt": prompt, "stream": False,
                         "keep_alive": 0, "format": SCHEMA,
                         "options": {"num_predict": 160, "temperature": 0, "seed": 20261103}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - started
    raw = payload.get("response", "")
    try:
        parsed, error = json.loads(raw), None
    except Exception as exc:
        parsed, error = None, f"{type(exc).__name__}: {exc}"
    return {"raw_text": raw, "parsed": parsed, "parse_error": error, "wall_seconds": elapsed,
            "prompt_eval_count": payload.get("prompt_eval_count"), "eval_count": payload.get("eval_count"),
            "stop_reason": payload.get("done_reason")}


def validate(broker: CapabilityBroker, proposal: dict | None, snapshot: dict) -> dict | None:
    if not isinstance(proposal, dict):
        return None
    observation = asyncio.run(broker.dispatch(CapabilityEnvelope(action="execute", capability="impression",
        operation=str(proposal.get("action", "")), arguments={
            "configuration_id": proposal.get("configuration_id"), "partition_id": PARTITION,
            "placement": proposal.get("placement"), "origin": "local_[BASE_MODEL]",
            "resource_snapshot": snapshot,
        })))
    return observation.to_dict()


def main() -> None:
    profiles = measured_profiles()
    candidates = candidate_summaries(profiles)
    dense = next(p for p in profiles if p.config_id == "dense_cpu1")
    registry = json.loads((ROOT / "experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json").read_text(encoding="utf-8"))
    partition = registry["partitions"][PARTITION]
    snapshot = actual_snapshot()
    case = {"case_id": "prompt_ablation_actual_snapshot", "snapshot_kind": "actual_live_telemetry",
            "hardware_snapshot": snapshot,
            "partition_metadata": {"block_id": PARTITION, "component": partition["component"],
                "source_layer": partition["source_layer"], "parameter_count": partition["parameter_count"],
                "registered_connections": partition["connections"], "trainable": partition["trainable"]},
            "objective": {"task_id": dense.task_id, "objective_fingerprint": dense.objective_fingerprint,
                "description": "minimize measured one-step wall time while preserving the held-out quality bound",
                "training_ids": list(dense.training_ids), "heldout_ids": list(dense.heldout_ids)},
            "baseline_validation_after": dense.validation_before, "max_quality_regression": 0.0001}

    # Reconstruct the old candidate payload exactly, then change one presentation
    # feature at a time. The compact table preserves all measured costs/quality.
    original_prompt = prompt_for(case, candidates)
    payload = json.loads(original_prompt.split("Input data:\n", 1)[1])
    no_anchor = json.loads(json.dumps(payload))
    no_anchor["constraints"].pop("baseline_configuration_id", None)
    no_anchor["candidate_measured_profiles"] = list(reversed(no_anchor["candidate_measured_profiles"]))
    no_anchor_prompt = "Select the lowest-cost feasible measured configuration. Input data:\n" + json.dumps(no_anchor, separators=(",", ":"))

    compact_rows = []
    # Deliberately put a slower option first, so a correct answer cannot be
    # obtained by selecting the first row.
    order = ["sparse_supervised_cpu2", "dense_cpu1", "sparse_cuda_head_cpu_suffix", "sparse_supervised_cpu1"]
    candidate_by_id = {row["configuration_id"]: row for row in candidates}
    for cid in order:
        row = candidate_by_id[cid]
        compact_rows.append({"id": cid, "placement": row["placement"], "median_wall_s": row["wall_seconds_median"],
            "median_validation_loss": row["validation_after_median"], "max_rss_bytes": row["peak_rss_bytes_max"],
            "max_vram_bytes": row["peak_vram_bytes_max"], "reload_verified": row["fresh_process_reload_verified"],
            "frozen_backbone_unchanged": row["frozen_backbone_unchanged"], "repeat_count": row["repeat_count"]})
    compact_data = {"snapshot_kind": "actual_live_telemetry", "hardware": snapshot,
        "objective": "Select the fastest resource-feasible configuration whose held-out loss is <= baseline + 0.0001.",
        "baseline_validation_loss": dense.validation_before,
        "candidates_in_deliberately_mixed_order": compact_rows}
    compact_prompt = "Compare every row numerically. Reject infeasible options, then choose the eligible row with minimum median_wall_s. Return the required JSON proposal only. Data:\n" + json.dumps(compact_data, separators=(",", ":"))

    fastest_first = json.loads(json.dumps(compact_data))
    fastest_first["candidates_in_deliberately_mixed_order"] = sorted(compact_rows, key=lambda row: row["median_wall_s"])
    fastest_prompt = "Compare every row numerically. Reject infeasible options, then choose the eligible row with minimum median_wall_s. Return the required JSON proposal only. Data:\n" + json.dumps(fastest_first, separators=(",", ":"))

    variants = [("historical_prompt_reconstructed", original_prompt),
                ("baseline_anchor_removed_and_reversed", no_anchor_prompt),
                ("compact_mixed_order_comparative_table", compact_prompt),
                ("compact_fastest_first_table", fastest_prompt)]
    broker = CapabilityBroker()
    NativeImpressionBrokerAdapter(broker)
    calls = []
    for name, prompt in variants:
        for repeat in range(1, REPEATS + 1):
            try:
                output = run_model(prompt)
            except Exception as exc:
                output = {"raw_text": "", "parsed": None, "parse_error": f"{type(exc).__name__}: {exc}", "wall_seconds": None}
            proposal = output.get("parsed")
            broker_result = validate(broker, proposal, snapshot)
            calls.append({"variant": name, "repeat": repeat, "exact_prompt": prompt,
                "available_configuration_ids": [row["configuration_id"] for row in candidates],
                "available_measured_evidence": candidates, "model_proposal": output,
                "broker_decision": broker_result,
                "proposal_configuration_id": proposal.get("configuration_id") if isinstance(proposal, dict) else None,
                "broker_rejection_or_acceptance": ((broker_result or {}).get("status")),
                "execution_performed": False, "graph_modified": False})
    summary = {name: {"calls": len([x for x in calls if x["variant"] == name]),
        "proposals": [x["proposal_configuration_id"] for x in calls if x["variant"] == name],
        "broker_statuses": [x["broker_rejection_or_acceptance"] for x in calls if x["variant"] == name],
        "mean_latency_seconds": statistics.mean(x["model_proposal"]["wall_seconds"] for x in calls
            if x["variant"] == name and x["model_proposal"].get("wall_seconds") is not None)} for name, _ in variants}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema_version": 1, "experiment": "IMPRESSION-006-followup",
        "snapshot_kind": "actual_live_telemetry", "snapshot": snapshot,
        "historical_candidate_profiles_reused_for_diagnostic_only": candidates,
        "candidate_profile_source": "verified IMPRESSION-004 receipts; no new training cycles executed",
        "variants": list(summary), "summary": summary, "calls": calls,
        "interpretation": "Prompt ablations test evidence presentation and baseline/order anchoring. This is not a workload-family generalization benchmark or a training-policy result."}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
