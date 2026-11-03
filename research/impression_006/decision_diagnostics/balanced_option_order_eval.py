"""Balanced option-order sensitivity diagnostic (proposal-only, no execution)."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from core.capability_broker import CapabilityBroker, CapabilityEnvelope
from experiments.impression_005.native_controller import (
    NativeImpressionBrokerAdapter, candidate_summaries, deterministic_choice,
    is_feasible, measured_profiles,
)

MODEL = "[BASE_MODEL]-3b:latest"
PARTITION = "IMP_L27_ATTENTION_QUERY_PROJECTION"
REPEATS_PER_ORDER = 2
WALL_CAP = 30
MAX_QUALITY_REGRESSION = 0.0001
SYSTEM = (
    "You are [SYSTEM_NAME] making a configuration proposal from measured evidence. Compare every listed configuration. "
    "Choose the minimum measured median wall time only among configurations whose every repeated profile passes "
    "RAM/VRAM reserve checks, reload parity, unchanged backbone integrity, and the held-out loss limit. "
    "The broker will independently recheck the proposal. Return only the requested typed JSON."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["partition.select", "partition.place"]},
        "configuration_id": {"type": "string", "enum": ["dense_cpu1", "sparse_supervised_cpu1", "sparse_supervised_cpu2", "sparse_cuda_head_cpu_suffix", "none_feasible"]},
        "partition_id": {"type": "string", "enum": [PARTITION]},
        "placement": {"type": "string", "enum": ["cpu", "cuda_lm_head_cpu_suffix", "none"]},
        "reason": {"type": "string"},
    }, "required": ["action", "configuration_id", "partition_id", "placement", "reason"],
    "additionalProperties": False,
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot() -> dict:
    mem = psutil.virtual_memory(); disk = psutil.disk_io_counters()
    cmd=["nvidia-smi","--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu","--format=csv,noheader,nounits"]
    try:
        f=[x.strip() for x in subprocess.run(cmd,capture_output=True,text=True,timeout=5,check=True).stdout.splitlines()[0].split(",")]
        gpu={"name":f[0],"vram_total_bytes":int(f[1])*1024**2,"vram_available_bytes":int(f[2])*1024**2,
             "vram_used_bytes":int(f[3])*1024**2,"gpu_utilization_percent":int(f[4]),"temperature_c":int(f[5]),"sensor_available":True}
    except Exception as exc:
        gpu={"sensor_available":False,"sensor_error":f"{type(exc).__name__}: {exc}","vram_total_bytes":0,"vram_available_bytes":0}
    return {"captured_utc":datetime.now(timezone.utc).isoformat(),"snapshot_kind":"actual_live_telemetry",
        "ram_total_bytes":int(mem.total),"ram_available_bytes":int(mem.available),"ram_used_percent":mem.percent,
        "cpu_utilization_percent":psutil.cpu_percent(interval=0.25),"logical_cpu_count":psutil.cpu_count(logical=True),
        "disk_read_bytes":int(disk.read_bytes) if disk else None,"disk_write_bytes":int(disk.write_bytes) if disk else None,**gpu}

def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())

def call_model(prompt: str) -> dict:
    req=urllib.request.Request("http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model":MODEL,"system":SYSTEM,"prompt":prompt,"stream":True,"keep_alive":0,"format":SCHEMA,
                         "options":{"num_predict":160,"temperature":0,"seed":20261103}}).encode(),
        headers={"Content-Type":"application/json"},method="POST")
    started=time.perf_counter(); chunks=[]; first=None; final={}
    with urllib.request.urlopen(req,timeout=WALL_CAP) as response:
        for raw in response:
            if not raw.strip(): continue
            item=json.loads(raw); part=item.get("response","")
            if part and first is None: first=time.perf_counter()-started
            chunks.append(part)
            if item.get("done"): final=item
    elapsed=time.perf_counter()-started; text="".join(chunks)
    try: parsed=json.loads(text); error=None
    except Exception as exc: parsed=None; error=f"{type(exc).__name__}: {exc}"
    return {"status":"complete","raw_text":text,"parsed":parsed,"parse_error":error,"wall_seconds":elapsed,
        "time_to_first_content_seconds":first,"prompt_eval_count":final.get("prompt_eval_count"),
        "eval_count":final.get("eval_count"),"stop_reason":final.get("done_reason")}

def summarize_feasible(profiles, snapshot_row, summaries, baseline_loss):
    grouped={}
    for p in profiles: grouped.setdefault(p.config_id,[]).append(p)
    out={}
    for row in summaries:
        cid=row["configuration_id"]; group=grouped[cid]
        resource=all(is_feasible(p,snapshot_row)[0] for p in group)
        quality=(float(row["validation_after_median"]) <= baseline_loss+MAX_QUALITY_REGRESSION)
        out[cid]={"resource_feasible_all_repeats":resource,"quality_eligible":quality,
                  "eligible":resource and quality,"median_wall_seconds":row["wall_seconds_median"],
                  "repeat_count":len(group)}
    feasible=[(cid,x["median_wall_seconds"]) for cid,x in out.items() if x["eligible"]]
    best=min(feasible,key=lambda p:p[1])[0] if feasible else "none_feasible"
    return out,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",required=True)
    args=ap.parse_args(); out=Path(args.run_dir).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"run directory is not empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    profiles=measured_profiles(); summaries=candidate_summaries(profiles)
    dense=next(p for p in profiles if p.config_id=="dense_cpu1")
    # The preregistered historical controller policy compares candidates to
    # the dense baseline's post-update held-out loss, not its pre-update loss.
    baseline_loss=float(dense.validation_after)
    orders=[
      ["dense_cpu1","sparse_supervised_cpu1","sparse_supervised_cpu2","sparse_cuda_head_cpu_suffix"],
      ["sparse_supervised_cpu1","sparse_supervised_cpu2","sparse_cuda_head_cpu_suffix","dense_cpu1"],
      ["sparse_supervised_cpu2","sparse_cuda_head_cpu_suffix","dense_cpu1","sparse_supervised_cpu1"],
      ["sparse_cuda_head_cpu_suffix","dense_cpu1","sparse_supervised_cpu1","sparse_supervised_cpu2"],
    ]
    by_id={row["configuration_id"]:row for row in summaries}
    registry=json.loads((ROOT/"experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json").read_text(encoding="utf-8"))
    part=registry["partitions"][PARTITION]
    broker=CapabilityBroker(); NativeImpressionBrokerAdapter(broker)
    config={"diagnostic":"balanced_option_order_prompt_sensitivity","model":MODEL,
      "prompt_order_ids":orders,"repeats_per_order":REPEATS_PER_ORDER,"max_decision_seconds":WALL_CAP,
      "temperature":0,"seed":20261103,"execution_authority":"proposal_only; broker resource checks enabled",
      "workload_source":"historical verified IMPRESSION-004 one-family profiles, used only to diagnose prompt order sensitivity",
      "counterfactual_scenarios":"none","training_or_cycle_execution":False,
      "profile_sources":{cid:[p.profile_id for p in profiles if p.config_id==cid]
                         for cid in sorted({p.config_id for p in profiles})},
      "profile_ledger_sha256":sha(ROOT/"experiments/impression_004/scheduler_decisions.jsonl"),
      "native_runner_sha256":sha(ROOT/"experiments/impression_005/run_native_decision_evaluation.py")}
    (out/"run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    events=[]; call_index=0
    for order_index,order in enumerate(orders):
      for repeat in range(1,REPEATS_PER_ORDER+1):
        call_index+=1
        prompt_snapshot=snapshot()
        feasibility,best=summarize_feasible(profiles,prompt_snapshot,summaries,baseline_loss)
        table=[{**by_id[cid],"live_resource_feasible_all_repeats":feasibility[cid]["resource_feasible_all_repeats"],
                "quality_eligible":feasibility[cid]["quality_eligible"]} for cid in order]
        prompt_payload={"hardware_snapshot":prompt_snapshot,"registered_partition":{"block_id":PARTITION,
            "component":part["component"],"source_layer":part["source_layer"],"parameter_count":part["parameter_count"]},
            "objective":{"task_id":dense.task_id,"objective_fingerprint":dense.objective_fingerprint,
                "baseline_validation_after":baseline_loss,"max_validation_regression":MAX_QUALITY_REGRESSION},
            "candidate_profiles_in_balanced_order":table,
            "constraints":{"ram_reserve_fraction":0.2,"vram_reserve_fraction":0.2,"vram_guard_bytes":256*1024**2},
            "instruction":"Select the fastest eligible configuration. If none is eligible, abstain with none_feasible."}
        prompt="Compare the measured candidates and return a typed proposal.\n"+json.dumps(prompt_payload,separators=(",",":"))
        try: proposal=call_model(prompt)
        except Exception as exc: proposal={"status":"error","raw_text":"","parsed":None,"parse_error":f"{type(exc).__name__}: {exc}","wall_seconds":None}
        raw_event={"event":"raw_model_proposal","call_index":call_index,"order_index":order_index,"repeat":repeat,
            "order":order,"origin":"local_[BASE_MODEL]","prompt":prompt,"prompt_hardware_snapshot":prompt_snapshot,
            "model":MODEL,"proposal":proposal,"saved_before_broker_processing":True}
        append_jsonl(out/"raw_model_proposals.jsonl",raw_event)
        parsed=proposal.get("parsed")
        broker_snapshot=snapshot()
        if isinstance(parsed,dict):
            args={"configuration_id":parsed.get("configuration_id"),"partition_id":PARTITION,
                "placement":parsed.get("placement"),"origin":"local_[BASE_MODEL]","resource_snapshot":broker_snapshot}
            obs=asyncio.run(broker.dispatch(CapabilityEnvelope(action="execute",capability="impression",
                operation=str(parsed.get("action","")),arguments=args))).to_dict()
            chosen=parsed.get("configuration_id")
        else:
            obs=None; chosen=None
        choice_score={"proposal_is_best_feasible_by_measured_median":chosen==best,
            "proposal_is_resource_feasible_in_prompt_snapshot":feasibility.get(chosen,{}).get("resource_feasible_all_repeats",False),
            "proposal_is_quality_eligible_in_prompt_snapshot":feasibility.get(chosen,{}).get("quality_eligible",False),
            "best_feasible_configuration_for_prompt_snapshot":best}
        broker_event={"event":"broker_validation","call_index":call_index,"proposal_configuration_id":chosen,
            "broker_snapshot":broker_snapshot,"broker_result":obs,"choice_quality_score":choice_score,
            "execution_performed":False,"fallback_performed":False,"correction_performed":False,"graph_modified":False}
        append_jsonl(out/"broker_events.jsonl",broker_event)
        events.append({"call_index":call_index,"order_index":order_index,"repeat":repeat,"order":order,
            "proposal_configuration_id":chosen,"broker_status":obs.get("status") if obs else None,
            "proposal_is_best_feasible":choice_score["proposal_is_best_feasible_by_measured_median"],
            "proposal_resource_feasible":choice_score["proposal_is_resource_feasible_in_prompt_snapshot"],
            "decision_latency_seconds":proposal.get("wall_seconds")})
        print(json.dumps(events[-1]),flush=True)
    grouped={}
    for e in events: grouped.setdefault(e["order_index"],[]).append(e)
    summary={"status":"completed","calls":len(events),"repeats_per_order":REPEATS_PER_ORDER,
        "choices_by_order":{str(i):[x["proposal_configuration_id"] for x in rows] for i,rows in grouped.items()},
        "first_position_counts":{cid:sum(e["order"][0]==cid for e in events) for cid in by_id},
        "correct_best_feasible_proposal_count":sum(e["proposal_is_best_feasible"] for e in events),
        "broker_rejection_count":sum(e["broker_status"] not in ("SUCCESS",None) for e in events),
        "execution_count":0,"counterfactual_challenge_count":0,
        "interpretation":"Prompt sensitivity diagnostic only; historical one-family profiles reused. No training-cycle performance claim.",
        "events":events}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="events"},indent=2))

if __name__=="__main__": main()
