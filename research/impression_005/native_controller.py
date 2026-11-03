"""Bounded IMPRESSION proposals exposed through the [SYSTEM_NAME] Capability Broker."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Mapping

from core.capability_broker import CapabilityBroker, CapabilityObservation
from experiments.impression_002.controller.efficiency_controller import (
    AdaptiveEfficiencyController,
    BlockCostProfile,
    ResourceEnvelope,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "experiments/impression_001/partitions/[BASE_MODEL]3_q4_registry.json"
PROFILE_LEDGER = ROOT / "experiments/impression_004/scheduler_decisions.jsonl"
SELECTED_CONFIG = ROOT / "experiments/impression_004/selected_configuration.json"
RAM_GUARD_BYTES = 256 * 1024**2
VRAM_GUARD_BYTES = 256 * 1024**2
VRAM_RESERVE_FRACTION = 0.20


def measured_profiles() -> list[BlockCostProfile]:
    """Load only successful, fresh-process-verified IMPRESSION-004 cycle receipts."""
    ledger_rows = []
    for line in PROFILE_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") != "complete_training_cycle_profile":
            continue
        row = event.get("payload", {})
        if (row.get("status") != "success" or row.get("reload_parity_passed") is not True
                or row.get("block_id") != "IMP_L27_ATTENTION_QUERY_PROJECTION"):
            continue
        ledger_rows.append(dict(row))
    dirs = ROOT / "experiments/impression_004/partition_training_receipts"
    run_to_config = {
        "cpu1_resume01": "dense_cpu1",
        "cpu1_sparse01": "sparse_supervised_cpu1",
        "cpu2_sparse01": "sparse_supervised_cpu2",
        "cuda_head_sparse01": "sparse_cuda_head_cpu_suffix",
        "cuda_head_sparse02": "sparse_cuda_head_cpu_suffix",
    }
    raw_rows = []
    for run_name, config_id in run_to_config.items():
        run_dir = dirs / run_name
        receipt = json.loads((run_dir / "training_receipt.json").read_text(encoding="utf-8"))
        cycle = json.loads((run_dir / "training_cycle_profile.json").read_text(encoding="utf-8"))
        verify = json.loads((run_dir / "fresh_process_verification.json").read_text(encoding="utf-8"))
        if receipt.get("status") != "passed" or verify.get("status") != "passed":
            continue
        template = next((row for row in ledger_rows if row.get("config_id") == config_id), None)
        if template is None:
            raise RuntimeError(f"No verified controller profile template for {config_id}")
        row = dict(template)
        row.update({
            "profile_id": f"{cycle['run_identity_sha256']}:{run_name}",
            "config_id": config_id,
            "status": "success",
            "total_wall_seconds": float(cycle["wall_seconds"]),
            "peak_rss_bytes": int(receipt["peak_process_rss_bytes_sampled"]),
            "ram_available_before_bytes": int(receipt["minimum_available_ram_sampled_bytes"]),
            "validation_before": float(receipt["heldout"]["loss_before"]),
            "validation_after": float(receipt["heldout"]["loss_after"]),
            "training_loss": float(receipt["training"]["loss_after"]),
            "adapter_sha256": receipt["adapter"]["sha256_file"],
            "adapter_version": receipt["adapter"]["post_fingerprint_sha256"],
            "reload_parity_passed": verify.get("status") == "passed",
            "base_fingerprint_before": verify.get("frozen_backbone_hash_before", row.get("base_fingerprint_before", "")),
            "base_fingerprint_after": verify.get("frozen_backbone_hash_after", row.get("base_fingerprint_after", "")),
            "extra": {**row.get("extra", {}),
                      "minimum_system_ram_available_bytes": int(receipt["minimum_available_ram_sampled_bytes"]),
                      "cpu_threads": int(receipt.get("cpu_threads", 1)),
                      "run_directory": run_name},
        })
        if cycle.get("highest_nvidia_smi_spot_sample_memory_used_bytes") is not None:
            row["peak_vram_bytes"] = int(cycle["highest_nvidia_smi_spot_sample_memory_used_bytes"])
        row["steps"] = int(cycle.get("optimizer_steps", 1))
        raw_rows.append(row)
    if not raw_rows:
        raise RuntimeError("No successful fresh-process-verified measured profiles found")
    return [BlockCostProfile.from_dict(row) for row in raw_rows]


def candidate_summaries(profiles: list[BlockCostProfile] | None = None) -> list[dict[str, Any]]:
    profiles = profiles or measured_profiles()
    by_id: dict[str, list[BlockCostProfile]] = {}
    for profile in profiles:
        by_id.setdefault(profile.config_id, []).append(profile)
    rows = []
    for config_id, group in sorted(by_id.items()):
        wall = sorted(float(p.total_measured_cost_seconds) for p in group)
        validation = sorted(float(p.validation_after) for p in group if p.validation_after is not None)
        rss = max(int(p.peak_rss_bytes or 0) for p in group)
        vram = max(int(p.peak_vram_bytes or 0) for p in group)
        rows.append({
            "configuration_id": config_id,
            "profile_ids": [p.profile_id for p in group],
            "repeat_count": len(group),
            "placement": group[0].placement,
            "wall_seconds_median": statistics.median(wall),
            "validation_after_median": statistics.median(validation) if validation else None,
            "peak_rss_bytes_max": rss,
            "peak_vram_bytes_max": vram,
            "frozen_backbone_unchanged": group[0].base_fingerprint_before == group[0].base_fingerprint_after,
            "fresh_process_reload_verified": all(p.reload_parity_passed is True for p in group),
            "task_id": group[0].task_id,
            "objective_fingerprint": group[0].objective_fingerprint,
            "training_ids": list(group[0].training_ids),
            "heldout_ids": list(group[0].heldout_ids),
        })
    return rows


def is_feasible(profile: BlockCostProfile, snapshot: Mapping[str, Any]) -> tuple[bool, str]:
    ram_total = int(snapshot["ram_total_bytes"])
    ram_available = int(snapshot["ram_available_bytes"])
    ram_reserve = max(int(ram_total * 0.20), RAM_GUARD_BYTES)
    peak_rss = int(profile.peak_rss_bytes or 0)
    if not peak_rss or ram_available - peak_rss < ram_reserve:
        return False, "20 percent RAM reserve plus measured peak RSS would not fit"
    if profile.placement.lower().startswith(("cuda", "gpu")):
        vram_total = int(snapshot["vram_total_bytes"])
        vram_available = int(snapshot["vram_available_bytes"])
        vram_reserve = max(int(vram_total * VRAM_RESERVE_FRACTION), VRAM_GUARD_BYTES)
        peak_vram = int(profile.peak_vram_bytes or 0)
        if not peak_vram or vram_available - peak_vram < vram_reserve:
            return False, "measured peak VRAM plus the 20 percent VRAM reserve and 256 MiB guard would not fit"
    if profile.base_fingerprint_before and profile.base_fingerprint_after and profile.base_fingerprint_before != profile.base_fingerprint_after:
        return False, "frozen backbone fingerprint changed"
    if profile.reload_parity_passed is not True:
        return False, "fresh-process adapter reload parity is not verified"
    return True, "feasible under measured profile and supplied resource envelope"


class NativeImpressionBrokerAdapter:
    """Read-only, proposal-validating adapter. It does not train or mutate topology."""
    SUPPORTED = ("partition.inspect", "partition.profile", "partition.select", "partition.place")

    def __init__(self, broker: CapabilityBroker | None = None):
        self.broker = broker or CapabilityBroker.get_instance()
        self.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.profiles = measured_profiles()
        self.profiles_by_config: dict[str, list[BlockCostProfile]] = {}
        for profile in self.profiles:
            self.profiles_by_config.setdefault(profile.config_id, []).append(profile)
        # Retain a representative for read-only profile lookups. Executable
        # proposal checks below always evaluate every repeated profile.
        self.profile_by_id = {key: group[-1] for key, group in self.profiles_by_config.items()}
        self.broker.register_capability_extension("impression", list(self.SUPPORTED), self.handle)

    def handle(self, operation: str, args: dict[str, Any]) -> CapabilityObservation | dict[str, Any]:
        if operation == "partition.inspect":
            block_id = args.get("partition_id")
            partitions = self.registry.get("partitions", {})
            if block_id is None:
                return {"model": self.registry.get("model"), "partition_count": len(partitions),
                        "partitions": sorted(partitions), "structural_reorganization_enabled": False}
            if block_id not in partitions:
                return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                              f"Unknown registered partition: {block_id}")
            return {"partition": partitions[block_id], "source": "registered_[BASE_MODEL]_graph"}
        if operation == "partition.profile":
            block_id = args.get("partition_id")
            if block_id and block_id not in self.registry.get("partitions", {}):
                return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                              f"Unknown registered partition: {block_id}")
            summaries = candidate_summaries(self.profiles)
            return {"profiles": [item for item in summaries if not block_id or any(
                self.profile_by_id[cid].block_id == block_id for cid in [item["configuration_id"]]
            )], "source": "verified_impression_004_cycle_receipts"}
        if operation == "partition.select":
            return self._validate_proposal(operation, args, require_config=True)
        if operation == "partition.place":
            return self._validate_proposal(operation, args, require_config=True)
        return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                     "No adapter is registered for this operation")

    def _validate_proposal(self, operation: str, args: dict[str, Any], *, require_config: bool) -> CapabilityObservation:
        if args.get("origin") not in {"local_[BASE_MODEL]", "deterministic_controller", "fixed_baseline", "external_orchestrator"}:
            return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                         "Decision origin is missing or unrecognized")
        block_id = args.get("partition_id")
        if block_id not in self.registry.get("partitions", {}):
            return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                         "Proposal references an unregistered partition")
        config_id = args.get("configuration_id")
        snapshot = args.get("resource_snapshot")
        if config_id == "none_feasible":
            if not isinstance(snapshot, dict):
                return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                             "A measured resource snapshot is required")
            feasible = [config for config, group in self.profiles_by_config.items()
                        if all(is_feasible(profile, snapshot)[0] for profile in group)]
            if feasible:
                return CapabilityObservation("GATING_ERROR", "impression", operation,
                    {"status": "invalid_abstention", "feasible_configuration_ids": sorted(set(feasible))}, 0.0,
                    "The proposal abstained although at least one measured configuration passed resource preflight")
            return CapabilityObservation("SUCCESS", "impression", operation,
                {"status": "validated_abstention", "origin": args["origin"],
                 "configuration_id": "none_feasible", "feasible": False,
                 "execution_performed": False, "graph_modified": False}, 0.0)
        if not require_config or config_id not in self.profile_by_id:
            return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                         "Proposal references an unknown measured configuration")
        group = self.profiles_by_config[config_id]
        if any(profile.block_id != block_id for profile in group):
            return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                         "Candidate configuration is registered for a different partition")
        placements = {profile.placement for profile in group}
        if operation == "partition.place" and args.get("placement") not in placements:
            return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                         "Placement does not match the measured configuration")
        if len(placements) != 1:
            return CapabilityObservation("GATING_ERROR", "impression", operation, None, 0.0,
                                         "Repeated profiles disagree on the placement for this configuration")
        if not isinstance(snapshot, dict):
            return CapabilityObservation("VALIDATION_ERROR", "impression", operation, None, 0.0,
                                         "A measured resource snapshot is required")
        checks = [{"profile_id": profile.profile_id, "feasible": is_feasible(profile, snapshot)[0],
                   "reason": is_feasible(profile, snapshot)[1]} for profile in group]
        failed = [check for check in checks if not check["feasible"]]
        if failed:
            reason = "At least one repeated profile failed the live resource/integrity preflight"
            return CapabilityObservation("GATING_ERROR", "impression", operation,
                {"configuration_id": config_id, "feasible": False, "repeated_profile_checks": checks}, 0.0, reason)
        profile = group[-1]
        return CapabilityObservation("SUCCESS", "impression", operation,
            {"status": "validated_proposal_only", "origin": args["origin"],
             "configuration_id": config_id, "partition_id": block_id,
             "placement": profile.placement, "feasible": True,
             "checked_repeated_profile_count": len(group), "repeated_profile_checks": checks,
             "reason": "all repeated profiles passed resource and integrity preflight",
             "execution_performed": False, "graph_modified": False}, 0.0)


def deterministic_choice(profiles: list[BlockCostProfile], snapshot: Mapping[str, Any], *,
                         baseline_config_id: str, max_quality_regression: float) -> dict[str, Any]:
    feasible_ids = {p.config_id for p in profiles if is_feasible(p, snapshot)[0]}
    eligible = [p for p in profiles if p.config_id in feasible_ids]
    if baseline_config_id not in feasible_ids:
        return {"configuration_id": "none_feasible", "reason": "fixed baseline failed live resource preflight",
                "feasible_configuration_ids": sorted(feasible_ids)}
    controller = AdaptiveEfficiencyController(min_learning_gain=0.0,
                                               max_quality_regression=max_quality_regression)
    resource = ResourceEnvelope(
        ram_available_bytes=int(snapshot["ram_available_bytes"]),
        ram_reserve_bytes=max(int(snapshot["ram_total_bytes"] * 0.20), RAM_GUARD_BYTES),
        vram_available_bytes=int(snapshot["vram_available_bytes"]),
    )
    try:
        selection = controller.select(eligible, baseline_config_id=baseline_config_id, resources=resource)
        return {"configuration_id": selection.config_id, "block_id": selection.block_id,
                "basis": selection.basis, "score": selection.score, "reason": selection.reason,
                "evidence": dict(selection.evidence), "feasible_configuration_ids": sorted(feasible_ids)}
    except ValueError as exc:
        return {"configuration_id": "none_feasible", "reason": str(exc),
                "feasible_configuration_ids": sorted(feasible_ids)}
