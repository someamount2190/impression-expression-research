"""Inventory verified receipts and failures without manufacturing labels."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/impression_006/frozen_sets"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = ROOT / "experiments/impression_005/decision_learning_dataset.jsonl"
    verified, failures = [], []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("record_type") == "verified_candidate_outcome":
            verified.append({"record_type": row["record_type"], "workload_family": row["workload_family"],
                "target_configuration_label": row.get("target_configuration_label"),
                "measured_outcome": row["measured_outcome"], "provenance": row["provenance"],
                "eligible_for_decision_training": False, "reason": "Only one measured workload family; frozen splits empty."})
        elif row.get("record_type") == "numerical_verification_failure_and_correction":
            failures.append({"record_type": row["record_type"], "workload_family": row.get("workload_family"),
                "failed_attempt": row["failed_attempt"], "corrective_verification": row["corrective_verification"],
                "provenance": row["provenance"], "eligible_as_configuration_label": False,
                "reason": "Numerical tolerance failure/correction is not an end-to-end configuration outcome."})
    receipts = [
        ("experiments/impression_002e/streaming_training/run01/training_receipt.json", "streamed_trace_adapter_reintegration"),
        ("experiments/impression_002e/streaming_training/run02/training_receipt.json", "streamed_trace_adapter_reintegration"),
        ("experiments/impression_002e/streaming_training/run03/training_receipt.json", "streamed_trace_adapter_reintegration"),
        ("experiments/expression_growth_001/impression_training/run01/training_receipt.json", "expression_growth_training"),
        ("experiments/expression_growth_001/impression_training/run02/training_receipt.json", "expression_growth_training"),
    ]
    for rel, family in receipts:
        path = ROOT / rel
        if not path.exists(): continue
        receipt = json.loads(path.read_text(encoding="utf-8-sig"))
        status = receipt.get("status")
        item = {"record_type": "training_receipt_status", "workload_family": family,
            "status": status, "training_record_id": receipt.get("training_record_id"),
            "heldout_record_id": receipt.get("heldout_record_id"), "source_receipt": rel.replace("\\", "/"),
            "source_sha256": sha(path), "eligible_as_configuration_outcome": False,
            "interpretation": "Training/runtime durability evidence only; not a matched hardware-configuration comparison."}
        (verified if status == "passed" else failures).append(item)
    structural = ROOT / "experiments/impression_005/structural/failed_attempts.jsonl"
    if structural.exists():
        for line in structural.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            failures.append({"record_type": "structural_research_failed_attempt", "workload_family": "manual_connector_research",
                "status": row.get("status", "failed"), "exception": row.get("exception"), "cause": row.get("cause"),
                "optimizer_updates": row.get("optimizer_updates"), "source_receipt": str(structural.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": sha(structural), "eligible_as_configuration_outcome": False,
                "interpretation": "A failed implementation attempt, not a measured configuration candidate result."})
    (OUT / "verified_decision_outcomes.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in verified), encoding="utf-8")
    (OUT / "failed_attempt_inventory.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in failures), encoding="utf-8")
    counts = {"verified_candidate_cycles": len([x for x in verified if x.get("record_type") == "verified_candidate_outcome"]),
        "successful_other_training_receipts": len([x for x in verified if x.get("record_type") == "training_receipt_status"]),
        "failure_or_correction_records": len(failures),
        "decision_outcome_families": sorted({x.get("workload_family") for x in verified if x.get("record_type") == "verified_candidate_outcome"}),
        "failed_run_families": sorted({x.get("workload_family") for x in failures}),
        "counterfactuals_included_as_measurements": False, "eligible_split_family_count": 0,
        "decision_training_performed": False}
    (OUT / "outcome_inventory_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))

if __name__ == "__main__": main()
