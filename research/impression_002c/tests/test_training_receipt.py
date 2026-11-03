"""Receipt durability tests; the child process does no model work."""
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.impression_002c.training.receipt_utils import atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[3]


def test_receipt_and_immutable_logits_survive_verifier_process_exit(tmp_path):
    logits = ROOT / "experiments/impression_002b/checkpoints/streamed_lora_layer27_qproj_run02/trace_edge_0035_post_logits.npy"
    receipt_path = tmp_path / "receipts/receipt.json"
    receipt = {"kind": "legacy_training_receipt", "verification_status": "unverified",
               "logits": {"path": str(logits), "sha256": sha256_file(logits), "size_bytes": logits.stat().st_size}}
    atomic_write_json(receipt_path, receipt)

    proc = subprocess.run([sys.executable, "-c", "import os; os._exit(73)"], cwd=ROOT,
                          capture_output=True, timeout=15)
    assert proc.returncode == 73
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    ref = Path(saved["logits"]["path"])
    assert ref.is_file()
    assert ref.stat().st_size == saved["logits"]["size_bytes"]
    assert sha256_file(ref) == saved["logits"]["sha256"]
    assert saved["verification_status"] == "unverified"


def test_materialized_legacy_receipt_is_truthful_and_references_only(tmp_path):
    base = ROOT / "experiments/impression_002c/training/receipts"
    receipt = json.loads((base / "run02_legacy_training_receipt.json").read_text(encoding="utf-8"))
    assert receipt["training_update_performed"] is True
    assert receipt["adapter"]["size_bytes"] == 98488
    assert receipt["verification"]["status"] == "unverified"
    assert receipt["verification"]["last_os_exception"] == "0xC0000005"
    assert receipt["telemetry"]["gradient_l2_norm"] is None
    assert receipt["telemetry"]["backward_seconds"] is None
    assert receipt["telemetry"]["optimizer_seconds"] is None
    assert receipt["runtime_dependency_versions"]["packages"]["pandas"] == "2.3.3"
    assert receipt["runtime_dependency_versions"]["packages"]["pyarrow"] == "24.0.0"
    for sample in receipt["samples"].values():
        for phase in ("pre", "post"):
            ref = sample["logits"][phase]
            p = Path(ref["path"])
            assert p.is_file() and p.stat().st_size == ref["size_bytes"]
            assert sha256_file(p) == ref["sha256"]
            assert ref["causal_lm_cross_entropy"] > 0
    measurements = [json.loads(line) for line in (base / "resource_measurements.jsonl").read_text().splitlines()]
    assert {row["measurement_id"] for row in measurements} >= {
        "run02_training_preflight", "full_forward_cpu_run", "qproj_cpu_cuda_microbenchmark"
    }
