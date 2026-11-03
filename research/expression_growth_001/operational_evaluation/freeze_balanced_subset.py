"""Freeze a balanced partial evaluation manifest from the sealed 100-task set."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "curriculum/sealed_evaluation/manifest.jsonl"
OUT = ROOT / "operational_evaluation/sealed_balanced_30.jsonl"
SUMMARY = ROOT / "operational_evaluation/evaluation_manifest.json"
PER_FAMILY = 3


def main() -> dict:
    raw = SEALED.read_bytes()
    sealed_hash = hashlib.sha256(raw).hexdigest()
    hash_manifest = json.loads((ROOT / "curriculum/hash_manifest.json").read_text(encoding="utf-8"))
    if sealed_hash != hash_manifest["sealed_evaluation_manifest_sha256"]:
        raise RuntimeError("sealed evaluation input hash mismatch")
    groups = defaultdict(list)
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[row["family"]].append(row)
    selected = []
    for family in sorted(groups):
        values = sorted(groups[family], key=lambda item: item["example_id"])
        if len(values) < PER_FAMILY:
            raise ValueError(f"sealed family {family} lacks {PER_FAMILY} examples")
        selected.extend(values[:PER_FAMILY])
    if len(selected) != 30 or len({row["family"] for row in selected}) != 10:
        raise AssertionError("balanced subset must contain exactly 3 examples across each of 10 families")
    content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                     for row in selected).encode("utf-8")
    subset_hash = hashlib.sha256(content).hexdigest()
    if OUT.exists() and OUT.read_bytes() != content:
        raise RuntimeError("refusing to change frozen balanced evaluation subset")
    OUT.write_bytes(content)
    manifest = {"sealed_evaluation_sha256": sealed_hash, "subset_sha256": subset_hash,
        "subset_count": len(selected), "per_family_count": PER_FAMILY,
        "families": sorted(groups), "selection_rule": "first three example_id values per family, lexical order",
        "configurations": ["base_raw", "base_prompt_only", "trained_adapter_prompt_only"],
        "frozen_before_inference": True, "target_plan_exposed_to_model": False}
    if SUMMARY.exists():
        old = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("evaluation manifest was already frozen with different contents")
    SUMMARY.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
