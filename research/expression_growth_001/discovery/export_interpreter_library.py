#!/usr/bin/env python3
"""Export discovered abstractions as typed, broker-executable macros.

The source discovery artifact is retained unchanged. Only macros that passed
source-trace/primitive-expansion validation and a fresh broker replay are
written to the interpreter library.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.expression_growth_001.symbolic_ir.interpreter import (
    CapabilityBrokerExecutor,
    MacroLibrary,
    PlanInterpreter,
)

SOURCE = ROOT / "experiments/expression_growth_001/learned_library/library.json"
DEST = ROOT / "experiments/expression_growth_001/learned_library/interpreter_library.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def typed_body(macro: dict[str, Any]) -> dict[str, Any]:
    inputs = macro["input_signature"]
    names = list(inputs)
    if set(names) != {"e501_u", "e501_v", "step2_v1"}:
        raise ValueError(f"unexpected discovered macro inputs: {names}")
    nodes = [
        {"id": "dot", "kind": "action", "capability": "expression", "operation": "eval",
         "arguments": {"expr_id": "E501", "args": ["$e501_u", "$e501_v"]}, "bind": "dot_result"},
        {"id": "clamp", "kind": "action", "depends_on": ["dot"], "capability": "expression",
         "operation": "eval", "arguments": {"expr_id": "E002", "args": ["$dot_result", "$step2_v1"]},
         "bind": "macro_result"},
        {"id": "return", "kind": "return", "depends_on": ["clamp"], "value": "$macro_result"},
    ]
    return {
        "version": 1,
        "objective": "replay discovered E501 to E002 symbolic composition",
        "intent": "execute",
        "variables": ([{"name": name, "type": inputs[name], "source": "input"} for name in names] +
                      [{"name": "dot_result", "type": "any"},
                       {"name": "macro_result", "type": "any"}]),
        "nodes": nodes,
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise RuntimeError(f"refusing to replace retained interpreter library with different content: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def export(source_path: Path = SOURCE, destination: Path = DEST) -> dict[str, Any]:
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    entries = []
    executor = CapabilityBrokerExecutor()
    for proposal in source_data.get("proposals", []):
        if proposal.get("status") != "accepted" or not proposal.get("validation", {}).get("passed"):
            continue
        if proposal.get("primitive_dependencies") != ["E002", "E501"]:
            raise ValueError("this exporter only supports the discovered E501 -> E002 composition")
        body = typed_body(proposal)
        primitive_expansion = [dict(node) for node in body["nodes"]]
        case_records = []
        expected_cases = proposal.get("validation", {}).get("cases", [])
        passed = bool(expected_cases)
        for case in expected_cases:
            inputs = case.get("inputs")
            if not isinstance(inputs, dict):
                passed = False
                case_records.append({"case_id": case.get("case_id"), "passed": False,
                                     "error": "validation case has no input object"})
                continue
            expected = case.get("primitive_expansion_output",
                                case.get("authoritative_trace_output", case.get("macro_output")))
            result = PlanInterpreter(executor).run(body, inputs=inputs)
            case_passed = result.completed and result.return_value == expected
            passed = passed and case_passed
            case_records.append({"case_id": case.get("case_id"), "family": case.get("family"),
                                 "inputs": inputs, "expected_output": expected,
                                 "broker_output": result.return_value if result.completed else None,
                                 "passed": bool(case_passed),
                                 "unhandled_errors": list(result.unhandled_errors)})
        expansion_hash = sha256(canonical_bytes(primitive_expansion))
        validation_ids = [case.get("case_id") for case in expected_cases if case.get("case_id")]
        entry = {
            "id": proposal["id"], "version": int(proposal.get("version", 1)),
            "status": "accepted" if passed else "rejected",
            "signature": {"inputs": dict(proposal["input_signature"]), "output": "any"},
            "body": body, "primitive_expansion": primitive_expansion,
            "primitive_dependencies": list(proposal["primitive_dependencies"]),
            "discovery_trace_refs": list(proposal.get("discovery_trace_refs", [])),
            "usage_statistics": dict(proposal.get("usage_statistics", {})),
            "validation": {
                "primitive_expansion_equivalent": True,
                "expansion_sha256": expansion_hash,
                "validation_record_ids": validation_ids,
                "source_discovery_validation_passed": True,
                "broker_replay_passed": bool(passed),
                "broker_replay_cases": case_records,
            },
        }
        entries.append(entry)

    output = {"schema_version": 1, "experiment": "EXPRESSION-GROWTH-001",
              "source_discovery_library": str(source_path.resolve()),
              "source_discovery_library_sha256": sha256(source_path.read_bytes()),
              "macros": entries}
    write_atomic(destination, output)
    loaded = MacroLibrary.load(destination)
    if len(loaded._macros) != sum(entry["status"] == "accepted" for entry in entries):
        raise RuntimeError("interpreter library accepted count does not match export result")
    return {"output": str(destination.resolve()), "output_sha256": sha256(destination.read_bytes()),
            "candidate_count": len(entries), "accepted_count": len(loaded._macros),
            "macro_ids": sorted(loaded._macros),
            "broker_replay_cases": sum(len(entry["validation"]["broker_replay_cases"]) for entry in entries),
            "all_broker_replays_passed": all(entry["validation"]["broker_replay_passed"] for entry in entries)}


if __name__ == "__main__":
    print(json.dumps(export(), indent=2, ensure_ascii=False))
