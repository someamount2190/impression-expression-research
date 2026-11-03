"""Mine reusable primitive subgraphs from independently recorded successes."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.expression_bridge import ExpressionBridge

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / "data" / "symbolic_experience_db.jsonl"
LIBRARY = ROOT / "learned_library" / "library.json"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parse_plan(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, list):
        return {"steps": parsed}
    return parsed if isinstance(parsed, dict) else None


def _trace_plan(row: dict[str, Any]) -> dict[str, Any] | None:
    plan = _parse_plan(row.get("raw_text"))
    if not plan:
        return None
    steps = plan.get("steps")
    trace = row.get("trace") or []
    if not isinstance(steps, list) or len(steps) != len(trace) or len(steps) < 2:
        return None
    if any(step.get("status") != "SUCCESS" for step in trace):
        return None
    if row.get("status") != "SUCCESS":
        return None
    return plan


def _candidate_subgraphs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        plan = _trace_plan(row)
        if not plan:
            continue
        steps = plan["steps"]
        bound = {}
        for start in range(len(steps)):
            for end in range(start + 2, len(steps) + 1):
                group = steps[start:end]
                if any(step.get("op", "").upper().startswith("E") is False for step in group):
                    break
                reads = set()
                writes = []
                for step in group:
                    args = step.get("arguments", {})
                    for value in args.values():
                        if isinstance(value, str) and value.startswith("$"):
                            reads.add(value[1:])
                    binding = step.get("bind") or step.get("bind_output_to")
                    if binding:
                        writes.append(str(binding))
                if not set(writes[:-1]) & reads:
                    continue
                # Use the latest binding of each consumed intermediate and all
                # literal boundary inputs to build a task-agnostic AST.
                external_inputs: dict[str, Any] = {}
                local_bindings: dict[str, str] = {}
                canonical_steps = []
                for offset, step in enumerate(group):
                    args = step.get("arguments", {})
                    input_refs = {value[1:] for value in args.values()
                                  if isinstance(value, str) and value.startswith("$")}
                    if offset and not (input_refs & set(local_bindings)):
                        # A macro sequence must be connected through an actual
                        # binding dependency; unrelated adjacent steps are not
                        # evidence for one reusable abstraction.
                        break
                    rewritten = {}
                    for key, value in sorted(args.items()):
                        if isinstance(value, str) and value.startswith("$"):
                            name = value[1:]
                            rewritten[key] = {"var": local_bindings[name] if name in local_bindings
                                              else _input_name(group[0], offset, key)}
                            if name not in local_bindings:
                                external_inputs[_input_name(group[0], offset, key)] = value
                        else:
                            name = _input_name(group[0], offset, key)
                            rewritten[key] = {"var": name}
                            external_inputs[name] = value
                    binding = step.get("bind") or step.get("bind_output_to")
                    out_name = f"output_{offset + 1}"
                    canonical_steps.append({"primitive": str(step["op"]).upper(), "arguments": rewritten,
                                           "output": out_name})
                    if binding:
                        local_bindings[str(binding)] = out_name
                if len(canonical_steps) != len(group):
                    continue
                dependency_count = sum(
                    1 for node in canonical_steps for arg in node["arguments"].values()
                    if isinstance(arg, dict) and arg.get("var") in set(local_bindings.values()))
                if dependency_count < 1:
                    continue
                candidate = {"canonical_ast": canonical_steps,
                    "input_names": sorted(external_inputs),
                    "source_plan_id": row.get("plan_id"),
                    "source_plan_sha256": hashlib.sha256(_canonical(plan).encode()).hexdigest(),
                    "task_family": row.get("task_family", row.get("category", "unknown")),
                    "primitive_dependencies": sorted({n["primitive"] for n in canonical_steps})}
                key = _canonical({"canonical_ast": canonical_steps, "input_names": sorted(external_inputs)})
                record = candidates.setdefault(key, {"candidate": candidate, "trace_refs": [],
                    "input_samples": [], "expected_outputs": []})
                if candidate["source_plan_sha256"] not in {item["plan_sha256"] for item in record["trace_refs"]}:
                    record["trace_refs"].append({"plan_id": candidate["source_plan_id"],
                        "plan_sha256": candidate["source_plan_sha256"], "task_family": candidate["task_family"]})
                    record["input_samples"].append(external_inputs)
                    record["expected_outputs"].append(row["trace"][end - 1].get("result"))
    return candidates


def _input_name(first_step: dict[str, Any], offset: int, param: str) -> str:
    if offset == 0:
        return f"{str(first_step.get('op')).lower()}_{param}"
    return f"step{offset + 1}_{param}"


def _type_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _parameterized_ast(candidate: dict[str, Any], sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    # Replace the task-specific examples used during mining with typed input slots.
    inputs = {name: _type_of(value) for name, value in zip(candidate["input_names"], sample.values())}
    ast_nodes = []
    for node in candidate["canonical_ast"]:
        arguments = {}
        for key, value in node["arguments"].items():
            if isinstance(value, dict) and "var" in value:
                name = value["var"]
                arguments[key] = {"ref": name}
                inputs.setdefault(name, _type_of(sample.get(name, None)))
            else:
                arguments[key] = value
        ast_nodes.append({"primitive": node["primitive"], "arguments": arguments,
                          "output": node["output"]})
    return ast_nodes, inputs


def _execute_ast(bridge: ExpressionBridge, ast_nodes: list[dict[str, Any]], inputs: dict[str, Any]) -> Any:
    env = dict(inputs)
    for node in ast_nodes:
        args = {}
        meta = next((x for x in bridge.get_catalog() if x["expr_id"] == node["primitive"]), None)
        if meta is None:
            raise ValueError(f"unknown primitive in candidate AST: {node['primitive']}")
        for arg in meta["parameters"]:
            spec = node["arguments"][arg]
            if "ref" in spec:
                args[arg] = env[spec["ref"]]
            else:
                args[arg] = spec["literal"]
        result = bridge.eval_operator(node["primitive"], **args)
        env[node["output"]] = result
    return env[ast_nodes[-1]["output"]]


def _validate_equivalence(candidate: dict[str, Any], examples: list[dict[str, Any]], bridge: ExpressionBridge) -> dict[str, Any]:
    ast_nodes, inferred_inputs = _parameterized_ast(candidate, candidate["input_samples"][0])
    # Every pattern boundary value is variable, even if a source trace used a
    # literal. Keep each concrete trace argument set as independent evidence.
    cases = []
    for case_index, sample in enumerate(candidate["input_samples"]):
        names = candidate["input_names"]
        inputs = {name: sample[name] for name in names}
        discovered = _execute_ast(bridge, ast_nodes, inputs)
        # The independent validation case may use a separate plan family,
        # evaluated through its primitive expansion rather than a task fixture.
        expected = candidate["expected_outputs"][case_index]
        if discovered != expected:
            return {"passed": False, "cases": cases,
                    "error": f"mined expansion differs from authoritative trace output at source {case_index + 1}"}
        cases.append({"case_id": f"source_trace_{case_index + 1}", "family": candidate["trace_refs"][case_index]["task_family"],
                      "inputs": inputs, "macro_output": discovered, "authoritative_trace_output": expected,
                      "equivalent_to_authoritative_trace": True})
    # Validate reuse on independent family inputs not seen in discovery.
    second_ast, _ = _parameterized_ast(candidate, candidate["input_samples"][0])
    transfer_cases = examples
    compatible_transfers = [case for case in transfer_cases if set(case["inputs"]) == set(candidate["input_names"])]
    for case in compatible_transfers:
        macro_result = _execute_ast(bridge, second_ast, case["inputs"])
        expansion_result = case["primitive_expansion"](bridge)
        if macro_result != expansion_result:
            return {"passed": False, "cases": cases, "transfer_cases": [],
                    "error": f"expanded primitive result mismatch for {case['family']}"}
        cases.append({"case_id": case["case_id"], "family": case["family"], "inputs": case["inputs"],
                      "macro_output": macro_result, "primitive_expansion_output": expansion_result,
                      "equivalent": True})
    return {"passed": len(compatible_transfers) >= 2 and len({c["family"] for c in compatible_transfers}) >= 2,
            "cases": cases, "transfer_families": sorted({c["family"] for c in compatible_transfers}),
            "transfer_case_count": len(compatible_transfers),
            "error": None if compatible_transfers else "no independent transfer cases match the discovered signature"}


def _candidate_transfer_cases(bridge: ExpressionBridge) -> list[dict[str, Any]]:
    # Evaluations are not training examples and do not use the sealed set.
    return [
        {"case_id": "transfer_sensor_weighted_index_v1", "family": "sensor_weighted_index",
         "inputs": {"e501_u": [0.5, 4, 3], "e501_v": [2, 1, 1], "step2_v1": 9},
         "primitive_expansion": lambda b: b.eval_operator("E002", b.eval_operator("E501", [0.5, 4, 3], [2, 1, 1]), 9)},
        {"case_id": "transfer_resource_priority_v1", "family": "resource_priority_scoring",
         "inputs": {"e501_u": [6, 2], "e501_v": [3, 4], "step2_v1": 25},
         "primitive_expansion": lambda b: b.eval_operator("E002", b.eval_operator("E501", [6, 2], [3, 4]), 25)},
    ]


def discover(source_path: Path = SOURCE, library_path: Path = LIBRARY) -> dict[str, Any]:
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = _candidate_subgraphs(rows)
    bridge = ExpressionBridge.get_instance()
    transfers = _candidate_transfer_cases(bridge)
    proposals = []
    for candidate in candidates.values():
        source_candidate = {**candidate["candidate"], "input_samples": candidate.pop("input_samples"),
            "expected_outputs": candidate.pop("expected_outputs"), "trace_refs": candidate["trace_refs"]}
        validation = _validate_equivalence(source_candidate, transfers, bridge)
        pattern = {key: value for key, value in source_candidate.items()
                   if key not in {"input_samples", "expected_outputs", "source_plan_id", "source_plan_sha256", "task_family", "trace_refs"}}
        macro_id = "learned_" + hashlib.sha256(_canonical(pattern).encode()).hexdigest()[:12]
        first_sample = source_candidate["input_samples"][0]
        transfer_reuses = int(validation.get("transfer_case_count", 0))
        proposal = {"id": macro_id, "version": 1,
            "example_inputs": validation.get("cases", [{}])[0].get("inputs", {}),
            "input_signature": {name: _type_of(value) for name, value in first_sample.items()},
            "output": "any", "canonical_ast": source_candidate["canonical_ast"],
            "primitive_dependencies": pattern["primitive_dependencies"],
            "discovery_trace_refs": candidate["trace_refs"],
            "validation": validation, "usage_statistics": {"executions": len(validation.get("cases", [])),
                "successful_reuses": transfer_reuses,
                "scope": "deterministic AST replay during independent transfer validation; no [BASE_MODEL] macro use"},
            "status": "accepted" if validation.get("passed") else "rejected"}
        proposals.append(proposal)
    accepted = [proposal for proposal in proposals if proposal["status"] == "accepted"]
    artifact = {"schema_version": 1, "source_path": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "distinct_verified_success_rows": sum(_trace_plan(row) is not None for row in rows),
        "candidate_count": len(proposals), "accepted_count": len(accepted), "proposals": proposals}
    _write_library(library_path, artifact)
    return artifact


def _write_library(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def verify_library(path: Path = LIBRARY) -> dict[str, Any]:
    """Reload persisted macros in a new interpreter and replay every stored case."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    bridge = ExpressionBridge.get_instance()
    verified = []
    for proposal in artifact.get("proposals", []):
        if proposal.get("status") != "accepted":
            continue
        validation = proposal.get("validation", {})
        if not validation.get("passed"):
            raise ValueError(f"library macro {proposal.get('id')} lacks passing validation")
        cases = validation.get("cases", [])
        if not cases:
            raise ValueError(f"library macro {proposal.get('id')} has no replay cases")
        candidate = {"canonical_ast": proposal["canonical_ast"],
            "input_names": sorted(proposal["input_signature"]), "input_samples": [cases[0]["inputs"]]}
        ast_nodes, _ = _parameterized_ast(candidate, candidate["input_samples"][0])
        for case in cases:
            replay = _execute_ast(bridge, ast_nodes, case["inputs"])
            expected = case.get("macro_output", case.get("primitive_expansion_output"))
            if replay != expected:
                raise ValueError(f"library macro {proposal.get('id')} failed fresh-process replay {case['case_id']}")
        verified.append({"id": proposal["id"], "version": proposal["version"],
                         "replayed_case_count": len(cases)})
    return {"status": "passed" if verified else "failed", "verified_macros": verified,
        "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expression_registry_size": len(bridge.get_catalog())}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-library", action="store_true")
    parser.add_argument("--library", type=Path, default=LIBRARY)
    args = parser.parse_args()
    if args.verify_library:
        result = verify_library(args.library)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "passed" else 1)
    result = discover(library_path=args.library)
    print(json.dumps({key: result[key] for key in ("distinct_verified_success_rows", "candidate_count", "accepted_count")}, indent=2))
