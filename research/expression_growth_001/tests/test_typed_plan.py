import pytest

from experiments.expression_growth_001.symbolic_ir.typed_plan import (
    PlanValidationError, parse_plan,
)


def valid_plan():
    return {
        "version": 1,
        "objective": "create and verify a script",
        "intent": "execute",
        "variables": [
            {"name": "sandbox", "type": "string"},
            {"name": "exit_code", "type": "integer"},
            {"name": "run_observation", "type": "observation"},
            {"name": "macro_result", "type": "string"},
        ],
        "macros": [{"id": "remember_sandbox", "version": 1,
                    "inputs": {"sandbox_id": "string"}, "output": "string"}],
        "nodes": [
            {"id": "create", "kind": "action", "capability": "filesystem",
             "operation": "create_sandbox", "arguments": {"name": "demo"},
             "bind": {"name": "sandbox", "field": "sandbox_id"}},
            {"id": "write", "kind": "action", "depends_on": ["create"],
             "capability": "filesystem", "operation": "write_file",
             "arguments": {"sandbox_id": "$sandbox", "filename": "main.py", "content": "print(1)"}},
            {"id": "execute", "kind": "action", "depends_on": ["write"],
             "capability": "process", "operation": "execute",
             "arguments": {"sandbox_id": "$sandbox", "script_name": "main.py"},
             "bind": {"name": "exit_code", "field": "exit_code"}},
            {"id": "branch", "kind": "if", "depends_on": ["execute"],
             "condition": {"var": "exit_code", "equals": 0},
             "then": ["assert_ok"], "else": ["observe"]},
            {"id": "assert_ok", "kind": "assert", "depends_on": ["execute", "branch"],
             "condition": {"var": "exit_code", "equals": 0}},
            {"id": "observe", "kind": "observe", "depends_on": ["execute", "branch"],
             "source": "execute", "bind": "run_observation"},
            {"id": "recover", "kind": "recover", "depends_on": ["execute"],
             "handlers": {"EXECUTION_ERROR": "repair_write"}},
            {"id": "repair_write", "kind": "action", "depends_on": ["recover", "write"],
             "capability": "filesystem", "operation": "write_file",
             "arguments": {"sandbox_id": "$sandbox", "filename": "main.py", "content": "print(2)"}},
            {"id": "macro", "kind": "macro_call", "depends_on": ["create"],
             "macro_id": "remember_sandbox", "arguments": {"sandbox_id": "$sandbox"},
             "bind": "macro_result"},
            {"id": "done", "kind": "return", "depends_on": ["observe", "assert_ok"],
             "value": "$run_observation"},
        ],
    }


def test_parses_typed_actions_dependencies_observations_control_and_macros():
    plan = parse_plan(valid_plan())
    assert plan.objective == "create and verify a script"
    assert plan.intent == "execute"
    assert [node.kind for node in plan.nodes] == [
        "action", "action", "action", "if", "assert", "observe", "recover", "action", "macro_call", "return"
    ]


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["nodes"][0].update(operation="delete_all_files"), "unsupported operation"),
    (lambda p: p["nodes"][1]["arguments"].update(sandbox_id=4), "invalid value type"),
    (lambda p: p["nodes"][1]["arguments"].update(sandbox_id="$missing"), "undefined or not-yet-bound"),
    (lambda p: p["nodes"][1].update(depends_on=[]), "without depending on its producer"),
    (lambda p: p["nodes"][0].update(receipt={"status": "SUCCESS"}), "cannot supply authoritative receipts"),
    (lambda p: p["nodes"][0].update(observation={"status": "SUCCESS"}), "cannot supply authoritative receipts"),
    (lambda p: p["nodes"][1].update(capability="web", operation="execute"), "unsupported operation"),
    (lambda p: next(n for n in p["nodes"] if n["kind"] == "macro_call").update(macro_id="invented"), "not declared"),
])
def test_rejects_bad_operations_types_dependencies_receipts_and_macros(mutate, match):
    raw = valid_plan()
    mutate(raw)
    with pytest.raises(PlanValidationError, match=match):
        parse_plan(raw)


def test_rejects_untyped_operation_even_if_capability_broker_lists_it():
    raw = valid_plan()
    raw["nodes"][0].update(capability="evaluation", operation="run_test_suite")
    with pytest.raises(PlanValidationError, match="no typed OPERATION_SPECS"):
        parse_plan(raw)


def test_macro_argument_types_and_branch_references_are_checked():
    raw = valid_plan()
    next(n for n in raw["nodes"] if n["kind"] == "macro_call")["arguments"]["sandbox_id"] = 1
    with pytest.raises(PlanValidationError, match="must have type string"):
        parse_plan(raw)
    raw = valid_plan()
    next(n for n in raw["nodes"] if n["kind"] == "if")["then"] = ["unknown"]
    with pytest.raises(PlanValidationError, match="branches must reference known future nodes"):
        parse_plan(raw)


def test_condition_variable_must_be_produced_by_a_dependency():
    raw = valid_plan()
    next(n for n in raw["nodes"] if n["kind"] == "assert")["depends_on"] = []
    with pytest.raises(PlanValidationError, match="without depending on its producer"):
        parse_plan(raw)


def test_legacy_prior_node_recovery_reference_remains_parseable():
    raw = {
        "objective": "legacy feedback route", "intent": "recover",
        "variables": [],
        "nodes": [
            {"id": "write", "kind": "action", "capability": "filesystem", "operation": "write_file",
             "arguments": {"sandbox_id": "sb", "filename": "x.txt", "content": "x"}},
            {"id": "recover", "kind": "recover", "depends_on": ["write"],
             "handlers": {"EXECUTION_ERROR": "write"}},
            {"id": "done", "kind": "return", "depends_on": ["recover"], "value": "pending external review"},
        ],
    }
    assert parse_plan(raw).nodes[1].data["handlers"]["EXECUTION_ERROR"] == "write"
