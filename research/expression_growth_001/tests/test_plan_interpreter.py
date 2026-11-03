import hashlib
import json
from pathlib import Path

import pytest

from core.capability_broker import CapabilityObservation
from experiments.expression_growth_001.symbolic_ir.interpreter import (
    MacroLibrary, PlanExecutionError, PlanInterpreter,
)
from experiments.expression_growth_001.symbolic_ir.typed_plan import PlanValidationError


class FakeExecutor:
    """In-memory fake only; tests never call the CapabilityBroker or OS tools."""

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def dispatch(self, capability, operation, arguments):
        self.calls.append((capability, operation, dict(arguments)))
        if self.outcomes:
            value = self.outcomes.pop(0)
            if isinstance(value, Exception):
                raise value
            if isinstance(value, CapabilityObservation):
                return value
            status, observation = value
            return CapabilityObservation(status, capability, operation, observation, 0.0)
        return CapabilityObservation("SUCCESS", capability, operation, {"ok": True}, 0.0)


def workflow_plan():
    return {
        "objective": "take the successful branch",
        "intent": "execute",
        "variables": [{"name": "exit_code", "type": "integer"}],
        "nodes": [
            {"id": "run", "kind": "action", "capability": "process", "operation": "execute",
             "arguments": {"sandbox_id": "sb_test", "script_name": "main.py"},
             "bind": {"name": "exit_code", "field": "exit_code"}},
            {"id": "choose", "kind": "if", "depends_on": ["run"],
             "condition": {"var": "exit_code", "equals": 0}, "then": ["ok_path"], "else": ["error_path"]},
            {"id": "ok_path", "kind": "action", "depends_on": ["choose"],
             "capability": "filesystem", "operation": "list_sandboxes", "arguments": {}},
            {"id": "error_path", "kind": "action", "depends_on": ["choose"],
             "capability": "filesystem", "operation": "read_file",
             "arguments": {"sandbox_id": "sb_test", "filename": "main.py"}},
            {"id": "finish", "kind": "return", "depends_on": ["ok_path", "error_path"], "value": "branch_join"},
        ],
    }


def test_if_executes_only_the_selected_future_action_and_joins_skipped_branch():
    executor = FakeExecutor([
        ("SUCCESS", {"exit_code": 0}),
        ("SUCCESS", {"sandboxes": []}),
    ])
    result = PlanInterpreter(executor).run(workflow_plan())
    assert result.completed is True
    assert [call[1] for call in executor.calls] == ["execute", "list_sandboxes"]
    assert result.node_executions["error_path"].state == "skipped"
    assert result.return_value == "branch_join"


def test_if_false_executes_else_future_action():
    executor = FakeExecutor([
        ("SUCCESS", {"exit_code": 1}),
        ("SUCCESS", {"cleaned": True}),
    ])
    result = PlanInterpreter(executor).run(workflow_plan())
    assert result.completed is True
    assert [call[1] for call in executor.calls] == ["execute", "read_file"]
    assert result.node_executions["ok_path"].state == "skipped"


def test_async_executor_can_run_from_existing_event_loop():
    class AsyncFake:
        async def dispatch(self, capability, operation, arguments):
            return CapabilityObservation("SUCCESS", capability, operation, {"exit_code": 0}, 0.0)

    import asyncio
    result = asyncio.run(PlanInterpreter(AsyncFake()).run_async(workflow_plan()))
    assert result.completed is True


def recovery_plan(*, insert_repair=True):
    recovery_target = "repair" if insert_repair else "retry"
    nodes = [
        {"id": "run1", "kind": "action", "capability": "process", "operation": "execute",
         "arguments": {"sandbox_id": "$sandbox", "script_name": "job.py"}},
        {"id": "recover", "kind": "recover", "depends_on": ["run1"],
         "handlers": {"EXECUTION_ERROR": recovery_target}},
    ]
    if insert_repair:
        nodes.append({"id": "repair", "kind": "action", "depends_on": ["recover"],
                      "capability": "filesystem", "operation": "write_file",
                      "arguments": {"sandbox_id": "$sandbox", "filename": "job.py", "content": "print(42)"}})
        retry_deps = ["repair"]
    else:
        retry_deps = ["recover"]
    nodes.extend([
        {"id": "retry", "kind": "action", "depends_on": retry_deps,
         "capability": "process", "operation": "execute",
         "arguments": {"sandbox_id": "$sandbox", "script_name": "job.py"}},
        {"id": "observe_retry", "kind": "observe", "depends_on": ["retry"], "source": "retry",
         "bind": "retry_observation"},
        {"id": "done", "kind": "return", "depends_on": ["observe_retry"], "value": "$retry_observation"},
    ])
    return {
        "objective": "repair an execution failure and verify retry",
        "intent": "recover",
        "variables": [{"name": "sandbox", "type": "string", "source": "input"},
                       {"name": "retry_observation", "type": "observation"}],
        "nodes": nodes,
    }


def test_recovery_routes_to_future_repair_and_allows_same_execution_after_write():
    executor = FakeExecutor([
        ("EXECUTION_ERROR", {"exit_code": 1, "stderr": "bad code"}),
        ("SUCCESS", {"written": True}),
        ("SUCCESS", {"exit_code": 0, "stdout": "42"}),
    ])
    result = PlanInterpreter(executor).run(recovery_plan(), inputs={"sandbox": "sb_test"})
    assert result.completed is True
    assert [call[1] for call in executor.calls] == ["execute", "write_file", "execute"]
    assert result.runtime_results["run1"].status == "EXECUTION_ERROR"
    assert result.return_value["observation"]["stdout"] == "42"


def test_identical_failed_action_without_repair_is_not_dispatched_twice():
    executor = FakeExecutor([("EXECUTION_ERROR", {"exit_code": 1})])
    result = PlanInterpreter(executor).run(recovery_plan(insert_repair=False), inputs={"sandbox": "sb_test"})
    assert result.completed is False
    assert [call[1] for call in executor.calls] == ["execute"]
    assert "REPEATED_ACTION" in " ".join(result.unhandled_errors)


def test_legacy_prior_recovery_reference_is_non_executable_and_never_retries():
    plan = {
        "objective": "route error to external handler", "intent": "recover", "variables": [],
        "nodes": [
            {"id": "write", "kind": "action", "capability": "filesystem", "operation": "write_file",
             "arguments": {"sandbox_id": "sb", "filename": "job.py", "content": "print(1)"}},
            {"id": "run", "kind": "action", "depends_on": ["write"], "capability": "process",
             "operation": "execute", "arguments": {"sandbox_id": "sb", "script_name": "job.py"}},
            {"id": "recover", "kind": "recover", "depends_on": ["write", "run"],
             "handlers": {"EXECUTION_ERROR": "write"}},
            {"id": "done", "kind": "return", "depends_on": ["recover"], "value": "await external recovery"},
        ],
    }
    executor = FakeExecutor([("SUCCESS", {"written": True}), ("EXECUTION_ERROR", {"exit_code": 1})])
    result = PlanInterpreter(executor).run(plan)
    assert result.completed is False
    assert [call[1] for call in executor.calls] == ["write_file", "execute"]
    assert "deprecated non-executable" in result.node_executions["recover"].detail


def test_failed_assert_is_a_real_failure_and_model_receipt_cannot_complete_plan():
    raw = {
        "objective": "assert runtime result", "intent": "execute",
        "variables": [{"name": "code", "type": "integer"}],
        "nodes": [
            {"id": "run", "kind": "action", "capability": "process", "operation": "execute",
             "arguments": {"sandbox_id": "sb", "script_name": "x.py"},
             "bind": {"name": "code", "field": "exit_code"}},
            {"id": "assert", "kind": "assert", "depends_on": ["run"],
             "condition": {"var": "code", "equals": 0}},
            {"id": "done", "kind": "return", "depends_on": ["assert"], "value": "verified"},
        ],
    }
    executor = FakeExecutor([("SUCCESS", {"exit_code": 2})])
    result = PlanInterpreter(executor).run(raw)
    assert result.completed is False
    assert result.node_executions["assert"].error_class == "ASSERTION_FAILED"
    raw["nodes"][0]["observation"] = {"status": "SUCCESS", "exit_code": 0}
    with pytest.raises(PlanValidationError, match="cannot supply authoritative receipts"):
        PlanInterpreter(FakeExecutor()).run(raw)


def _macro_fixture():
    body = {
        "objective": "clamp an input", "intent": "compute",
        "variables": [{"name": "score", "type": "integer", "source": "input"},
                      {"name": "value", "type": "integer"}],
        "nodes": [
            {"id": "clamp", "kind": "action", "capability": "expression", "operation": "eval",
             "arguments": {"expr_id": "E002", "args": ["$score", 10]}, "bind": "value"},
            {"id": "return", "kind": "return", "depends_on": ["clamp"], "value": "$value"},
        ],
    }
    expansion = body["nodes"]
    digest = hashlib.sha256(json.dumps(expansion, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": 1,
        "macros": [{"id": "clamp_score", "version": 1, "status": "accepted",
                    "signature": {"inputs": {"score": "integer"}, "output": "integer"},
                    "body": body, "primitive_expansion": expansion,
                    "validation": {"primitive_expansion_equivalent": True,
                                   "expansion_sha256": digest, "validation_record_ids": ["val-a", "val-b"]}}],
    }


def test_accepted_macro_loads_checks_expansion_and_executes_through_injected_executor(tmp_path):
    path = tmp_path / "library.json"
    path.write_text(json.dumps(_macro_fixture()), encoding="utf-8")
    library = MacroLibrary.load(path)
    plan = {
        "objective": "use accepted clamp macro", "intent": "compute",
        "variables": [{"name": "score_in", "type": "integer", "source": "input"},
                      {"name": "result", "type": "integer"}],
        "macros": [{"id": "clamp_score", "version": 1,
                    "inputs": {"score": "integer"}, "output": "integer"}],
        "nodes": [
            {"id": "call", "kind": "macro_call", "macro_id": "clamp_score",
             "arguments": {"score": "$score_in"}, "bind": "result"},
            {"id": "return", "kind": "return", "depends_on": ["call"], "value": "$result"},
        ],
    }
    executor = FakeExecutor([("SUCCESS", 7)])
    result = PlanInterpreter(executor, macro_library=library).run(plan, inputs={"score_in": 13})
    assert result.completed is True
    assert result.return_value == 7
    assert executor.calls == [("expression", "eval", {"expr_id": "E002", "args": [13, 10]})]


def test_macro_library_rejects_expansion_mismatch(tmp_path):
    library = _macro_fixture()
    library["macros"][0]["primitive_expansion"] = []
    path = tmp_path / "library.json"
    path.write_text(json.dumps(library), encoding="utf-8")
    with pytest.raises(PlanValidationError, match="differs from macro body AST"):
        MacroLibrary.load(path)

