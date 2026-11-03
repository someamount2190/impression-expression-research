"""Injected, deterministic executor for validated EXPRESSION plans.

The interpreter never fabricates tool receipts. Action results come only from
the injected executor; model text can describe intent but cannot populate the
runtime result channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from core.capability_broker import CapabilityBroker, CapabilityEnvelope, CapabilityObservation
from core.tool_registry import ToolRegistry
from experiments.expression_growth_001.symbolic_ir.typed_plan import (
    DeterministicPlanExecutor, PlanNode, PlanValidationError, TypedPlan,
    parse_plan, value_matches_type,
)


class PlanExecutionError(RuntimeError):
    """Plan-level errors that are exposed to recover nodes by error class."""


class RepeatedActionError(PlanExecutionError):
    error_class = "REPEATED_ACTION"


@dataclass(frozen=True)
class NodeExecution:
    node_id: str
    kind: str
    state: str
    runtime_status: str | None = None
    error_class: str | None = None
    detail: str | None = None
    # This contains only data returned by the injected executor or derived
    # control-flow state. It is not an authoritative receipt object.
    value: Any = None


@dataclass(frozen=True)
class PlanExecutionResult:
    completed: bool
    return_value: Any
    node_executions: Mapping[str, NodeExecution]
    runtime_results: Mapping[str, Any]
    variables: Mapping[str, Any]
    unhandled_errors: tuple[str, ...] = ()


class CapabilityBrokerExecutor:
    """Adapter that routes validated interpreter actions through [SYSTEM_NAME]."""

    def __init__(self, broker: CapabilityBroker | None = None,
                 registry: ToolRegistry | None = None):
        self.broker = broker or CapabilityBroker.get_instance()
        self.registry = registry or ToolRegistry(broker=self.broker)

    def dispatch(self, capability: str, operation: str, arguments: Mapping[str, Any]) -> Any:
        envelope = CapabilityEnvelope(action="execute", capability=capability,
                                      operation=operation, arguments=dict(arguments))
        valid, error, _failure, _correction = self.registry.validate_envelope(envelope)
        if not valid:
            return CapabilityObservation("VALIDATION_ERROR", capability, operation, None, 0.0, error=error)
        return self.broker.dispatch(envelope)


def _resolve_dispatch(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("async executor used from an active event loop; call await PlanInterpreter.run_async()")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _runtime_status(result: Any) -> str | None:
    if isinstance(result, Mapping):
        value = result.get("status")
    else:
        value = getattr(result, "status", None)
    return str(value).upper() if value is not None else None


def _runtime_payload(result: Any) -> Any:
    if isinstance(result, Mapping):
        # Executor contract: this mapping is trusted runtime output.
        if "observation" in result:
            return result["observation"]
        return result
    return getattr(result, "observation", result)


def _authoritative_observation(result: Any) -> Any:
    """Return only an observation object supplied/serialized by the executor."""
    serializer = getattr(result, "to_dict", None)
    if callable(serializer):
        return serializer()
    return result


def _error_class(result: Any) -> str:
    status = _runtime_status(result) or "EXECUTION_ERROR"
    aliases = {
        "EXECUTION_ERROR": "EXECUTION_ERROR", "FAILED": "EXECUTION_ERROR",
        "ERROR": "EXECUTION_ERROR", "TIMEOUT": "EXECUTION_ERROR",
        "VALIDATION_ERROR": "SCHEMA_ERROR", "SCHEMA_ERROR": "SCHEMA_ERROR",
        "PERMISSION_DENIED": "PERMISSION_DENIED", "GATING_ERROR": "PERMISSION_DENIED",
    }
    return aliases.get(status, status)


def _substitute(value: Any, variables: Mapping[str, Any], where: str) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in variables:
            raise PlanExecutionError(f"unbound variable {value!r} at {where}")
        return variables[name]
    if isinstance(value, Mapping):
        return {key: _substitute(child, variables, f"{where}.{key}") for key, child in value.items()}
    if isinstance(value, list):
        return [_substitute(child, variables, f"{where}[{index}]") for index, child in enumerate(value)]
    return value


def _read_field(payload: Any, field_name: str) -> Any:
    current = payload
    for segment in field_name.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                raise PlanExecutionError(f"runtime output does not contain bound field {field_name!r}")
            current = current[segment]
        else:
            if not hasattr(current, segment):
                raise PlanExecutionError(f"runtime output does not contain bound field {field_name!r}")
            current = getattr(current, segment)
    return current


def _condition(condition: Mapping[str, Any], variables: Mapping[str, Any]) -> bool:
    if set(condition) == {"var", "equals"}:
        name = str(condition["var"]).removeprefix("$")
        if name not in variables:
            raise PlanExecutionError(f"condition references unavailable variable ${name}")
        expected = _substitute(condition["equals"], variables, "condition.equals")
        return variables[name] == expected
    if set(condition) == {"all"}:
        return all(_condition(child, variables) for child in condition["all"])
    if set(condition) == {"any"}:
        return any(_condition(child, variables) for child in condition["any"])
    if set(condition) == {"not"}:
        return not _condition(condition["not"], variables)
    raise PlanExecutionError("invalid condition AST (plan should have been validated first)")


class MacroLibrary:
    """Load accepted macro plans with checked typed signatures and expansion ASTs."""

    def __init__(self, macros: Mapping[str, Mapping[str, Any]] | None = None):
        self._macros = dict(macros or {})

    @classmethod
    def load(cls, path: Path | str) -> "MacroLibrary":
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping) or data.get("schema_version") != 1:
            raise PlanValidationError("macro library schema_version must be 1")
        entries = data.get("macros")
        if not isinstance(entries, list):
            raise PlanValidationError("macro library macros must be an array")
        macros: dict[str, Mapping[str, Any]] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise PlanValidationError(f"macro library macros[{index}] must be an object")
            macro_id = entry.get("id")
            if not isinstance(macro_id, str) or not macro_id:
                raise PlanValidationError(f"macro library macros[{index}].id is required")
            if macro_id in macros:
                raise PlanValidationError(f"duplicate accepted macro {macro_id!r}")
            if entry.get("status") != "accepted":
                continue
            macros[macro_id] = _validate_macro_entry(entry)
        return cls(macros)

    def get(self, macro_id: str) -> Mapping[str, Any] | None:
        return self._macros.get(macro_id)

    def __contains__(self, macro_id: str) -> bool:
        return macro_id in self._macros


def _validate_macro_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    macro_id = str(entry["id"])
    version = entry.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PlanValidationError(f"macro {macro_id}: version must be a positive integer")
    signature = entry.get("signature")
    if not isinstance(signature, Mapping) or not isinstance(signature.get("inputs"), Mapping):
        raise PlanValidationError(f"macro {macro_id}: typed signature.inputs is required")
    output_type = signature.get("output")
    body = entry.get("body")
    expansion = entry.get("primitive_expansion")
    if not isinstance(body, Mapping) or not isinstance(expansion, list):
        raise PlanValidationError(f"macro {macro_id}: body and primitive_expansion are required")
    parsed_body = parse_plan(body)
    if any(node.kind == "macro_call" for node in parsed_body.nodes):
        raise PlanValidationError(f"macro {macro_id}: nested macro calls are not supported")
    if not parsed_body.nodes or parsed_body.nodes[-1].kind != "return":
        raise PlanValidationError(f"macro {macro_id}: body must end with a return node")
    body_inputs = {v.name: v.type for v in parsed_body.variables if v.source == "input"}
    if body_inputs != dict(signature["inputs"]):
        raise PlanValidationError(f"macro {macro_id}: body input bindings differ from declared signature")
    if not isinstance(output_type, str):
        raise PlanValidationError(f"macro {macro_id}: signature.output is required")
    raw_return = parsed_body.nodes[-1].data["value"]
    output_declared = None
    if isinstance(raw_return, str) and raw_return.startswith("$"):
        var_name = raw_return[1:]
        output_declared = next((var.type for var in parsed_body.variables if var.name == var_name), None)
    elif raw_return is None:
        output_declared = "null"
    elif isinstance(raw_return, bool):
        output_declared = "boolean"
    elif isinstance(raw_return, int):
        output_declared = "integer"
    elif isinstance(raw_return, float):
        output_declared = "number"
    elif isinstance(raw_return, str):
        output_declared = "string"
    elif isinstance(raw_return, list):
        output_declared = "array"
    elif isinstance(raw_return, Mapping):
        output_declared = "object"
    if output_declared is None or (output_type != "any" and output_declared != output_type
                                   and not (output_declared == "integer" and output_type == "number")):
        raise PlanValidationError(f"macro {macro_id}: body return type conflicts with declared output {output_type!r}")
    canonical_expansion = [dict(node) for node in body["nodes"]]
    expansion_hash = hashlib.sha256(_canonical(expansion)).hexdigest()
    validation = entry.get("validation")
    if not isinstance(validation, Mapping) or validation.get("primitive_expansion_equivalent") is not True:
        raise PlanValidationError(f"macro {macro_id}: independent primitive expansion validation is absent")
    if expansion != canonical_expansion:
        raise PlanValidationError(f"macro {macro_id}: stored primitive expansion differs from macro body AST")
    if validation.get("expansion_sha256") != expansion_hash:
        raise PlanValidationError(f"macro {macro_id}: primitive expansion hash mismatch")
    if not validation.get("validation_record_ids"):
        raise PlanValidationError(f"macro {macro_id}: validation record references are required")
    return dict(entry)


class PlanInterpreter:
    def __init__(self, executor: DeterministicPlanExecutor, *, macro_library: MacroLibrary | None = None):
        if executor is None or not callable(getattr(executor, "dispatch", None)):
            raise TypeError("an injected deterministic executor with dispatch() is required")
        self.executor = executor
        self.macro_library = macro_library or MacroLibrary()

    def run(self, plan: Mapping[str, Any] | TypedPlan, *, inputs: Mapping[str, Any] | None = None) -> PlanExecutionResult:
        raw_plan = plan
        if isinstance(plan, TypedPlan):
            raw_plan = {
                "version": plan.version, "objective": plan.objective, "intent": plan.intent,
                "variables": [{"name": v.name, "type": v.type, "source": v.source} for v in plan.variables],
                "nodes": [{"id": node.id, "kind": node.kind, **dict(node.data)} for node in plan.nodes],
                "macros": [{"id": m.id, "version": m.version, "inputs": dict(m.inputs), "output": m.output}
                           for m in plan.macros],
            }
            parsed = parse_plan(raw_plan)
        else:
            parsed = parse_plan(plan)
        input_values = dict(inputs or {})
        input_decls = {var.name: var for var in parsed.variables if var.source == "input"}
        if set(input_values) - set(input_decls):
            raise PlanValidationError(f"undeclared runtime inputs: {sorted(set(input_values) - set(input_decls))}")
        missing = set(input_decls) - set(input_values)
        if missing:
            raise PlanValidationError(f"missing runtime inputs: {sorted(missing)}")
        for name, value in input_values.items():
            if not value_matches_type(value, input_decls[name].type):
                raise PlanValidationError(f"runtime input {name!r} must have type {input_decls[name].type}")

        nodes = {node.id: node for node in parsed.nodes}
        by_id_order = [node.id for node in parsed.nodes]
        variables = dict(input_values)
        node_values: dict[str, Any] = {}
        runtime_results: dict[str, Any] = {}
        statuses: dict[str, NodeExecution] = {}
        guard_targets: dict[str, str] = {}
        selected_targets: dict[str, bool] = {}
        handled_errors: set[str] = set()
        action_history: dict[str, dict[str, Any]] = {}
        return_value: Any = None
        returned = False

        for control_node in parsed.nodes:
            if control_node.kind == "if":
                for target in list(control_node.data.get("then", [])) + list(control_node.data.get("else", [])):
                    guard_targets[target] = control_node.id
                    selected_targets[target] = False
            elif control_node.kind == "recover":
                for target in control_node.data.get("handlers", {}).values():
                    if target in nodes and by_id_order.index(target) > by_id_order.index(control_node.id):
                        guard_targets[target] = control_node.id
                        selected_targets[target] = False

        for node in parsed.nodes:
            if returned:
                statuses[node.id] = NodeExecution(node.id, node.kind, "skipped", detail="plan_returned")
                continue
            guard = guard_targets.get(node.id)
            if guard is not None and not selected_targets.get(node.id, False):
                statuses[node.id] = NodeExecution(node.id, node.kind, "skipped", detail=f"guard {guard} did not select node")
                continue
            dep_states = [statuses.get(dep) for dep in node.data.get("depends_on", [])]
            failed_deps = [state for state in dep_states if state and state.state == "failed"]
            skipped_deps = [state for state in dep_states if state and state.state == "skipped"]
            if skipped_deps and node.kind != "return":
                statuses[node.id] = NodeExecution(node.id, node.kind, "skipped", detail="dependency_path_not_selected")
                continue
            if skipped_deps and node.kind == "return" and not any(state and state.state == "succeeded" for state in dep_states):
                statuses[node.id] = NodeExecution(node.id, node.kind, "skipped", detail="no selected branch reached return")
                continue
            if failed_deps and node.kind not in {"observe", "if", "recover"}:
                statuses[node.id] = NodeExecution(node.id, node.kind, "failed", error_class="DEPENDENCY_ERROR",
                                                  detail="a required dependency failed")
                continue

            try:
                if node.kind == "action":
                    cap = str(node.data["capability"])
                    op = str(node.data["operation"])
                    args = _substitute(node.data.get("arguments", {}), variables, f"{node.id}.arguments")
                    self._prevent_duplicate_failed_action(parsed, node, cap, op, args, action_history)
                    try:
                        result = _resolve_dispatch(self.executor.dispatch(cap, op, args))
                    except Exception as dispatch_error:
                        action_history[node.id] = {"capability": cap, "operation": op, "arguments": dict(args),
                                                   "status": "failed", "runtime_status": None}
                        raise dispatch_error
                    runtime_results[node.id] = result
                    node_values[node.id] = _runtime_payload(result)
                    runtime_status = _runtime_status(result)
                    state = "succeeded" if runtime_status in {"SUCCESS", "OK", "COMPLETED"} else "failed"
                    error_class = None if state == "succeeded" else _error_class(result)
                    action_history[node.id] = {"capability": cap, "operation": op, "arguments": dict(args),
                                               "status": state, "runtime_status": runtime_status}
                    if state == "succeeded":
                        self._bind(node, node_values[node.id], variables, parsed)
                    else:
                        self._bind_if_available(node, node_values[node.id], variables, parsed)
                    statuses[node.id] = NodeExecution(node.id, node.kind, state, runtime_status,
                                                      error_class, getattr(result, "error", None), node_values[node.id])
                elif node.kind == "observe":
                    source_id = str(node.data["source"])
                    if source_id not in runtime_results:
                        raise PlanExecutionError(f"no executor result exists for observation source {source_id!r}")
                    result = runtime_results[source_id]
                    observation = _authoritative_observation(result)
                    node_values[node.id] = observation
                    self._bind(node, observation, variables, parsed)
                    statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", _runtime_status(result), value=observation)
                elif node.kind == "assert":
                    passed = _condition(node.data["condition"], variables)
                    if not passed:
                        statuses[node.id] = NodeExecution(node.id, node.kind, "failed", error_class="ASSERTION_FAILED",
                                                          detail="plan assertion evaluated false", value=False)
                    else:
                        statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", value=True)
                elif node.kind == "if":
                    passed = _condition(node.data["condition"], variables)
                    selected = set(node.data.get("then", [])) if passed else set(node.data.get("else", []))
                    other = (set(node.data.get("else", [])) if passed else set(node.data.get("then", [])))
                    for target in selected:
                        guard_targets[target] = node.id
                        selected_targets[target] = True
                    for target in other:
                        guard_targets[target] = node.id
                        selected_targets[target] = False
                    statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", value=passed)
                elif node.kind == "recover":
                    errors = [state for state in dep_states if state and state.state == "failed"]
                    if not errors:
                        statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", detail="no_recovery_needed")
                    else:
                        handlers = node.data["handlers"]
                        match = next((error for error in reversed(errors) if error.error_class in handlers), None)
                        if match is None and "*" in handlers:
                            match = errors[-1]
                            handler_key = "*"
                        elif match is not None:
                            handler_key = str(match.error_class)
                        else:
                            statuses[node.id] = NodeExecution(node.id, node.kind, "failed", error_class="RECOVERY_UNAVAILABLE",
                                                              detail="no handler matches dependency failure")
                            continue
                        target = handlers[handler_key]
                        if by_id_order.index(target) < by_id_order.index(node.id):
                            statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded",
                                detail=f"deprecated non-executable external recovery reference {target}", value=target)
                        else:
                            selected_targets[target] = True
                            guard_targets[target] = node.id
                            handled_errors.add(match.node_id)
                            statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", detail=f"selected recovery target {target}", value=target)
                elif node.kind == "return":
                    return_value = _substitute(node.data["value"], variables, f"{node.id}.value")
                    statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", value=return_value)
                    returned = True
                elif node.kind == "macro_call":
                    return_value = self._call_macro(node, parsed, variables, runtime_results, node_values, statuses)
                    statuses[node.id] = NodeExecution(node.id, node.kind, "succeeded", value=return_value)
                    self._bind(node, return_value, variables, parsed)
                else:
                    raise PlanExecutionError(f"unsupported node kind {node.kind!r}")
            except Exception as exc:
                error_class = getattr(exc, "error_class", None) or _classify_exception(exc)
                statuses[node.id] = NodeExecution(node.id, node.kind, "failed", error_class=error_class,
                                                  detail=f"{type(exc).__name__}: {exc}")

        unhandled = tuple(f"{node_id}:{state.error_class}" for node_id, state in statuses.items()
                          if state.state == "failed" and node_id not in handled_errors)
        completed = returned and not unhandled
        return PlanExecutionResult(completed, return_value, dict(statuses), dict(runtime_results),
                                   dict(variables), unhandled)

    async def run_async(self, plan: Mapping[str, Any] | TypedPlan, *,
                        inputs: Mapping[str, Any] | None = None) -> PlanExecutionResult:
        """Run without blocking an existing asyncio loop; broker calls remain injectable."""
        return await asyncio.to_thread(self.run, plan, inputs=inputs)

    def _bind(self, node: PlanNode, payload: Any, variables: dict[str, Any], plan: TypedPlan) -> None:
        binding = node.data.get("bind")
        if binding is None:
            return
        if isinstance(binding, Mapping):
            name = str(binding["name"])
            value = _read_field(payload, str(binding["field"]))
        else:
            name = str(binding)
            value = payload
        declaration = next(var for var in plan.variables if var.name == name)
        if not value_matches_type(value, declaration.type):
            raise PlanExecutionError(f"bound result for {name!r} does not match declared type {declaration.type!r}")
        variables[name] = value

    def _bind_if_available(self, node: PlanNode, payload: Any, variables: dict[str, Any], plan: TypedPlan) -> None:
        binding = node.data.get("bind")
        if binding is None:
            return
        try:
            self._bind(node, payload, variables, plan)
        except PlanExecutionError:
            # Failures often have no normal output fields; the error class and
            # runtime result remain available to observe/recover nodes instead.
            return

    def _prevent_duplicate_failed_action(self, plan: TypedPlan, node: PlanNode, capability: str,
                                         operation: str, arguments: Mapping[str, Any],
                                         history: Mapping[str, Mapping[str, Any]]) -> None:
        signature = _canonical([capability, operation, arguments])
        current_ancestors = _ancestor_ids(plan, node.id)
        positions = {item.id: index for index, item in enumerate(plan.nodes)}
        for old_id, old in history.items():
            if old_id not in current_ancestors or old.get("status") != "failed":
                continue
            old_signature = _canonical([old["capability"], old["operation"], old["arguments"]])
            if old_signature != signature:
                continue
            # An identical operation is only a valid retry when a successful
            # filesystem write repaired the exact script in the same sandbox.
            if capability == "process" and operation == "execute":
                sandbox = arguments.get("sandbox_id")
                script = arguments.get("script_name")
                repaired = False
                for candidate_id, candidate in history.items():
                    if (candidate_id not in current_ancestors or candidate.get("status") != "succeeded"
                            or positions.get(candidate_id, -1) <= positions.get(old_id, -1)
                            or positions.get(candidate_id, -1) >= positions.get(node.id, -1)):
                        continue
                    if candidate.get("capability") == "filesystem" and candidate.get("operation") == "write_file":
                        repair_args = candidate.get("arguments", {})
                        if repair_args.get("sandbox_id") == sandbox and repair_args.get("filename") == script:
                            repaired = True
                if repaired:
                    return
            raise RepeatedActionError(f"identical failed action {capability}.{operation} cannot be repeated without a registered repair")

    def _call_macro(self, node: PlanNode, plan: TypedPlan, variables: Mapping[str, Any],
                    runtime_results: dict[str, Any], node_values: dict[str, Any],
                    statuses: dict[str, NodeExecution]) -> Any:
        macro_id = str(node.data["macro_id"])
        entry = self.macro_library.get(macro_id)
        if entry is None:
            raise PlanValidationError(f"macro {macro_id!r} is not accepted or unavailable")
        signature = next(m for m in plan.macros if m.id == macro_id)
        if entry["version"] != signature.version:
            raise PlanValidationError(f"macro {macro_id!r} version differs from the plan signature")
        library_sig = entry["signature"]
        if dict(library_sig["inputs"]) != dict(signature.inputs) or library_sig["output"] != signature.output:
            raise PlanValidationError(f"macro {macro_id!r} signature differs from accepted library")
        args = _substitute(node.data.get("arguments", {}), variables, f"{node.id}.arguments")
        nested = PlanInterpreter(self.executor, macro_library=self.macro_library).run(entry["body"], inputs=args)
        if not nested.completed:
            raise PlanExecutionError(f"macro {macro_id!r} failed: {nested.unhandled_errors}")
        output_type = str(entry["signature"]["output"])
        if not value_matches_type(nested.return_value, output_type):
            raise PlanExecutionError(f"macro {macro_id!r} return value does not match declared type {output_type!r}")
        for nested_id, result in nested.runtime_results.items():
            runtime_results[f"{node.id}::{nested_id}"] = result
        return nested.return_value


def _classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "permission" in name:
        return "PERMISSION_DENIED"
    if "validation" in name or "schema" in name:
        return "SCHEMA_ERROR"
    if "assert" in name:
        return "ASSERTION_FAILED"
    if "dependency" in name:
        return "DEPENDENCY_ERROR"
    return "EXECUTION_ERROR"


def _ancestor_ids(plan: TypedPlan, node_id: str) -> set[str]:
    by_id = {node.id: node for node in plan.nodes}
    found: set[str] = set()
    todo = list(by_id[node_id].data.get("depends_on", []))
    while todo:
        item = todo.pop()
        if item in found:
            continue
        found.add(item)
        todo.extend(by_id[item].data.get("depends_on", []))
    return found


