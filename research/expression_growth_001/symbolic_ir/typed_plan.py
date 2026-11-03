"""Typed, side-effect-free EXPRESSION plan parser and validator.

This module validates plans against [SYSTEM_NAME]'s live capability and operation
registries. It deliberately does not execute plans or accept model-authored
execution receipts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from core.capability_broker import CapabilityBroker
from core.tool_registry import OPERATION_SPECS, ToolOperationSpec


class PlanValidationError(ValueError):
    """Raised when a symbolic plan is malformed or unsafe to dispatch."""


_JSON_TYPES = {"any", "null", "boolean", "integer", "number", "string", "array", "object", "observation"}
_FORBIDDEN_RECEIPT_KEYS = {
    "receipt", "receipts", "authoritative_receipt", "authoritative_receipts",
    "observation", "observation_data", "authoritative_observation",
    "verified_facts", "execution_receipt", "verification_receipt", "evidence",
}
_NODE_KINDS = {"action", "observe", "assert", "if", "recover", "return", "macro_call"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def value_matches_type(value: Any, type_name: str) -> bool:
    """JSON-compatible type check; integer and number follow Python numeric rules."""
    checks = {
        "any": lambda _: True,
        "null": lambda v: v is None,
        "boolean": lambda v: isinstance(v, bool),
        "integer": _is_int,
        "number": lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)),
        "string": lambda v: isinstance(v, str),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "observation": lambda v: isinstance(v, dict),
    }
    return type_name in checks and checks[type_name](value)


def _python_type_names(expected: type | tuple[type, ...]) -> set[str]:
    values = expected if isinstance(expected, tuple) else (expected,)
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
    return {mapping[t] for t in values if t in mapping}


def _compatible(actual: str, expected: type | tuple[type, ...]) -> bool:
    allowed = _python_type_names(expected)
    return (actual == "any" or actual in allowed or
            (actual == "integer" and "number" in allowed) or
            (actual == "observation" and "object" in allowed))


@dataclass(frozen=True)
class VariableDecl:
    name: str
    type: str
    source: str = "node"


@dataclass(frozen=True)
class PlanNode:
    id: str
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MacroSignature:
    id: str
    version: int
    inputs: Mapping[str, str]
    output: str


@dataclass(frozen=True)
class TypedPlan:
    objective: str
    intent: str
    variables: tuple[VariableDecl, ...]
    nodes: tuple[PlanNode, ...]
    macros: tuple[MacroSignature, ...] = ()
    version: int = 1

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "TypedPlan":
        return parse_plan(raw)


@runtime_checkable
class DeterministicPlanExecutor(Protocol):
    """Trusted runtime boundary. Implementations own authorization and receipts."""

    def dispatch(self, capability: str, operation: str, arguments: Mapping[str, Any]) -> Any:
        """Dispatch one already-validated action and return runtime-owned data."""
        ...


def _check_no_receipts(value: Any, path: str = "plan") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_RECEIPT_KEYS:
                raise PlanValidationError(f"{path}.{key}: model plans cannot supply authoritative receipts or evidence")
            _check_no_receipts(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_no_receipts(child, f"{path}[{index}]")


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlanValidationError(f"{where} must be an object")
    return value


def _require_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{where} must be a non-empty string")
    return value.strip()


def parse_plan(raw: Mapping[str, Any]) -> TypedPlan:
    """Parse and statically validate a JSON-like plan against live registries."""
    raw = _require_mapping(raw, "plan")
    _check_no_receipts(raw)
    objective = _require_name(raw.get("objective"), "objective")
    intent = _require_name(raw.get("intent"), "intent")
    version = raw.get("version", 1)
    if not _is_int(version) or version != 1:
        raise PlanValidationError("version must be integer 1")

    variables: dict[str, VariableDecl] = {}
    raw_vars = raw.get("variables", [])
    if not isinstance(raw_vars, list):
        raise PlanValidationError("variables must be an array")
    for index, item in enumerate(raw_vars):
        item = _require_mapping(item, f"variables[{index}]")
        name = _require_name(item.get("name"), f"variables[{index}].name")
        type_name = _require_name(item.get("type"), f"variables[{index}].type").lower()
        source = _require_name(item.get("source", "node"), f"variables[{index}].source").lower()
        if name in variables:
            raise PlanValidationError(f"duplicate variable declaration {name!r}")
        if type_name not in _JSON_TYPES:
            raise PlanValidationError(f"variables[{index}].type must be one of {sorted(_JSON_TYPES)}")
        if source not in {"node", "input"}:
            raise PlanValidationError(f"variables[{index}].source must be 'node' or 'input'")
        variables[name] = VariableDecl(name, type_name, source)

    macros: dict[str, MacroSignature] = {}
    raw_macros = raw.get("macros", [])
    if not isinstance(raw_macros, list):
        raise PlanValidationError("macros must be an array")
    for index, item in enumerate(raw_macros):
        item = _require_mapping(item, f"macros[{index}]")
        macro_id = _require_name(item.get("id"), f"macros[{index}].id")
        macro_version = item.get("version", 1)
        if not _is_int(macro_version) or macro_version < 1:
            raise PlanValidationError(f"macros[{index}].version must be a positive integer")
        inputs_raw = item.get("inputs", {})
        if not isinstance(inputs_raw, Mapping):
            raise PlanValidationError(f"macros[{index}].inputs must be an object of input names to types")
        inputs: dict[str, str] = {}
        for name, type_name in inputs_raw.items():
            name = _require_name(name, f"macros[{index}].inputs key")
            type_name = _require_name(type_name, f"macros[{index}].inputs.{name}").lower()
            if type_name not in _JSON_TYPES:
                raise PlanValidationError(f"invalid macro input type {type_name!r}")
            inputs[name] = type_name
        output = _require_name(item.get("output"), f"macros[{index}].output").lower()
        if output not in _JSON_TYPES:
            raise PlanValidationError(f"invalid macro output type {output!r}")
        if macro_id in macros:
            raise PlanValidationError(f"duplicate macro declaration {macro_id!r}")
        macros[macro_id] = MacroSignature(macro_id, macro_version, inputs, output)

    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise PlanValidationError("nodes must be a non-empty array")
    nodes: list[PlanNode] = []
    ids: set[str] = set()
    produced: dict[str, str] = {name: decl.type for name, decl in variables.items() if decl.source == "input"}
    binding_nodes: dict[str, str] = {}
    ancestors: dict[str, set[str]] = {}
    previous_ids: set[str] = set()
    raw_node_ids = [_require_name(_require_mapping(node, f"nodes[{i}]").get("id"), f"nodes[{i}].id")
                    for i, node in enumerate(raw_nodes)]
    if len(set(raw_node_ids)) != len(raw_node_ids):
        raise PlanValidationError("node IDs must be unique")
    node_positions = {node_id: index for index, node_id in enumerate(raw_node_ids)}
    guard_targets: dict[str, str] = {}

    def check_refs(value: Any, where: str) -> None:
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            if not var_name or var_name not in produced:
                raise PlanValidationError(f"{where} references undefined or not-yet-bound variable {value!r}")
        elif isinstance(value, Mapping):
            for child_key, child_value in value.items():
                check_refs(child_value, f"{where}.{child_key}")
        elif isinstance(value, list):
            for child_index, child_value in enumerate(value):
                check_refs(child_value, f"{where}[{child_index}]")

    def result_binding(node: Mapping[str, Any], node_id: str, result_type: str,
                       capability: str | None = None, operation: str | None = None) -> None:
        binding = node.get("bind")
        if binding is None:
            return
        field_type = result_type
        if isinstance(binding, Mapping):
            bind_obj = binding
            binding = _require_name(bind_obj.get("name"), f"node {node_id}.bind.name")
            field_name = _require_name(bind_obj.get("field"), f"node {node_id}.bind.field")
            field_type = _operation_field_type(capability, operation, field_name)
            if "type" in bind_obj and str(bind_obj["type"]).lower() != field_type:
                raise PlanValidationError(f"node {node_id}.bind.type conflicts with registered output field type {field_type!r}")
        else:
            binding = _require_name(binding, f"node {node_id}.bind")
        if binding not in variables:
            raise PlanValidationError(f"node {node_id} binds undeclared variable {binding!r}")
        if variables[binding].source == "input":
            raise PlanValidationError(f"node {node_id} cannot overwrite input variable {binding!r}")
        if field_type != "any" and not _compatible(field_type, _type_spec(variables[binding].type)):
            raise PlanValidationError(f"node {node_id} result type {field_type!r} conflicts with variable {binding!r} type {variables[binding].type!r}")
        produced[binding] = field_type
        binding_nodes[binding] = node_id

    for index, source in enumerate(raw_nodes):
        source = _require_mapping(source, f"nodes[{index}]")
        node_id = _require_name(source.get("id"), f"nodes[{index}].id")
        kind = _require_name(source.get("kind"), f"nodes[{index}].kind").lower()
        if node_id in ids:
            raise PlanValidationError(f"duplicate node id {node_id!r}")
        if kind not in _NODE_KINDS:
            raise PlanValidationError(f"node {node_id}: unsupported kind {kind!r}")
        deps = source.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            raise PlanValidationError(f"node {node_id}.depends_on must be an array of node IDs")
        if len(set(deps)) != len(deps):
            raise PlanValidationError(f"node {node_id}.depends_on contains duplicates")
        unknown_deps = set(deps) - previous_ids
        if unknown_deps:
            raise PlanValidationError(f"node {node_id} has forward or unknown dependencies: {sorted(unknown_deps)}")
        data = {k: v for k, v in source.items() if k not in {"id", "kind", "depends_on"}}
        check_refs(data, f"node {node_id}")
        node_ancestors = set(deps)
        for dep in deps:
            node_ancestors.update(ancestors.get(dep, set()))
        for ref in _variable_refs(data) | _condition_refs(data):
            producer = binding_nodes.get(ref)
            if producer and producer not in node_ancestors:
                raise PlanValidationError(f"node {node_id} uses ${ref} without depending on its producer node {producer!r}")

        result_type = "any"
        if kind == "action":
            capability = _require_name(data.get("capability"), f"node {node_id}.capability").lower()
            operation = _require_name(data.get("operation"), f"node {node_id}.operation").lower()
            if operation not in CapabilityBroker.CAPABILITIES.get(capability, ()):
                raise PlanValidationError(f"node {node_id}: unsupported operation {capability}.{operation}")
            spec = OPERATION_SPECS.get((capability, operation))
            if spec is None:
                raise PlanValidationError(f"node {node_id}: {capability}.{operation} has no typed OPERATION_SPECS contract")
            arguments = data.get("arguments", {})
            if not isinstance(arguments, Mapping):
                raise PlanValidationError(f"node {node_id}.arguments must be an object")
            _validate_arguments(arguments, spec, node_id, produced, variables)
            result_type = _operation_result_type(capability, operation)
            result_binding(data, node_id, result_type, capability, operation)
        elif kind == "observe":
            source_id = _require_name(data.get("source"), f"node {node_id}.source")
            if source_id not in previous_ids:
                raise PlanValidationError(f"node {node_id}.source must reference an earlier node")
            if source_id not in deps:
                raise PlanValidationError(f"node {node_id}.source must also appear in depends_on")
            result_type = "observation"
            result_binding(data, node_id, result_type)
        elif kind == "assert":
            assertion = data.get("condition")
            _validate_condition(assertion, f"node {node_id}.condition", produced, variables)
        elif kind == "if":
            _validate_condition(data.get("condition"), f"node {node_id}.condition", produced, variables)
            then_ids = data.get("then", [])
            else_ids = data.get("else", [])
            if not isinstance(then_ids, list) or not isinstance(else_ids, list):
                raise PlanValidationError(f"node {node_id}: then and else must be arrays of node IDs")
            targets = then_ids + else_ids
            if any(not isinstance(branch, str) or branch not in node_positions
                   or node_positions[branch] <= index for branch in targets):
                raise PlanValidationError(f"node {node_id}: branches must reference known future nodes")
            if set(then_ids) & set(else_ids):
                raise PlanValidationError(f"node {node_id}: a node cannot belong to both branches")
            for target_id in targets:
                if target_id in guard_targets:
                    raise PlanValidationError(f"node {node_id}: branch target {target_id!r} already has a control guard")
                guard_targets[target_id] = node_id
        elif kind == "recover":
            handlers = data.get("handlers")
            if not isinstance(handlers, Mapping) or not handlers:
                raise PlanValidationError(f"node {node_id}.handlers must map error classes to recovery node references")
            for error_class, target_id in handlers.items():
                if not isinstance(error_class, str) or not error_class.strip():
                    raise PlanValidationError(f"node {node_id}: error handler keys must be non-empty strings")
                if not isinstance(target_id, str) or target_id not in node_positions:
                    raise PlanValidationError(f"node {node_id}: each recovery target must reference a known node")
                if node_positions[target_id] < index:
                    # Compatibility form: a prior node is an external routing
                    # reference only, never an executable retry target.
                    if target_id not in deps:
                        raise PlanValidationError(f"node {node_id}: legacy recovery references must be dependencies")
                    continue
                if node_positions[target_id] == index:
                    raise PlanValidationError(f"node {node_id}: recovery target cannot be itself")
                if target_id in guard_targets:
                    raise PlanValidationError(f"node {node_id}: recovery target {target_id!r} already has a control guard")
                guard_targets[target_id] = node_id
        elif kind == "return":
            if "value" not in data:
                raise PlanValidationError(f"node {node_id}.value is required")
            check_refs(data["value"], f"node {node_id}.value")
        elif kind == "macro_call":
            macro_id = _require_name(data.get("macro_id"), f"node {node_id}.macro_id")
            macro = macros.get(macro_id)
            if macro is None:
                raise PlanValidationError(f"node {node_id}: macro {macro_id!r} is not declared in this plan")
            args = data.get("arguments", {})
            if not isinstance(args, Mapping):
                raise PlanValidationError(f"node {node_id}.arguments must be an object")
            if set(args) != set(macro.inputs):
                raise PlanValidationError(f"node {node_id}: macro arguments must match signature {sorted(macro.inputs)}")
            for arg_name, expected_type in macro.inputs.items():
                _validate_value(args[arg_name], expected_type, f"node {node_id}.arguments.{arg_name}", produced)
            result_type = macro.output
            result_binding(data, node_id, result_type)

        node = PlanNode(node_id, kind, {**data, "depends_on": list(deps)})
        nodes.append(node)
        ids.add(node_id)
        previous_ids.add(node_id)
        ancestors[node_id] = node_ancestors

    # Guarded nodes must explicitly depend on the control decision so an
    # executor cannot run them before the if/recover node has selected them.
    for target_id, guard_id in guard_targets.items():
        target_deps = next(node.data["depends_on"] for node in nodes if node.id == target_id)
        target_ancestors = set(target_deps)
        for dep in target_deps:
            target_ancestors.update(ancestors.get(dep, set()))
        if guard_id not in target_ancestors:
            raise PlanValidationError(f"guarded node {target_id!r} must depend on control node {guard_id!r}")

    return TypedPlan(objective, intent, tuple(variables.values()), tuple(nodes), tuple(macros.values()), version)


def _type_spec(type_name: str) -> type | tuple[type, ...]:
    mapping = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict, "observation": dict, "null": type(None), "any": object}
    return mapping[type_name]


def _variable_refs(value: Any) -> set[str]:
    if isinstance(value, str) and value.startswith("$"):
        return {value[1:]}
    if isinstance(value, Mapping):
        refs: set[str] = set()
        for child in value.values():
            refs.update(_variable_refs(child))
        return refs
    if isinstance(value, list):
        refs: set[str] = set()
        for child in value:
            refs.update(_variable_refs(child))
        return refs
    return set()


def _condition_refs(value: Any) -> set[str]:
    """Collect variable names used by the typed condition AST's `var` leaves."""
    if isinstance(value, Mapping):
        refs = set()
        if "var" in value and isinstance(value["var"], str):
            refs.add(value["var"].removeprefix("$"))
        for child in value.values():
            refs.update(_condition_refs(child))
        return refs
    if isinstance(value, list):
        refs = set()
        for child in value:
            refs.update(_condition_refs(child))
        return refs
    return set()


def _operation_field_type(capability: str | None, operation: str | None, field_name: str) -> str:
    # Only stable, documented output fields can be bound as typed variables.
    fields = {
        ("filesystem", "create_sandbox"): {"sandbox_id": "string", "name": "string", "path": "string", "target_node": "string"},
        ("filesystem", "write_file"): {"sandbox_id": "string", "filename": "string", "path": "string"},
        ("filesystem", "read_file"): {"sandbox_id": "string", "filename": "string", "content": "string"},
        ("process", "execute"): {"sandbox_id": "string", "script_name": "string", "stdout": "string", "stderr": "string", "exit_code": "integer"},
        ("web", "search"): {"query": "string", "results": "array"},
        ("web", "fetch"): {"url": "string", "content": "string"},
    }
    result = fields.get((capability, operation), {}).get(field_name)
    if result is None:
        raise PlanValidationError(f"cannot bind unknown output field {field_name!r} from {capability}.{operation}")
    return result


def _validate_value(value: Any, expected: str, where: str, produced: Mapping[str, str]) -> None:
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in produced:
            raise PlanValidationError(f"{where} references undefined or not-yet-bound variable {value!r}")
        if expected != "any" and not _compatible(produced[name], _type_spec(expected)):
            raise PlanValidationError(f"{where} expects {expected}, but ${name} has type {produced[name]}")
        return
    if expected == "any":
        return
    if not value_matches_type(value, expected):
        raise PlanValidationError(f"{where} must have type {expected}")


def _validate_arguments(arguments: Mapping[str, Any], spec: ToolOperationSpec, node_id: str,
                         produced: Mapping[str, str], variables: Mapping[str, VariableDecl]) -> None:
    extra = set(arguments) - set(spec.params)
    if extra:
        raise PlanValidationError(f"node {node_id}: unsupported arguments for {spec.capability}.{spec.operation}: {sorted(extra)}")
    missing = {name for name, param in spec.params.items() if param.required and name not in arguments}
    if missing:
        raise PlanValidationError(f"node {node_id}: missing required arguments {sorted(missing)}")
    for name, value in arguments.items():
        param = spec.params[name]
        allowed = _python_type_names(param.type_)
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            if var_name not in produced:
                raise PlanValidationError(f"node {node_id}.arguments.{name} references undefined or not-yet-bound {value!r}")
            actual = produced[var_name]
            if actual != "any" and not (actual in allowed or (actual == "integer" and "number" in allowed)):
                raise PlanValidationError(f"node {node_id}.arguments.{name} expects {sorted(allowed)}, but ${var_name} has type {actual}")
        else:
            if not any(value_matches_type(value, type_name) for type_name in allowed):
                raise PlanValidationError(f"node {node_id}.arguments.{name} has invalid value type {type(value).__name__}; expected {sorted(allowed)}")


def _operation_result_type(capability: str, operation: str) -> str:
    if (capability, operation) in {("expression", "eval"), ("symbolic", "eval")}:
        return "any"
    if (capability, operation) in {("expression", "catalog"), ("symbolic", "catalog"), ("filesystem", "list_sandboxes")}:
        return "array"
    return "observation"


def _validate_condition(condition: Any, where: str, produced: Mapping[str, str], variables: Mapping[str, VariableDecl]) -> None:
    condition = _require_mapping(condition, where)
    if set(condition) == {"var", "equals"}:
        var = _require_name(condition["var"], f"{where}.var")
        if var not in produced:
            raise PlanValidationError(f"{where} references undefined or not-yet-bound variable ${var}")
        _validate_value(condition["equals"], produced[var], f"{where}.equals", produced)
        return
    if set(condition) == {"all"} or set(condition) == {"any"}:
        parts = next(iter(condition.values()))
        if not isinstance(parts, list) or not parts:
            raise PlanValidationError(f"{where} logical operands must be a non-empty array")
        for index, child in enumerate(parts):
            _validate_condition(child, f"{where}[{index}]", produced, variables)
        return
    if set(condition) == {"not"}:
        _validate_condition(condition["not"], f"{where}.not", produced, variables)
        return
    raise PlanValidationError(f"{where} must be {{var, equals}}, {{all}}, {{any}}, or {{not}}")

