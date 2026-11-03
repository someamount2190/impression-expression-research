"""
Symbolic Cognition Adapter (SCA) for the host execution runtime.
Implements the Symbolic Cognition Integration (SCI) Layer:
- Interprets model-generated symbolic plans and expressions.
- Classifies the 41 verified Expression IR operators (E001–E502) into semantic categories.
- Type-checks and validates plan grammar before execution.
- Dispatches pure symbolic operations to ExpressionBridge deterministically.
- Routes privileged external operations (filesystem, process, web) through CapabilityBroker.
- Provides inspectability: decompiles symbolic plans into human-readable explanations.
- Records all plan executions (valid and invalid) into an append-only Experience Database.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from core.expression_bridge import ExpressionBridge
from core.capability_broker import CapabilityBroker, CapabilityEnvelope, CapabilityObservation

logger = logging.getLogger("SymbolicCognitionAdapter")

# Default path for Experience Database
DEFAULT_EXPERIENCE_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "symbolic_experience_db.jsonl")
)

# ---------------------------------------------------------------------------
# Operator Semantic Classification
# ---------------------------------------------------------------------------
OPERATOR_CATEGORIES = {
    "MATHEMATICAL_VECTOR": [
        "E002",  # CLAMP_RANGE
        "E304",  # L2_NORM
        "E404",  # MATRIX_DIAGONAL_SUM
        "E501",  # DOT_PRODUCT
    ],
    "AGGREGATION_STATISTICS": [
        "E006",  # SUM_MAP
        "E012",  # FILTER_SUM
        "E013",  # COUNT_FILTER
        "E016",  # MAX_MAP
        "E018",  # SUM_FILTER_MAP
        "E022",  # ZIP_SUM
        "E102",  # HIERARCHICAL_SUM_MAP
        "E201",  # HIERARCHICAL_SUM_FILTER_MAP
        "E202",  # CLAMPED_SUM_FILTER_MAP
        "E403",  # GROUP_SUM
        "E502",  # FREQUENCY_COUNTER
    ],
    "TRANSFORMATION_MAPPING": [
        "E001",  # MAP
        "E003",  # FILTER
        "E004",  # DICT_COMPREHENSION
        "E005",  # SAFE_DIV_OP
        "E009",  # SORTED_SET
        "E010",  # COND_ZERO
        "E011",  # FILTER_MAP
        "E014",  # FLATTEN_MAP
        "E015",  # SORTED_FILTER
        "E017",  # DICT_FILTER_MAP
        "E019",  # ENUMERATE_MAP
        "E020",  # ENUMERATE_FILTER
        "E021",  # ZIP_MAP
        "E101",  # HIERARCHICAL_FILTER_MAP
        "E203",  # NESTED_MATRIX_FILTER_MAP
        "E301",  # SLIDING_WINDOW
        "E302",  # PREFIX_SCAN
        "E303",  # GROUP_BY
        "E401",  # WINDOW_INDEX_FILTER
        "E402",  # ADJACENT_STEP_FILTER
    ],
    "QUANTIFICATION_BOOLEAN": [
        "E007",  # ALL
        "E008",  # ANY
        "E103",  # HIERARCHICAL_ANY_FILTER_MAP
        "E104",  # HIERARCHICAL_ALL_FILTER_MAP
    ],
    "STRUCTURAL_COMPRESSION": [
        "E305",  # RUN_LENGTH_ENCODE
        "E405",  # BIGRAM_EXTRACTOR
    ]
}


class SymbolicPlanStep:
    """Represents a single atomic step within a symbolic execution plan."""

    def __init__(
        self,
        step_id: Union[int, str],
        op_or_capability: str,
        operation: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        bind_output_to: Optional[str] = None,
        is_pure_symbolic: bool = True
    ):
        self.step_id = step_id
        self.op_or_capability = op_or_capability  # e.g., 'E501' or 'filesystem'
        self.operation = operation                # e.g., 'read_file' if external
        self.arguments = arguments or {}
        self.bind_output_to = bind_output_to      # variable name in plan execution context
        self.is_pure_symbolic = is_pure_symbolic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "op": self.op_or_capability,
            "operation": self.operation,
            "arguments": self.arguments,
            "bind_output_to": self.bind_output_to,
            "is_pure_symbolic": self.is_pure_symbolic
        }


class SymbolicPlan:
    """Represents an ordered, validated sequence of symbolic or external operations."""

    def __init__(self, plan_id: str, steps: List[SymbolicPlanStep], raw_text: str = ""):
        self.plan_id = plan_id
        self.steps = steps
        self.raw_text = raw_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "raw_text": self.raw_text
        }


class SymbolicCognitionAdapter:
    """
    Symbolic Cognition Adapter.
    Translates, validates, executes, and records neural-symbolic plans.
    """

    def __init__(
        self,
        bridge: Optional[ExpressionBridge] = None,
        broker: Optional[CapabilityBroker] = None,
        experience_db_path: str = DEFAULT_EXPERIENCE_DB_PATH
    ):
        self.bridge = bridge or ExpressionBridge.get_instance()
        self.broker = broker or CapabilityBroker.get_instance()
        self.experience_db_path = experience_db_path
        os.makedirs(os.path.dirname(self.experience_db_path), exist_ok=True)

    def parse_typed_plan(self, plan_input: Union[str, Dict[str, Any]]) -> Any:
        """Parse the extensible typed-plan IR against the live operation schemas.

        The original flat SymbolicPlan parser remains available for backward
        compatibility. Typed plans use the EXPRESSION-GROWTH IR validator and
        are dispatched by its interpreter only after validation succeeds.
        """
        from experiments.expression_growth_001.symbolic_ir.typed_plan import parse_plan

        if isinstance(plan_input, str):
            try:
                plan_input = json.loads(plan_input)
            except json.JSONDecodeError as exc:
                raise ValueError(f"typed symbolic plan must be valid JSON: {exc}") from exc
        return parse_plan(plan_input)

    def validate_typed_plan(self, plan_input: Union[str, Dict[str, Any]]) -> Tuple[bool, List[str], Any]:
        """Return typed-plan validity and diagnostics without executing actions."""
        from experiments.expression_growth_001.symbolic_ir.typed_plan import PlanValidationError

        try:
            plan = self.parse_typed_plan(plan_input)
            return True, [], plan
        except PlanValidationError as exc:
            return False, [str(exc)], None

    def classify_operator(self, op_id: str) -> Dict[str, Any]:
        """Returns semantic category and classification details for a given operator ID."""
        op_norm = op_id.upper().strip()
        matched_category = "UNKNOWN"
        for cat, ops in OPERATOR_CATEGORIES.items():
            if op_norm in ops:
                matched_category = cat
                break

        # Check if known in Expression Bridge registry
        catalog = self.bridge.get_catalog()
        op_meta = next((item for item in catalog if item["expr_id"].upper() == op_norm), None)

        return {
            "expr_id": op_norm,
            "category": matched_category,
            "is_pure_symbolic": True,
            "exists_in_registry": op_meta is not None,
            "parameters": op_meta.get("parameters", []) if op_meta else [],
            "arity": op_meta.get("arity", 0) if op_meta else 0,
            "description": op_meta.get("description", "") if op_meta else ""
        }

    def parse_plan(self, plan_input: Union[str, Dict[str, Any], List[Any]]) -> Tuple[Optional[SymbolicPlan], Optional[str]]:
        """
        Parses a plan representation into a validated SymbolicPlan.
        Accepts:
        - Structured JSON / dict: {"plan_id": ..., "steps": [{"op": "E501", ...}]}
        - List of step dictionaries
        - Compact functional plan notation: PLAN(SEQ(E501(u, v), E002(x, y)))
        """
        raw_text = str(plan_input)
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Case 1: Dict payload
        if isinstance(plan_input, dict):
            steps_data = plan_input.get("steps") or plan_input.get("plan")
            if not isinstance(steps_data, list):
                # Check if it's a single operation envelope
                if "capability" in plan_input or "action" in plan_input:
                    cap = plan_input.get("capability", "expression")
                    op = plan_input.get("operation") or plan_input.get("op")
                    args = plan_input.get("arguments", {})
                    is_sym = (cap == "expression" or (isinstance(op, str) and op.upper().startswith("E")))
                    step = SymbolicPlanStep(
                        step_id=1,
                        op_or_capability=op if is_sym else cap,
                        operation=None if is_sym else op,
                        arguments=args,
                        is_pure_symbolic=is_sym
                    )
                    return SymbolicPlan(plan_id=plan_id, steps=[step], raw_text=raw_text), None
                return None, "Invalid dict format: missing 'steps' or 'plan' list"
            
            steps = []
            for idx, s in enumerate(steps_data):
                if not isinstance(s, dict):
                    return None, f"Step {idx} is not a valid dictionary"
                op_raw = s.get("op") or s.get("operator") or s.get("capability")
                if not op_raw:
                    return None, f"Step {idx} missing operator or capability"
                
                op_upper = str(op_raw).upper()
                is_pure_symbolic = op_upper.startswith("E") or s.get("capability") == "expression"
                
                step = SymbolicPlanStep(
                    step_id=s.get("step_id", idx + 1),
                    op_or_capability=op_raw,
                    operation=s.get("operation"),
                    arguments=s.get("arguments", s.get("args", {})),
                    bind_output_to=s.get("bind") or s.get("bind_output_to"),
                    is_pure_symbolic=is_pure_symbolic
                )
                steps.append(step)
            return SymbolicPlan(plan_id=plan_input.get("plan_id", plan_id), steps=steps, raw_text=raw_text), None

        # Case 2: List payload
        elif isinstance(plan_input, list):
            steps = []
            for idx, s in enumerate(plan_input):
                if not isinstance(s, dict):
                    return None, f"Step {idx} is not a valid dictionary"
                op_raw = s.get("op") or s.get("capability")
                is_pure_symbolic = str(op_raw).upper().startswith("E") or s.get("capability") == "expression"
                step = SymbolicPlanStep(
                    step_id=s.get("step_id", idx + 1),
                    op_or_capability=op_raw,
                    operation=s.get("operation"),
                    arguments=s.get("arguments", s.get("args", {})),
                    bind_output_to=s.get("bind"),
                    is_pure_symbolic=is_pure_symbolic
                )
                steps.append(step)
            return SymbolicPlan(plan_id=plan_id, steps=steps, raw_text=raw_text), None

        # Case 3: String payload (JSON or symbolic string)
        elif isinstance(plan_input, str):
            clean = plan_input.strip()
            # Try parsing as JSON first
            if clean.startswith("{") or clean.startswith("["):
                try:
                    parsed = json.loads(clean)
                    return self.parse_plan(parsed)
                except Exception as e:
                    return None, f"JSON parse error: {str(e)}"
            
            # Simple symbolic string notation parser: e.g. "SEQ(E501(u, v), E002(a, b))"
            if clean.startswith("PLAN(") or clean.startswith("SEQ("):
                return self._parse_symbolic_text_plan(clean, plan_id)

            return None, f"Unrecognized plan string format: '{clean[:60]}...'"

        return None, f"Unsupported plan input type: {type(plan_input).__name__}"

    def _parse_symbolic_text_plan(self, text: str, plan_id: str) -> Tuple[Optional[SymbolicPlan], Optional[str]]:
        """Parses a functional SEQ(...) plan string."""
        import re
        inner = text.strip()
        if inner.startswith("PLAN("):
            inner = inner[5:-1].strip()
        if inner.startswith("SEQ("):
            inner = inner[4:-1].strip()

        # Split top-level operator invocations
        pattern = r"([A-Za-z0-9_]+)\((.*?)\)"
        matches = list(re.finditer(pattern, inner))
        if not matches:
            return None, f"No operators found in symbolic plan text: {text}"

        steps = []
        for idx, m in enumerate(matches):
            op_name = m.group(1).strip()
            args_str = m.group(2).strip()
            # Try to safely evaluate simple literals or keep as string references
            args_list = [a.strip() for a in args_str.split(",") if a.strip()]
            
            is_sym = op_name.upper().startswith("E")
            step = SymbolicPlanStep(
                step_id=idx + 1,
                op_or_capability=op_name,
                operation=None if is_sym else op_name,
                arguments={"args": args_list},
                is_pure_symbolic=is_sym
            )
            steps.append(step)

        return SymbolicPlan(plan_id=plan_id, steps=steps, raw_text=text), None

    def validate_plan(self, plan: SymbolicPlan) -> Tuple[bool, List[str]]:
        """
        Validates grammar, operator existence, and parameter signatures for a plan.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        if not plan.steps:
            return False, ["Plan contains zero steps."]

        for step in plan.steps:
            if step.is_pure_symbolic:
                op_id = step.op_or_capability.upper()
                classification = self.classify_operator(op_id)
                if not classification["exists_in_registry"]:
                    errors.append(f"Step {step.step_id}: Operator '{op_id}' is not in the Expression registry.")
                else:
                    # Check arity if arguments is a dict with exact parameter names or 'args' list
                    expected_arity = classification["arity"]
                    if "args" in step.arguments and isinstance(step.arguments["args"], list):
                        actual_arity = len(step.arguments["args"])
                        if actual_arity != expected_arity:
                            errors.append(
                                f"Step {step.step_id}: Arity mismatch for {op_id}. "
                                f"Expected {expected_arity}, got {actual_arity}."
                            )
            else:
                # External capability validation
                cap = step.op_or_capability
                if cap not in self.broker.CAPABILITIES:
                    errors.append(f"Step {step.step_id}: Unknown external capability '{cap}'.")
                elif step.operation not in self.broker.CAPABILITIES[cap]:
                    errors.append(
                        f"Step {step.step_id}: Unknown operation '{step.operation}' for capability '{cap}'."
                    )

        return (len(errors) == 0), errors

    def explain_plan(self, plan: SymbolicPlan) -> str:
        """
        Decompiles a symbolic plan into a human-readable English explanation.
        Preserves inspectability and reversibility.
        """
        lines = [f"=== Symbolic Execution Plan [{plan.plan_id}] ==="]
        for step in plan.steps:
            if step.is_pure_symbolic:
                op_id = step.op_or_capability.upper()
                meta = self.classify_operator(op_id)
                desc = meta.get("description", "Deterministic symbolic operation")
                cat = meta.get("category", "SYMBOLIC")
                bind_str = f" -> ${step.bind_output_to}" if step.bind_output_to else ""
                lines.append(f"Step {step.step_id}: [Symbolic / {cat}] {op_id}({step.arguments}){bind_str}")
                lines.append(f"        Semantics: {desc}")
            else:
                bind_str = f" -> ${step.bind_output_to}" if step.bind_output_to else ""
                lines.append(
                    f"Step {step.step_id}: [External Privileged] {step.op_or_capability}.{step.operation}({step.arguments}){bind_str}"
                )
        return "\n".join(lines)

    async def execute_plan(
        self,
        plan: SymbolicPlan,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
        ledger: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes a validated SymbolicPlan step-by-step.
        Maintains an execution environment/context across steps.
        Logs the complete trace to the Experience Database.
        """
        t0 = time.perf_counter()
        exec_context = dict(context or {})
        trace = []
        overall_status = "SUCCESS"
        error_msg = None

        # Validate first
        is_valid, validation_errors = self.validate_plan(plan)
        if not is_valid:
            overall_status = "VALIDATION_FAILED"
            error_msg = "; ".join(validation_errors)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            result = {
                "plan_id": plan.plan_id,
                "status": overall_status,
                "error": error_msg,
                "duration_ms": duration_ms,
                "context": exec_context,
                "trace": trace
            }
            self._record_experience(plan, result)
            return result

        # Execute steps
        for step in plan.steps:
            step_t0 = time.perf_counter()
            step_record = {
                "step_id": step.step_id,
                "op": step.op_or_capability,
                "operation": step.operation,
                "status": "PENDING",
                "result": None,
                "error": None
            }

            try:
                # Resolve variable references in arguments
                resolved_args = self._resolve_arguments(step.arguments, exec_context)

                if step.is_pure_symbolic:
                    op_id = step.op_or_capability.upper()
                    # Execute purely deterministically via ExpressionBridge
                    obs = await self._execute_symbolic_operator(op_id, resolved_args)
                    if obs.status == "SUCCESS":
                        step_record["status"] = "SUCCESS"
                        step_record["result"] = obs.observation
                        if step.bind_output_to:
                            exec_context[step.bind_output_to] = obs.observation
                    else:
                        step_record["status"] = "ERROR"
                        step_record["error"] = obs.error
                        overall_status = "EXECUTION_ERROR"
                        error_msg = obs.error
                else:
                    # Privileged external call via CapabilityBroker
                    envelope = CapabilityEnvelope(
                        action="execute",
                        capability=step.op_or_capability,
                        operation=step.operation or "",
                        arguments=resolved_args
                    )
                    obs = await self.broker.dispatch(envelope, ledger=ledger)
                    if obs.status == "SUCCESS":
                        step_record["status"] = "SUCCESS"
                        step_record["result"] = obs.observation
                        if step.bind_output_to:
                            exec_context[step.bind_output_to] = obs.observation
                    else:
                        step_record["status"] = "ERROR"
                        step_record["error"] = obs.error
                        overall_status = "EXECUTION_ERROR"
                        error_msg = obs.error

            except Exception as e:
                step_record["status"] = "EXCEPTION"
                step_record["error"] = str(e)
                overall_status = "EXCEPTION"
                error_msg = str(e)

            step_record["duration_ms"] = (time.perf_counter() - step_t0) * 1000.0
            trace.append(step_record)

            # Abort plan on step failure
            if step_record["status"] != "SUCCESS":
                break

        duration_ms = (time.perf_counter() - t0) * 1000.0
        final_result = {
            "plan_id": plan.plan_id,
            "status": overall_status,
            "error": error_msg,
            "duration_ms": duration_ms,
            "context": exec_context,
            "trace": trace
        }

        # Log trace to research experience database
        self._record_experience(plan, final_result)
        return final_result

    def _resolve_arguments(self, raw_args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitutes variable references like '$result1' with values from context."""
        resolved = {}
        for k, v in raw_args.items():
            if isinstance(v, str) and v.startswith("$"):
                var_name = v[1:]
                resolved[k] = context.get(var_name, v)
            elif isinstance(v, list):
                resolved[k] = [context.get(elem[1:], elem) if isinstance(elem, str) and elem.startswith("$") else elem for elem in v]
            else:
                resolved[k] = v
        return resolved

    async def _execute_symbolic_operator(self, op_id: str, args: Dict[str, Any]) -> CapabilityObservation:
        """Executes a pure Expression IR operator deterministically."""
        meta = self.classify_operator(op_id)
        params = meta.get("parameters", [])

        # Format positional arguments
        call_args = []
        if "args" in args and isinstance(args["args"], list):
            call_args = args["args"]
        elif params:
            for p in params:
                if p in args:
                    call_args.append(args[p])
                else:
                    return CapabilityObservation(
                        status="VALIDATION_ERROR",
                        capability="expression",
                        operation="eval",
                        observation=None,
                        duration_ms=0.0,
                        error=f"Missing argument '{p}' for operator {op_id}"
                    )
        else:
            call_args = list(args.values())

        # Standard function resolution for higher-order arguments (e.g. 'sum', 'len')
        standard_funcs = {
            "sum": sum, "len": len, "min": min, "max": max, "abs": abs,
            "sorted": sorted, "str": str, "int": int, "float": float
        }
        converted_args = []
        for a in call_args:
            if isinstance(a, str) and a in standard_funcs:
                converted_args.append(standard_funcs[a])
            elif isinstance(a, str) and a.startswith("lambda "):
                # Model-provided lambdas are code, not symbolic data.
                return CapabilityObservation("VALIDATION_ERROR", "expression", "eval", None, 0.0,
                                             "Executable lambda arguments are not permitted")
            else:
                converted_args.append(a)
        call_args = converted_args

        expr_str = f"{op_id}({', '.join(repr(a) for a in call_args)})"
        try:
            res = self.bridge.eval_operator(op_id, *call_args)
            return CapabilityObservation(
                status="SUCCESS",
                capability="expression",
                operation="eval",
                observation=res,
                duration_ms=0.0
            )
        except Exception as e:
            return CapabilityObservation(
                status="EXECUTION_ERROR",
                capability="expression",
                operation="eval",
                observation=None,
                duration_ms=0.0,
                error=str(e)
            )

    def _record_experience(self, plan: SymbolicPlan, execution_result: Dict[str, Any]):
        """Persists the executed plan into the append-only Experience Database."""
        entry = {
            "plan_id": plan.plan_id,
            "timestamp": time.time(),
            "raw_text": plan.raw_text,
            "step_count": len(plan.steps),
            "status": execution_result.get("status"),
            "error": execution_result.get("error"),
            "duration_ms": execution_result.get("duration_ms"),
            "trace": execution_result.get("trace", [])
        }
        try:
            with open(self.experience_db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to write to experience db: %s", e)
