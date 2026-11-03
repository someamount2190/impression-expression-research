"""
[SYSTEM_NAME] Symbolic Planner.
Phase 5 of [SYSTEM_NAME] Autonomous Execution Architecture.

Integrates task planning with Symbolic Cognition Adapter and Expression IR:
- Formulates compact, typed multi-step execution plans.
- Supports sequencing, dependencies, variable bindings, conditional branching, and verification.
- Compiles symbolic plans into standard CapabilityBroker execution envelopes.
- Measures token reduction of symbolic plans vs verbose natural language plans.
"""

from __future__ import annotations

import logging
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from core.symbolic_cognition_adapter import SymbolicCognitionAdapter, SymbolicPlan, SymbolicPlanStep
from core.capability_broker import CapabilityBroker, CapabilityEnvelope, CapabilityObservation
from core.task_ledger import TaskLedger, TaskStep, TaskReceipt, StepStatus

logger = logging.getLogger("[PROJECT_NAME].SymbolicPlanner")


class SymbolicPlanner:
    """Compiles and executes symbolic plans through standard CapabilityBroker envelopes."""

    def __init__(
        self,
        sca: Optional[SymbolicCognitionAdapter] = None,
        broker: Optional[CapabilityBroker] = None
    ):
        self.broker = broker or CapabilityBroker.get_instance()
        self.sca = sca or SymbolicCognitionAdapter(broker=self.broker)

    def compile_to_ledger(self, symbolic_plan: SymbolicPlan, objective: str, user_id: str = "default_user") -> TaskLedger:
        """Compiles a validated SymbolicPlan into an authoritative TaskLedger."""
        ledger = TaskLedger(
            task_id=f"task_sym_{symbolic_plan.plan_id[:8]}",
            user_id=user_id,
            objective=objective
        )

        prev_step_id = None
        for step in symbolic_plan.steps:
            step_id = f"step_{step.step_id}"
            deps = [prev_step_id] if prev_step_id else []

            # Determine capability & operation
            if step.is_pure_symbolic:
                cap = "expression"
                op = "eval"
                args = {
                    "expr_id": step.op_or_capability,
                    "args": step.arguments.get("args", []),
                    "kwargs": step.arguments.get("kwargs", {})
                }
            else:
                cap = step.op_or_capability
                op = step.operation or "execute"
                args = dict(step.arguments)

            envelope = CapabilityEnvelope(action="execute", capability=cap, operation=op, arguments=args)
            task_step = TaskStep(
                step_id=step_id,
                description=f"Symbolic Step {step.step_id}: {cap}.{op}",
                action_envelope=envelope,
                dependencies=deps,
                expected_outcome=f"Output bound to {step.bind_output_to}" if step.bind_output_to else f"Success on {cap}.{op}"
            )
            ledger.steps.append(task_step)
            prev_step_id = step_id

        return ledger

    async def execute_symbolic_plan(
        self,
        plan: SymbolicPlan,
        ledger: TaskLedger
    ) -> Dict[str, Any]:
        """
        Executes a symbolic plan step-by-step, resolving variable references and updating ledger.
        """
        context_vars: Dict[str, Any] = {}
        step_results: List[Dict[str, Any]] = []

        for step in plan.steps:
            # Resolve variable bindings in arguments (e.g. $sb_id or context references)
            resolved_args = self._resolve_variable_bindings(step.arguments, context_vars)

            if step.is_pure_symbolic:
                op_id = step.op_or_capability.upper()
                call_args = resolved_args.get("args", [])
                call_kwargs = resolved_args.get("kwargs", {})
                try:
                    res = self.sca.bridge.eval_operator(op_id, *call_args, **call_kwargs)
                    obs = CapabilityObservation(
                        status="SUCCESS",
                        capability="expression",
                        operation="eval",
                        observation={"result": res},
                        duration_ms=0.5
                    )
                except Exception as e:
                    obs = CapabilityObservation(
                        status="EXECUTION_ERROR",
                        capability="expression",
                        operation="eval",
                        observation=None,
                        duration_ms=0.5,
                        error=str(e)
                    )
            else:
                envelope = CapabilityEnvelope(
                    action="execute",
                    capability=step.op_or_capability,
                    operation=step.operation or "execute",
                    arguments=resolved_args
                )
                obs = await self.broker.dispatch(envelope)

            # Store in context variables if bound
            if step.bind_output_to:
                if isinstance(obs.observation, dict) and "result" in obs.observation:
                    context_vars[step.bind_output_to] = obs.observation["result"]
                elif isinstance(obs.observation, dict):
                    context_vars[step.bind_output_to] = obs.observation
                    for k, v in obs.observation.items():
                        context_vars[f"{step.bind_output_to}.{k}"] = v
                else:
                    context_vars[step.bind_output_to] = obs.observation

            # Create TaskReceipt
            receipt = TaskReceipt(
                receipt_id=f"rec_{uuid.uuid4().hex[:8]}",
                capability=obs.capability,
                operation=obs.operation,
                status=obs.status,
                target_node="HOST",
                duration_ms=obs.duration_ms,
                summary=f"Symbolic step {step.step_id} executed: {obs.status}",
                evidence=obs.observation if isinstance(obs.observation, dict) else {"result": obs.observation}
            )
            ledger.receipts.append(receipt)

            step_results.append({
                "step_id": step.step_id,
                "capability": obs.capability,
                "operation": obs.operation,
                "status": obs.status,
                "observation": obs.observation,
                "error": obs.error
            })

            # Check for failure
            if obs.status != "SUCCESS":
                ledger.status = StepStatus.FAILED.name
                return {
                    "status": "FAILED",
                    "failed_step": step.step_id,
                    "error": obs.error,
                    "results": step_results,
                    "context": context_vars
                }

        ledger.status = "SUCCEEDED"
        return {
            "status": "SUCCESS",
            "results": step_results,
            "context": context_vars
        }

    def _resolve_variable_bindings(self, args: Dict[str, Any], context_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves references like '$sandbox_id' or '$res.sandbox_id' from context_vars."""
        resolved = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("$"):
                var_name = v[1:]
                resolved[k] = context_vars.get(var_name, v)
            elif isinstance(v, list):
                resolved[k] = [
                    context_vars.get(item[1:], item) if isinstance(item, str) and item.startswith("$") else item
                    for item in v
                ]
            else:
                resolved[k] = v
        return resolved

    def measure_token_efficiency(self, plan: SymbolicPlan, verbose_nl_prompt: str) -> Dict[str, Any]:
        """
        Measures token reduction of compact symbolic plan vs verbose natural language prompt.
        Uses ~4 characters per token heuristic standard.
        """
        compact_json = json.dumps(plan.to_dict(), separators=(',', ':'))
        sym_tokens = max(1, len(compact_json) // 4)
        nl_tokens = max(1, len(verbose_nl_prompt) // 4)
        savings_pct = round(((nl_tokens - sym_tokens) / nl_tokens) * 100.0, 1) if nl_tokens > sym_tokens else 0.0

        return {
            "symbolic_characters": len(compact_json),
            "symbolic_estimated_tokens": sym_tokens,
            "nl_characters": len(verbose_nl_prompt),
            "nl_estimated_tokens": nl_tokens,
            "token_reduction_percent": savings_pct,
            "efficiency_ratio": round(nl_tokens / max(1, sym_tokens), 2)
        }
