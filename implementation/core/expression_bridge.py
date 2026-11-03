"""
EXPRESSION Bridge: Neural-Symbolic Integration Adapter.
Bridges the EXPRESSION autonomous program discovery & IR compiler system
(from the sibling symbolic-expression project) into the host execution runtime:
- Loads the verified 41+ Expression IR definitions (E001–E502) from the registry.
- Provides deterministic beta-reduction transpilation from Expression IR to pure Python.
- Provides direct evaluation of verified expressions.
- Supplies live telemetry and sensory context to the host runtime.
"""

from __future__ import annotations

import hashlib
import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("ExpressionBridge")

# Attempt to locate and import expression package
DEFAULT_EXPRESSION_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "expression")
)
DEFAULT_REGISTRY_JSON = os.path.join(
    DEFAULT_EXPRESSION_ROOT, "reports", "expression_registry.json"
)


from core.expression_sandbox import (  # noqa: E402  (policy lives in the stdlib-only worker module)
    MAX_OUTPUT_BYTES, PASSABLE_BUILTINS, RESULT_MARKER, UnsafeExpressionError,
    _SAFE_BUILTINS, _compile_restricted, run_restricted,
)

_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expression_sandbox.py")
EVAL_TIMEOUT_SECONDS = float(os.environ.get("EXPRESSION_EVAL_TIMEOUT", "15"))


def _encode_for_worker(value):
    """JSON-encode call arguments. Returns (encoded, ok); ok is False for non-portable objects."""
    for name, fn in PASSABLE_BUILTINS.items():
        if value is fn:
            return {"__builtin__": name}, True
    if isinstance(value, (list, tuple)):
        items = [_encode_for_worker(v) for v in value]
        return [i for i, _ in items], all(ok for _, ok in items)
    if isinstance(value, dict):
        items = {k: _encode_for_worker(v) for k, v in value.items()}
        return {k: i for k, (i, _) in items.items()}, all(ok for _, ok in items.values())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, True
    return None, False


class ExpressionBridge:
    _instance: Optional[ExpressionBridge] = None

    def __init__(self, expression_root: Optional[str] = None, registry_path: Optional[str] = None):
        self._compiled_impl_cache: Dict[str, Any] = {}
        self.expression_root = expression_root or os.environ.get("EXPRESSION_ROOT", DEFAULT_EXPRESSION_ROOT)
        self.registry_path = registry_path or os.environ.get("EXPRESSION_REGISTRY_PATH", DEFAULT_REGISTRY_JSON)
        
        self.registry = None
        self.compiler = None
        self._compiled_count = 0
        self._evaluated_count = 0
        self._last_error: Optional[str] = None
        
        self._init_engine()

    @classmethod
    def get_instance(cls) -> ExpressionBridge:
        if cls._instance is None:
            cls._instance = ExpressionBridge()
        return cls._instance

    def _init_engine(self):
        """Initializes the ExpressionRegistry and ExpressionCompiler."""
        src_path = os.path.join(self.expression_root, "src")
        # Append (never prepend) so this directory cannot shadow stdlib/installed modules.
        if os.path.exists(src_path) and src_path not in sys.path:
            sys.path.append(src_path)

        try:
            from expression.registry.store import ExpressionRegistry
            from expression.compiler.transpiler import ExpressionCompiler

            # Optional integrity pin: set EXPRESSION_REGISTRY_SHA256 to refuse a tampered registry.
            pinned = os.environ.get("EXPRESSION_REGISTRY_SHA256", "").strip().lower()
            if pinned and os.path.exists(self.registry_path):
                with open(self.registry_path, "rb") as fh:
                    actual = hashlib.sha256(fh.read()).hexdigest()
                if actual != pinned:
                    raise UnsafeExpressionError(
                        f"Registry hash mismatch for {self.registry_path}: refusing to load")

            if os.path.exists(self.registry_path):
                self.registry = ExpressionRegistry.load(self.registry_path)
                self.compiler = ExpressionCompiler(self.registry)
                logger.info("Successfully loaded ExpressionRegistry with %d expressions", len(self.registry._latest_version))
            else:
                logger.warning("Expression registry JSON not found at %s", self.registry_path)
                self.registry = ExpressionRegistry()
                self.compiler = ExpressionCompiler(self.registry)
        except Exception as e:
            self._last_error = str(e)
            logger.error("Failed to initialize Expression engine: %s", e)

    @property
    def is_available(self) -> bool:
        return self.registry is not None and self.compiler is not None and len(self.registry._latest_version) > 0

    def get_catalog(self, filter_query: str = "") -> List[Dict[str, Any]]:
        """Returns catalog metadata for accepted expression operators."""
        if not self.registry:
            return []
        
        catalog = []
        q = filter_query.lower().strip()
        for eid in sorted(self.registry._latest_version.keys()):
            defn = self.registry.get(eid)
            if not defn:
                continue
            
            p_names = [p.name for p in defn.parameters]
            desc = getattr(defn, "description", "") or defn.expr_id
            
            # Simple substring filtering
            if q and q not in eid.lower() and q not in desc.lower():
                continue
            
            catalog.append({
                "expr_id": eid,
                "version": defn.version,
                "parameters": p_names,
                "arity": len(p_names),
                "description": desc,
                "python_implementation": defn.python_implementation,
                "usage_count": defn.usage_count,
            })
        return catalog

    def compile_ir(self, source_code: str) -> Dict[str, Any]:
        """
        Transpiles code containing Expression IR (e.g. E401(coll, pred))
        into deterministically beta-reduced pure Python code.
        """
        if not self.compiler:
            return {
                "success": False,
                "original_source": source_code,
                "compiled_python": source_code,
                "used_expressions": [],
                "source_map": [],
                "error": "ExpressionCompiler not initialized"
            }

        try:
            res = self.compiler.compile(source_code)
            self._compiled_count += 1
            return {
                "success": True,
                "original_source": res.original_source,
                "compiled_python": res.compiled_python,
                "used_expressions": res.used_expressions,
                "source_map": [
                    {
                        "expr_id": sm.expr_id,
                        "version": sm.version,
                        "original_call": sm.original_call,
                        "expanded_python": sm.expanded_python,
                        "line_number": sm.line_number
                    } for sm in res.source_map
                ],
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "original_source": source_code,
                "compiled_python": source_code,
                "used_expressions": [],
                "source_map": [],
                "error": str(e)
            }

    def eval_operator(self, expr_id: str, *args, **kwargs) -> Any:
        """Directly executes an expression operator with provided arguments."""
        if not self.registry:
            raise RuntimeError("ExpressionRegistry not available")

        defn = self.registry.get(expr_id)
        if not defn:
            raise KeyError(f"Expression {expr_id} not found in registry")

        # Registry code runs in a separate, resource-limited interpreter (see core/expression_sandbox.py).
        # Only callables that cannot cross a process boundary (e.g. in-process lambdas supplied by
        # trusted callers) take the in-process path, which still enforces the AST policy and
        # minimal builtins.
        enc_args, ok_a = _encode_for_worker(list(args))
        enc_kwargs, ok_k = _encode_for_worker(dict(kwargs))
        if ok_a and ok_k:
            result = self._eval_in_worker(defn.python_implementation, expr_id, enc_args, enc_kwargs)
        else:
            logger.debug("Expression %s has non-portable arguments; using in-process restricted path", expr_id)
            result = run_restricted(defn.python_implementation, expr_id, args, kwargs)
        self._evaluated_count += 1
        return result

    def _eval_in_worker(self, source: str, expr_id: str, args, kwargs) -> Any:
        import subprocess
        request = json.dumps({"source": source, "expr_id": expr_id, "args": args, "kwargs": kwargs})
        env = {"PYTHONIOENCODING": "utf-8"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "C:\\Windows")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", _WORKER_PATH],
                input=request.encode("utf-8"), capture_output=True, timeout=EVAL_TIMEOUT_SECONDS,
                env=env, cwd=os.path.dirname(_WORKER_PATH), check=False)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Expression {expr_id} exceeded {EVAL_TIMEOUT_SECONDS:.0f}s and was terminated")
        out = proc.stdout[:MAX_OUTPUT_BYTES + 4096].decode("utf-8", errors="replace")
        if RESULT_MARKER not in out:
            raise RuntimeError(f"Expression {expr_id} worker died (exit {proc.returncode}): "
                               f"{proc.stderr.decode('utf-8', 'replace')[-300:].strip()}")
        reply = json.loads(out.split(RESULT_MARKER, 1)[1])
        if not reply.get("ok"):
            etype, msg = reply.get("error_type"), reply.get("error")
            exc = {"UnsafeExpressionError": UnsafeExpressionError, "KeyError": KeyError,
                   "TypeError": TypeError, "ValueError": ValueError,
                   "ZeroDivisionError": ZeroDivisionError, "IndexError": IndexError,
                   "NameError": NameError}.get(etype, RuntimeError)
            raise exc(msg if exc is not RuntimeError else f"{etype}: {msg}")
        return reply["result"]

    def get_telemetry(self) -> Dict[str, Any]:
        """Telemetry snapshot for UI and coordinator broadcasts."""
        op_count = len(self.registry._latest_version) if self.registry else 0
        return {
            "available": self.is_available,
            "operator_count": op_count,
            "compiled_scripts_count": self._compiled_count,
            "evaluated_calls_count": self._evaluated_count,
            "registry_path": self.registry_path,
            "last_error": self._last_error
        }
