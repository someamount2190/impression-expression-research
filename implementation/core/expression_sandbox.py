"""
Policy + out-of-process runner for Expression IR operators.

Stdlib-only on purpose: this file is both imported by core.expression_bridge (for the
static policy) and executed directly (``python -I core/expression_sandbox.py``) as the
isolated worker. Running with ``-I`` means the project tree is NOT on sys.path in the
worker, so nothing here may import project modules.

Defence in depth, outermost first:
  1. Separate interpreter process: scrubbed environment, wall-clock timeout, stdout cap.
  2. Worker self-limits: memory cap and "no child processes" (Windows Job Object /
     POSIX rlimits) applied before any registry code is parsed.
  3. AST policy + minimal __builtins__ (see _compile_restricted).
This is a containment layer, not a substitute for OS-level container isolation.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import sys
from typing import Any, Dict

MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
CPU_SECONDS = 10
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
RESULT_MARKER = "\x00HIVE_EXPR_RESULT\x00"

_SAFE_BUILTINS: Dict[str, Any] = {
    name: getattr(builtins, name) for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "len", "list", "map", "max", "min", "range", "reversed", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
        "isinstance", "divmod", "pow",
    )
}

# Builtin callables that may be passed *by name* across the process boundary.
PASSABLE_BUILTINS = {
    n: getattr(builtins, n) for n in ("sum", "len", "min", "max", "abs", "sorted", "str", "int", "float")
}

# AST nodes that have no place in a pure expression operator.
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
    ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.Try,
    ast.With, ast.AsyncWith, ast.AsyncFor, ast.Delete, ast.Raise,
)
# Attributes that enable format-string or frame/generator introspection escapes.
_FORBIDDEN_ATTRS = {"format", "format_map", "mro"}
_FORBIDDEN_ATTR_PREFIXES = ("_", "gi_", "cr_", "ag_", "f_", "tb_")


class UnsafeExpressionError(ValueError):
    """Raised when a registry implementation violates the pure-expression policy."""


def _compile_restricted(source: str, expr_id: str):
    """Parse, validate and compile a registry implementation. Never executes it."""
    try:
        tree = ast.parse(source, filename=f"<expression {expr_id}>", mode="exec")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Expression {expr_id} has invalid syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeExpressionError(
                f"Expression {expr_id} uses forbidden construct {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _FORBIDDEN_ATTRS or attr.startswith(_FORBIDDEN_ATTR_PREFIXES):
                raise UnsafeExpressionError(
                    f"Expression {expr_id} accesses forbidden attribute '{attr}'")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeExpressionError(
                f"Expression {expr_id} references forbidden name '{node.id}'")
    return compile(tree, f"<expression {expr_id}>", "exec")


def run_restricted(source: str, expr_id: str, args, kwargs):
    """In-process execution of an already-policy-checked implementation."""
    import collections
    import functools
    import itertools
    import math
    ns: Dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "math": math, "itertools": itertools, "functools": functools, "collections": collections,
    }
    exec(_compile_restricted(source, expr_id), ns)
    fn = ns.get(expr_id) or ns.get(expr_id.lower())
    if not callable(fn):
        raise TypeError(f"Expression {expr_id} is not callable in its definition")
    return fn(*args, **kwargs)


# --------------------------------------------------------------------------- worker side

def _apply_self_limits() -> None:
    """Best effort; failures are reported to the parent, which then refuses the result."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes as w

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = w.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        k32.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        k32.GetCurrentProcess.restype = w.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObject failed")
        lim = EXTENDED()
        # ACTIVE_PROCESS(0x8) | PROCESS_MEMORY(0x100) | KILL_ON_JOB_CLOSE(0x2000) | PER_PROCESS_TIME(0x2)
        lim.Basic.LimitFlags = 0x8 | 0x100 | 0x2000 | 0x2
        lim.Basic.ActiveProcessLimit = 1
        lim.Basic.PerProcessUserTimeLimit = CPU_SECONDS * 10_000_000  # 100ns units
        lim.ProcessMemoryLimit = MEMORY_LIMIT_BYTES
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(lim), ctypes.sizeof(lim)):
            raise OSError("SetInformationJobObject failed")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            raise OSError("AssignProcessToJobObject failed")
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


def _decode(value):
    if isinstance(value, dict) and set(value) == {"__builtin__"}:
        name = value["__builtin__"]
        if name not in PASSABLE_BUILTINS:
            raise UnsafeExpressionError(f"Builtin '{name}' may not be passed to an operator")
        return PASSABLE_BUILTINS[name]
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    return value


def _json_default(obj):
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(obj)
        except TypeError:
            return list(obj)
    if isinstance(obj, (tuple, bytes, bytearray)):
        return list(obj)
    return str(obj)


def _worker_main() -> int:
    out = sys.stdout
    try:
        request = json.loads(sys.stdin.buffer.read(MAX_OUTPUT_BYTES).decode("utf-8"))
        _apply_self_limits()
        result = run_restricted(
            request["source"], request["expr_id"],
            _decode(request.get("args", [])), _decode(request.get("kwargs", {})))
        payload = {"ok": True, "result": result}
        text = json.dumps(payload, default=_json_default)
    except BaseException as exc:  # report every failure, including MemoryError
        text = json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:2000]})
    out.write(RESULT_MARKER + text)
    out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
