from core.symbolic_cognition_adapter import SymbolicCognitionAdapter


def test_existing_adapter_exposes_typed_validator_without_execution():
    adapter = object.__new__(SymbolicCognitionAdapter)
    valid, errors, parsed = adapter.validate_typed_plan({
        "version": 1, "objective": "compute a primitive", "intent": "compute",
        "variables": [], "nodes": [{"id": "return", "kind": "return", "value": "not executed"}],
    })
    assert valid and not errors
    assert parsed.objective == "compute a primitive"
    valid, errors, parsed = adapter.validate_typed_plan({
        "version": 1, "objective": "invalid", "intent": "execute", "variables": [],
        "nodes": [{"id": "action", "kind": "action", "capability": "web",
                   "operation": "execute", "arguments": {}}],
    })
    assert not valid and parsed is None
    assert "unsupported operation" in errors[0]
