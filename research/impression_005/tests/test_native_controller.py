import asyncio
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from core.capability_broker import CapabilityBroker, CapabilityEnvelope
from experiments.impression_005.native_controller import NativeImpressionBrokerAdapter
from experiments.impression_005.structural_runtime import (
    IdentityResidualConnector,
    install_cached_prefix,
    install_cached_prefix_live_connector,
    restore_original_layers,
)


def test_capability_broker_registers_and_dispatches_domain_extension():
    broker = CapabilityBroker()
    broker.register_capability_extension("impression", ["partition.inspect"],
                                         lambda operation, args: {"operation": operation})
    assert "partition.inspect" in broker.get_registered_tools()
    parsed = broker.parse_envelope(
        '{"action":"execute","capability":"impression","operation":"partition.inspect","arguments":{}}'
    )
    assert parsed is not None
    observation = asyncio.run(broker.dispatch(parsed[0]))
    assert observation.status == "SUCCESS"
    assert observation.observation == {"operation": "partition.inspect"}


def test_registered_adapter_validates_feasibility_without_execution():
    broker = CapabilityBroker()
    adapter = NativeImpressionBrokerAdapter(broker)
    snapshot = {
        "ram_total_bytes": 24_919_851_008,
        "ram_available_bytes": 13_000_000_000,
        "vram_total_bytes": 4 * 1024**3,
        "vram_available_bytes": 3 * 1024**3,
    }
    valid = CapabilityEnvelope("execute", "impression", "partition.select", {
        "origin": "local_[BASE_MODEL]", "configuration_id": "sparse_supervised_cpu1",
        "partition_id": "IMP_L27_ATTENTION_QUERY_PROJECTION", "resource_snapshot": snapshot,
    })
    result = asyncio.run(broker.dispatch(valid))
    assert result.status == "SUCCESS"
    assert result.observation["status"] == "validated_proposal_only"
    assert result.observation["execution_performed"] is False

    unsafe = CapabilityEnvelope("execute", "impression", "partition.select", {
        "origin": "local_[BASE_MODEL]", "configuration_id": "sparse_cuda_head_cpu_suffix",
        "partition_id": "IMP_L27_ATTENTION_QUERY_PROJECTION",
        "resource_snapshot": {**snapshot, "vram_available_bytes": 900 * 1024**2},
    })
    rejected = asyncio.run(broker.dispatch(unsafe))
    assert rejected.status == "GATING_ERROR"


def test_expansion_and_reconnection_are_not_registered():
    broker = CapabilityBroker()
    NativeImpressionBrokerAdapter(broker)
    assert "partition.expand" not in broker.get_registered_tools()
    assert "partition.connect" not in broker.get_registered_tools()
    unsupported = CapabilityEnvelope("execute", "impression", "partition.expand", {})
    result = asyncio.run(broker.dispatch(unsupported))
    assert result.status == "VALIDATION_ERROR"


def test_proposal_checks_every_repeated_profile_for_resource_safety():
    broker = CapabilityBroker()
    adapter = NativeImpressionBrokerAdapter(broker)
    group = adapter.profiles_by_config["sparse_supervised_cpu1"]
    adapter.profiles_by_config["sparse_supervised_cpu1"] = [
        group[0], replace(group[0], profile_id="synthetic-safety-test-repeat",
                          peak_rss_bytes=2 * 1024**3)
    ]
    snapshot = {
        "ram_total_bytes": 24_919_851_008,
        "ram_available_bytes": 6_800_000_000,
        "vram_total_bytes": 4 * 1024**3,
        "vram_available_bytes": 3 * 1024**3,
    }
    proposal = CapabilityEnvelope("execute", "impression", "partition.select", {
        "origin": "local_[BASE_MODEL]", "configuration_id": "sparse_supervised_cpu1",
        "partition_id": "IMP_L27_ATTENTION_QUERY_PROJECTION", "resource_snapshot": snapshot,
    })
    result = asyncio.run(broker.dispatch(proposal))
    assert result.status == "GATING_ERROR"
    assert result.observation["repeated_profile_checks"][-1]["feasible"] is False


def test_repeated_hybrid_proposal_rejects_mixed_pass_fail_profiles():
    broker = CapabilityBroker()
    adapter = NativeImpressionBrokerAdapter(broker)
    first, second = adapter.profiles_by_config["sparse_cuda_head_cpu_suffix"]
    adapter.profiles_by_config["sparse_cuda_head_cpu_suffix"] = [
        first, replace(second, peak_vram_bytes=2_400_000_000)
    ]
    snapshot = {
        "ram_total_bytes": 24_919_851_008,
        "ram_available_bytes": 12_000_000_000,
        "vram_total_bytes": 4 * 1024**3,
        "vram_available_bytes": 3 * 1024**3,
    }
    proposal = CapabilityEnvelope("execute", "impression", "partition.select", {
        "origin": "local_[BASE_MODEL]", "configuration_id": "sparse_cuda_head_cpu_suffix",
        "partition_id": "IMP_L27_ATTENTION_QUERY_PROJECTION", "resource_snapshot": snapshot,
    })
    result = asyncio.run(broker.dispatch(proposal))
    assert result.status == "GATING_ERROR"
    checks = result.observation["repeated_profile_checks"]
    assert len(checks) == 2
    assert checks[0]["feasible"] is True
    assert checks[1]["feasible"] is False


def test_structural_connector_is_tensor_compatible_and_identity_initialized():
    model = SimpleNamespace(model=SimpleNamespace(layers=nn.ModuleList([nn.Identity() for _ in range(28)])))
    activations = [torch.randn(1, 3, 8) for _ in range(27)]
    original = install_cached_prefix(model, activations, IdentityResidualConnector(hidden_size=8, rank=2))
    assert torch.equal(model.model.layers[26](torch.zeros_like(activations[26])), activations[26])
    assert torch.equal(model.model.layers[0](torch.zeros_like(activations[0])), activations[0])
    restore_original_layers(model, original)
    assert isinstance(model.model.layers[26], nn.Identity)

    original = install_cached_prefix_live_connector(
        model, activations, IdentityResidualConnector(hidden_size=8, rank=2)
    )
    live = model.model.layers[26](activations[25])
    assert isinstance(live, torch.Tensor)
    assert torch.equal(live, activations[25])
    restore_original_layers(model, original)
