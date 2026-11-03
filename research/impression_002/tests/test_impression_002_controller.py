from types import SimpleNamespace

import pytest

from experiments.impression_002.configurations.structural_config import (
    ConfigurationStore,
    StructuralConfiguration,
)
from experiments.impression_002.controller.efficiency_controller import (
    AdaptiveEfficiencyController,
    BlockCostProfile,
    ExperienceLedger,
    ResourceEnvelope,
)


REV = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
REGISTRY = "a" * 64


def profile(profile_id, config_id, block_id, seconds, before, after, *, placement="cpu", rss=7_000_000_000, vram=None):
    return BlockCostProfile(
        profile_id=profile_id,
        config_id=config_id,
        config_version=1 if config_id == "baseline" else 2,
        task_id="tool-sft-v1",
        block_id=block_id,
        block_version=1,
        checkpoint_sha256="b" * 64,
        placement=placement,
        status="success",
        parameter_count=25_165_824,
        trainable_parameter_count=45_056,
        forward_seconds=seconds * 0.65,
        backward_seconds=seconds * 0.25,
        optimizer_seconds=seconds * 0.05,
        total_wall_seconds=seconds,
        setup_seconds=10.0,
        peak_rss_bytes=rss,
        ram_available_before_bytes=16_000_000_000,
        ram_total_bytes=24_000_000_000,
        peak_vram_bytes=vram,
        vram_available_before_bytes=4_000_000_000,
        transferred_bytes=0 if placement == "cpu" else 1024,
        transfer_seconds=0.0 if placement == "cpu" else 0.002,
        activation_bytes=8_000_000,
        gradient_l2_norm=0.25,
        training_loss=1.1,
        validation_before=before,
        validation_after=after,
        heldout_ids=("heldout-1", "heldout-2"),
        training_ids=("train-1",),
        objective_fingerprint="objective-v1",
        steps=1,
        adapter_sha256="c" * 64,
        adapter_version="lora-r4-alpha8-v1",
        dependencies=("IMP_L00_FEED_FORWARD_INPUT_NORM",),
        dtype="bfloat16",
        hardware={"cpu": "test-cpu", "gpu": "none"},
        runtime_versions={"torch": "2.5.1"},
        base_fingerprint_before="d" * 64,
        base_fingerprint_after="d" * 64,
        reload_parity_passed=True,
    )


def test_controller_selects_measured_best_efficiency_without_hardcoded_region():
    baseline1 = profile("b1", "baseline", "IMP_L00_FEED_FORWARD_GATE_PROJECTION", 100, 1.0, 0.95)
    baseline2 = profile("b2", "baseline", "IMP_L00_FEED_FORWARD_GATE_PROJECTION", 102, 1.0, 0.95)
    candidate1 = profile("c1", "candidate-q", "IMP_L00_ATTENTION_QUERY_PROJECTION", 50, 1.0, 0.90)
    candidate2 = profile("c2", "candidate-q", "IMP_L00_ATTENTION_QUERY_PROJECTION", 52, 1.0, 0.90)
    other = profile("d1", "candidate-up", "IMP_L00_FEED_FORWARD_UP_PROJECTION", 80, 1.0, 0.97)
    choice = AdaptiveEfficiencyController().select(
        [baseline1, baseline2, candidate1, candidate2, other],
        baseline_config_id="baseline",
        resources=ResourceEnvelope(ram_available_bytes=16_000_000_000, ram_reserve_bytes=4_000_000_000),
    )
    assert choice.config_id == "candidate-q"
    assert choice.block_id == "IMP_L00_ATTENTION_QUERY_PROJECTION"
    assert choice.basis == "learning_gain_per_second"
    assert choice.evidence["repeat_count"] == 2


def test_controller_rejects_infeasible_or_unmatched_profiles():
    baseline = profile("b", "baseline", "gate", 100, 1.0, 0.9)
    gpu = profile("g", "gpu-candidate", "query", 20, 1.0, 0.5, placement="cuda:0", vram=5_000_000_000)
    changed_data = profile("x", "other-candidate", "up", 40, 1.0, 0.5)
    changed_data = BlockCostProfile.from_dict({**changed_data.to_dict(), "heldout_ids": ["different-heldout"]})
    choice = AdaptiveEfficiencyController().select(
        [baseline, gpu, changed_data], baseline_config_id="baseline",
        resources=ResourceEnvelope(ram_available_bytes=16_000_000_000, ram_reserve_bytes=4_000_000_000, vram_available_bytes=4_000_000_000),
    )
    assert choice.config_id == "baseline"
    assert choice.basis == "retain_measured_baseline"
    assert choice.evidence["rejected_alternatives"]


def test_controller_rejects_slower_candidate_that_has_worse_gain_per_second():
    baseline = profile("b", "baseline", "gate", 100, 1.0, 0.9)
    slower = profile("s", "slower", "query", 110, 1.0, 0.9)
    choice = AdaptiveEfficiencyController().select(
        [baseline, slower], baseline_config_id="baseline",
        resources=ResourceEnvelope(ram_available_bytes=16_000_000_000, ram_reserve_bytes=4_000_000_000),
    )
    assert choice.config_id == "baseline"
    assert choice.basis == "retain_measured_baseline"


def _config(config_id, blocks, trainable, version, parent=None, placements=None):
    return StructuralConfiguration(
        config_id=config_id,
        config_version=version,
        checkpoint_revision=REV,
        registry_sha256=REGISTRY,
        active_blocks=tuple(blocks),
        trainable_regions=tuple(trainable),
        dependencies={blocks[1]: (blocks[0],)} if len(blocks) > 1 else {},
        placements=placements or {block: "cpu" for block in blocks},
        adapter_refs={trainable[0]: "adapter.safetensors"} if trainable else {},
        parent_config_id=parent,
    )


def test_config_store_retains_or_rolls_back_transitions_and_restores_after_restart(tmp_path):
    store = ConfigurationStore(tmp_path / "configurations")
    baseline = _config("baseline", ["block-a"], ["block-a"], 1)
    events = []
    # Seed persisted current state through an accepted configuration transition.
    result = store.transition(
        baseline,
        apply=lambda config: events.append(("apply", config.config_id)),
        validate=lambda _: True,
        benchmark=lambda _: {"seconds": 2.0},
        accept=lambda _: True,
        rollback=lambda previous: events.append(("rollback", getattr(previous, "config_id", None))),
    )
    assert result["status"] == "retained"
    with pytest.raises(FileExistsError):
        store.save(baseline)

    rejected = _config("rejected", ["block-a", "block-b"], ["block-b"], 2, "baseline")
    result = store.transition(
        rejected,
        apply=lambda config: events.append(("apply", config.config_id)),
        validate=lambda _: True,
        benchmark=lambda _: {"quality": 0.4},
        accept=lambda _: False,
        rollback=lambda previous: events.append(("rollback", previous.config_id)),
    )
    assert result["status"] == "reverted"
    assert store.current().config_id == "baseline"
    assert events[-1] == ("rollback", "baseline")

    retained = _config("retained", ["block-a", "block-c"], ["block-c"], 2, "baseline", {"block-a": "cpu", "block-c": "cuda:0"})
    result = store.transition(
        retained,
        apply=lambda config: events.append(("apply", config.config_id)),
        validate=lambda _: True,
        benchmark=lambda _: {"quality": 0.5, "cost": 3.0},
        accept=lambda metrics: metrics["quality"] >= 0.5,
        rollback=lambda previous: events.append(("rollback", getattr(previous, "config_id", None))),
    )
    assert result["status"] == "retained"
    restarted = ConfigurationStore(tmp_path / "configurations")
    applied = []
    restored = restarted.restore_current(apply=lambda c: applied.append(c.config_id), validate=lambda _: True)
    assert restored.config_id == "retained"
    assert applied == ["retained"]


def test_config_rejects_unknown_registry_blocks_and_dangling_dependencies():
    config = _config("valid", ["known"], ["known"], 1)
    with pytest.raises(ValueError, match="absent from registry"):
        config.validate_against_registry(SimpleNamespace(partitions={"other": object()}, source_sha256=REGISTRY))
    with pytest.raises(ValueError, match="inactive block"):
        StructuralConfiguration(
            config_id="dangling", config_version=1, checkpoint_revision=REV, registry_sha256=REGISTRY,
            active_blocks=("known",), trainable_regions=(), dependencies={"known": ("missing",)},
        )


def test_experience_ledger_is_append_only_and_rejects_duplicate_event_id(tmp_path):
    ledger = ExperienceLedger(tmp_path / "experience.jsonl")
    ledger.append("profile", {"block_id": "block-a", "forward_seconds": 1.25}, event_id="profile-1")
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.append("profile", {"block_id": "block-a"}, event_id="profile-1")
    ledger.append("selection", {"config_id": "cfg-a", "basis": "measured"}, event_id="selection-1")
    rows = ledger.read()
    assert [row["event_id"] for row in rows] == ["profile-1", "selection-1"]
    assert rows[0]["payload"]["forward_seconds"] == 1.25
