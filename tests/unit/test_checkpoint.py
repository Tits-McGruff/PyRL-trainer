"""Checkpoint durability, schema, and compatibility tests."""

# pylint: disable=import-error

import copy

import pytest
import torch

from pyrl_trainer.checkpointing import load_checkpoint_if_present, save_checkpoint
from pyrl_trainer.config import CHECKPOINT_SCHEMA_VERSION, Config
from pyrl_trainer.learner import SharedState
from pyrl_trainer.sensor_contract import SensorContract

pytestmark = pytest.mark.unit


def _config(tmp_path, **kwargs) -> Config:
    values = {
        "ckpt_dir": str(tmp_path),
        "keep_last": 5,
        "net_hidden": 8,
        "net_layers": 1,
        "train_device": "cpu",
        "infer_device": "cpu",
        "lr": 3e-4,
    }
    values.update(kwargs)
    return Config(**values)


def _state(cfg: Config, contract=None) -> SharedState:
    selected_contract = contract or SensorContract("v3", ("a", "b"))
    return SharedState(2, cfg, sensor_contract=selected_contract)


def _clone_model(state: SharedState):
    return {
        key: value.detach().clone()
        for key, value in state.train_model.state_dict().items()
    }


def test_checkpoint_recovers_from_corrupt_latest_files(tmp_path):
    """Newest numbered checkpoint recovers when both latest pointers are corrupt."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.fill_(0.25)
    expected = _clone_model(state)
    state.update_steps = 7
    save_checkpoint(cfg, state)

    assert not list(tmp_path.glob(".*.tmp"))
    (tmp_path / "latest.pt").write_bytes(b"corrupt")
    (tmp_path / "latest_h8_l1.pt").write_bytes(b"corrupt")

    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded == tmp_path / "ckpt_h8_l1_00000007.pt"
    assert restored.update_steps == 7
    for key, value in restored.train_model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_checkpoint_rejects_same_width_different_sensor_order(tmp_path):
    """Input width alone cannot make a checkpoint sensor-compatible."""
    cfg = _config(tmp_path)
    saved = _state(cfg, SensorContract("v3", ("a", "b")))
    saved.update_steps = 3
    save_checkpoint(cfg, saved)

    restored = _state(cfg, SensorContract("v3", ("b", "a")))

    assert load_checkpoint_if_present(cfg, restored) is None
    assert restored.update_steps == 0


def test_stale_latest_pointer_does_not_hide_newer_numbered_checkpoint(tmp_path):
    """Candidate ranking uses update_steps rather than latest-pointer role."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    state.update_steps = 7
    save_checkpoint(cfg, state)
    old_latest = (tmp_path / "latest.pt").read_bytes()
    old_arch = (tmp_path / "latest_h8_l1.pt").read_bytes()

    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.add_(1.0)
    expected = _clone_model(state)
    state.update_steps = 8
    save_checkpoint(cfg, state)

    (tmp_path / "latest.pt").write_bytes(old_latest)
    (tmp_path / "latest_h8_l1.pt").write_bytes(old_arch)

    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded == tmp_path / "ckpt_h8_l1_00000008.pt"
    assert restored.update_steps == 8
    for key, value in restored.train_model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_new_checkpoint_contains_reward_schema(tmp_path):
    """Every newly written checkpoint carries schema and reward metadata."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    state.update_steps = 1
    save_checkpoint(cfg, state)

    payload = torch.load(tmp_path / "latest.pt", map_location="cpu")

    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["reward_config_version"] == 1
    assert payload["reward_config"]["reward_death_penalty_magnitude"] == pytest.approx(
        0.5
    )


def test_reward_mismatch_rejects_schema_v2_checkpoint(tmp_path):
    """Automatic resume rejects a checkpoint produced under different rewards."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    state.update_steps = 4
    save_checkpoint(cfg, state)

    changed = _config(tmp_path, reward_growth_scale=11.0)
    restored = _state(changed)

    assert load_checkpoint_if_present(changed, restored) is None
    assert restored.update_steps == 0


def test_legacy_checkpoint_loads_only_under_canonical_defaults(tmp_path):
    """Legacy checkpoints are accepted only when current rewards are old defaults."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    state.update_steps = 5
    legacy = {
        "update_steps": 5,
        "obs_dim": 2,
        "net_hidden": 8,
        "net_layers": 1,
        "model": state.train_model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        **state.sensor_contract.checkpoint_fields(),
    }
    torch.save(legacy, tmp_path / "latest.pt")

    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)
    assert loaded == tmp_path / "latest.pt"
    assert restored.update_steps == 5

    changed = _config(tmp_path, reward_survival_bonus=0.001)
    changed_state = _state(changed)
    assert load_checkpoint_if_present(changed, changed_state) is None
    assert changed_state.update_steps == 0


def test_future_schema_is_skipped_without_mutation(tmp_path):
    """An unknown future schema is skipped and cannot displace a valid candidate."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    state.update_steps = 2
    save_checkpoint(cfg, state)

    payload = torch.load(tmp_path / "latest.pt", map_location="cpu")
    payload["checkpoint_schema_version"] = 999
    payload["update_steps"] = 99
    torch.save(payload, tmp_path / "ckpt_h8_l1_00000099.pt")

    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded is not None
    assert restored.update_steps == 2


def test_failed_application_rolls_back_before_fallback(tmp_path):
    """A failed optimizer restore cannot leave the model partially replaced."""
    cfg = _config(tmp_path)
    state = _state(cfg)
    state.update_steps = 7
    save_checkpoint(cfg, state)
    expected = _clone_model(state)

    bad = copy.deepcopy(torch.load(tmp_path / "latest.pt", map_location="cpu"))
    bad["update_steps"] = 9
    bad["optimizer"] = {"broken": True}
    with torch.no_grad():
        for value in bad["model"].values():
            if hasattr(value, "add_"):
                value.add_(3.0)
    torch.save(bad, tmp_path / "ckpt_h8_l1_00000009.pt")

    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded is not None
    assert restored.update_steps == 7
    for key, value in restored.train_model.state_dict().items():
        assert torch.equal(value, expected[key])
