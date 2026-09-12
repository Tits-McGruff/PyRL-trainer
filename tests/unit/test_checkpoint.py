"""Checkpoint durability and sensor-contract tests."""

# pylint: disable=import-error

import pytest
import torch

from pyrl_trainer.config import Config
from pyrl_trainer.checkpointing import load_checkpoint_if_present, save_checkpoint
from pyrl_trainer.learner import SharedState
from pyrl_trainer.sensor_contract import SensorContract

pytestmark = pytest.mark.unit


def _config(tmp_path) -> Config:
    return Config(
        ckpt_dir=str(tmp_path),
        keep_last=5,
        net_hidden=8,
        net_layers=1,
        train_device="cpu",
        infer_device="cpu",
        lr=3e-4,
    )


def test_checkpoint_recovers_from_corrupt_latest_files(tmp_path):
    """Newest numbered checkpoint recovers when both latest pointers are corrupt."""
    cfg = _config(tmp_path)
    contract = SensorContract("v3", ("a", "b"))
    state = SharedState(2, cfg)
    state.sensor_contract = contract

    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.fill_(0.25)
    expected = {
        key: value.detach().clone()
        for key, value in state.train_model.state_dict().items()
    }
    state.update_steps = 7
    save_checkpoint(cfg, state)

    assert not list(tmp_path.glob(".*.tmp"))
    (tmp_path / "latest.pt").write_bytes(b"corrupt")
    (tmp_path / "latest_h8_l1.pt").write_bytes(b"corrupt")

    restored = SharedState(2, cfg)
    restored.sensor_contract = contract
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded == tmp_path / "ckpt_h8_l1_00000007.pt"
    assert restored.update_steps == 7
    for key, value in restored.train_model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_checkpoint_rejects_same_width_different_sensor_order(tmp_path):
    """Input width alone cannot make a checkpoint sensor-compatible."""
    cfg = _config(tmp_path)
    saved_contract = SensorContract("v3", ("a", "b"))
    state = SharedState(2, cfg)
    state.sensor_contract = saved_contract
    state.update_steps = 3
    save_checkpoint(cfg, state)

    changed_contract = SensorContract("v3", ("b", "a"))
    restored = SharedState(2, cfg)
    restored.sensor_contract = changed_contract

    assert load_checkpoint_if_present(cfg, restored) is None
    assert restored.update_steps == 0
