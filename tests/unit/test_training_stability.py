"""Regression tests for PPO stability safeguards and best-policy retention."""

# pylint: disable=import-error,protected-access

import pytest
import torch

from pyrl_trainer.checkpointing import (
    load_checkpoint_if_present,
    save_best_checkpoint,
    save_checkpoint,
)
from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState
from pyrl_trainer.sensor_contract import SensorContract

pytestmark = pytest.mark.unit


def _config(tmp_path=None, **kwargs) -> Config:
    values = {
        "batch_size": 4,
        "minibatch": 2,
        "epochs": 4,
        "net_hidden": 8,
        "net_layers": 1,
        "train_device": "cpu",
        "infer_device": "cpu",
        "turn_std": 0.2,
    }
    if tmp_path is not None:
        values["ckpt_dir"] = str(tmp_path)
    values.update(kwargs)
    return Config(**values)


def _state(cfg: Config) -> SharedState:
    return SharedState(
        obs_dim=2,
        cfg=cfg,
        sensor_contract=SensorContract("v3", ("a", "b")),
    )


def test_ppo_skips_update_when_policy_is_already_over_target_kl():
    """A stale/high-drift minibatch cannot push the policy even farther away."""
    torch.manual_seed(123)
    cfg = _config()
    state = _state(cfg)

    obs = torch.zeros((4, 2), dtype=torch.float32)
    latent = torch.zeros(4, dtype=torch.float32)
    boost = torch.zeros(4, dtype=torch.float32)
    with torch.no_grad():
        turn_mean, boost_logit, _value = state.train_model(obs)
        _turn, current_logp, _turn_entropy, _boost_entropy = (
            state._joint_logp_entropy(
                turn_mean,
                boost_logit,
                latent,
                boost,
                cfg.turn_std,
            )
        )

    batch = {
        "obs": obs,
        "act_turn_latent": latent,
        "act_boost": boost,
        "old_logp": current_logp + 1.0,
        "returns": torch.zeros(4, dtype=torch.float32),
        "adv": torch.ones(4, dtype=torch.float32),
    }
    before = {
        key: value.detach().clone()
        for key, value in state.train_model.state_dict().items()
    }

    metrics = state._ppo_update_batch_sync(batch)

    assert metrics["kl_early_stopped"] == 1.0
    assert metrics["optimizer_steps"] == 0.0
    assert metrics["approx_kl"] > metrics["target_kl"]
    assert state.update_steps == 0
    for key, value in state.train_model.state_dict().items():
        assert torch.equal(value, before[key])


def test_best_checkpoint_tracks_lifetime_record_and_survives_resume(tmp_path):
    """A long-lived policy is kept separately and its record persists in latest saves."""
    cfg = _config(tmp_path)
    state = _state(cfg)

    state.record_episode(1.3, 17996)
    assert state.best_checkpoint_pending
    assert state.best_lifetime_ticks == 17996
    assert state.best_episode_return == pytest.approx(1.3)

    state.update_steps = 42
    best_path = save_best_checkpoint(cfg, state)
    state.mark_best_checkpoint_saved()

    assert best_path == tmp_path / "best_h8_l1.pt"
    assert best_path.exists()
    assert (tmp_path / "best.pt").exists()
    best_payload = torch.load(best_path, map_location="cpu")
    assert best_payload["best_lifetime_ticks"] == 17996
    assert best_payload["best_episode_return"] == pytest.approx(1.3)

    state.record_episode(-0.4, 360)
    assert not state.best_checkpoint_pending
    assert state.best_lifetime_ticks == 17996

    save_checkpoint(cfg, state)
    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded is not None
    assert restored.best_lifetime_ticks == 17996
    assert restored.best_episode_return == pytest.approx(1.3)
    assert not restored.best_checkpoint_pending
