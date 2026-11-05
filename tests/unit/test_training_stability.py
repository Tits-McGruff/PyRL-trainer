"""Regression tests for PPO policy coherence and best-policy retention."""

# pylint: disable=import-error,protected-access

import numpy as np
import pytest
import torch

from pyrl_trainer.agent import Transition
from pyrl_trainer.checkpointing import (
    load_checkpoint_if_present,
    save_best_checkpoint,
    save_checkpoint,
)
from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState, split_rollout_by_policy_version
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


def _transition(version: int, value: float) -> Transition:
    return Transition(
        obs=np.zeros(2, dtype=np.float32),
        action_turn=0.0,
        turn_latent=0.0,
        action_boost=0.0,
        logp=0.0,
        value=value,
        reward=0.1,
        done=0.0,
        policy_version=version,
    )


def test_stale_batch_is_rejected_before_any_optimizer_step():
    """Already-off-policy behavior is discarded before PPO mutates the model."""
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

    assert metrics["stale_batch"] == 1.0
    assert metrics["kl_early_stopped"] == 0.0
    assert metrics["optimizer_steps"] == 0.0
    assert metrics["initial_kl"] > metrics["target_kl"]
    assert state.update_steps == 0
    for key, value in state.train_model.state_dict().items():
        assert torch.equal(value, before[key])


def test_rollout_is_split_at_policy_version_boundaries():
    """Mixed live rollouts become contiguous behavior-policy segments before PPO."""
    rollout = [
        _transition(10, 0.1),
        _transition(10, 0.2),
        _transition(12, 0.3),
        _transition(12, 0.4),
    ]

    segments = split_rollout_by_policy_version(rollout, bootstrap_value=9.0)

    assert [version for version, _segment, _boot in segments] == [10, 12]
    assert [len(segment) for _version, segment, _boot in segments] == [2, 2]
    assert segments[0][2] == pytest.approx(0.3)
    assert segments[1][2] == pytest.approx(9.0)


def test_frozen_evaluator_policy_does_not_change_when_training_publishes():
    """An evaluator episode keeps one immutable policy while training advances."""
    cfg = _config()
    state = _state(cfg)
    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.fill_(0.25)
    state.sync_infer_from_train()

    snapshot, evaluator_model = state.begin_evaluation_policy()
    frozen = {
        key: value.detach().clone()
        for key, value in evaluator_model.state_dict().items()
    }

    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.add_(1.0)
    state.update_steps = 9
    state.sync_infer_from_train()

    assert snapshot["update_steps"] == 0
    assert state.published_policy_version == 9
    for key, value in evaluator_model.state_dict().items():
        assert torch.equal(value, frozen[key])
        assert torch.equal(value.cpu(), snapshot["model"][key])


def test_best_checkpoint_contains_exact_evaluator_policy_and_survives_resume(tmp_path):
    """The archived best model is the frozen policy that actually ran the episode."""
    cfg = _config(tmp_path)
    state = _state(cfg)

    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.fill_(0.25)
    state.sync_infer_from_train()
    snapshot, _evaluator_model = state.begin_evaluation_policy()

    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.fill_(0.75)
    state.update_steps = 42
    state.sync_infer_from_train()

    state.record_episode(
        1.3,
        17996,
        policy_snapshot=snapshot,
        include_runtime_stats=False,
    )
    assert state.best_checkpoint_pending
    assert state.best_lifetime_ticks == 17996
    assert state.best_episode_return == pytest.approx(1.3)

    best_path = save_best_checkpoint(cfg, state)
    state.mark_best_checkpoint_saved()

    assert best_path == tmp_path / "best_h8_l1.pt"
    assert best_path.exists()
    assert (tmp_path / "best.pt").exists()
    best_payload = torch.load(best_path, map_location="cpu")
    assert best_payload["best_lifetime_ticks"] == 17996
    assert best_payload["best_episode_return"] == pytest.approx(1.3)
    assert best_payload["update_steps"] == snapshot["update_steps"]
    assert best_payload["policy_snapshot"] is True
    assert best_payload["optimizer_reset_for_policy_snapshot"] is True
    for key, value in best_payload["model"].items():
        assert torch.equal(value, snapshot["model"][key])
        assert not torch.equal(value, state.train_model.state_dict()[key].cpu())

    state.record_episode(-0.4, 360)
    assert not state.best_checkpoint_pending
    assert state.best_lifetime_ticks == 17996

    save_checkpoint(cfg, state)
    restored = _state(cfg)
    loaded = load_checkpoint_if_present(cfg, restored)

    assert loaded is not None
    assert restored.update_steps == 42
    assert restored.best_lifetime_ticks == 17996
    assert restored.best_episode_return == pytest.approx(1.3)
    assert restored.best_policy_snapshot is None
    assert not restored.best_checkpoint_pending


def test_evaluator_promotion_uses_return_and_accepts_negative_first_result(tmp_path):
    """A long idle life cannot displace a shorter policy with greater task return."""
    state = _state(_config(tmp_path))
    snapshot, _model = state.begin_evaluation_policy()
    state.record_episode(-0.4, 100, policy_snapshot=snapshot, include_runtime_stats=False)
    assert state.best_episode_return == pytest.approx(-0.4)
    assert state.best_checkpoint_pending
    state.mark_best_checkpoint_saved()
    state.record_episode(-0.45, 20_000, policy_snapshot=snapshot, include_runtime_stats=False)
    assert not state.best_checkpoint_pending
    assert state.best_lifetime_ticks == 100
    state.record_episode(0.2, 50, policy_snapshot=snapshot, include_runtime_stats=False)
    assert state.best_checkpoint_pending
    assert state.best_lifetime_ticks == 50
    assert state.best_episode_return == pytest.approx(0.2)
