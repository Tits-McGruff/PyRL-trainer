"""Focused PPO policy and minibatch tests."""

# pylint: disable=import-error,protected-access

import numpy as np
import pytest
import torch

from pyrl_trainer.agent import Transition
from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState, collate_rollouts

pytestmark = pytest.mark.unit


def _config(**kwargs) -> Config:
    values = {
        "batch_size": 8,
        "minibatch": 3,
        "epochs": 2,
        "net_hidden": 8,
        "net_layers": 1,
        "train_device": "cpu",
        "infer_device": "cpu",
        "turn_std": 0.2,
    }
    values.update(kwargs)
    return Config(**values)


def test_actor_logp_matches_stored_latent_evaluation():
    """Actor and learner evaluate the same latent action with one formula."""
    torch.manual_seed(1234)
    cfg = _config()
    state = SharedState(obs_dim=3, cfg=cfg)
    obs = np.array([0.2, -0.1, 0.4], dtype=np.float32)

    turn, boost, logp, _value, latent = state.act(obs, cfg.turn_std)

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        turn_mean, boost_logit, _ = state.infer_model(obs_t)
        replay_turn, replay_logp, _ = state._joint_logp_entropy(
            turn_mean,
            boost_logit,
            torch.tensor([latent], dtype=torch.float32),
            torch.tensor([boost], dtype=torch.float32),
            cfg.turn_std,
        )

    assert float(replay_turn.item()) == pytest.approx(turn, abs=1e-6)
    assert float(replay_logp.item()) == pytest.approx(logp, abs=1e-5)


def test_ppo_update_uses_all_shuffled_minibatches():
    """Each epoch covers the batch in minibatches, including the short tail."""
    torch.manual_seed(7)
    cfg = _config(batch_size=8, minibatch=3, epochs=2)
    state = SharedState(obs_dim=2, cfg=cfg)

    transitions = []
    for index in range(8):
        obs = np.array([index / 10.0, -index / 20.0], dtype=np.float32)
        turn, boost, logp, value, latent = state.act(obs, cfg.turn_std)
        transitions.append(
            Transition(
                obs=obs,
                action_turn=turn,
                turn_latent=latent,
                action_boost=boost,
                logp=logp,
                value=value,
                reward=0.1,
                done=0.0,
            )
        )

    batch = collate_rollouts([(transitions, 0.0)], cfg, "cpu")
    metrics = state._ppo_update_batch_sync(batch)

    assert metrics["samples"] == 8.0
    assert metrics["optimizer_steps"] == 6.0
    assert state.update_steps == 6
    assert np.isfinite(metrics["approx_kl"])
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
    assert np.isfinite(metrics["explained_variance"])


def test_runtime_diagnostics_are_bounded_and_drained():
    """Episode and reward stats reset after one log snapshot."""
    state = SharedState(obs_dim=1, cfg=_config())

    state.record_reward_components(
        {"growth": 1.5, "food_approach": 0.25, "death": -0.5}
    )
    state.record_episode(2.0, 120)

    first = state.drain_runtime_stats()
    second = state.drain_runtime_stats()

    assert first["episodes"] == 1.0
    assert first["episode_return_mean"] == pytest.approx(2.0)
    assert first["episode_lifetime_mean"] == pytest.approx(120.0)
    assert first["reward_growth"] == pytest.approx(1.5)
    assert first["reward_food_approach"] == pytest.approx(0.25)
    assert first["reward_death"] == pytest.approx(-0.5)
    assert second["episodes"] == 0.0
    assert second["reward_growth"] == 0.0
