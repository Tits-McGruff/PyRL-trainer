"""Unit tests for actor rate limiting and reconnect state."""

# pylint: disable=import-error,protected-access

import asyncio

import numpy as np
import pytest

from pyrl_trainer.agent import ActorClient, Transition
from pyrl_trainer.config import Config
from pyrl_trainer.sensor_contract import SensorContract, SensorContractError

pytestmark = pytest.mark.unit


class DummySharedState:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for SharedState."""

    sensor_contract = None

    def act(self, _obs, **_kwargs):
        """Return fixed policy outputs."""
        return 0.0, 0.0, 0.0, 0.0, 0.0


def _make_actor(cfg: Config) -> ActorClient:
    """Build a test actor with dummy shared state."""
    return ActorClient(0, cfg, DummySharedState(), asyncio.Queue())


def test_should_send_action_stride():
    """Stride-based gating only sends on allowed ticks."""
    cfg = Config(max_actions_per_second=120)
    actor = _make_actor(cfg)
    actor.stride = 2

    assert actor._should_send_action(1) is True
    actor.last_sent_tick = 1

    assert actor._should_send_action(2) is False
    assert actor._should_send_action(3) is True


def test_should_send_action_time_gate(monkeypatch):
    """Monotonic time gating prevents sending too quickly."""
    cfg = Config(max_actions_per_second=2)
    actor = _make_actor(cfg)
    actor.stride = 1
    actor.last_sent_tick = 10
    actor.last_sent_time = 1.0

    monkeypatch.setattr("pyrl_trainer.agent.time.monotonic", lambda: 1.1)
    assert actor._should_send_action(11) is False

    monkeypatch.setattr("pyrl_trainer.agent.time.monotonic", lambda: 1.6)
    assert actor._should_send_action(11) is True


def test_welcome_rejects_changed_sensor_contract():
    """An actor refuses a server layout that disagrees with its model contract."""
    shared_state = DummySharedState()
    shared_state.sensor_contract = SensorContract("v3", ("a", "b"))
    actor = ActorClient(0, Config(), shared_state, asyncio.Queue())

    welcome = {
        "protocolVersion": 2,
        "tickRate": 60,
        "sensorSpec": {
            "sensorCount": 2,
            "layoutVersion": "v3",
            "order": ["b", "a"],
        },
    }

    with pytest.raises(SensorContractError):
        actor._apply_welcome(welcome)


@pytest.mark.asyncio
async def test_disconnect_truncates_rollout_and_reclaim_preserves_episode():
    """Disconnect keeps completed PPO work and reclaim keeps the same episode."""
    actor = _make_actor(Config(max_actions_per_second=120))
    obs = np.array([0.25], dtype=np.float32)

    actor.snake_id = 7
    actor.resume_token = "a" * 32
    actor.episodes = 3
    actor.episode_return = 1.25
    actor.last_obs = obs
    actor.rollout.append(
        Transition(
            obs=obs,
            action_turn=0.1,
            turn_latent=0.1003,
            action_boost=0.0,
            logp=-0.2,
            value=0.4,
            reward=0.3,
            done=0.0,
        )
    )
    actor.pending_transition = Transition(
        obs=obs,
        action_turn=0.2,
        turn_latent=0.2027,
        action_boost=1.0,
        logp=-0.3,
        value=0.75,
        reward=0.0,
        done=0.0,
    )

    await actor._truncate_disconnected_rollout()

    actor_id, transitions, bootstrap = actor.experience_q.get_nowait()
    assert actor_id == 0
    assert len(transitions) == 1
    assert bootstrap == pytest.approx(0.75)
    assert not actor.rollout
    assert actor.pending_transition is None
    assert actor.last_obs is None
    assert actor.snake_id == 7
    assert actor.resume_token == "a" * 32
    assert actor.episode_return == pytest.approx(1.25)

    await actor._on_assign(7, "b" * 32, reclaimed=True)

    assert actor.snake_id == 7
    assert actor.resume_token == "b" * 32
    assert actor.episodes == 3
    assert actor.episode_return == pytest.approx(1.25)
