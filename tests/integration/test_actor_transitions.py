"""Integration tests for actor transitions."""

# pylint: disable=import-error,protected-access

import asyncio
import json

import numpy as np
import pytest

from pyrl_trainer.agent import ActorClient, Transition
from pyrl_trainer.config import Config

pytestmark = pytest.mark.integration


class DummySharedState:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for SharedState."""

    def act(self, _obs, **_kwargs):
        """Return fixed policy outputs."""
        return 0.1, 1.0, -0.5, 0.25, 0.1003


class DummyWS:  # pylint: disable=too-few-public-methods
    """Capture outgoing messages."""

    def __init__(self, received=None):
        self.sent = []
        self.received = list(received or [])

    async def send(self, data):
        """Store outgoing payloads."""
        self.sent.append(data)

    async def recv(self):
        """Return the next queued server payload."""
        return json.dumps(self.received.pop(0))


@pytest.mark.asyncio
async def test_transitions_align_with_stride(monkeypatch):
    """Pending transitions align with sent actions."""
    cfg = Config(max_actions_per_second=0, horizon=10)
    actor = ActorClient(0, cfg, DummySharedState(), asyncio.Queue())
    actor.snake_id = 1
    actor.stride = 2
    actor.sensor_order = ["points_delta_norm"]
    actor.sensor_idx = {"points_delta_norm": 0}

    monkeypatch.setattr(
        "pyrl_trainer.agent.default_reward_components",
        lambda prev, obs, idx, reward_cfg: {
            "growth": 1.0,
            "food_approach": 0.0,
            "survival": 0.0,
            "wall": 0.0,
            "hazard": 0.0,
        },
    )

    ws = DummyWS()

    for tick in [1, 2, 3]:
        msg = {
            "type": "sensors",
            "tick": tick,
            "snakeId": 1,
            "sensors": [float(tick)],
        }
        await actor._handle_sensors(msg, ws)

    assert len(ws.sent) == 2
    assert len(actor.rollout) == 1
    assert abs(actor.rollout[0].reward - 2.0) < 1e-6
    assert actor.pending_transition is not None
    assert abs(actor.pending_transition.reward) < 1e-6


@pytest.mark.asyncio
async def test_import_replacement_rejoins_without_stale_token():
    """A live import ends the old episode and sends one fresh Protocol 2 join."""
    actor = ActorClient(
        7,
        Config(max_actions_per_second=20),
        DummySharedState(),
        asyncio.Queue(),
    )
    actor.snake_id = 91
    actor.resume_token = "stale-token"
    actor.sensor_order = ["old"]
    actor.sensor_idx = {"old": 0}
    ws = DummyWS()

    await actor._handle_state_replaced(
        {
            "type": "stateReplaced",
            "welcome": {
                "protocolVersion": 2,
                "tickRate": 60,
                "sensorSpec": {"order": ["food_proximity"]},
            },
        },
        ws,
        "trainer-007",
    )

    assert actor.snake_id is None
    assert actor.resume_token is None
    assert actor.sensor_order == ["food_proximity"]
    assert json.loads(ws.sent[-1]) == {
        "type": "join",
        "mode": "player",
        "name": "trainer-007",
    }


@pytest.mark.asyncio
async def test_import_replacement_during_initial_assignment_wait():
    """A replacement arriving before the first assignment sends a fresh join."""
    actor = ActorClient(
        8,
        Config(max_actions_per_second=20),
        DummySharedState(),
        asyncio.Queue(),
    )
    actor.resume_token = "stale-token"
    ws = DummyWS(
        [
            {
                "type": "welcome",
                "protocolVersion": 2,
                "tickRate": 60,
                "sensorSpec": {"order": ["old"]},
            },
            {
                "type": "stateReplaced",
                "welcome": {
                    "protocolVersion": 2,
                    "tickRate": 30,
                    "sensorSpec": {"order": ["food_proximity"]},
                },
            },
            {"type": "assign", "snakeId": 12, "resumeToken": "fresh-token"},
        ]
    )

    await actor._handshake(ws, "trainer-008")

    sent = [json.loads(value) for value in ws.sent]
    assert sent[1]["resumeToken"] == "stale-token"
    assert sent[2] == {
        "type": "join",
        "mode": "player",
        "name": "trainer-008",
    }
    assert actor.snake_id == 12
    assert actor.resume_token == "fresh-token"
    assert actor.sensor_order == ["food_proximity"]


@pytest.mark.asyncio
async def test_terminal_death_uses_configured_penalty_once():
    """A genuine terminal episode receives exactly one configured death penalty."""
    cfg = Config(reward_death_penalty_magnitude=0.75)
    experience_q = asyncio.Queue()
    actor = ActorClient(0, cfg, DummySharedState(), experience_q)
    actor.snake_id = 1
    actor.pending_transition = Transition(
        obs=np.zeros(1, dtype=np.float32),
        action_turn=0.0,
        turn_latent=0.0,
        action_boost=0.0,
        logp=0.0,
        value=0.0,
        reward=0.25,
        done=0.0,
    )

    await actor._finalize_terminal_episode()
    _actor_id, rollout, bootstrap = await experience_q.get()

    assert bootstrap == 0.0
    assert len(rollout) == 1
    assert rollout[0].reward == pytest.approx(-0.5)
    assert rollout[0].done == 1.0


@pytest.mark.asyncio
async def test_state_replacement_applies_zero_death_penalty():
    """State replacement terminates the transition without a death contribution."""
    cfg = Config(reward_death_penalty_magnitude=0.75)
    experience_q = asyncio.Queue()
    actor = ActorClient(0, cfg, DummySharedState(), experience_q)
    actor.snake_id = 1
    actor.pending_transition = Transition(
        obs=np.zeros(1, dtype=np.float32),
        action_turn=0.0,
        turn_latent=0.0,
        action_boost=0.0,
        logp=0.0,
        value=0.0,
        reward=0.25,
        done=0.0,
    )
    actor.sensor_order = ["x"]
    actor.sensor_idx = {"x": 0}
    ws = DummyWS()

    await actor._handle_state_replaced(
        {
            "welcome": {
                "protocolVersion": 2,
                "tickRate": 60,
                "sensorSpec": {"order": ["x"]},
            }
        },
        ws,
        "trainer",
    )
    _actor_id, rollout, _bootstrap = await experience_q.get()

    assert rollout[0].reward == pytest.approx(0.25)
    assert rollout[0].done == 1.0
