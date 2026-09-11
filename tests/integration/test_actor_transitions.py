"""Integration tests for actor transitions."""

# pylint: disable=import-error,protected-access

import asyncio
import json

import pytest

from pyrl_trainer.agent import ActorClient
from pyrl_trainer.config import Config

pytestmark = pytest.mark.integration


class DummySharedState:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for SharedState."""

    def act(self, _obs, **_kwargs):
        """Return fixed policy outputs."""
        return 0.1, 1.0, -0.5, 0.25


class DummyWS:  # pylint: disable=too-few-public-methods
    """Capture outgoing messages."""
    def __init__(self):
        self.sent = []

    async def send(self, data):
        """Store outgoing payloads."""
        self.sent.append(data)


@pytest.mark.asyncio
async def test_transitions_align_with_stride(monkeypatch):
    """Pending transitions align with sent actions."""
    cfg = Config(max_actions_per_second=0, horizon=10)
    actor = ActorClient(0, cfg, DummySharedState(), asyncio.Queue())
    actor.snake_id = 1
    actor.stride = 2
    actor.sensor_order = ["points_delta_norm"]
    actor.sensor_idx = {"points_delta_norm": 0}

    monkeypatch.setattr("pyrl_trainer.agent.default_reward", lambda prev, obs, idx: 1.0)

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
    assert abs(actor.pending_transition.reward - 0.0) < 1e-6


@pytest.mark.asyncio
async def test_import_replacement_rejoins_without_stale_token():
    """A live import ends the old episode and sends one fresh Protocol 2 join."""
    actor = ActorClient(7, Config(max_actions_per_second=20), DummySharedState(), asyncio.Queue())
    actor.snake_id = 91
    actor.resume_token = "stale-token"
    actor.sensor_order = ["old"]
    actor.sensor_idx = {"old": 0}
    ws = DummyWS()

    await actor._handle_state_replaced({
        "type": "stateReplaced",
        "welcome": {
            "protocolVersion": 2,
            "tickRate": 60,
            "sensorSpec": {"order": ["food_proximity"]},
        },
    }, ws, "trainer-007")

    assert actor.snake_id is None
    assert actor.resume_token is None
    assert actor.sensor_order == ["food_proximity"]
    assert json.loads(ws.sent[-1]) == {
        "type": "join", "mode": "player", "name": "trainer-007"
    }
