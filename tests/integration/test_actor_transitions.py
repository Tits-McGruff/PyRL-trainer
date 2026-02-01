import asyncio

import numpy as np
import pytest

from pyrl_trainer.agent import ActorClient
from pyrl_trainer.config import Config

pytestmark = pytest.mark.integration


class DummySharedState:
    def act(self, obs, turn_std):
        return 0.1, 1.0, -0.5, 0.25


class DummyWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


@pytest.mark.asyncio
async def test_transitions_align_with_stride(monkeypatch):
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
