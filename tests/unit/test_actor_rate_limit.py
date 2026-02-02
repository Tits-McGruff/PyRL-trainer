"""Unit tests for actor rate limiting."""

# pylint: disable=import-error,protected-access

import asyncio

import pytest

from pyrl_trainer.agent import ActorClient
from pyrl_trainer.config import Config

pytestmark = pytest.mark.unit


class DummySharedState:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for SharedState."""

    def act(self, _obs, _turn_std):
        """Return fixed policy outputs."""
        return 0.0, 0.0, 0.0, 0.0


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
    """Time gating prevents sending too quickly."""
    cfg = Config(max_actions_per_second=2)
    actor = _make_actor(cfg)
    actor.stride = 1
    actor.last_sent_tick = 10
    actor.last_sent_time = 1.0

    monkeypatch.setattr("pyrl_trainer.agent.time.time", lambda: 1.1)
    assert actor._should_send_action(11) is False

    monkeypatch.setattr("pyrl_trainer.agent.time.time", lambda: 1.6)
    assert actor._should_send_action(11) is True
