"""Integration tests for actor transitions."""

# pylint: disable=import-error,protected-access

import asyncio
import json
from importlib.resources import files

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

    def infer_model(self, obs):
        """Make the bootstrap depend on the final state, not the pending value."""
        return None, None, obs[:, 0] * 0.9


def _reward(tick, *, initial=False, kind="live", events=True):
    """Authoritative native event fixture with explicit test interval boundaries."""
    value = json.loads(files("slither_contracts").joinpath(
        "external-reward-v1-fixture.json").read_text(encoding="utf-8"))[0]
    value.update(snakeId=1, observationSequence=f"{1000 + tick:016x}",
                 previousObservationSequence=None if initial else f"{999 + tick:016x}",
                 fromStep=f"{tick if initial else tick - 1:016x}", toStep=f"{tick:016x}",
                 transitionKind="initial" if initial else kind,
                 reason=(None if initial or kind == "live"
                         else ("death" if kind == "terminal" else "generation")))
    if initial or not events:
        value["deltas"] = {name: 0 for name in value["deltas"]}
    if not initial:
        value["deltas"]["elapsedSeconds"] = 1 / 60
    growth = tick if initial or events else tick - 1
    value["state"]["bodyLength"] = value["state"]["startingLength"] + growth
    value["state"]["retainedGrowth"] = growth
    return value


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
async def test_transitions_align_with_stride():
    """Pending transitions align with sent actions."""
    cfg = Config(max_actions_per_second=0, horizon=10)
    actor = ActorClient(0, cfg, DummySharedState(), asyncio.Queue())
    actor.snake_id = 1
    actor.stride = 2
    actor.sensor_order = ["points_delta_norm"]
    actor.sensor_idx = {"points_delta_norm": 0}

    ws = DummyWS()

    for tick in [1, 2, 3]:
        msg = {
            "type": "sensors",
            "tick": tick,
            "snakeId": 1,
            "sensors": [float(tick)],
            "reward": _reward(tick, initial=tick == 1),
        }
        await actor._handle_sensors(msg, ws)

    assert len(ws.sent) == 2
    assert len(actor.rollout) == 1
    assert actor.rollout[0].reward == pytest.approx(0.7)
    assert actor.pending_transition is not None
    assert abs(actor.pending_transition.reward) < 1e-6


@pytest.mark.asyncio
async def test_burst_observation_stream_processes_every_sensor_between_actions():
    """Queued sensor bursts remain ordered even when action pacing skips ticks."""
    cfg = Config(max_actions_per_second=0, horizon=20)
    actor = ActorClient(0, cfg, DummySharedState(), asyncio.Queue())
    actor.snake_id = 1
    actor.stride = 3
    actor.sensor_order = ["points_delta_norm"]
    actor.sensor_idx = {"points_delta_norm": 0}

    ws = DummyWS()
    for tick in [10, 11, 12, 13, 14]:
        await actor._handle_sensors(
            {
                "type": "sensors",
                "tick": tick,
                "snakeId": 1,
                "sensors": [float(tick)],
                "reward": _reward(tick, initial=tick == 10),
            },
            ws,
        )

    assert len(ws.sent) == 2
    assert [json.loads(value)["tick"] for value in ws.sent] == [10, 13]
    assert actor.last_sensor_tick == 14
    assert actor.last_obs is not None
    assert actor.last_obs[0] == pytest.approx(14.0)
    assert len(actor.rollout) == 1
    assert actor.rollout[0].reward == pytest.approx(1.05)
    assert actor.pending_transition is not None
    assert actor.pending_transition.reward == pytest.approx(0.35)


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
                "rewardVersion": 1, "binarySensorsVersion": 2,
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
                "rewardVersion": 1, "binarySensorsVersion": 2,
                "tickRate": 60,
                "sensorSpec": {"order": ["old"]},
            },
            {
                "type": "stateReplaced",
                "welcome": {
                    "protocolVersion": 2,
                    "rewardVersion": 1, "binarySensorsVersion": 2,
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

    actor._reward_stream.accept(_reward(0, initial=True), snake_id=1)
    await actor._handle_transition({"type": "transition",
                                    "reward": _reward(1, kind="terminal", events=False)})
    _actor_id, rollout, bootstrap = await experience_q.get()

    assert bootstrap == 0.0
    assert len(rollout) == 1
    assert rollout[0].reward == pytest.approx(-0.5)
    assert rollout[0].done == 1.0


@pytest.mark.asyncio
async def test_generation_replacement_truncates_without_death_penalty():
    """Generation rollover bootstraps completed PPO work without a fake death."""
    cfg = Config(reward_death_penalty_magnitude=0.75)
    experience_q = asyncio.Queue()
    actor = ActorClient(0, cfg, DummySharedState(), experience_q)
    actor.snake_id = 1
    actor.last_assign_tick = 100
    actor.last_sensor_tick = 160
    actor.rollout = [
        Transition(
            obs=np.zeros(1, dtype=np.float32),
            action_turn=0.0,
            turn_latent=0.0,
            action_boost=0.0,
            logp=0.0,
            value=0.1,
            reward=0.25,
            done=0.0,
        )
    ]
    actor.pending_transition = Transition(
        obs=np.ones(1, dtype=np.float32),
        action_turn=0.0,
        turn_latent=0.0,
        action_boost=0.0,
        logp=0.0,
        value=0.4,
        reward=0.0,
        done=0.0,
    )

    actor.sensor_order = ["motion"]
    actor._reward_stream.accept(_reward(159, initial=True), snake_id=1)
    actor._reward_has_observation = True
    await actor._handle_transition({"type": "transition",
                                    "reward": _reward(160, kind="truncated", events=False),
                                    "finalSensors": [1.0]})

    await actor._on_assign(
        2,
        "next-token",
        replacement_reason="generation",
    )
    actor_id, rollout, bootstrap = await experience_q.get()

    assert actor_id == 0
    assert len(rollout) == 2
    assert rollout[0].reward == pytest.approx(0.25)
    assert rollout[0].done == 0.0
    assert bootstrap == pytest.approx(0.9)
    assert actor.snake_id == 2
    assert actor.resume_token == "next-token"


@pytest.mark.asyncio
async def test_state_replacement_applies_zero_death_penalty():
    """State replacement keeps completed work and discards its unresolved tail."""
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
    actor.rollout = [Transition(obs=np.zeros(1, dtype=np.float32), action_turn=0.0,
                                turn_latent=0.0, action_boost=0.0, logp=0.0,
                                value=0.0, reward=0.1, done=0.0)]
    ws = DummyWS()

    await actor._handle_state_replaced(
        {
            "welcome": {
                "protocolVersion": 2,
                "rewardVersion": 1, "binarySensorsVersion": 2,
                "tickRate": 60,
                "sensorSpec": {"order": ["x"]},
            }
        },
        ws,
        "trainer",
    )
    _actor_id, rollout, _bootstrap = await experience_q.get()

    assert len(rollout) == 1
    assert rollout[0].reward == pytest.approx(0.1)
    assert rollout[0].done == 0.0
