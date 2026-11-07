"""Actual PPO actor admission against native reward facts and Motion metadata."""

# pylint: disable=protected-access,import-error

from copy import deepcopy
import asyncio
import gzip
from importlib.resources import files
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pyrl_trainer.agent import ActorClient
from pyrl_trainer.config import Config
from pyrl_trainer.evaluation import EvaluationEpisodeComplete, policy_snapshot_sha256
from pyrl_trainer.checkpointing import save_best_checkpoint
from pyrl_trainer.reward_contract import RewardContractError, RewardDiscontinuityError, RewardStream
from pyrl_trainer.sensor_contract import SensorContract
from pyrl_trainer.sync_ppo import (
    SynchronousActorClient, SynchronousSharedState, _coherent_policy_version,
    _write_exact_best_metadata, restore_exact_best_metadata,
)

pytestmark = pytest.mark.unit


class Socket:  # pylint: disable=too-few-public-methods
    """Capture issued commands without replacing actor reward or policy code."""

    def __init__(self, received=()):
        self.messages = []
        self.received = list(received)

    async def send(self, raw):
        """Retain the exact issued command for pacing assertions."""
        self.messages.append(json.loads(raw))

    async def recv(self):
        """Return one queued server envelope in its wire order."""
        return json.dumps(self.received.pop(0))


@pytest.fixture(scope="module", name="motion_spec")
def native_motion_spec():
    """Reuse native canonical Motion metadata from the existing compact corpus."""
    path = (Path(__file__).resolve().parents[1]
            / "fixtures/tactical_profiles_v1/motion-b32/corpus.json.gz")
    corpus = json.loads(gzip.decompress(path.read_bytes()))
    raw = json.loads(corpus["sensorCanonicalJson"])
    return {
        "layoutVersion": "v4", "profile": "motion", "order": raw["order"],
        "sensorCount": corpus["inputSize"],
        "contract": {"canonicalJson": corpus["sensorCanonicalJson"],
                     "sha256": corpus["sensorSha256"]},
    }


def reward(kind="initial", *, snake_id=7, applied=False):
    """Use quantities already verified by the real Rust food/death physics test."""
    payload = json.loads(files("slither_contracts").joinpath(
        "external-reward-v1-fixture.json").read_text(encoding="utf-8"))[0]
    payload["snakeId"] = snake_id
    if kind == "initial":
        payload.update(transitionKind="initial", reason=None,
                       observationSequence="0000000000000001",
                       previousObservationSequence=None, toStep=payload["fromStep"])
        payload["deltas"] = {name: 0 for name in payload["deltas"]}
        payload["state"]["bodyLength"] = payload["state"]["startingLength"]
        payload["state"]["retainedGrowth"] = 0
    else:
        payload["transitionKind"] = kind
        payload["reason"] = {
            "live": None, "terminal": "death", "truncated": "manualGeneration"
        }[kind]
    if applied:
        payload["control"] = tracked_reward(1, 1, 1, 1)["control"]
    return payload


def state_for(spec):
    """Use the real frozen/inference model; V(final ones)=3, V(start zeros)=0."""
    cfg = Config(net_hidden=8, net_layers=1, train_device="cpu", infer_device="cpu",
                 max_actions_per_second=0, horizon=256)
    contract = SensorContract.from_spec(spec)
    state = SynchronousSharedState(contract.sensor_count, cfg, sensor_contract=contract)
    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.zero_()
        state.train_model.net[0].weight[0, 0] = 1
        state.train_model.value_head.weight[0, 0] = 3
    state.sync_infer_from_train()
    return cfg, state


def welcome(spec):
    """Advertise the independent reward and observation transport extensions."""
    return {"protocolVersion": 2, "tickRate": 60, "sensorSpec": spec,
            "rewardVersion": 1, "binarySensorsVersion": 2}


async def start(actor, spec, socket, snake_id=7):
    """Establish the real handshake, assignment and zero-reward initial boundary."""
    actor._apply_welcome(welcome(spec))
    await actor._on_assign(snake_id, "a" * 32)
    await actor._handle_sensors({"type": "sensors", "snakeId": snake_id, "tick": 0,
                                 "sensors": np.zeros(spec["sensorCount"], dtype=np.float32),
                                 "reward": reward(snake_id=snake_id)}, socket)


@pytest.mark.asyncio
async def test_handshake_requests_reward_and_binary_versions(motion_spec):
    """The real handshake negotiates the transport used by reward collection."""
    cfg, state = state_for(motion_spec)
    actor = SynchronousActorClient(0, cfg, state, asyncio.Queue())
    socket = Socket([dict(welcome(motion_spec), type="welcome"),
                     {"type": "assign", "snakeId": 7, "resumeToken": "a" * 32}])
    await actor._handshake(socket, "reward-test")
    assert socket.messages[0] == {
        "type": "hello", "clientType": "bot", "version": 2,
        "binarySensors": True, "binarySensorsVersion": 2, "rewardVersion": 1,
    }
    assert socket.messages[1]["type"] == "join"


@pytest.mark.asyncio
async def test_unsupported_reward_stops_actor_instead_of_reconnecting(monkeypatch, motion_spec):
    """Contract errors bypass the generic socket-retry loop."""
    cfg, state = state_for(motion_spec)
    actor = SynchronousActorClient(0, cfg, state, asyncio.Queue())
    unsupported = dict(welcome(motion_spec), type="welcome", rewardVersion=2)
    socket = Socket([unsupported])
    attempts = []

    class Connection:  # pylint: disable=too-few-public-methods
        """Enter the fake socket while exercising the real actor retry owner."""
        async def __aenter__(self):
            return socket

        async def __aexit__(self, *_args):
            return False

    def connect(*_args, **_kwargs):
        attempts.append(True)
        return Connection()

    monkeypatch.setattr("pyrl_trainer.agent.websockets.connect", connect)
    with pytest.raises(RewardContractError, match="rewardVersion 1"):
        await actor.run()
    assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_type", [ActorClient, SynchronousActorClient])
@pytest.mark.parametrize("kind,expected,bootstrap", [("terminal", -0.15, 0.0),
                                                    ("truncated", 0.35, 3.0)])
async def test_native_final_tail_reaches_pending_action_and_correct_bootstrap(
    actor_type, kind, expected, bootstrap, motion_spec,
):
    """Closure preserves food/growth/points and uses the actual final old-life value."""
    cfg, state = state_for(motion_spec)
    queue = asyncio.Queue()
    actor = actor_type(0, cfg, state, queue)
    await start(actor, motion_spec, Socket())
    assert "points_delta_norm" not in actor.sensor_idx
    assert actor.pending_transition is None
    message = {"type": "transition", "reward": reward(kind, applied=True)}
    if kind == "truncated":
        message["finalSensors"] = np.ones(motion_spec["sensorCount"], dtype=np.float32)
    await actor._handle_transition(message)
    contribution = queue.get_nowait()
    rollout, value = contribution[-2:]
    assert len(rollout) == 1
    assert rollout[0].reward == pytest.approx(expected)
    assert rollout[0].done == (1 if kind == "terminal" else 0)
    assert value == pytest.approx(bootstrap)
    assert actor.episode_return == pytest.approx(expected)
    totals = state.drain_runtime_stats()
    assert totals["reward_food"] == pytest.approx(0.1)
    assert totals["reward_growth"] == pytest.approx(0.05)
    assert totals["reward_points"] == pytest.approx(0.2)
    assert queue.empty()
    with pytest.raises(RewardDiscontinuityError):
        await actor._handle_transition(message)


@pytest.mark.asyncio
async def test_multiple_observations_accumulate_without_new_decisions(motion_spec):
    """Action pacing cannot drop intermediate authoritative reward intervals."""
    cfg, state = state_for(motion_spec)
    actor = SynchronousActorClient(0, cfg, state, asyncio.Queue())
    socket = Socket()
    await start(actor, motion_spec, socket)
    actor.stride = 10
    first = reward("live", applied=True)
    await actor._handle_sensors({"type": "sensors", "snakeId": 7, "tick": 1,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": first}, socket)
    second = deepcopy(first)
    second["control"]["selectedAtStep"] = "0000000000000001"
    second["state"]["bodyLength"] += 1
    second["state"]["retainedGrowth"] += 1
    second.update(observationSequence="0000000000000004",
                  previousObservationSequence=first["observationSequence"],
                  fromStep=first["toStep"], toStep="0000000000000002")
    await actor._handle_sensors({"type": "sensors", "snakeId": 7, "tick": 2,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": second}, socket)
    assert len(socket.messages) == 1
    assert actor.pending_transition.reward == pytest.approx(0.7)
    assert actor.episode_return == pytest.approx(0.7)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_type", [ActorClient, SynchronousActorClient])
async def test_dropped_pending_reward_is_received_but_never_admitted(actor_type, motion_spec):
    """A disconnect after a paced observation cannot inflate PPO reward logs."""
    cfg, state = state_for(motion_spec)
    queue = asyncio.Queue()
    actor = actor_type(0, cfg, state, queue)
    await start(actor, motion_spec, Socket())
    actor.stride = 10
    await actor._handle_sensors({"type": "sensors", "snakeId": 7, "tick": 1,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": reward("live")}, Socket())
    assert actor.pending_transition is None
    await actor._truncate_disconnected_rollout()
    stats = state.drain_runtime_stats()
    assert stats["received_foodValueDelta"] == 1
    assert stats["received_intervals"] == 1
    assert stats["reward_food"] == 0
    assert stats["admitted_transitions"] == 0
    if not queue.empty():
        assert queue.get_nowait()[-2] == []


@pytest.mark.asyncio
async def test_multi_step_interval_with_a_final_applied_marker_is_not_credited(motion_spec):
    """The last selection cannot identify the earlier physics in a multi-step interval."""
    cfg, state = state_for(motion_spec)
    cfg.horizon = 1
    queue, socket = asyncio.Queue(), Socket()
    actor = SynchronousActorClient(0, cfg, state, queue)
    await start(actor, motion_spec, socket)
    payload = reward("live", applied=True)
    payload["toStep"] = "0000000000000002"
    payload["control"]["selectedAtStep"] = "0000000000000001"
    payload["deltas"]["elapsedSeconds"] = 2 / 60
    await actor._handle_sensors({"snakeId": 7, "tick": 2, "reward": payload,
                                 "sensors": np.zeros(motion_spec["sensorCount"])}, socket)
    assert actor.pending_transition is None and queue.empty()
    assert actor.episode_return == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_barrier_wait_counts_received_events_without_mutating_submitted_reward(motion_spec):
    """Waiting actors keep event diagnostics, while a queued contribution stays fixed."""
    cfg, state = state_for(motion_spec)
    cfg.horizon = 1
    queue = asyncio.Queue()
    actor = SynchronousActorClient(0, cfg, state, queue)
    await start(actor, motion_spec, Socket())
    first = reward("live", applied=True)
    await actor._handle_sensors({"snakeId": 7, "tick": 1,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": first}, Socket())
    contribution = queue.get_nowait()
    second = deepcopy(first)
    second["control"]["selectedAtStep"] = "0000000000000001"
    second["state"]["bodyLength"] += 1
    second["state"]["retainedGrowth"] += 1
    second.update(observationSequence="0000000000000004",
                  previousObservationSequence=first["observationSequence"],
                  fromStep=first["toStep"], toStep="0000000000000002")
    await actor._handle_sensors({"snakeId": 7, "tick": 2,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": second}, Socket())
    stats = state.drain_runtime_stats()
    assert stats["received_foodValueDelta"] == 2
    assert stats["reward_food"] == pytest.approx(0.1)
    assert stats["reward_total_food"] == pytest.approx(0.1)
    assert stats["admitted_transitions"] == 1
    assert stats["admitted_elapsed_seconds"] == pytest.approx(1 / 60)
    assert stats["decision_seconds_mean"] == pytest.approx(1 / 60)
    assert contribution[-2][0].reward == pytest.approx(0.35)
    assert queue.empty()
    drained = state.drain_runtime_stats()
    assert drained["reward_food"] == 0
    assert drained["received_foodValueDelta"] == 0
    assert drained["received_total_foodValueDelta"] == 2


@pytest.mark.asyncio
async def test_missing_reward_and_missing_closure_fail_without_survival_fallback(motion_spec):
    """A modern actor cannot infer death from assignment or reward from labels."""
    cfg, state = state_for(motion_spec)
    actor = SynchronousActorClient(0, cfg, state, asyncio.Queue())
    await start(actor, motion_spec, Socket())
    with pytest.raises(RewardContractError):
        await actor._handle_sensors({"snakeId": 7, "tick": 1,
                                     "sensors": np.zeros(motion_spec["sensorCount"])}, Socket())
    with pytest.raises(RewardContractError, match="omitted the old-life closure"):
        await actor._on_assign(8, "b" * 32, replacement_reason="generation")
    assert actor.pending_transition is None


@pytest.mark.asyncio
async def test_reconnect_followed_by_death_discards_unresolved_reward(motion_spec):
    """A closure after a transport gap retires the life without a synthetic reward."""
    cfg, state = state_for(motion_spec)
    queue = asyncio.Queue()
    actor = SynchronousActorClient(0, cfg, state, queue)
    await start(actor, motion_spec, Socket())
    await actor._truncate_disconnected_rollout()
    assert queue.get_nowait()[-2] == []
    await actor._on_assign(7, "b" * 32, reclaimed=True)
    await actor._handle_transition({"type": "transition", "reward": reward("terminal")})
    assert actor.discarded_reward_intervals == 1
    assert actor.episode_return == 0
    assert queue.empty()
    await actor._on_assign(8, "c" * 32)
    assert actor.pending_transition is None


@pytest.mark.asyncio
async def test_four_actor_barrier_preserves_native_reward_totals(motion_spec):
    """Each actor contributes one coherent version with its complete terminal tail."""
    cfg, state = state_for(motion_spec)
    queue = asyncio.Queue()
    for actor_id in range(4):
        snake_id = 7 + actor_id
        actor = SynchronousActorClient(actor_id, cfg, state, queue)
        await start(actor, motion_spec, Socket(), snake_id)
        await actor._handle_transition({"type": "transition",
                                        "reward": reward("terminal", snake_id=snake_id,
                                                         applied=True)})
    contributions = {}
    while not queue.empty():
        actor_id, epoch, rollout, bootstrap = queue.get_nowait()
        assert epoch == state.collection_epoch
        contributions[actor_id] = (rollout, bootstrap)
    assert len(contributions) == 4
    assert _coherent_policy_version(contributions) == state.published_policy_version
    total = sum(item.reward for rollout, _ in contributions.values() for item in rollout)
    assert total == pytest.approx(-0.6)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_type", [ActorClient, SynchronousActorClient])
async def test_evaluator_caps_at_complete_reward_boundary_without_death(actor_type, motion_spec):
    """A local evaluation cutoff keeps its full interval and never invents a terminal."""
    cfg, state = state_for(motion_spec)
    cfg.evaluation_max_seconds = 0.01
    cfg.evaluation_seed = 41
    queue, socket = asyncio.Queue(), Socket()
    actor = actor_type(4, cfg, state, queue, evaluation=True)
    await start(actor, motion_spec, socket)
    frozen = actor.episode_policy_snapshot
    with pytest.raises(EvaluationEpisodeComplete):
        await actor._handle_sensors({"snakeId": 7, "tick": 1,
                                     "sensors": np.zeros(motion_spec["sensorCount"]),
                                     "reward": reward("live")}, socket)
    result = state.best_evaluation_result
    assert result["transition_kind"] == "truncated"
    assert result["reason"] == "evaluationTimeLimit"
    assert result["elapsed_seconds"] == pytest.approx(1 / 60)
    assert result["duration_overshoot_seconds"] == pytest.approx(1 / 60 - 0.01)
    assert result["raw_deltas"]["foodValueDelta"] == 1
    assert result["raw_deltas"]["retainedGrowthDelta"] == 1
    assert result["starting_body_length"] == 12
    assert result["initial_observed_body_length"] == 12
    assert result["final_body_length"] == 13
    assert result["retained_growth"] == 1
    assert result["reward_components"]["death"] == 0
    assert result["episode_return"] == pytest.approx(0.35)
    assert result["sampling_seed"] == 41
    assert result["policy_sha256"] == policy_snapshot_sha256(frozen)
    assert result["cutoff_step"] == "0000000000000001"
    assert len(socket.messages) == 1
    assert queue.empty()


@pytest.mark.asyncio
async def test_evaluator_records_terminal_tail_and_saves_exact_result(motion_spec, tmp_path):
    """A negative first result can win and its metrics accompany the evaluated model."""
    cfg, state = state_for(motion_spec)
    cfg.ckpt_dir = str(tmp_path)
    actor = SynchronousActorClient(4, cfg, state, asyncio.Queue(), evaluation=True)
    await start(actor, motion_spec, Socket())
    await actor._handle_transition({"type": "transition", "reward": reward("terminal")})
    result = state.best_evaluation_result
    assert result["reason"] == "death"
    assert result["lifetime_ticks"] == 1
    assert result["raw_deltas"]["foodCountDelta"] == 1
    assert result["raw_deltas"]["pointsDelta"] == 20
    assert result["reward_components"]["death"] == -0.5
    assert state.best_episode_return == pytest.approx(-0.15)
    path = save_best_checkpoint(cfg, state)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["best_evaluation_result"] == result
    assert policy_snapshot_sha256({"model": payload["model"]}) == result["policy_sha256"]


@pytest.mark.asyncio
async def test_interrupted_evaluation_is_recorded_but_excluded_from_promotion(motion_spec):
    """A received food event cannot make an incomplete evaluator episode eligible."""
    cfg, state = state_for(motion_spec)
    queue = asyncio.Queue()
    actor = SynchronousActorClient(4, cfg, state, queue, evaluation=True)
    await start(actor, motion_spec, Socket())
    await actor._handle_sensors({"snakeId": 7, "tick": 1,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": reward("live")}, Socket())
    await actor._truncate_disconnected_rollout()
    assert state.last_evaluation_result["episode_return"] == pytest.approx(0.35)
    assert state.last_evaluation_result["reason"] == "transportGap"
    assert state.last_evaluation_result["integrity"] is False
    assert state.best_episode_return == float("-inf")
    assert state.best_evaluation_result is None
    assert queue.empty()


@pytest.mark.asyncio
async def test_capped_evaluator_reconnects_with_fresh_identity_and_no_reclaim(
    monkeypatch, motion_spec,
):
    """The intentional cap follows a clean reconnect path instead of reclaiming the old life."""
    cfg, state = state_for(motion_spec)
    cfg.evaluation_max_seconds = 0.01
    actor = SynchronousActorClient(4, cfg, state, asyncio.Queue(), evaluation=True)

    class EndingSocket(Socket):
        """Cancel the test after the successor's zero-reward baseline is processed."""
        async def recv(self):
            if not self.received:
                raise asyncio.CancelledError()
            return await super().recv()

    def messages(snake_id):
        return [dict(welcome(motion_spec), type="welcome"),
                {"type": "assign", "snakeId": snake_id, "resumeToken": "a" * 32},
                {"type": "sensors", "snakeId": snake_id, "tick": 0,
                 "sensors": [0.0] * motion_spec["sensorCount"],
                 "reward": reward(snake_id=snake_id)}]

    first = EndingSocket(messages(7) + [{
        "type": "sensors", "snakeId": 7, "tick": 1,
        "sensors": [0.0] * motion_spec["sensorCount"], "reward": reward("live"),
    }])
    second = EndingSocket(messages(8))
    sockets = [first, second]

    class Connection:  # pylint: disable=too-few-public-methods
        """Exercise the actual run-loop exit and fresh handshake between episodes."""
        async def __aenter__(self):
            return sockets.pop(0)

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr("pyrl_trainer.agent.websockets.connect", lambda *_a, **_k: Connection())
    with pytest.raises(asyncio.CancelledError):
        await actor.run()
    joins = [next(message for message in peer.messages if message["type"] == "join")
             for peer in (first, second)]
    assert joins[0]["name"] != joins[1]["name"]
    assert all("resumeToken" not in join for join in joins)
    assert actor.snake_id == 8
    assert actor.episode_return == 0
    assert state.best_evaluation_result["snake_id"] == 7


@pytest.mark.asyncio
async def test_valid_zero_step_death_can_be_first_negative_incumbent(motion_spec, tmp_path):
    """Incumbent presence comes from a valid score, not a positive lifetime sentinel."""
    cfg, state = state_for(motion_spec)
    cfg.ckpt_dir = str(tmp_path)
    actor = SynchronousActorClient(4, cfg, state, asyncio.Queue(), evaluation=True)
    await start(actor, motion_spec, Socket())
    closure = reward("terminal")
    closure["toStep"] = closure["fromStep"]
    closure["deltas"] = {name: 0 for name in closure["deltas"]}
    closure["state"]["bodyLength"] = closure["state"]["startingLength"]
    closure["state"]["retainedGrowth"] = 0
    await actor._handle_transition({"type": "transition", "reward": closure})
    assert state.best_episode_return == -0.5
    assert state.best_lifetime_ticks == 0
    _write_exact_best_metadata(cfg, state)
    _cfg, restored = state_for(motion_spec)
    restored.cfg.ckpt_dir = str(tmp_path)
    restore_exact_best_metadata(restored.cfg, restored)
    assert restored.best_episode_return == -0.5
    assert restored.best_evaluation_result == state.best_evaluation_result
    restored.cfg.evaluation_max_seconds = 120
    restore_exact_best_metadata(restored.cfg, restored)
    assert restored.best_episode_return == float("-inf")
    assert restored.best_evaluation_result is None


def tracked_reward(tick, accepted, applied, applied_count, superseded=0):
    """Complete ordered fixture; command 3 expires and is superseded before selection."""
    payload = reward("live")
    payload.update(observationSequence=f"{tick + 2:016x}",
                   previousObservationSequence=f"{1 if tick == 1 else tick + 1:016x}",
                   fromStep=f"{tick - 1:016x}", toStep=f"{tick:016x}")
    payload["state"].update(bodyLength=12 + tick, retainedGrowth=tick)

    def command(identity):
        at_step = {1: 0, 2: 1}.get(identity, identity)
        return {"sequence": f"{100 + identity:016x}", "clientActionId": f"{identity:016x}",
                "clientTick": f"{identity - 1:016x}", "acceptedAtStep": f"{at_step:016x}"}

    first = {1: 0, 2: 2}.get(applied, applied)
    payload["control"] = {
        "source": "neutral" if tick == 4 else "held", "selectedAtStep": f"{tick - 1:016x}",
        "acceptedCommand": command(accepted),
        "appliedCommand": {"command": command(applied), "firstAppliedStep": f"{first:016x}"},
        "acceptedCount": accepted, "appliedCount": applied_count, "supersededCount": superseded,
    }
    return payload


@pytest.mark.asyncio
async def test_rejects_untracked_ppo_command(motion_spec):
    """A missing echo cannot leave the PPO epoch fence waiting indefinitely."""
    cfg, state = state_for(motion_spec)
    actor = SynchronousActorClient(0, cfg, state, asyncio.Queue())
    socket = Socket()
    await start(actor, motion_spec, socket)
    payload = tracked_reward(1, 1, 1, 1)
    payload["control"]["acceptedCommand"]["clientActionId"] = None
    payload["control"]["appliedCommand"]["command"]["clientActionId"] = None
    with pytest.raises(RewardContractError, match="without its client action identity"):
        await actor._handle_sensors({"snakeId": 7, "tick": 1,
                                     "sensors": np.zeros(motion_spec["sensorCount"]),
                                     "reward": payload}, socket)


def test_native_receipt_count_requires_a_new_sequence():
    """Counter conservation alone cannot validate a replayed command marker."""
    stream = RewardStream()
    stream.rebase(tracked_reward(1, 1, 1, 1), snake_id=7)
    replay = tracked_reward(2, 1, 1, 2)
    replay["control"]["acceptedCount"] = 2
    with pytest.raises(RewardDiscontinuityError, match="without a new sequence"):
        stream.accept(replay, snake_id=7)
    stream.accept(tracked_reward(2, 2, 1, 1), snake_id=7)


def test_reclaim_retires_pending_command_without_erasing_acceptance():
    """A reclaim can account for the latest receipt without an application."""
    stream = RewardStream()
    before = tracked_reward(2, 2, 1, 1)
    stream.rebase(before, snake_id=7)
    reclaimed = stream.accept(tracked_reward(3, 2, 1, 1, 1), snake_id=7)
    assert reclaimed["control"]["acceptedCommand"] == before["control"]["acceptedCommand"]
    assert reclaimed["control"]["appliedCommand"] == before["control"]["appliedCommand"]
    stream.accept(tracked_reward(4, 3, 1, 1, 1), snake_id=7)


@pytest.mark.asyncio
async def test_epoch_fence_waits_for_physics_and_allows_superseding_new_command(motion_spec):
    """Acceptance alone cannot open collection, and skipped first IDs cannot deadlock it."""
    cfg, state = state_for(motion_spec)
    cfg.horizon = 1
    queue, socket = asyncio.Queue(), Socket()
    actor = SynchronousActorClient(0, cfg, state, queue)
    await start(actor, motion_spec, socket)

    async def deliver(tick, accepted, applied, count, superseded=0):
        await actor._handle_sensors({"snakeId": 7, "tick": tick,
                                     "sensors": np.zeros(motion_spec["sensorCount"]),
                                     "reward": tracked_reward(tick, accepted, applied, count,
                                                              superseded)}, socket)

    await deliver(1, 1, 1, 1)
    assert queue.get_nowait()[1] == 0
    state.drain_runtime_stats()
    state.update_steps = 1
    state.sync_infer_from_train()
    state.advance_collection_epoch()
    await deliver(2, 2, 1, 1)
    assert actor._fence_action_id == 3
    await deliver(3, 3, 2, 2)
    assert actor._fence_required  # Command 3 was accepted, but only old command 2 ran.
    await deliver(4, 4, 2, 2, 1)
    assert actor.pending_transition is None
    await deliver(5, 5, 4, 3, 1)
    assert not actor._fence_required
    assert actor.pending_transition is None
    await deliver(6, 6, 5, 4, 1)
    assert queue.empty()
    assert actor.pending_transition is None
    await deliver(7, 7, 6, 5, 1)
    actor_id, epoch, rollout, _bootstrap = queue.get_nowait()
    assert actor_id == 0 and epoch == 1
    assert len(rollout) == 1 and rollout[0].policy_version == 1
    assert rollout[0].action_id == "0000000000000006"
    assert rollout[0].reward == pytest.approx(0.35)
    stats = state.drain_runtime_stats()
    assert stats["epoch_fence_intervals"] == 4
    assert stats["received_foodValueDelta"] == 6
    assert stats["reward_food"] == pytest.approx(0.1)
    assert stats["commands_superseded"] == 1
    assert stats["command_delay_samples"] == 4
    assert stats["command_delay_steps_mean"] == 1
    assert stats["command_delay_steps_max"] == 1
    assert queue.empty()


@pytest.mark.asyncio
async def test_closure_before_epoch_observation_contributes_empty_only_once(motion_spec):
    """An early closure cannot reset an already submitted empty contribution on assignment."""
    cfg, state = state_for(motion_spec)
    cfg.horizon = 1
    queue = asyncio.Queue()
    actor = SynchronousActorClient(0, cfg, state, queue)
    await start(actor, motion_spec, Socket())
    await actor._handle_sensors({"snakeId": 7, "tick": 1,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": reward("live", applied=True)}, Socket())
    queue.get_nowait()
    state.advance_collection_epoch()
    closure = reward("terminal", applied=True)
    closure.update(observationSequence="0000000000000004",
                   previousObservationSequence="0000000000000003",
                   fromStep="0000000000000001", toStep="0000000000000002")
    closure["state"].update(bodyLength=14, retainedGrowth=2)
    await actor._handle_transition({"reward": closure})
    assert queue.get_nowait()[1:3] == (1, [])
    await actor._on_assign(8, "b" * 32)
    await actor._handle_sensors({"snakeId": 8, "tick": 0,
                                 "sensors": np.zeros(motion_spec["sensorCount"]),
                                 "reward": reward(snake_id=8)}, Socket())
    assert actor._submitted_collection_epoch == 1
    assert actor.pending_transition is None
    assert queue.empty()
