"""Real native Slither -> binary protocol -> synchronized PPO and frozen evaluation."""

# pylint: disable=protected-access

import asyncio
import json
import math
import os
from pathlib import Path

import pytest
import numpy as np
import torch

from pyrl_trainer.__main__ import discover_sensor_contract, main
from pyrl_trainer.config import Config
from pyrl_trainer.checkpointing import (
    save_checkpoint, load_checkpoint_if_present, save_best_checkpoint,
)
from pyrl_trainer.evaluation import policy_snapshot_sha256
from pyrl_trainer.learner import collate_rollouts
from pyrl_trainer.sync_ppo import (
    SynchronousActorClient, SynchronousSharedState, _coherent_policy_version,
    _write_exact_best_metadata,
)

_SLITHER_ROOT = os.environ.get("PYRL_SLITHER_ROOT")
pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    not _SLITHER_ROOT, reason="set PYRL_SLITHER_ROOT to exercise the real native server"
)]


async def read_fixture_packet(process, prefix=b"PYRL_TEST_SERVER "):
    """Bound startup log consumption and fail if configured native startup fails."""
    consumed = 0
    tail = b""
    while consumed < 65_536:
        line = await process.stdout.readline()
        if not line:
            raise RuntimeError("native test server exited before readiness: "
                               + tail.decode(errors="replace"))
        consumed += len(line)
        tail = (tail + line)[-4096:]
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError("native test server exceeded bounded startup output")


async def stop_server(process):
    """Close the task's process, preserving useful failures and never accepting a hung shutdown."""
    if process.returncode is None:
        try:
            process.stdin.write(b"stop\n")
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
    try:
        await asyncio.wait_for(process.communicate(), timeout=15)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise RuntimeError("native test server did not complete shutdown") from exc


async def fixture_command(process, action, snake_id=None):
    """Drive a real UI placement or cutoff; this transport never supplies reward values."""
    request = {"id": action, "action": action}
    if snake_id is not None:
        request["snakeId"] = snake_id
    process.stdin.write(json.dumps(request).encode() + b"\n")
    await process.stdin.drain()
    result = await asyncio.wait_for(read_fixture_packet(process, b"PYRL_TEST_RESULT "), timeout=15)
    assert result.get("ok"), result
    assert result["id"] == action


async def until(predicate):
    """Wait for one task-owned actor observation without imposing server step acknowledgements."""
    deadline = asyncio.get_running_loop().time() + 10
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("native PPO actors did not reach the expected collection boundary")


async def update_round(state, queue, epoch):
    """Collect all four contributions and exercise the production PPO update path."""
    async def collect():
        contributions = {}
        for _ in range(4):
            actor_id, collection_epoch, rollout, bootstrap = await queue.get()
            assert collection_epoch == epoch
            assert actor_id not in contributions
            contributions[actor_id] = (rollout, bootstrap)
        return contributions

    contributions = await asyncio.wait_for(collect(), timeout=20)
    assert _coherent_policy_version(contributions) == state.published_policy_version
    batch = collate_rollouts(list(contributions.values()), state.cfg, "cpu")
    before = {key: tensor.clone() for key, tensor in state.train_model.state_dict().items()}
    metrics = await state.ppo_update_async(batch)
    assert abs(metrics["initial_kl"]) < 1e-5
    assert metrics["stale_batch"] == 0
    assert metrics["optimizer_steps"] > 0
    assert all(math.isfinite(metrics[key]) for key in ("loss", "turn_std", "approx_kl"))
    assert any(not torch.equal(before[key], tensor)
               for key, tensor in state.train_model.state_dict().items())
    return contributions


async def training_state(port, tmp_path):
    """Use a small CI policy; forced boost ensures actual expenditure is exercised."""
    url = f"ws://127.0.0.1:{port}"
    contract = await discover_sensor_contract(url)
    assert contract.profile == "motion"
    cfg = Config(ws_url=url, actors=4, horizon=256, net_hidden=8, net_layers=1,
                 train_device="cpu", infer_device="cpu", evaluation_max_seconds=0.05,
                 ckpt_dir=str(tmp_path / "checkpoints"), max_actions_per_second=0)
    torch.manual_seed(42)
    state = SynchronousSharedState(contract.sensor_count, cfg, sensor_contract=contract)
    with torch.no_grad():
        state.train_model.boost_head.weight.zero_()
        state.train_model.boost_head.bias.fill_(100)
    state.sync_infer_from_train()
    return cfg, contract, state


def assert_checkpoint_roundtrip(state, contract):
    """The actual evaluator winner stays exact after training publishes newer weights."""
    cfg = state.cfg
    best = save_best_checkpoint(cfg, state)
    _write_exact_best_metadata(cfg, state)
    saved = torch.load(best, map_location="cpu", weights_only=False)
    assert policy_snapshot_sha256({"model": saved["model"]}) == (
        state.best_evaluation_result["policy_sha256"]
    )
    save_checkpoint(cfg, state)
    restored = SynchronousSharedState(contract.sensor_count, cfg, sensor_contract=contract)
    assert load_checkpoint_if_present(cfg, restored) is not None
    assert restored.update_steps == state.update_steps
    for key, value in state.train_model.state_dict().items():
        assert torch.equal(restored.train_model.state_dict()[key], value)
    assert np.isfinite(float(restored.train_model.steering_std().detach()))


def assert_terminal_food(contribution, runtime):
    """Check the admitted terminal tail and independent survival/boost exposure."""
    victim, bootstrap = contribution
    assert victim[-1].done == 1 and bootstrap == 0
    assert sum(item.reward_components.get("death", 0) for item in victim) == -0.5
    assert sum(item.reward_components.get("food", 0) for item in victim) > 0
    assert sum(item.reward_components.get("growth", 0) for item in victim) > 0
    assert runtime["received_boostPointsSpentDelta"] > 0
    assert runtime["received_aliveSecondsDelta"] > 0
    assert runtime["reward_survival"] > 0


@pytest.mark.asyncio
async def test_real_binary_evaluator_caps_then_joins_a_distinct_native_life(tmp_path):
    """Both capped episodes traverse the real engine and socket, without synthetic rewards."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/reward_smoke_server.mjs"
    loader = (Path(_SLITHER_ROOT) / "node_modules/tsx/dist/loader.mjs").resolve().as_uri()
    process = await asyncio.create_subprocess_exec(
        "node", "--import", loader, str(fixture), str(_SLITHER_ROOT),
        str(tmp_path / "server.sqlite"),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    task = None
    try:
        port = (await asyncio.wait_for(read_fixture_packet(process), timeout=30))["port"]
        url = f"ws://127.0.0.1:{port}"
        contract = await discover_sensor_contract(url)
        cfg = Config(ws_url=url, net_hidden=8, net_layers=1,
                     train_device="cpu", infer_device="cpu", evaluation_max_seconds=0.01,
                     ckpt_dir=str(tmp_path / "checkpoints"), max_actions_per_second=0)
        state = SynchronousSharedState(contract.sensor_count, cfg, sensor_contract=contract)
        complete, results = asyncio.Event(), []
        original = state.record_evaluation_result

        def record(result, snapshot):
            original(result, snapshot)
            results.append(result)
            if len(results) >= 2:
                complete.set()

        state.record_evaluation_result = record
        queue = asyncio.Queue()
        actor = SynchronousActorClient(4, cfg, state, queue, evaluation=True)
        task = asyncio.create_task(actor.run())
        await asyncio.wait_for(complete.wait(), timeout=20)
        first, second = results[0], results[1]
        assert all(item["integrity"] for item in results[:2])
        assert all(item["reason"] == "evaluationTimeLimit" for item in results[:2])
        assert all(item["reward_components"]["death"] == 0 for item in results[:2])
        assert first["physical_snake_id"] != second["physical_snake_id"]
        assert first["lease_id"] != second["lease_id"]
        assert first["run_id"] == second["run_id"]
        assert first["policy_sha256"] == second["policy_sha256"]
        assert [first["sampling_seed"], second["sampling_seed"]] == [0, 1]
        assert first["elapsed_seconds"] >= cfg.evaluation_max_seconds
        for item in results[:2]:
            assert item["starting_body_length"] > 0
            assert item["final_body_length"] == (
                item["initial_observed_body_length"] + item["raw_deltas"]["lengthDelta"]
            )
            assert item["retained_growth"] == max(
                item["final_body_length"] - item["starting_body_length"], 0
            )
        assert queue.empty()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await stop_server(process)


def assert_epoch_fence_opened(actors):
    """Confirm the real application marker that opened each actor's next epoch."""
    for actor in actors:
        envelope = actor.last_reward_envelope
        marker = envelope["control"]["appliedCommand"]
        assert int(marker["command"]["clientActionId"], 16) >= actor._fence_action_id
        assert int(marker["firstAppliedStep"], 16) < int(envelope["toStep"], 16)


@pytest.mark.asyncio
async def test_native_events_reach_two_synchronized_ppo_updates_and_exact_checkpoints(tmp_path):
    """Known real events train four actors, with terminal and cutoff bootstraps kept distinct."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/reward_smoke_server.mjs"
    loader = (Path(_SLITHER_ROOT) / "node_modules/tsx/dist/loader.mjs").resolve().as_uri()
    process = await asyncio.create_subprocess_exec(
        "node", "--import", loader, str(fixture), str(_SLITHER_ROOT),
        str(tmp_path / "server.sqlite"), "ppo", stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    tasks = []
    try:
        port = (await asyncio.wait_for(read_fixture_packet(process), timeout=30))["port"]
        cfg, contract, state = await training_state(port, tmp_path)
        queue = asyncio.Queue(maxsize=8)
        actors = [SynchronousActorClient(index, cfg, state, queue) for index in range(4)]
        tasks.extend(asyncio.create_task(actor.run()) for actor in actors)
        evaluator = SynchronousActorClient(4, cfg, state, queue, evaluation=True)
        eval_task = asyncio.create_task(evaluator.run())
        tasks.append(eval_task)
        await until(lambda: all(actor.pending_transition is not None for actor in actors))
        await fixture_command(process, "food", actors[0].snake_id)
        await until(lambda: actors[0].pending_transition is not None and
                    actors[0].pending_transition.reward_components.get("food", 0) > 0
                    or any(item.reward_components.get("food", 0) > 0 for item in actors[0].rollout))
        await fixture_command(process, "death", actors[0].snake_id)
        await until(lambda: actors[0].episodes >= 2)
        await until(lambda: state.best_evaluation_result is not None)
        eval_task.cancel()
        await asyncio.gather(eval_task, return_exceptions=True)
        await fixture_command(process, "generation")
        first = await update_round(state, queue, 0)
        assert_terminal_food(first[0], state.drain_runtime_stats())

        cutoffs = {}
        for actor in actors:
            original = actor._cutoff_value

            def observe_cutoff(obs, value=original, actor_id=actor.actor_id):
                result = value(obs)
                cutoffs[actor_id] = result
                return result

            actor._cutoff_value = observe_cutoff
        state.advance_collection_epoch()
        await until(lambda: all(actor.pending_transition is not None for actor in actors))
        assert_epoch_fence_opened(actors)
        await fixture_command(process, "generation")
        second = await update_round(state, queue, 1)
        stats = state.drain_runtime_stats()
        assert stats["epoch_fence_intervals"] >= 4
        assert stats["commands_applied"] >= 4
        assert set(cutoffs) == set(range(4))
        for actor_id, (rollout, value) in second.items():
            assert rollout[-1].done == 0
            assert value == pytest.approx(cutoffs[actor_id])
            assert all(item.reward_components.get("death", 0) == 0 for item in rollout)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert_checkpoint_roundtrip(state, contract)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await stop_server(process)


@pytest.mark.asyncio
async def test_seeded_native_run_stops_and_preserves_both_policies(tmp_path, monkeypatch, capsys):
    """The qualification entry point completes over real sockets and keeps the baseline."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/reward_smoke_server.mjs"
    loader = (Path(_SLITHER_ROOT) / "node_modules/tsx/dist/loader.mjs").resolve().as_uri()
    process = await asyncio.create_subprocess_exec(
        "node", "--import", loader, str(fixture), str(_SLITHER_ROOT),
        str(tmp_path / "server.sqlite"), "ppo", stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        ready = await asyncio.wait_for(read_fixture_packet(process), timeout=15)
        checkpoints = tmp_path / "checkpoints"
        monkeypatch.setenv("SLITHER_CONFIG", str(tmp_path / "trainer.toml"))
        for key, value in {
            "WS_URL": f"ws://127.0.0.1:{ready['port']}", "CKPT_DIR": str(checkpoints),
            "ACTORS": "4", "HORIZON": "4", "NET_HIDDEN": "8", "NET_LAYERS": "1",
            "TRAIN_DEVICE": "cpu", "INFER_DEVICE": "cpu", "MAX_APS": "0",
        }.items():
            monkeypatch.setenv(f"SLITHER_{key}", value)
        await asyncio.wait_for(main(training_seed=17, decision_budget=5), timeout=20)
        initial = torch.load(checkpoints / "initial/latest.pt", weights_only=True)
        candidate = torch.load(checkpoints / "latest.pt", weights_only=True)
        assert initial["update_steps"] == 0 < candidate["update_steps"]
        assert policy_snapshot_sha256(initial) != policy_snapshot_sha256(candidate)
        log = capsys.readouterr().out
        assert f"initial_policy_sha256={policy_snapshot_sha256(initial)}" in log
        assert "admitted_decisions=16, decision_budget=5" in log
    finally:
        await stop_server(process)
