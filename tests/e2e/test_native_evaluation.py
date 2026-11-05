"""Real native Slither -> binary protocol -> frozen Python evaluator lifecycle."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from pyrl_trainer.__main__ import discover_sensor_contract
from pyrl_trainer.config import Config
from pyrl_trainer.sync_ppo import SynchronousActorClient, SynchronousSharedState

_SLITHER_ROOT = os.environ.get("PYRL_SLITHER_ROOT")
pytestmark = [pytest.mark.e2e, pytest.mark.skipif(
    not _SLITHER_ROOT, reason="set PYRL_SLITHER_ROOT to exercise the real native server"
)]


async def ready_port(process):
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
        if line.startswith(b"PYRL_TEST_SERVER "):
            return json.loads(line[len(b"PYRL_TEST_SERVER "):])["port"]
    raise RuntimeError("native test server exceeded bounded startup output")


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
        port = await asyncio.wait_for(ready_port(process), timeout=30)
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
        assert queue.empty()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if process.returncode is None:
            try:
                process.stdin.write(b"stop\n")
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                await asyncio.wait_for(process.communicate(), timeout=15)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
