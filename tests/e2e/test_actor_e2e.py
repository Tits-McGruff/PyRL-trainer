"""End-to-end tests for actor client."""

# pylint: disable=import-error

import asyncio
import json

import pytest
import websockets

from pyrl_trainer.agent import ActorClient
from pyrl_trainer.config import Config

pytestmark = pytest.mark.e2e


class DummySharedState:  # pylint: disable=too-few-public-methods
    """Minimal stand-in for SharedState."""

    def act(self, _obs, **_kwargs):
        """Return fixed policy outputs."""
        return 0.0, 0.0, 0.0, 0.25, 0.0


@pytest.mark.asyncio
async def test_actor_end_to_end_handshake_and_actions():
    """Actor falls back from an expired token and sends actions."""
    actions = []
    server_ready = asyncio.Event()

    async def handler(ws):
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello"
        assert hello["clientType"] == "bot"
        assert hello["version"] == 2

        await ws.send(json.dumps({
            "type": "welcome",
            "protocolVersion": 2,
            "tickRate": 20,
            "sensorSpec": {"order": ["points_delta_norm"]},
        }))

        reclaim_join = json.loads(await ws.recv())
        assert reclaim_join["type"] == "join"
        assert reclaim_join["resumeToken"] == "expired-token"
        await ws.send(json.dumps({
            "type": "reclaimResult",
            "reclaimed": False,
            "reason": "invalid",
        }))

        fresh_join = json.loads(await ws.recv())
        assert fresh_join["type"] == "join"
        assert "resumeToken" not in fresh_join

        await ws.send(json.dumps({
            "type": "assign",
            "snakeId": 1,
            "controller": "bot",
            "resumeToken": "replacement-token",
        }))

        server_ready.set()

        for tick in [1, 2, 3]:
            await ws.send(json.dumps({
                "type": "sensors",
                "tick": tick,
                "snakeId": 1,
                "sensors": [0.0],
            }))
            msg = json.loads(await ws.recv())
            if msg["type"] == "action":
                actions.append(msg)

        await ws.close()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    cfg = Config(
        ws_url=f"ws://127.0.0.1:{port}",
        horizon=1,
        max_actions_per_second=0,
        train_device="cpu",
        infer_device="cpu",
    )
    experience_q: asyncio.Queue = asyncio.Queue()
    actor = ActorClient(0, cfg, DummySharedState(), experience_q)
    actor.resume_token = "expired-token"

    task = asyncio.create_task(actor.run())

    try:
        await asyncio.wait_for(server_ready.wait(), timeout=2.0)
        rollout = await asyncio.wait_for(experience_q.get(), timeout=2.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()

    assert actions
    assert actor.resume_token == "replacement-token"
    _actor_id, transitions, bootstrap = rollout
    assert len(transitions) == 1
    assert isinstance(bootstrap, float)


@pytest.mark.asyncio
async def test_successful_reconnect_reclaims_same_episode():
    """A socket loss reclaims the same snake without inventing a new episode."""
    connection_count = 0
    reclaimed = asyncio.Event()

    async def handler(ws):
        nonlocal connection_count
        connection_count += 1
        attempt = connection_count

        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello"
        await ws.send(json.dumps({
            "type": "welcome",
            "protocolVersion": 2,
            "tickRate": 20,
            "sensorSpec": {"order": ["points_delta_norm"]},
        }))

        join = json.loads(await ws.recv())
        if attempt == 1:
            assert "resumeToken" not in join
            await ws.send(json.dumps({
                "type": "assign",
                "snakeId": 7,
                "controller": "bot",
                "resumeToken": "token-a",
                "reclaimed": False,
            }))
            for tick in [1, 2]:
                await ws.send(json.dumps({
                    "type": "sensors",
                    "tick": tick,
                    "snakeId": 7,
                    "sensors": [0.0],
                }))
                action = json.loads(await ws.recv())
                assert action["type"] == "action"
            await ws.close()
            return

        assert join["resumeToken"] == "token-a"
        await ws.send(json.dumps({
            "type": "reclaimResult",
            "reclaimed": True,
            "reason": "reclaimed",
            "snakeId": 7,
        }))
        await ws.send(json.dumps({
            "type": "assign",
            "snakeId": 7,
            "controller": "bot",
            "resumeToken": "token-b",
            "reclaimed": True,
        }))
        await ws.send(json.dumps({
            "type": "sensors",
            "tick": 3,
            "snakeId": 7,
            "sensors": [0.0],
        }))
        action = json.loads(await ws.recv())
        assert action["type"] == "action"
        reclaimed.set()
        await asyncio.sleep(0.1)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    cfg = Config(
        ws_url=f"ws://127.0.0.1:{port}",
        horizon=16,
        max_actions_per_second=0,
        train_device="cpu",
        infer_device="cpu",
    )
    experience_q: asyncio.Queue = asyncio.Queue()
    actor = ActorClient(0, cfg, DummySharedState(), experience_q)

    task = asyncio.create_task(actor.run())

    try:
        await asyncio.wait_for(reclaimed.wait(), timeout=4.0)
        actor_id, transitions, bootstrap = experience_q.get_nowait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        server.close()
        await server.wait_closed()

    assert connection_count >= 2
    assert actor_id == 0
    assert len(transitions) == 1
    assert bootstrap == pytest.approx(0.25)
    assert actor.snake_id == 7
    assert actor.resume_token == "token-b"
    assert actor.episodes == 1
