"""End-to-end test for actor client."""

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

    def act(self, _obs, _turn_std):
        """Return fixed policy outputs."""
        return 0.0, 0.0, 0.0, 0.0


@pytest.mark.asyncio
async def test_actor_end_to_end_handshake_and_actions():
    """Actor connects, receives sensors, and sends actions."""
    actions = []
    server_ready = asyncio.Event()

    async def handler(ws):
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello"

        await ws.send(
            json.dumps(
                {
                    "type": "welcome",
                    "tickRate": 20,
                    "sensorSpec": {"order": ["points_delta_norm"]},
                }
            )
        )

        seen_join = False
        seen_viz = False
        for _ in range(2):
            msg = json.loads(await ws.recv())
            if msg["type"] == "join":
                seen_join = True
            if msg["type"] == "viz":
                seen_viz = True
        assert seen_join
        assert seen_viz

        await ws.send(
            json.dumps(
                {
                    "type": "assign",
                    "snakeId": 1,
                    "controller": "bot",
                }
            )
        )

        server_ready.set()

        for tick in [1, 2, 3]:
            await ws.send(
                json.dumps(
                    {
                        "type": "sensors",
                        "tick": tick,
                        "snakeId": 1,
                        "sensors": [0.0],
                    }
                )
            )
            data = await ws.recv()
            msg = json.loads(data)
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

    assert len(actions) >= 1
    _actor_id, transitions, bootstrap = rollout
    assert len(transitions) == 1
    assert isinstance(bootstrap, float)
