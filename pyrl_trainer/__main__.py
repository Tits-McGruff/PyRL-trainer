"""Entrypoint for the trainer process."""

import asyncio
import json
import os
from typing import Iterable

import torch
import websockets

from .agent import ActorClient
from .checkpointing import load_checkpoint_if_present
from .config import MAX_WS_MESSAGE_BYTES, PROTOCOL_VERSION, load_or_create_config
from .learner import SharedState, learner_loop
from .sensor_contract import SensorContract

# pylint: disable=duplicate-code


async def discover_sensor_contract(ws_url: str) -> SensorContract:
    """Connect once and read the exact server sensor contract without joining."""
    async with websockets.connect(ws_url, max_size=MAX_WS_MESSAGE_BYTES) as ws:
        hello = {
            "type": "hello",
            "clientType": "bot",
            "version": PROTOCOL_VERSION,
        }
        await ws.send(json.dumps(hello))
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "welcome":
                if msg.get("protocolVersion") != PROTOCOL_VERSION:
                    raise RuntimeError("server welcome did not confirm Protocol 2")
                return SensorContract.from_spec(msg.get("sensorSpec") or {})
            if msg.get("type") == "error":
                raise RuntimeError(
                    f"server error during welcome: {msg.get('message')}"
                )


async def discover_obs_dim(ws_url: str) -> int:
    """Compatibility helper returning the current sensor contract width."""
    contract = await discover_sensor_contract(ws_url)
    return contract.sensor_count


async def _cancel_and_wait(tasks: Iterable[asyncio.Task]) -> None:
    """Cancel unfinished owned tasks and wait until every task settles."""
    task_list = list(tasks)
    for task in task_list:
        if not task.done():
            task.cancel()
    if task_list:
        await asyncio.gather(*task_list, return_exceptions=True)


async def _supervise_tasks(tasks: Iterable[asyncio.Task]) -> None:
    """Propagate the primary fatal failure after settling all sibling tasks."""
    task_list = list(tasks)
    try:
        await asyncio.gather(*task_list)
    except (Exception, asyncio.CancelledError):
        await _cancel_and_wait(task_list)
        raise
    finally:
        if any(not task.done() for task in task_list):
            await _cancel_and_wait(task_list)


async def main() -> None:
    """Start the learner and actor tasks."""
    cfg = load_or_create_config(os.environ.get("SLITHER_CONFIG", "config.toml"))

    if str(cfg.train_device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass

    try:
        sensor_contract = await discover_sensor_contract(cfg.ws_url)
    except (OSError, RuntimeError, websockets.WebSocketException) as exc:
        print(f"[main] Failed to connect to {cfg.ws_url}: {exc}")
        print("[main] Ensure the game server is running.")
        return

    obs_dim = sensor_contract.sensor_count
    print(
        f"[main] ws={cfg.ws_url}, obs_dim={obs_dim}, "
        f"sensor_layout={sensor_contract.layout_version}, actors={cfg.actors}, "
        f"train_device={cfg.train_device}, infer_device={cfg.infer_device}, "
        f"network={cfg.net_hidden}x{cfg.net_layers}"
    )

    shared_state = SharedState(
        obs_dim=obs_dim,
        cfg=cfg,
        sensor_contract=sensor_contract,
    )

    try:
        loaded = load_checkpoint_if_present(cfg, shared_state)
        if loaded is not None:
            print(
                f"[main] resumed from {loaded} "
                f"at update_steps={shared_state.update_steps}"
            )
    except (OSError, RuntimeError) as exc:
        print(f"[main] checkpoint load failed: {type(exc).__name__}: {exc}")

    experience_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.actors * 4)

    tasks = [
        asyncio.create_task(learner_loop(cfg, shared_state, experience_q))
    ]
    for actor_id in range(cfg.actors):
        actor = ActorClient(actor_id, cfg, shared_state, experience_q)
        tasks.append(asyncio.create_task(actor.run()))

    await _supervise_tasks(tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
