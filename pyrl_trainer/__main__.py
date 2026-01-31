import asyncio
import os
import torch
import json
import websockets
from .config import load_or_create_config, Config, PROTOCOL_VERSION, MAX_WS_MESSAGE_BYTES
from .learner import SharedState, learner_loop, load_checkpoint_if_present
from .agent import ActorClient


async def discover_obs_dim(ws_url: str) -> int:
    """
    Connect once, read welcome, return sensorCount; does not join as player.
    This avoids hardcoding obs dim and honors sensorSpec order.
    """
    async with websockets.connect(ws_url, max_size=MAX_WS_MESSAGE_BYTES) as ws:
        hello = {"type": "hello", "clientType": "bot", "version": PROTOCOL_VERSION}
        await ws.send(json.dumps(hello))
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "welcome":
                spec = msg.get("sensorSpec") or {}
                sensor_count = int(spec.get("sensorCount", 0))
                if sensor_count <= 0:
                    order = spec.get("order") or []
                    sensor_count = len(order)
                return sensor_count
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during welcome: {msg.get('message')}")


async def main() -> None:
    cfg = load_or_create_config(os.environ.get("SLITHER_CONFIG", "config.toml"))

    # GPU fast-path
    if str(cfg.train_device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
            
    try:
        obs_dim = await discover_obs_dim(cfg.ws_url)
    except Exception as e:
        print(f"[main] Failed to connect to {cfg.ws_url}: {e}")
        print("[main] Ensure the game server is running.")
        return

    print(
        f"[main] ws={cfg.ws_url}, obs_dim={obs_dim}, actors={cfg.actors}, "
        f"train_device={cfg.train_device}, infer_device={cfg.infer_device}"
    )

    shared_state = SharedState(obs_dim=obs_dim, cfg=cfg)

    # Auto-resume from latest checkpoint if present
    try:
        loaded = load_checkpoint_if_present(cfg, shared_state)
        if loaded is not None:
            print(f"[main] resumed from {loaded} at update_steps={shared_state.update_steps}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[main] checkpoint load failed: {type(e).__name__}: {e}")

    experience_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.actors * 4)

    learner = asyncio.create_task(learner_loop(cfg, shared_state, experience_q))

    actors = []
    for i in range(cfg.actors):
        actors.append(asyncio.create_task(ActorClient(i, cfg, shared_state, experience_q).run()))

    await asyncio.gather(learner, *actors)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
