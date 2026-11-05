"""Entrypoint selecting independent PPO and canonical distillation owners."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .sensor_contract import SensorContract

# Owner-specific imports stay behind mode selection.
# pylint: disable=duplicate-code,import-outside-toplevel


async def discover_sensor_contract(ws_url: str) -> SensorContract:
    """Connect once and read the exact server sensor contract without joining."""
    import websockets
    from .config import MAX_WS_MESSAGE_BYTES, PROTOCOL_VERSION
    from .sensor_contract import SensorContract
    from .reward_contract import require_reward_version, RewardContractError

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
                require_reward_version(msg)
                if msg.get("binarySensorsVersion") != 2:
                    raise RewardContractError(
                        "server must support reward-bearing binarySensorsVersion 2"
                    )
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


def _log_reward_configuration(cfg, shared_state) -> None:
    """Report the effective loaded distribution and objective before actors start."""
    from .config import REWARD_CONFIG_VERSION, reward_config_record
    from .network import POLICY_PARAMETERIZATION_VERSION

    print(
        f"[main] reward_config_version={REWARD_CONFIG_VERSION}, "
        f"reward_coefficients={json.dumps(reward_config_record(cfg), sort_keys=True)}, "
        f"policy_parameterization={POLICY_PARAMETERIZATION_VERSION}, "
        f"turn_log_std={shared_state.infer_model.turn_log_std.item():.6f}, "
        f"turn_std={shared_state.infer_model.steering_std().item():.6f}, "
        f"optimizer_lr={shared_state.optimizer.param_groups[0]['lr']:g}, "
        f"horizon={cfg.horizon}, epochs={cfg.epochs}, minibatch={cfg.minibatch}, "
        f"evaluation_max_seconds={cfg.evaluation_max_seconds:g}, "
        f"evaluation_seed={cfg.evaluation_seed}"
    )


async def main() -> None:
    """Start synchronized PPO training actors plus one frozen-policy evaluator."""
    import torch
    import websockets
    from .checkpointing import load_checkpoint_if_present
    from .config import load_or_create_config
    from .reward_contract import RewardContractError
    from .sensor_contract import SensorContractError
    from .sync_ppo import (
        SynchronousActorClient,
        SynchronousSharedState,
        apply_stable_learning_rate_default,
        enforce_optimizer_learning_rate,
        restore_exact_best_metadata,
        synchronous_learner_loop,
    )

    config_path = os.environ.get("SLITHER_CONFIG", "config.toml")
    cfg = load_or_create_config(config_path)
    apply_stable_learning_rate_default(cfg, config_path)

    if str(cfg.train_device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass

    try:
        sensor_contract = await discover_sensor_contract(cfg.ws_url)
    except (RewardContractError, SensorContractError):
        raise
    except (OSError, RuntimeError, websockets.WebSocketException) as exc:
        print(f"[main] Failed to connect to {cfg.ws_url}: {exc}")
        print("[main] Ensure the game server is running.")
        return

    obs_dim = sensor_contract.sensor_count
    print(
        f"[main] ws={cfg.ws_url}, obs_dim={obs_dim}, "
        f"sensor_layout={sensor_contract.layout_version}, actors={cfg.actors}, "
        f"sensor_profile={sensor_contract.profile}, reward_version=1, binary_sensors=2, "
        f"evaluator=1, train_device={cfg.train_device}, "
        f"infer_device={cfg.infer_device}, lr={cfg.lr:g}, "
        f"network={cfg.net_hidden}x{cfg.net_layers}, collection=barrier"
    )

    shared_state = SynchronousSharedState(
        obs_dim=obs_dim,
        cfg=cfg,
        sensor_contract=sensor_contract,
    )

    try:
        loaded = load_checkpoint_if_present(cfg, shared_state)
        if loaded is None:
            from pathlib import Path
            if any(Path(cfg.ckpt_dir).glob("*.pt")):
                raise RuntimeError(
                    "checkpoint directory contains no compatible repaired policy; "
                    "select a fresh checkpoint namespace to preserve historical files"
                )
        enforce_optimizer_learning_rate(shared_state)
        restore_exact_best_metadata(cfg, shared_state)
        if loaded is not None:
            print(
                f"[main] resumed from {loaded} "
                f"at update_steps={shared_state.update_steps}, "
                f"exact_best={shared_state.best_lifetime_ticks}, "
                f"legacy_best={shared_state.legacy_best_lifetime_ticks}, "
                f"lr={shared_state.optimizer.param_groups[0]['lr']:g}"
            )
    except (OSError, RuntimeError) as exc:
        print(f"[main] checkpoint load failed: {type(exc).__name__}: {exc}")
        raise

    _log_reward_configuration(cfg, shared_state)

    experience_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.actors * 2)

    tasks = [
        asyncio.create_task(
            synchronous_learner_loop(cfg, shared_state, experience_q)
        )
    ]
    for actor_id in range(cfg.actors):
        actor = SynchronousActorClient(
            actor_id, cfg, shared_state, experience_q
        )
        tasks.append(asyncio.create_task(actor.run()))

    evaluator = SynchronousActorClient(
        cfg.actors,
        cfg,
        shared_state,
        experience_q,
        evaluation=True,
    )
    tasks.append(asyncio.create_task(evaluator.run()))

    await _supervise_tasks(tasks)


def run(argv=None):
    """Select a trainer owner before loading PPO config or starting actors/tasks."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--mode", choices=("ppo", "distillation"), default="ppo",
        help="PPO actors (default) or a canonical graph distillation job runner",
    )
    selected, remaining = selector.parse_known_args(argv)
    if selected.mode == "distillation":
        # The dedicated runner owns its arguments, checkpoints and bounded HTTP client.
        from .distillation_run import run as run_distillation  # pylint: disable=import-outside-toplevel

        return run_distillation(remaining)
    parser = argparse.ArgumentParser(
        parents=[selector], allow_abbrev=False,
        description="Train PPO actors using config.toml/SLITHER_CONFIG, or select "
        "--mode distillation for the independent canonical graph job runner.",
    )
    parser.parse_args(argv)
    asyncio.run(main())
    return None


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
