"""Synchronous collection barrier for stable on-policy PPO training."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent import ActorClient, Transition
from .checkpointing import save_best_checkpoint, save_checkpoint
from .config import Config
from .learner import SharedState, collate_rollouts


_STABLE_PPO_LR = 1e-4
_LEGACY_DEFAULT_PPO_LR = 3e-4
_AUTO_CONFIG_MARKER = "# Auto-generated because it was missing."
_EXACT_BEST_META = "best_evaluator.json"


def apply_stable_learning_rate_default(cfg: Config, config_path: str) -> None:
    """Migrate the old auto-generated 3e-4 PPO default to the safer 1e-4 default."""
    if "SLITHER_LR" in os.environ:
        return
    if abs(float(cfg.lr) - _LEGACY_DEFAULT_PPO_LR) > 1e-12:
        return

    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""

    if _AUTO_CONFIG_MARKER not in text:
        return

    cfg.lr = _STABLE_PPO_LR
    migrated = re.sub(
        r"(?m)^(\s*lr\s*=\s*)(?:0\.0003|3(?:\.0+)?[eE]-4)(\s*)$",
        rf"\g<1>{_STABLE_PPO_LR:.4f}\g<2>",
        text,
        count=1,
    )
    if migrated != text:
        try:
            path.write_text(migrated, encoding="utf-8")
        except OSError:
            pass


def _best_meta_path(cfg: Config) -> Path:
    return Path(cfg.ckpt_dir) / _EXACT_BEST_META


def restore_exact_best_metadata(cfg: Config, shared_state: "SynchronousSharedState") -> None:
    """Separate legacy training-actor records from exact frozen-evaluator records."""
    shared_state.legacy_best_lifetime_ticks = max(
        0, int(getattr(shared_state, "best_lifetime_ticks", 0))
    )
    legacy_return = float(getattr(shared_state, "best_episode_return", float("-inf")))
    shared_state.legacy_best_episode_return = legacy_return

    shared_state.best_lifetime_ticks = 0
    shared_state.best_episode_return = float("-inf")
    shared_state.best_policy_snapshot = None
    shared_state.best_checkpoint_pending = False

    path = _best_meta_path(cfg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return

    if payload.get("source") != "frozen_evaluator":
        return
    try:
        lifetime = int(payload["lifetime_ticks"])
        episode_return = float(payload["episode_return"])
    except (KeyError, TypeError, ValueError):
        return
    if lifetime <= 0:
        return

    shared_state.best_lifetime_ticks = lifetime
    shared_state.best_episode_return = episode_return


def _write_exact_best_metadata(cfg: Config, shared_state: "SynchronousSharedState") -> None:
    path = _best_meta_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = shared_state.best_policy_snapshot or {}
    payload = {
        "source": "frozen_evaluator",
        "lifetime_ticks": int(shared_state.best_lifetime_ticks),
        "episode_return": float(shared_state.best_episode_return),
        "policy_version": int(snapshot.get("update_steps", -1)),
    }
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


class SynchronousSharedState(SharedState):
    """Shared PPO state with a collection epoch that advances only after one update."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_epoch = 0
        self.legacy_best_lifetime_ticks = 0
        self.legacy_best_episode_return = float("-inf")

    def advance_collection_epoch(self) -> None:
        """Release actors to collect one new rollout under the newly published policy."""
        self.collection_epoch += 1

    def inference_value(self, obs: np.ndarray) -> float:
        """Evaluate the currently published inference policy value without sampling."""
        with torch.no_grad():
            tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.cfg.infer_device
            ).unsqueeze(0)
            _turn_mean, _boost_logit, value = self.infer_model(tensor)
        return float(value.item())


class SynchronousActorClient(ActorClient):
    """Actor that contributes at most one rollout to each collection epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._submitted_collection_epoch: Optional[int] = None

    def _collection_open(self) -> bool:
        if self.evaluation:
            return False
        current = int(self.shared_state.collection_epoch)
        return self._submitted_collection_epoch != current

    async def _submit_collection_rollout(
        self,
        bootstrap: float,
        *,
        allow_empty: bool = False,
    ) -> None:
        """Submit this actor's sole contribution for the current collection epoch."""
        if self.evaluation or not self._collection_open():
            self.rollout.clear()
            return
        if not self.rollout and not allow_empty:
            return

        epoch = int(self.shared_state.collection_epoch)
        await self.experience_q.put(
            (self.actor_id, epoch, list(self.rollout), float(bootstrap))
        )
        self.rollout.clear()
        self._submitted_collection_epoch = epoch

    def _refresh_collection_epoch(self) -> None:
        if self.evaluation or self._submitted_collection_epoch is None:
            return
        if int(self.shared_state.collection_epoch) > self._submitted_collection_epoch:
            self._submitted_collection_epoch = None

    async def _truncate_disconnected_rollout(self) -> None:
        """Contribute partial work once, then reconnect without inventing a terminal."""
        bootstrap = (
            float(self.pending_transition.value)
            if self.pending_transition is not None
            else 0.0
        )
        self.pending_transition = None
        if not self.evaluation:
            await self._submit_collection_rollout(bootstrap, allow_empty=True)
        self.rollout.clear()
        self.last_obs = None
        self.last_sent_tick = None

    async def _finalize_truncated_episode(self) -> None:
        """Treat a server generation cutoff as a bootstrap truncation and barrier contribution."""
        had_episode = (
            self.snake_id is not None
            or self.episode_actions > 0
            or self.pending_transition is not None
            or bool(self.rollout)
        )
        bootstrap = (
            float(self.pending_transition.value)
            if self.pending_transition is not None
            else 0.0
        )
        self.pending_transition = None

        if not self.evaluation:
            await self._submit_collection_rollout(bootstrap, allow_empty=True)
        self.rollout.clear()

        lifetime_ticks = 0
        if self.last_assign_tick is not None and self.last_sensor_tick is not None:
            lifetime_ticks = max(0, int(self.last_sensor_tick - self.last_assign_tick))
        if had_episode:
            self._record_episode(lifetime_ticks)

    async def _finalize_terminal_episode(
        self, death_penalty: Optional[float] = None
    ) -> None:
        """Finish a real death, preserving the death reward only for real terminals."""
        had_episode = (
            self.snake_id is not None
            or self.episode_actions > 0
            or self.pending_transition is not None
            or bool(self.rollout)
        )
        penalty = (
            -float(self.cfg.reward_death_penalty_magnitude)
            if death_penalty is None
            else float(death_penalty)
        )

        if self.evaluation:
            if had_episode:
                self.episode_return += penalty
        elif self.pending_transition is not None and self._collection_open():
            self.pending_transition.reward += penalty
            self.pending_transition.done = 1.0
            self.rollout.append(self.pending_transition)
            self.pending_transition = None
            self.episode_return += penalty
            if penalty:
                self._record_reward_components({"death": penalty})
        else:
            self.pending_transition = None

        if not self.evaluation:
            await self._submit_collection_rollout(0.0, allow_empty=True)
        self.rollout.clear()

        lifetime_ticks = 0
        if self.last_assign_tick is not None and self.last_sensor_tick is not None:
            lifetime_ticks = max(0, int(self.last_sensor_tick - self.last_assign_tick))
        if had_episode:
            self._record_episode(lifetime_ticks)

    async def _handle_sensors(self, msg: Dict[str, Any], ws) -> None:
        if self.snake_id is None:
            return
        if int(msg.get("snakeId", -1)) != self.snake_id:
            return

        tick = int(msg.get("tick", 0))
        self.last_sensor_tick = tick
        if self.last_assign_tick is None:
            self.last_assign_tick = tick

        sensors = msg.get("sensors")
        if sensors is None:
            sensors = []
        obs = np.asarray(sensors, dtype=np.float32)
        if obs.shape[0] != len(self.sensor_order):
            return

        self._refresh_collection_epoch()

        reward_ready = self.pending_transition is not None or (
            self.evaluation and self.episode_actions > 0
        )
        if self.last_obs is not None and reward_ready:
            components = self._reward_components(self.last_obs, obs)
            reward = float(sum(components.values()))
            if self.pending_transition is not None:
                self.pending_transition.reward += reward
            self.episode_return += reward
            self._record_reward_components(components)

        self.last_obs = obs
        if not self._should_send_action(tick):
            return

        if not self.evaluation and self.pending_transition is not None:
            if self._collection_open():
                self.rollout.append(self.pending_transition)
            self.pending_transition = None

        if (
            not self.evaluation
            and self._collection_open()
            and len(self.rollout) >= self.cfg.horizon
        ):
            bootstrap = self.shared_state.inference_value(obs)
            await self._submit_collection_rollout(bootstrap)

        sampled_action = self._sample_action(obs)
        turn, boost, logp, value, turn_latent, policy_version = sampled_action
        if not self.evaluation and self._collection_open():
            self.pending_transition = Transition(
                obs=obs,
                action_turn=turn,
                turn_latent=turn_latent,
                action_boost=boost,
                logp=logp,
                value=value,
                reward=0.0,
                done=0.0,
                policy_version=policy_version,
            )

        self.steps += 1
        self.episode_actions += 1
        action_msg = {
            "type": "action",
            "tick": tick,
            "snakeId": self.snake_id,
            "turn": max(-1.0, min(1.0, float(turn))),
            "boost": max(0.0, min(1.0, float(boost))),
        }
        await ws.send(json.dumps(action_msg))
        self.last_sent_tick = tick
        self.last_sent_time = asyncio.get_running_loop().time()

    def _reward_components(self, previous: np.ndarray, current: np.ndarray) -> Dict[str, float]:
        """Call the canonical reward helper while keeping the overridden sensor loop compact."""
        from .utils import default_reward_components  # pylint: disable=import-outside-toplevel

        return default_reward_components(previous, current, self.sensor_idx, self.cfg)


def _save_pending_exact_best(cfg: Config, shared_state: SynchronousSharedState) -> None:
    if not shared_state.best_checkpoint_pending:
        return
    try:
        best_path = save_best_checkpoint(cfg, shared_state)
        _write_exact_best_metadata(cfg, shared_state)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"[learner] best checkpoint save failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    shared_state.mark_best_checkpoint_saved()
    snapshot = shared_state.best_policy_snapshot or {}
    print(
        f"[learner] best checkpoint saved path={best_path}, "
        f"lifetime_ticks={shared_state.best_lifetime_ticks}, "
        f"ep_return={shared_state.best_episode_return:.3f}, "
        f"policy_v={int(snapshot.get('update_steps', -1))}, source=frozen_evaluator"
    )


def _coherent_policy_version(
    contributions: Dict[int, Tuple[List[Transition], float]]
) -> Optional[int]:
    versions = {
        int(transition.policy_version)
        for rollout, _bootstrap in contributions.values()
        for transition in rollout
    }
    if not versions:
        return None
    if len(versions) != 1:
        raise RuntimeError(f"collection epoch mixed policy versions: {sorted(versions)}")
    return next(iter(versions))


async def synchronous_learner_loop(
    cfg: Config,
    shared_state: SynchronousSharedState,
    experience_q: asyncio.Queue,
) -> None:
    """Wait for one contribution from every actor, update once, then publish the next policy."""
    import time  # pylint: disable=import-outside-toplevel

    last_log = time.time()
    update_rounds = 0
    total_steps = 0
    pending: Dict[int, Dict[int, Tuple[List[Transition], float]]] = {}

    while True:
        actor_id, epoch, rollout, bootstrap = await experience_q.get()
        _save_pending_exact_best(cfg, shared_state)

        current_epoch = int(shared_state.collection_epoch)
        if int(epoch) < current_epoch:
            continue
        if int(epoch) > current_epoch:
            pending.setdefault(int(epoch), {})[int(actor_id)] = (
                list(rollout), float(bootstrap)
            )
            continue

        contributions = pending.setdefault(current_epoch, {})
        contributions.setdefault(int(actor_id), (list(rollout), float(bootstrap)))
        if len(contributions) < cfg.actors:
            continue

        contributions = pending.pop(current_epoch)
        total_steps += sum(len(item[0]) for item in contributions.values())
        nonempty = [item for item in contributions.values() if item[0]]
        rollout_version = _coherent_policy_version(contributions)
        metrics: Optional[Dict[str, float]] = None

        if nonempty:
            batch = collate_rollouts(nonempty, cfg, device=cfg.train_device)
            metrics = await shared_state.ppo_update_async(batch)
            update_rounds += 1

        shared_state.advance_collection_epoch()
        _save_pending_exact_best(cfg, shared_state)

        if (
            metrics is not None
            and cfg.save_every_updates > 0
            and (update_rounds % cfg.save_every_updates) == 0
        ):
            try:
                save_checkpoint(cfg, shared_state)
            except (OSError, RuntimeError) as exc:
                print(
                    f"[learner] checkpoint save failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        now = time.time()
        if metrics is None or now - last_log < cfg.log_every_seconds:
            continue

        runtime = shared_state.drain_runtime_stats()
        rewards = (
            f"growth={runtime['reward_growth']:.3f},"
            f"food={runtime['reward_food_approach']:.3f},"
            f"survival={runtime['reward_survival']:.3f},"
            f"wall={runtime['reward_wall']:.3f},"
            f"hazard={runtime['reward_hazard']:.3f},"
            f"death={runtime['reward_death']:.3f}"
        )
        print(
            f"[learner] updates={update_rounds}, steps={total_steps}, "
            f"collection={current_epoch}, actors_ready={len(contributions)}/{cfg.actors}, "
            f"policy_v={shared_state.published_policy_version}, "
            f"rollout_v={rollout_version if rollout_version is not None else -1}, "
            f"samples={int(metrics['samples'])}, "
            f"opt_steps={int(metrics['optimizer_steps'])}, "
            f"loss={metrics['loss']:.4f}, "
            f"policy={metrics['policy_loss']:.4f}, "
            f"value={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            f"entropy_turn={metrics['entropy_turn']:.4f}, "
            f"entropy_boost={metrics['entropy_boost']:.4f}, "
            f"initial_kl={metrics['initial_kl']:.5f}, "
            f"kl={metrics['approx_kl']:.5f}, "
            f"kl_stop={int(metrics['kl_early_stopped'])}, "
            f"stale_batch={int(metrics['stale_batch'])}, "
            f"clip={metrics['clip_fraction']:.3f}, "
            f"ev={metrics['explained_variance']:.3f}, "
            f"turn_abs_mean={metrics['turn_mean_abs_mean']:.3f}, "
            f"turn_abs_max={metrics['turn_mean_abs_max']:.3f}, "
            f"turn_sat={metrics['turn_saturated_fraction']:.3f}, "
            f"episodes={int(runtime['episodes'])}, "
            f"ep_return={runtime['episode_return_mean']:.3f}, "
            f"lifetime_ticks={runtime['episode_lifetime_mean']:.1f}, "
            f"reward[{rewards}]"
        )
        last_log = now
