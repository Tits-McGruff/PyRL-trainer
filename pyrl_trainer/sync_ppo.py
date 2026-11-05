"""Synchronous collection barrier for stable on-policy PPO training."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent import ActorClient, Transition
from .checkpointing import save_best_checkpoint, save_checkpoint
from .config import Config, REWARD_CONFIG_VERSION, reward_config_record
from .learner import SharedState, collate_rollouts, format_reward_exposure
from .evaluation import EVALUATION_CONFIG_VERSION, evaluation_config_record


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


def enforce_optimizer_learning_rate(shared_state: "SynchronousSharedState") -> None:
    """Keep a resumed Adam optimizer on the configured LR instead of checkpoint LR."""
    target = float(shared_state.cfg.lr)
    for param_group in shared_state.optimizer.param_groups:
        param_group["lr"] = target


def _best_meta_path(cfg: Config) -> Path:
    return Path(cfg.ckpt_dir) / _EXACT_BEST_META


def restore_exact_best_metadata(cfg: Config, shared_state: "SynchronousSharedState") -> None:
    """Separate legacy training-actor records from exact frozen-evaluator records."""
    checkpoint_lifetime = max(
        0, int(getattr(shared_state, "best_lifetime_ticks", 0))
    )
    checkpoint_return = float(
        getattr(shared_state, "best_episode_return", float("-inf"))
    )

    shared_state.best_lifetime_ticks = 0
    shared_state.best_episode_return = float("-inf")
    shared_state.best_policy_snapshot = None
    shared_state.best_evaluation_result = None
    shared_state.best_checkpoint_pending = False
    shared_state.legacy_best_lifetime_ticks = checkpoint_lifetime
    shared_state.legacy_best_episode_return = checkpoint_return

    path = _best_meta_path(cfg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return

    if payload.get("source") != "frozen_evaluator":
        return
    if (payload.get("reward_config_version") != REWARD_CONFIG_VERSION
            or payload.get("reward_config") != reward_config_record(cfg)):
        return
    if (payload.get("evaluation_config_version") != EVALUATION_CONFIG_VERSION
            or payload.get("evaluation_config") != evaluation_config_record(cfg)):
        return
    try:
        lifetime = int(payload["lifetime_ticks"])
        episode_return = float(payload["episode_return"])
        legacy_lifetime = int(payload.get("legacy_lifetime_ticks", 0))
        legacy_return = float(payload.get("legacy_episode_return", float("-inf")))
    except (KeyError, TypeError, ValueError):
        return
    if lifetime < 0 or not math.isfinite(episode_return):
        return

    shared_state.best_lifetime_ticks = lifetime
    shared_state.best_episode_return = episode_return
    if isinstance(payload.get("evaluation_result"), dict):
        shared_state.best_evaluation_result = payload["evaluation_result"]
    if legacy_lifetime > 0:
        shared_state.legacy_best_lifetime_ticks = legacy_lifetime
        shared_state.legacy_best_episode_return = legacy_return
    elif checkpoint_lifetime == lifetime:
        shared_state.legacy_best_lifetime_ticks = 0
        shared_state.legacy_best_episode_return = float("-inf")


def _write_exact_best_metadata(cfg: Config, shared_state: "SynchronousSharedState") -> None:
    path = _best_meta_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = shared_state.best_policy_snapshot or {}
    payload = {
        "source": "frozen_evaluator",
        "reward_config_version": REWARD_CONFIG_VERSION,
        "reward_config": reward_config_record(cfg),
        "evaluation_config_version": EVALUATION_CONFIG_VERSION,
        "evaluation_config": evaluation_config_record(cfg),
        "evaluation_result": shared_state.best_evaluation_result,
        "lifetime_ticks": int(shared_state.best_lifetime_ticks),
        "episode_return": float(shared_state.best_episode_return),
        "policy_version": int(snapshot.get("update_steps", -1)),
        "legacy_lifetime_ticks": int(shared_state.legacy_best_lifetime_ticks),
    }
    if math.isfinite(shared_state.legacy_best_episode_return):
        payload["legacy_episode_return"] = float(
            shared_state.legacy_best_episode_return
        )
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


class SynchronousActorClient(ActorClient):  # pylint: disable=too-few-public-methods
    """Actor that contributes at most one rollout to each collection epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._submitted_collection_epoch: Optional[int] = None
        self._active_collection_epoch = int(self.shared_state.collection_epoch)
        self._fence_required = False
        self._fence_action_id: Optional[int] = None

    def _collection_open(self) -> bool:
        if self.evaluation:
            return False
        current = int(self.shared_state.collection_epoch)
        return self._submitted_collection_epoch != current

    def _training_collection_open(self) -> bool:
        return self._collection_open() and not self._fence_required

    def _account_reward(self, reward) -> None:
        if (reward is not None and reward["transitionKind"] != "initial"
                and self._fence_required and self._collection_open()):
            callback = getattr(self.shared_state, "record_epoch_fence_interval", None)
            if callable(callback):
                callback(reward["deltas"]["elapsedSeconds"])
        super()._account_reward(reward)

    async def _handle_transition(self, msg) -> None:
        self._refresh_collection_epoch()
        await super()._handle_transition(msg)

    async def _submit_environment_closure(self, bootstrap: float) -> None:
        if not self.evaluation:
            await self._submit_collection_rollout(bootstrap, allow_empty=True)
        self.rollout.clear()

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
        if self._fence_required and self.rollout:
            raise RuntimeError("cannot submit data before the collection control fence")

        epoch = int(self.shared_state.collection_epoch)
        await self.experience_q.put(
            (self.actor_id, epoch, list(self.rollout), float(bootstrap))
        )
        self._record_admitted_rollout()
        self.rollout.clear()
        self._submitted_collection_epoch = epoch

    def _refresh_collection_epoch(self) -> None:
        if self.evaluation:
            return
        current = int(self.shared_state.collection_epoch)
        if current != self._active_collection_epoch:
            if self.rollout or self.pending_transition is not None:
                raise RuntimeError("unsubmitted experience crossed a collection epoch")
            self._active_collection_epoch = current
            self._submitted_collection_epoch = None
            self._fence_required = True
            self._fence_action_id = None

    def _observe_collection_fence(self) -> None:
        if self.evaluation or not self._collection_open() or not self._fence_required:
            return
        reward = self.last_reward_envelope
        if reward is None:
            return
        applied = reward["control"]["appliedCommand"]
        identity = None if applied is None else applied["command"]["clientActionId"]
        if (identity is not None and self._fence_action_id is not None
                and int(identity, 16) >= self._fence_action_id):
            self._fence_required = False

    async def _truncate_disconnected_rollout(self) -> None:
        """Contribute partial work once, then reconnect without inventing a terminal."""
        self._refresh_collection_epoch()
        self._finish_evaluation("incomplete", "transportGap", integrity=False)
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
        self._reward_rebase = True
        self._episode_integrity = False



    async def _handle_sensors(self, msg: Dict[str, object], ws) -> None:
        if self.snake_id is None:
            return
        self._refresh_collection_epoch()
        obs = self._observation_and_reward(msg)
        self._check_evaluation_limit()
        self._observe_collection_fence()
        tick = int(msg["tick"])

        self.last_obs = obs
        if not self._should_send_action(tick):
            return

        if not self.evaluation and self.pending_transition is not None:
            if self._training_collection_open():
                self.rollout.append(self.pending_transition)
            self.pending_transition = None

        if (
            not self.evaluation
            and self._training_collection_open()
            and len(self.rollout) >= self.cfg.horizon
        ):
            bootstrap = self.shared_state.inference_value(obs)
            await self._submit_collection_rollout(bootstrap)

        sampled_action = self._sample_action(obs)
        turn, boost, logp, value, turn_latent, policy_version = sampled_action
        if not self.evaluation and self._training_collection_open():
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
        identity = await self._send_action(ws, tick, turn, boost)
        if (self._fence_required and self._collection_open() and self._fence_action_id is None
                and self._active_collection_epoch == self.shared_state.collection_epoch):
            self._fence_action_id = int(identity, 16)


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
    *,
    decision_budget: Optional[int] = None,
) -> None:
    """Wait for one contribution from every actor, update once, then publish the next policy."""
    last_log = time.time()
    update_rounds = 0
    total_steps = 0
    total_alive_seconds = 0.0
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

        if decision_budget is not None:
            total_alive_seconds += sum(
                item.alive_seconds for rollout, _value in nonempty for item in rollout
            )
            if metrics is not None and metrics["stale_batch"]:
                raise RuntimeError("bounded PPO run received a stale batch")
            if total_steps >= decision_budget:
                _save_pending_exact_best(cfg, shared_state)
                save_checkpoint(cfg, shared_state)
                print(
                    f"[qualification] admitted_decisions={total_steps}, "
                    f"decision_budget={decision_budget}, alive_seconds={total_alive_seconds:.6f}, "
                    f"updates={update_rounds}, policy_v={shared_state.published_policy_version}"
                )
                return

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
            f"food={runtime['reward_food']:.3f},"
            f"survival={runtime['reward_survival']:.3f},"
            f"points={runtime['reward_points']:.3f},"
            f"kills={runtime['reward_kills']:.3f},"
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
            f"entropy_turn_gaussian={metrics['entropy_turn_gaussian']:.4f}, "
            f"entropy_boost={metrics['entropy_boost']:.4f}, "
            f"initial_kl={metrics['initial_kl']:.5f}, "
            f"kl={metrics['approx_kl']:.5f}, "
            f"kl_stop={int(metrics['kl_early_stopped'])}, "
            f"stale_batch={int(metrics['stale_batch'])}, "
            f"clip={metrics['clip_fraction']:.3f}, "
            f"ev={metrics['explained_variance']:.3f}, "
            f"turn_abs_mean={metrics['turn_mean_abs_mean']:.3f}, "
            f"turn_abs_max={metrics['turn_mean_abs_max']:.3f}, "
            f"turn_mean_sat={metrics['turn_saturated_fraction']:.3f}, "
            f"turn_std={metrics['turn_std']:.3f}, "
            f"turn_sample_abs={metrics['turn_sample_abs_mean']:.3f}, "
            f"turn_sample_sat={metrics['turn_sample_saturated_fraction']:.3f}, "
            f"episodes={int(runtime['episodes'])}, "
            f"ep_return={runtime['episode_return_mean']:.3f}, "
            f"lifetime_ticks={runtime['episode_lifetime_mean']:.1f}, "
            f"reward[{rewards}] {format_reward_exposure(runtime)}"
        )
        last_log = now
