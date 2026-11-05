"""Learner loop and PPO update utilities."""

import asyncio
import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import optim
from slither_contracts.models import ExternalRewardDeltasV1, ExternalRewardEnvelopeV1

from .agent import Transition  # Circular import if not careful, but Transition is data
from .checkpointing import save_best_checkpoint, save_checkpoint
from .config import Config
from .network import PolicyValueNet, MIN_TURN_LOG_STD, MAX_TURN_LOG_STD
from .sensor_contract import SensorContract
from .reward_contract import REWARD_COMPONENT_KEYS
from .evaluation import policy_snapshot_sha256


_TARGET_KL = 0.025
_LOGGER = logging.getLogger(__name__)


def format_reward_exposure(stats: Dict[str, float]) -> str:
    """Describe received events separately from the rollout-admission denominator."""
    rates = ",".join(f"{key}={stats[f'reward_{key}_per_alive_second']:.4f}"
                     for key in REWARD_COMPONENT_KEYS)
    cumulative = ",".join(f"{key}={stats[f'reward_total_{key}']:.3f}"
                          for key in REWARD_COMPONENT_KEYS)
    return (
        f"received[intervals={int(stats['received_intervals'])},"
        f"total_intervals={int(stats['received_interval_count'])},"
        f"elapsed_s={stats['received_elapsedSeconds']:.3f},"
        f"alive_s={stats['received_aliveSecondsDelta']:.3f},"
        f"food_value={stats['received_foodValueDelta']:.3f},"
        f"food_count={int(stats['received_foodCountDelta'])},"
        f"growth={stats['received_retainedGrowthDelta']:.3f},"
        f"task_points={stats['received_pointsDelta'] - stats['received_survivalPointsDelta']:.3f},"
        f"kills={int(stats['received_killsDelta'])},"
        f"boost_spent={stats['received_boostPointsSpentDelta']:.3f},"
        f"total_food={stats['received_total_foodValueDelta']:.3f},"
        f"total_alive_s={stats['received_total_aliveSecondsDelta']:.3f}] "
        f"admitted[transitions={int(stats['admitted_transitions'])},"
        f"elapsed_s={stats['admitted_elapsed_seconds']:.3f},"
        f"alive_s={stats['admitted_alive_seconds']:.3f},"
        f"decision_s={stats['decision_seconds_mean']:.5f}] "
        f"reward_per_admitted_alive_second[{rates}] reward_total[{cumulative}]"
    )


class SharedState:  # pylint: disable=too-many-instance-attributes
    """
    Shared policy/value network weights.
    Learner updates the train_model; actors use an inference copy to avoid training contention.
    """

    def __init__(
        self,
        obs_dim: int,
        cfg: Config,
        sensor_contract: Optional[SensorContract] = None,
    ):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.sensor_contract = sensor_contract

        self.train_model = PolicyValueNet(
            obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers, initial_turn_std=cfg.turn_std
        ).to(cfg.train_device)
        self.infer_model = PolicyValueNet(
            obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers, initial_turn_std=cfg.turn_std
        ).to(cfg.infer_device)

        self.optimizer = optim.Adam(self.train_model.parameters(), lr=cfg.lr)
        self.update_steps = 0

        self._reward_totals = {
            **{key: 0.0 for key in REWARD_COMPONENT_KEYS},
            "death": 0.0,
        }
        self._reward_cumulative = dict(self._reward_totals)
        self._admitted_seconds = 0.0
        self._admitted_alive_seconds = 0.0
        self._admitted_transitions = 0
        self._received_totals = {key: 0.0 for key in ExternalRewardDeltasV1.__annotations__}
        self._received_cumulative = dict(self._received_totals)
        self._received_intervals = 0
        self._received_interval_count = 0
        self._no_task_intervals = 0
        self._no_task_alive_seconds = 0.0
        self._episode_count = 0
        self._episode_return_sum = 0.0
        self._episode_lifetime_sum = 0.0

        self.best_lifetime_ticks = 0
        self.best_episode_return = float("-inf")
        self.best_checkpoint_pending = False
        self.best_policy_snapshot: Optional[Dict[str, Any]] = None
        self.best_evaluation_result: Optional[Dict[str, Any]] = None
        self.last_evaluation_result: Optional[Dict[str, Any]] = None
        self._published_policy_snapshot: Dict[str, Any] = {}
        self.sync_infer_from_train()

    def _sync_infer_from_train(self) -> None:
        """Copy training weights to the actor inference model."""
        target_device = self.cfg.infer_device
        state_dict = {
            key: value.to(target_device)
            for key, value in self.train_model.state_dict().items()
        }
        self.infer_model.load_state_dict(state_dict)

    def _capture_published_policy_snapshot(self) -> Dict[str, Any]:
        """Freeze the currently published actor policy on CPU for exact evaluation."""
        return {
            "update_steps": int(self.update_steps),
            "model": {
                key: value.detach().to("cpu").clone()
                for key, value in self.train_model.state_dict().items()
            },
        }

    def sync_infer_from_train(self) -> None:
        """Publish training weights and retain an immutable policy snapshot."""
        self._sync_infer_from_train()
        self._published_policy_snapshot = self._capture_published_policy_snapshot()

    @property
    def published_policy_version(self) -> int:
        """Return the optimizer-step version of the policy actors currently see."""
        return int(self._published_policy_snapshot.get("update_steps", self.update_steps))

    def begin_evaluation_policy(self) -> Tuple[Dict[str, Any], PolicyValueNet]:
        """Create a frozen local model for one complete evaluator episode."""
        snapshot = self._published_policy_snapshot
        # Clone without random initialization. Saving/restoring a global RNG here
        # would race the PPO update thread's minibatch sampling.
        model = copy.deepcopy(self.infer_model)
        model.load_state_dict(snapshot["model"])
        model.eval()
        return snapshot, model

    def fresh_optimizer_state_dict(self) -> Dict[str, Any]:
        """Return a valid empty Adam state for resuming an archived policy snapshot."""
        optimizer = optim.Adam(self.train_model.parameters(), lr=self.cfg.lr)
        return optimizer.state_dict()

    @staticmethod
    def _joint_logp_entropy(
        turn_mean: torch.Tensor,
        boost_logit: torch.Tensor,
        turn_latent: torch.Tensor,
        action_boost: torch.Tensor,
        turn_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a stored pre-tanh turn and boost under one policy."""
        std = torch.as_tensor(
            turn_std, dtype=turn_mean.dtype, device=turn_mean.device
        )
        base_dist = torch.distributions.Normal(turn_mean, std)
        boost_dist = torch.distributions.Bernoulli(logits=boost_logit)

        action_turn = torch.tanh(turn_latent)
        transform = torch.distributions.transforms.TanhTransform(cache_size=0)
        logp_turn = base_dist.log_prob(turn_latent) - transform.log_abs_det_jacobian(
            turn_latent, action_turn
        )
        logp_boost = boost_dist.log_prob(action_boost)
        entropy_turn = base_dist.entropy()
        entropy_boost = boost_dist.entropy()
        return action_turn, logp_turn + logp_boost, entropy_turn, entropy_boost

    def act(
        self,
        obs: np.ndarray,
        policy_model: Optional[PolicyValueNet] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[float, float, float, float, float]:
        """Sample turn/boost and return action, log-prob, value, and turn latent."""
        model = self.infer_model if policy_model is None else policy_model
        o = torch.tensor(
            obs, dtype=torch.float32, device=self.cfg.infer_device
        ).unsqueeze(0)
        turn_mean, boost_logit, value = model(o)

        std = model.steering_std()
        turn_latent = torch.normal(turn_mean, std.expand_as(turn_mean), generator=generator)
        boost_dist = torch.distributions.Bernoulli(logits=boost_logit)
        action_boost = torch.bernoulli(boost_dist.probs, generator=generator)

        action_turn, logp, _entropy_turn, _entropy_boost = self._joint_logp_entropy(
            turn_mean, boost_logit, turn_latent, action_boost, std
        )

        return (
            float(action_turn.item()),
            float(action_boost.item()),
            float(logp.item()),
            float(value.item()),
            float(turn_latent.item()),
        )

    def record_reward_components(self, components: Dict[str, float]) -> None:
        """Accumulate admitted reward components for the next learner log."""
        for key, value in components.items():
            if key in self._reward_totals:
                self._reward_totals[key] += float(value)
                self._reward_cumulative[key] += float(value)

    def record_admitted_rollout(self, rollout: List[Transition]) -> None:
        """Count a queued rollout, excluding pending or discarded decisions."""
        for transition in rollout:
            self.record_reward_components(transition.reward_components)
            self._admitted_seconds += transition.elapsed_seconds
            self._admitted_alive_seconds += transition.alive_seconds
        self._admitted_transitions += len(rollout)

    def record_received_reward(self, reward: ExternalRewardEnvelopeV1) -> None:
        """Track raw accepted intervals even when an actor is waiting at the barrier."""
        if reward["transitionKind"] == "initial":
            return
        delta = reward["deltas"]
        self._received_intervals += 1
        self._received_interval_count += 1
        for key, value in delta.items():
            self._received_totals[key] += value
            self._received_cumulative[key] += value
        # Category counters avoid treating subtraction roundoff between two
        # large survival balances as evidence of a real task event.
        has_task_event = any((delta["foodValueDelta"], delta["foodCountDelta"],
                              delta["retainedGrowthDelta"], delta["killsDelta"],
                              delta["foodPointsDelta"], delta["killPointsDelta"],
                              delta["otherPointsDelta"], delta["boostPointsSpentDelta"]))
        if has_task_event:
            self._no_task_intervals = 0
            self._no_task_alive_seconds = 0.0
            return
        self._no_task_intervals += 1
        self._no_task_alive_seconds += delta["aliveSecondsDelta"]
        if self._no_task_intervals >= 10_000 and self._no_task_alive_seconds >= 60.0:
            _LOGGER.warning(
                "No authoritative task events across %d received intervals and %.3f "
                "credited live seconds; check environment opportunities and reward delivery",
                self._no_task_intervals, self._no_task_alive_seconds,
            )
            self._no_task_intervals = 0
            self._no_task_alive_seconds = 0.0

    def record_evaluation_result(self, result: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        """Retain a bounded complete result and promote only an intact frozen episode."""
        self.last_evaluation_result = result
        if not result["integrity"] or result["transition_kind"] == "incomplete":
            return
        if (result["policy_version"] != int(snapshot["update_steps"])
                or result["policy_sha256"] != policy_snapshot_sha256(snapshot)):
            raise RuntimeError("evaluator result does not identify its exact frozen policy")
        previous_best = self.best_episode_return
        self.record_episode(result["episode_return"], result["lifetime_ticks"],
                            policy_snapshot=snapshot, include_runtime_stats=False)
        if self.best_episode_return > previous_best:
            self.best_evaluation_result = result

    def record_episode(
        self,
        episode_return: float,
        lifetime_ticks: int,
        policy_snapshot: Optional[Dict[str, Any]] = None,
        include_runtime_stats: bool = True,
    ) -> None:
        """Record diagnostics and retain an exact evaluator policy when it sets a record."""
        normalized_return = float(episode_return)
        normalized_lifetime = max(0, int(lifetime_ticks))

        if include_runtime_stats:
            self._episode_count += 1
            self._episode_return_sum += normalized_return
            self._episode_lifetime_sum += float(normalized_lifetime)

        if policy_snapshot is None:
            return
        if not isinstance(policy_snapshot.get("model"), dict):
            return

        if math.isfinite(normalized_return) and normalized_return > self.best_episode_return:
            self.best_lifetime_ticks = normalized_lifetime
            self.best_episode_return = normalized_return
            self.best_policy_snapshot = policy_snapshot
            self.best_checkpoint_pending = True

    def mark_best_checkpoint_saved(self) -> None:
        """Clear the pending best-save flag after a successful atomic save."""
        self.best_checkpoint_pending = False

    def drain_runtime_stats(self) -> Dict[str, float]:
        """Return and reset bounded actor-side training diagnostics."""
        count = self._episode_count
        stats = {
            "episodes": float(count),
            "episode_return_mean": (
                self._episode_return_sum / count if count else 0.0
            ),
            "episode_lifetime_mean": (
                self._episode_lifetime_sum / count if count else 0.0
            ),
        }
        for key, value in self._reward_totals.items():
            stats[f"reward_{key}"] = float(value)
            stats[f"reward_total_{key}"] = self._reward_cumulative[key]
            stats[f"reward_{key}_per_alive_second"] = (
                value / self._admitted_alive_seconds if self._admitted_alive_seconds else 0.0
            )
        stats.update(
            admitted_transitions=float(self._admitted_transitions),
            admitted_elapsed_seconds=self._admitted_seconds,
            admitted_alive_seconds=self._admitted_alive_seconds,
            decision_seconds_mean=(self._admitted_seconds / self._admitted_transitions
                                   if self._admitted_transitions else 0.0),
            received_intervals=float(self._received_intervals),
            received_interval_count=float(self._received_interval_count),
        )
        for key, value in self._received_totals.items():
            stats[f"received_{key}"] = value
            stats[f"received_total_{key}"] = self._received_cumulative[key]
            self._received_totals[key] = 0.0

        for key in self._reward_totals:
            self._reward_totals[key] = 0.0
        self._admitted_seconds = 0.0
        self._admitted_alive_seconds = 0.0
        self._admitted_transitions = 0
        self._received_intervals = 0
        self._episode_count = 0
        self._episode_return_sum = 0.0
        self._episode_lifetime_sum = 0.0
        return stats

    def _evaluate_policy_batch_sync(  # pylint: disable=too-many-locals
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Measure whole-batch PPO drift without building an autograd graph."""
        with torch.no_grad():
            obs = batch["obs"]
            act_turn_latent = batch["act_turn_latent"]
            act_boost = batch["act_boost"]
            old_logp = batch["old_logp"]
            returns = batch["returns"]
            adv = batch["adv"]

            turn_mean, boost_logit, value = self.train_model(obs)
            (
                _action_turn,
                logp,
                entropy_turn_values,
                entropy_boost_values,
            ) = self._joint_logp_entropy(
                turn_mean,
                boost_logit,
                act_turn_latent,
                act_boost,
                self.train_model.steering_std(),
            )

            log_ratio = logp - old_logp
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip
            )
            policy_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()
            value_loss = ((returns - value) ** 2).mean()
            entropy_turn = entropy_turn_values.mean()
            entropy_boost = entropy_boost_values.mean()
            entropy = entropy_turn + entropy_boost
            loss = (
                policy_loss
                + self.cfg.vf_coef * value_loss
                - self.cfg.ent_coef * entropy
            )
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.cfg.ppo_clip)
                .to(torch.float32)
                .mean()
            )

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "entropy_turn": float(entropy_turn.item()),
            "entropy_boost": float(entropy_boost.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
        }

    def _ppo_minibatch_sync(  # pylint: disable=too-many-locals
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run one PPO minibatch, stopping before a step when drift is too large."""
        obs = batch["obs"]
        act_turn_latent = batch["act_turn_latent"]
        act_boost = batch["act_boost"]
        old_logp = batch["old_logp"]
        returns = batch["returns"]
        adv = batch["adv"]

        turn_mean, boost_logit, value = self.train_model(obs)
        (
            _action_turn,
            logp,
            entropy_turn_values,
            entropy_boost_values,
        ) = self._joint_logp_entropy(
            turn_mean,
            boost_logit,
            act_turn_latent,
            act_boost,
            self.train_model.steering_std(),
        )

        log_ratio = logp - old_logp
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip
        )
        policy_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()
        value_loss = ((returns - value) ** 2).mean()
        entropy_turn = entropy_turn_values.mean()
        entropy_boost = entropy_boost_values.mean()
        entropy = entropy_turn + entropy_boost
        loss = (
            policy_loss
            + self.cfg.vf_coef * value_loss
            - self.cfg.ent_coef * entropy
        )

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.cfg.ppo_clip)
                .to(torch.float32)
                .mean()
            )

        should_step = float(approx_kl.item()) <= _TARGET_KL
        if should_step:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0,
                                           error_if_nonfinite=True)
            self.optimizer.step()
            self.train_model.constrain_turn_std()
            self.update_steps += 1

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "entropy_turn": float(entropy_turn.item()),
            "entropy_boost": float(entropy_boost.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "optimizer_step": 1.0 if should_step else 0.0,
        }

    def _post_update_diagnostics_sync(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Measure critic fit and action saturation after a PPO batch attempt."""
        with torch.no_grad():
            turn_mean, _boost_logit, values = self.train_model(batch["obs"])
            returns = batch["returns"]
            return_variance = torch.var(returns, unbiased=False)
            if float(return_variance.item()) > 1e-8:
                explained_variance = 1.0 - (
                    torch.var(returns - values, unbiased=False) / return_variance
                )
                explained = float(explained_variance.item())
            else:
                explained = 0.0

            turn_mean_abs = torch.abs(turn_mean)
            mean_action = torch.tanh(turn_mean)
            sampled_action = torch.tanh(batch["act_turn_latent"])
            learned_std = self.train_model.steering_std()
            return {
                "explained_variance": explained,
                "turn_mean_abs_mean": float(turn_mean_abs.mean().item()),
                "turn_mean_abs_max": float(turn_mean_abs.max().item()),
                "turn_std": float(learned_std.item()),
                "turn_sample_abs_mean": float(sampled_action.abs().mean().item()),
                "turn_sample_saturated_fraction": float(
                    (sampled_action.abs() > 0.95).float().mean().item()
                ),
                "turn_sample_sign_disagreement": float(
                    (sampled_action.sign() != mean_action.sign()).float().mean().item()
                ),
                "turn_log_std_bound_hit": float(
                    self.train_model.turn_log_std <= MIN_TURN_LOG_STD + 1e-6
                    or self.train_model.turn_log_std >= MAX_TURN_LOG_STD - 1e-6
                ),
                "entropy_turn_gaussian": float(
                    torch.distributions.Normal(turn_mean, learned_std).entropy().mean().item()
                ),
                "turn_saturated_fraction": float(
                    (torch.abs(mean_action) > 0.95)
                    .to(torch.float32)
                    .mean()
                    .item()
                ),
            }

    def _ppo_update_batch_sync(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run coherent PPO minibatches, rejecting already-stale behavior batches."""
        sample_count = int(batch["obs"].shape[0])
        if sample_count <= 0:
            raise ValueError("PPO batch must contain at least one sample")

        initial = self._evaluate_policy_batch_sync(batch)
        if initial["approx_kl"] > _TARGET_KL:
            metrics = dict(initial)
            metrics.update(self._post_update_diagnostics_sync(batch))
            metrics["optimizer_steps"] = 0.0
            metrics["samples"] = float(sample_count)
            metrics["target_kl"] = _TARGET_KL
            metrics["initial_kl"] = initial["approx_kl"]
            metrics["kl_early_stopped"] = 0.0
            metrics["stale_batch"] = 1.0
            return metrics

        minibatch_size = max(1, int(self.cfg.minibatch))
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "entropy_turn": 0.0,
            "entropy_boost": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        weighted_samples = 0
        optimizer_steps = 0
        kl_early_stopped = False

        for _ in range(max(1, int(self.cfg.epochs))):
            order = torch.randperm(sample_count, device=batch["obs"].device)
            for start in range(0, sample_count, minibatch_size):
                index = order[start:start + minibatch_size]
                mini = {
                    key: value.index_select(0, index)
                    for key, value in batch.items()
                }
                mini_metrics = self._ppo_minibatch_sync(mini)
                weight = int(index.numel())
                weighted_samples += weight
                optimizer_steps += int(mini_metrics["optimizer_step"])
                for key in totals:
                    totals[key] += mini_metrics[key] * weight

                if mini_metrics["optimizer_step"] < 0.5:
                    kl_early_stopped = True
                    break
            if kl_early_stopped:
                break

        denom = float(max(1, weighted_samples))
        metrics = {key: value / denom for key, value in totals.items()}
        metrics["entropy"] = metrics["entropy_turn"] + metrics["entropy_boost"]
        metrics.update(self._post_update_diagnostics_sync(batch))
        metrics["optimizer_steps"] = float(optimizer_steps)
        metrics["samples"] = float(sample_count)
        metrics["target_kl"] = _TARGET_KL
        metrics["initial_kl"] = initial["approx_kl"]
        metrics["kl_early_stopped"] = 1.0 if kl_early_stopped else 0.0
        metrics["stale_batch"] = 0.0
        return metrics

    async def ppo_update_async(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run one PPO batch off-loop, publishing only when weights actually changed."""
        metrics = await asyncio.to_thread(self._ppo_update_batch_sync, batch)
        if metrics["optimizer_steps"] > 0.0:
            self.sync_infer_from_train()
        return metrics


def gae(
    rollout: List[Transition],
    gamma: float,
    lam: float,
    bootstrap_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and returns."""
    t_len = len(rollout)
    adv = np.zeros(t_len, dtype=np.float32)

    last_gae = 0.0
    last_value = bootstrap_value

    for t in reversed(range(t_len)):
        reward = rollout[t].reward
        value = rollout[t].value
        done = rollout[t].done
        next_value = last_value if t == t_len - 1 else rollout[t + 1].value
        delta = reward + gamma * (1.0 - done) * next_value - value
        last_gae = delta + gamma * lam * (1.0 - done) * last_gae
        adv[t] = last_gae

    returns = adv + np.asarray(
        [transition.value for transition in rollout], dtype=np.float32
    )
    return adv, returns


def split_rollout_by_policy_version(
    rollout: List[Transition], bootstrap_value: float
) -> List[Tuple[int, List[Transition], float]]:
    """Split a live actor rollout into contiguous single-policy behavior segments."""
    if not rollout:
        return []

    segments: List[Tuple[int, List[Transition], float]] = []
    start = 0
    version = int(rollout[0].policy_version)
    for index in range(1, len(rollout)):
        next_version = int(rollout[index].policy_version)
        if next_version == version:
            continue
        segment = list(rollout[start:index])
        segments.append((version, segment, float(rollout[index].value)))
        start = index
        version = next_version

    segments.append((version, list(rollout[start:]), float(bootstrap_value)))
    return segments


def collate_rollouts(  # pylint: disable=too-many-locals
    rollouts_and_boots: List[Tuple[List[Transition], float]],
    cfg: Config,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Stack complete rollout segments and normalize advantages once per PPO batch."""
    rollouts = [rollout for rollout, _bootstrap in rollouts_and_boots]

    obs = np.concatenate(
        [
            np.stack([transition.obs for transition in rollout], axis=0)
            for rollout in rollouts
        ],
        axis=0,
    )
    act_turn = np.concatenate(
        [
            np.asarray(
                [transition.action_turn for transition in rollout],
                dtype=np.float32,
            )
            for rollout in rollouts
        ],
        axis=0,
    )
    act_turn_latent = np.concatenate(
        [
            np.asarray(
                [transition.turn_latent for transition in rollout],
                dtype=np.float32,
            )
            for rollout in rollouts
        ],
        axis=0,
    )
    act_boost = np.concatenate(
        [
            np.asarray(
                [transition.action_boost for transition in rollout],
                dtype=np.float32,
            )
            for rollout in rollouts
        ],
        axis=0,
    )
    old_logp = np.concatenate(
        [
            np.asarray(
                [transition.logp for transition in rollout],
                dtype=np.float32,
            )
            for rollout in rollouts
        ],
        axis=0,
    )

    advantages = []
    returns = []
    for rollout, bootstrap in rollouts_and_boots:
        advantage, ret = gae(
            rollout, cfg.gamma, cfg.gae_lambda, bootstrap_value=bootstrap
        )
        advantages.append(advantage)
        returns.append(ret)

    adv = np.concatenate(advantages, axis=0)
    ret = np.concatenate(returns, axis=0)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return {
        "obs": torch.tensor(obs, dtype=torch.float32, device=device),
        "act_turn": torch.tensor(act_turn, dtype=torch.float32, device=device),
        "act_turn_latent": torch.tensor(
            act_turn_latent, dtype=torch.float32, device=device
        ),
        "act_boost": torch.tensor(
            act_boost, dtype=torch.float32, device=device
        ),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32, device=device),
        "adv": torch.tensor(adv, dtype=torch.float32, device=device),
        "returns": torch.tensor(ret, dtype=torch.float32, device=device),
    }


def _save_pending_best_checkpoint(cfg: Config, shared_state: SharedState) -> None:
    if not shared_state.best_checkpoint_pending:
        return
    try:
        best_path = save_best_checkpoint(cfg, shared_state)
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
        f"policy_v={int(snapshot.get('update_steps', -1))}"
    )


async def learner_loop(  # pylint: disable=too-many-locals,too-many-branches
    cfg: Config, shared_state: SharedState, experience_q: asyncio.Queue
) -> None:
    """Consume policy-versioned rollout segments and train on coherent PPO batches."""
    last_log = time.time()
    updates = 0
    total_steps = 0
    dropped_stale_steps = 0
    stale_before_or_equal = -1
    pending_by_version: Dict[int, List[Tuple[List[Transition], float]]] = {}

    while True:
        _actor_id, rollout, bootstrap = await experience_q.get()
        if not rollout:
            _save_pending_best_checkpoint(cfg, shared_state)
            continue

        _save_pending_best_checkpoint(cfg, shared_state)
        for version, segment, segment_bootstrap in split_rollout_by_policy_version(
            rollout, bootstrap
        ):
            total_steps += len(segment)
            if version <= stale_before_or_equal:
                dropped_stale_steps += len(segment)
                continue
            pending_by_version.setdefault(version, []).append(
                (segment, segment_bootstrap)
            )

        ready_versions = [
            version
            for version, items in pending_by_version.items()
            if sum(len(segment) for segment, _boot in items) >= cfg.batch_size
        ]
        if not ready_versions:
            continue

        rollout_version = max(ready_versions)
        pending_data = pending_by_version.pop(rollout_version)
        batch = collate_rollouts(pending_data, cfg, device=cfg.train_device)
        metrics = await shared_state.ppo_update_async(batch)
        updates += 1

        if metrics["stale_batch"] >= 0.5:
            stale_before_or_equal = max(stale_before_or_equal, rollout_version)
            dropped_stale_steps += int(metrics["samples"])
        else:
            stale_before_or_equal = max(stale_before_or_equal, rollout_version - 1)

        obsolete_versions = [
            version
            for version in pending_by_version
            if version <= stale_before_or_equal
        ]
        for version in obsolete_versions:
            dropped_stale_steps += sum(
                len(segment) for segment, _boot in pending_by_version[version]
            )
            del pending_by_version[version]

        _save_pending_best_checkpoint(cfg, shared_state)

        if cfg.save_every_updates > 0 and (updates % cfg.save_every_updates) == 0:
            try:
                save_checkpoint(cfg, shared_state)
            except (OSError, RuntimeError) as exc:
                print(
                    f"[learner] checkpoint save failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        now = time.time()
        if now - last_log >= cfg.log_every_seconds:
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
                f"[learner] updates={updates}, steps={total_steps}, "
                f"policy_v={shared_state.published_policy_version}, "
                f"rollout_v={rollout_version}, "
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
                f"stale_drop={dropped_stale_steps}, "
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
            dropped_stale_steps = 0
            last_log = now
