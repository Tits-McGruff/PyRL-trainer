"""Learner loop and PPO update utilities."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import optim

from .config import Config
from .network import PolicyValueNet
from .agent import Transition  # Circular import if not careful, but Transition is data
from .utils import REWARD_COMPONENT_KEYS, ensure_dir


class SharedState:  # pylint: disable=too-many-instance-attributes
    """
    Shared policy/value network weights.
    Learner updates the train_model; actors use an inference copy to avoid training contention.
    """
    def __init__(self, obs_dim: int, cfg: Config):
        self.cfg = cfg
        self.obs_dim = obs_dim

        self.train_model = PolicyValueNet(
            obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers
        ).to(cfg.train_device)
        self.infer_model = PolicyValueNet(
            obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers
        ).to(cfg.infer_device)

        self._sync_infer_from_train()

        self.optimizer = optim.Adam(self.train_model.parameters(), lr=cfg.lr)
        self.update_steps = 0

        self._reward_totals = {
            **{key: 0.0 for key in REWARD_COMPONENT_KEYS},
            "death": 0.0,
        }
        self._episode_count = 0
        self._episode_return_sum = 0.0
        self._episode_lifetime_sum = 0.0

    def _sync_infer_from_train(self) -> None:
        """Copy training weights to the actor inference model."""
        target_device = self.cfg.infer_device
        state_dict = {
            key: value.to(target_device)
            for key, value in self.train_model.state_dict().items()
        }
        self.infer_model.load_state_dict(state_dict)

    def sync_infer_from_train(self) -> None:
        """Sync the inference model weights from the training model."""
        self._sync_infer_from_train()

    @staticmethod
    def _joint_logp_entropy(
        turn_mean: torch.Tensor,
        boost_logit: torch.Tensor,
        turn_latent: torch.Tensor,
        action_boost: torch.Tensor,
        turn_std: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        entropy_proxy = base_dist.entropy() + boost_dist.entropy()
        return action_turn, logp_turn + logp_boost, entropy_proxy

    def act(
        self, obs: np.ndarray, turn_std: float
    ) -> Tuple[float, float, float, float, float]:
        """Sample turn/boost and return action, log-prob, value, and turn latent."""
        o = torch.tensor(
            obs, dtype=torch.float32, device=self.cfg.infer_device
        ).unsqueeze(0)
        turn_mean, boost_logit, value = self.infer_model(o)

        std = torch.as_tensor(
            turn_std, dtype=turn_mean.dtype, device=turn_mean.device
        )
        turn_latent = torch.distributions.Normal(turn_mean, std).sample()
        boost_dist = torch.distributions.Bernoulli(logits=boost_logit)
        action_boost = boost_dist.sample()

        action_turn, logp, _entropy = self._joint_logp_entropy(
            turn_mean, boost_logit, turn_latent, action_boost, turn_std
        )

        return (
            float(action_turn.item()),
            float(action_boost.item()),
            float(logp.item()),
            float(value.item()),
            float(turn_latent.item()),
        )

    def record_reward_components(self, components: Dict[str, float]) -> None:
        """Accumulate bounded reward-component totals for the next learner log."""
        for key, value in components.items():
            if key in self._reward_totals:
                self._reward_totals[key] += float(value)

    def record_episode(self, episode_return: float, lifetime_ticks: int) -> None:
        """Accumulate one completed episode for interval diagnostics."""
        self._episode_count += 1
        self._episode_return_sum += float(episode_return)
        self._episode_lifetime_sum += float(max(0, lifetime_ticks))

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

        for key in self._reward_totals:
            self._reward_totals[key] = 0.0
        self._episode_count = 0
        self._episode_return_sum = 0.0
        self._episode_lifetime_sum = 0.0
        return stats

    def _ppo_minibatch_sync(  # pylint: disable=too-many-locals
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run one optimizer step for one shuffled PPO minibatch."""
        obs = batch["obs"]
        act_turn_latent = batch["act_turn_latent"]
        act_boost = batch["act_boost"]
        old_logp = batch["old_logp"]
        returns = batch["returns"]
        adv = batch["adv"]

        turn_mean, boost_logit, value = self.train_model(obs)
        _action_turn, logp, entropy_values = self._joint_logp_entropy(
            turn_mean,
            boost_logit,
            act_turn_latent,
            act_boost,
            self.cfg.turn_std,
        )

        log_ratio = logp - old_logp
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip
        )
        policy_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()
        value_loss = ((returns - value) ** 2).mean()
        entropy = entropy_values.mean()

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

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.optimizer.step()

        self.update_steps += 1
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
        }

    def _ppo_update_batch_sync(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run shuffled PPO minibatches for all configured epochs."""
        sample_count = int(batch["obs"].shape[0])
        if sample_count <= 0:
            raise ValueError("PPO batch must contain at least one sample")

        minibatch_size = max(1, int(self.cfg.minibatch))
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        weighted_samples = 0
        optimizer_steps = 0

        for _ in range(max(1, int(self.cfg.epochs))):
            order = torch.randperm(sample_count, device=batch["obs"].device)
            for start in range(0, sample_count, minibatch_size):
                index = order[start:start + minibatch_size]
                mini = {
                    key: value.index_select(0, index)
                    for key, value in batch.items()
                }
                metrics = self._ppo_minibatch_sync(mini)
                weight = int(index.numel())
                weighted_samples += weight
                optimizer_steps += 1
                for key in totals:
                    totals[key] += metrics[key] * weight

        with torch.no_grad():
            _turn_mean, _boost_logit, values = self.train_model(batch["obs"])
            returns = batch["returns"]
            return_variance = torch.var(returns, unbiased=False)
            if float(return_variance.item()) > 1e-8:
                explained_variance = 1.0 - (
                    torch.var(returns - values, unbiased=False) / return_variance
                )
                explained = float(explained_variance.item())
            else:
                explained = 0.0

        denom = float(max(1, weighted_samples))
        metrics = {key: value / denom for key, value in totals.items()}
        metrics["explained_variance"] = explained
        metrics["optimizer_steps"] = float(optimizer_steps)
        metrics["samples"] = float(sample_count)
        return metrics

    async def ppo_update_async(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run a complete PPO batch off-loop, then publish one new actor policy."""
        metrics = await asyncio.to_thread(self._ppo_update_batch_sync, batch)
        self._sync_infer_from_train()
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


def collate_rollouts(  # pylint: disable=too-many-locals
    rollouts_and_boots: List[Tuple[List[Transition], float]],
    cfg: Config,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Stack complete rollout segments and normalize advantages once per PPO batch."""
    rollouts = [rollout for rollout, _bootstrap in rollouts_and_boots]

    obs = np.concatenate(
        [np.stack([transition.obs for transition in rollout], axis=0)
         for rollout in rollouts],
        axis=0,
    )
    act_turn = np.concatenate(
        [np.asarray(
            [transition.action_turn for transition in rollout],
            dtype=np.float32,
        ) for rollout in rollouts],
        axis=0,
    )
    act_turn_latent = np.concatenate(
        [np.asarray(
            [transition.turn_latent for transition in rollout],
            dtype=np.float32,
        ) for rollout in rollouts],
        axis=0,
    )
    act_boost = np.concatenate(
        [np.asarray(
            [transition.action_boost for transition in rollout],
            dtype=np.float32,
        ) for rollout in rollouts],
        axis=0,
    )
    old_logp = np.concatenate(
        [np.asarray(
            [transition.logp for transition in rollout],
            dtype=np.float32,
        ) for rollout in rollouts],
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


def _list_ckpts(ckpt_dir: str) -> list[Path]:
    directory = Path(ckpt_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob("ckpt_*.pt"))


def _rotate_ckpts(ckpt_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = _list_ckpts(ckpt_dir)
    if len(ckpts) <= keep_last:
        return
    for path in ckpts[: max(0, len(ckpts) - keep_last)]:
        try:
            path.unlink()
        except OSError:
            pass


def save_checkpoint(cfg: Config, shared_state: SharedState) -> None:
    """Persist a checkpoint and rotate older files."""
    ensure_dir(cfg.ckpt_dir)
    step = int(shared_state.update_steps)

    payload = {
        "update_steps": step,
        "obs_dim": int(shared_state.obs_dim),
        "net_hidden": int(cfg.net_hidden),
        "net_layers": int(cfg.net_layers),
        "model": shared_state.train_model.state_dict(),
        "optimizer": shared_state.optimizer.state_dict(),
    }

    ckpt_dir = Path(cfg.ckpt_dir)
    latest_path = ckpt_dir / "latest.pt"
    latest_arch_path = ckpt_dir / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"

    torch.save(payload, latest_path)
    torch.save(payload, latest_arch_path)

    numbered = (
        ckpt_dir / f"ckpt_h{cfg.net_hidden}_l{cfg.net_layers}_{step:08d}.pt"
    )
    torch.save(payload, numbered)

    _rotate_ckpts(cfg.ckpt_dir, cfg.keep_last)


async def learner_loop(  # pylint: disable=too-many-locals
    cfg: Config, shared_state: SharedState, experience_q: asyncio.Queue
) -> None:
    """Consume rollout segments and train on shuffled PPO minibatches."""
    last_log = time.time()
    updates = 0
    total_steps = 0
    pending_data: List[Tuple[List[Transition], float]] = []

    while True:
        _actor_id, rollout, bootstrap = await experience_q.get()
        if not rollout:
            continue
        pending_data.append((rollout, bootstrap))
        total_steps += len(rollout)

        current_samples = sum(len(items) for items, _boot in pending_data)
        if current_samples < cfg.batch_size:
            continue

        batch = collate_rollouts(pending_data, cfg, device=cfg.train_device)
        pending_data = []
        metrics = await shared_state.ppo_update_async(batch)
        updates += 1

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
                f"food={runtime['reward_food_approach']:.3f},"
                f"survival={runtime['reward_survival']:.3f},"
                f"wall={runtime['reward_wall']:.3f},"
                f"hazard={runtime['reward_hazard']:.3f},"
                f"death={runtime['reward_death']:.3f}"
            )
            print(
                f"[learner] updates={updates}, steps={total_steps}, "
                f"samples={int(metrics['samples'])}, "
                f"opt_steps={int(metrics['optimizer_steps'])}, "
                f"loss={metrics['loss']:.4f}, "
                f"policy={metrics['policy_loss']:.4f}, "
                f"value={metrics['value_loss']:.4f}, "
                f"entropy={metrics['entropy']:.4f}, "
                f"kl={metrics['approx_kl']:.5f}, "
                f"clip={metrics['clip_fraction']:.3f}, "
                f"ev={metrics['explained_variance']:.3f}, "
                f"episodes={int(runtime['episodes'])}, "
                f"ep_return={runtime['episode_return_mean']:.3f}, "
                f"lifetime_ticks={runtime['episode_lifetime_mean']:.1f}, "
                f"reward[{rewards}]"
            )
            last_log = now


def load_checkpoint_if_present(cfg: Config, shared_state: SharedState) -> Optional[Path]:
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-locals
    """
    Load a checkpoint if present and compatible with the current model shape.
    Return the Path that was loaded, or None if no compatible checkpoint exists.
    """
    ckpt_dir = Path(cfg.ckpt_dir)
    latest_arch = ckpt_dir / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"
    latest = ckpt_dir / "latest.pt"

    candidate = None
    if latest_arch.exists():
        candidate = latest_arch
    elif latest.exists():
        candidate = latest
    else:
        return None

    try:
        payload = torch.load(candidate, map_location=cfg.train_device)
    except (OSError, RuntimeError):
        return None

    if int(payload.get("obs_dim", -1)) != int(shared_state.obs_dim):
        return None

    ckpt_hidden = payload.get("net_hidden", None)
    ckpt_layers = payload.get("net_layers", None)
    if ckpt_hidden is not None and int(ckpt_hidden) != int(cfg.net_hidden):
        return None
    if ckpt_layers is not None and int(ckpt_layers) != int(cfg.net_layers):
        return None

    model_sd = payload.get("model", None)
    opt_sd = payload.get("optimizer", None)
    if not isinstance(model_sd, dict) or not isinstance(opt_sd, dict):
        return None

    current_sd = shared_state.train_model.state_dict()
    if set(model_sd.keys()) != set(current_sd.keys()):
        return None

    for key, value in model_sd.items():
        if key not in current_sd:
            return None
        try:
            if tuple(value.shape) != tuple(current_sd[key].shape):
                return None
        except (AttributeError, TypeError):
            return None

    shared_state.train_model.load_state_dict(model_sd)
    shared_state.optimizer.load_state_dict(opt_sd)
    shared_state.update_steps = int(payload.get("update_steps", 0))
    shared_state.sync_infer_from_train()
    return candidate
