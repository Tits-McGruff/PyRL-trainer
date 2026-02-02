"""Learner loop and PPO update utilities."""

import asyncio
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import optim

from .config import Config
from .network import PolicyValueNet
from .agent import Transition  # Circular import if not careful, but Transition is data
from .utils import ensure_dir


class SharedState:  # pylint: disable=too-many-instance-attributes
    """
    Shared policy/value network weights.
    Learner updates the train_model; actors use an inference copy to avoid training contention.
    Includes async update method to prevent blocking the event loop.
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

    def _sync_infer_from_train(self) -> None:
        """Internal sync, assumes thread safety or called from safe context."""
        # Explicitly move tensors to the inference device to avoid runtime errors
        target_device = self.cfg.infer_device
        state_dict = {
            k: v.to(target_device) for k, v in self.train_model.state_dict().items()
        }
        self.infer_model.load_state_dict(state_dict)

    def sync_infer_from_train(self) -> None:
        """Sync the inference model weights from the training model."""
        self._sync_infer_from_train()

    def act(  # pylint: disable=too-many-locals
        self, obs: np.ndarray, turn_std: float
    ) -> Tuple[float, float, float, float]:
        """Return turn, boost, logp, and value for a single observation."""
        o = torch.tensor(obs, dtype=torch.float32, device=self.cfg.infer_device).unsqueeze(0)
        turn_mean, boost_logit, value = self.infer_model(o)
        turn_mean = turn_mean.item()
        boost_prob = torch.sigmoid(boost_logit).item()
        value = value.item()

        # TanhNormal: Sample from N(mean, std), then apply tanh.
        # Log prob needs Jacobian correction.
        u = np.random.normal(loc=turn_mean, scale=turn_std)
        turn = float(math.tanh(u))

        # Boost is Bernoulli
        boost = 1.0 if (np.random.rand() < boost_prob) else 0.0

        # Logp
        # log p(a) = log p(u) - log(1 - tanh^2(u))
        normal_logp = (
            -0.5 * (((u - turn_mean) / turn_std) ** 2)
            - math.log(turn_std)
            - 0.5 * math.log(2 * math.pi)
        )
        correction = math.log(1.0 - turn**2 + 1e-6)
        logp_turn = normal_logp - correction

        bern_logp = (
            math.log(boost_prob + 1e-8)
            if boost > 0.5
            else math.log(1.0 - boost_prob + 1e-8)
        )
        logp = float(logp_turn + bern_logp)

        return turn, boost, logp, value

    def _ppo_update_sync(  # pylint: disable=too-many-locals
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Run a synchronous PPO update step."""
        obs = batch["obs"]
        act_turn = batch["act_turn"]
        act_boost = batch["act_boost"]
        old_logp = batch["old_logp"]
        returns = batch["returns"]
        adv = batch["adv"]

        turn_mean, boost_logit, value = self.train_model(obs)
        boost_prob = torch.sigmoid(boost_logit)

        # Recompute logp of the TAKEN action under CURRRENT policy.
        # Note: act_turn is already tanh(u). We need to recover u or use a TanhNormal dist.
        # Inverse tanh: u = atanh(act_turn)
        # However, act_turn might be slightly clipped/noisy.
        # TanhNormal in PyTorch:
        turn_std = torch.tensor(self.cfg.turn_std, device=obs.device)

        # 1. Base distribution N(mean, std)
        base_dist = torch.distributions.Normal(turn_mean, turn_std)

        # 2. Transform: Tanh
        dist = torch.distributions.TransformedDistribution(
            base_dist, torch.distributions.transforms.TanhTransform(cache_size=1)
        )

        # 3. Log prob of the action
        # Epsilon prevents nan at boundaries
        act_turn_clamped = torch.clamp(act_turn, -0.999999, 0.999999)
        logp_turn = dist.log_prob(act_turn_clamped)

        bern = torch.distributions.Bernoulli(probs=boost_prob)
        logp_boost = bern.log_prob(act_boost)

        logp = logp_turn + logp_boost
        ratio = torch.exp(logp - old_logp)

        clipped = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
        policy_loss = -(torch.min(ratio * adv, clipped * adv)).mean()

        value_loss = ((returns - value) ** 2).mean()

        # Entropy of TanhNormal is tricky. It has no closed form.
        # We can approximate it by sampling, or just use the base Normal entropy (ignoring squash).
        # Standard practice: use base distribution entropy if strictly varying std.
        # OR just ignore entropy for Tanh part?
        # A common simple approximation is just base_dist.entropy(), knowing it's not exact.
        # But we want to encourage exploration properly.
        # Actually g.entropy() is not implemented for TransformedDistribution.
        # We'll use base_dist.entropy() - expected_log_det_jacobian? No, that's complex.
        # Let's use the entropy of the base Normal as a proxy for exploration width.
        entropy = (base_dist.entropy() + bern.entropy()).mean()

        loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

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
        }

    async def ppo_update_async(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Async wrapper for PPO update in a background thread."""
        # Run the training step in a default executor (thread pool)
        metrics = await asyncio.to_thread(self._ppo_update_sync, batch)

        # After thread returns, we are back on the main event loop thread (usually).
        # We can now safely sync the inference model for actors to use.
        if self.update_steps % 1 == 0:
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
    ret = np.zeros(t_len, dtype=np.float32)

    last_gae = 0.0
    last_value = bootstrap_value

    for t in reversed(range(t_len)):
        r = rollout[t].reward
        v = rollout[t].value
        d = rollout[t].done
        next_v = last_value if t == t_len - 1 else rollout[t + 1].value
        # If done=1, next_v is masked out by (1-d) anyway.
        # If done=0 (truncation), we use bootstrap_value as next_v for the last step.
        delta = r + gamma * (1.0 - d) * next_v - v
        last_gae = delta + gamma * lam * (1.0 - d) * last_gae
        adv[t] = last_gae

    ret = adv + np.asarray([tr.value for tr in rollout], dtype=np.float32)
    return adv, ret


def collate_rollouts(  # pylint: disable=too-many-locals
    rollouts_and_boots: List[Tuple[List[Transition], float]],
    cfg: Config,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Stack rollouts and compute normalized advantages."""
    # rollouts_and_boots is list of (rollout, bootstrap)
    rollouts = [r for r, _ in rollouts_and_boots]

    obs = np.concatenate(
        [np.stack([tr.obs for tr in ro], axis=0) for ro in rollouts], axis=0
    )
    act_turn = np.concatenate(
        [np.asarray([tr.action_turn for tr in ro], dtype=np.float32) for ro in rollouts],
        axis=0,
    )
    act_boost = np.concatenate(
        [np.asarray([tr.action_boost for tr in ro], dtype=np.float32) for ro in rollouts],
        axis=0,
    )
    old_logp = np.concatenate(
        [np.asarray([tr.logp for tr in ro], dtype=np.float32) for ro in rollouts],
        axis=0,
    )

    advs = []
    rets = []
    for ro, boot in rollouts_and_boots:
        a, r = gae(ro, cfg.gamma, cfg.gae_lambda, bootstrap_value=boot)
        advs.append(a)
        rets.append(r)

    adv = np.concatenate(advs, axis=0)
    ret = np.concatenate(rets, axis=0)

    # Normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    batch = {
        "obs": torch.tensor(obs, dtype=torch.float32, device=device),
        "act_turn": torch.tensor(act_turn, dtype=torch.float32, device=device),
        "act_boost": torch.tensor(act_boost, dtype=torch.float32, device=device),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32, device=device),
        "adv": torch.tensor(adv, dtype=torch.float32, device=device),
        "returns": torch.tensor(ret, dtype=torch.float32, device=device),
    }
    return batch


def _list_ckpts(ckpt_dir: str) -> list[Path]:
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    return sorted(d.glob("ckpt_*.pt"))


def _rotate_ckpts(ckpt_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = _list_ckpts(ckpt_dir)
    if len(ckpts) <= keep_last:
        return
    for p in ckpts[: max(0, len(ckpts) - keep_last)]:
        try:
            p.unlink()
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
    """Consume rollouts and update the policy."""
    last_log = time.time()
    updates = 0
    total_steps = 0

    pending_data: List[Tuple[List[Transition], float]] = []

    while True:
        # Unpack tuple including bootstrap value
        _actor_id, rollout, bootstrap = await experience_q.get()
        pending_data.append((rollout, bootstrap))
        total_steps += len(rollout)

        # Train when we have enough samples.
        current_samples = sum(len(r) for r, _ in pending_data)
        if current_samples < cfg.minibatch:
            continue

        batch = collate_rollouts(pending_data, cfg, device=cfg.train_device)
        pending_data = []

        metrics_accum = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        for _ in range(cfg.epochs):
            # *** CRITICAL FIX: Use async update to avoid blocking ***
            metrics = await shared_state.ppo_update_async(batch)
            for k in metrics_accum:
                metrics_accum[k] += metrics[k]

        updates += 1

        if cfg.save_every_updates > 0 and (updates % cfg.save_every_updates) == 0:
            try:
                save_checkpoint(cfg, shared_state)
            except (OSError, RuntimeError) as e:
                print(f"[learner] checkpoint save failed: {type(e).__name__}: {e}")

        now = time.time()
        if now - last_log >= cfg.log_every_seconds:
            denom = float(cfg.epochs)
            msg = (
                f"[learner] updates={updates}, steps={total_steps}, "
                f"loss={metrics_accum['loss']/denom:.4f}, "
                f"policy={metrics_accum['policy_loss']/denom:.4f}, "
                f"value={metrics_accum['value_loss']/denom:.4f}, "
                f"entropy={metrics_accum['entropy']/denom:.4f}"
            )
            print(msg)
            last_log = now


def load_checkpoint_if_present(cfg: Config, shared_state: SharedState) -> Optional[Path]:
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-locals
    """
    Loads a checkpoint if present and compatible with the current model shape.
    Returns the Path that was loaded, or None if no compatible checkpoint exists.
    """
    ckpt_dir = Path(cfg.ckpt_dir)
    latest_arch = ckpt_dir / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"
    latest = ckpt_dir / "latest.pt"

    cand = None
    if latest_arch.exists():
        cand = latest_arch
    elif latest.exists():
        cand = latest
    else:
        return None

    try:
        payload = torch.load(cand, map_location=cfg.train_device)
    except (OSError, RuntimeError):
        return None

    # Quick compatibility checks to avoid noisy state_dict tracebacks.
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

    # If keys differ, it's almost certainly an architecture mismatch.
    if set(model_sd.keys()) != set(current_sd.keys()):
        return None

    # If any tensor shape differs, skip.
    for k, v in model_sd.items():
        if k not in current_sd:
            return None
        try:
            if tuple(v.shape) != tuple(current_sd[k].shape):
                return None
        except (AttributeError, TypeError):
            return None

    shared_state.train_model.load_state_dict(model_sd)
    shared_state.optimizer.load_state_dict(opt_sd)
    shared_state.update_steps = int(payload.get("update_steps", 0))
    shared_state.sync_infer_from_train()
    return cand
