import asyncio
import time
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np

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

        self.train_model = PolicyValueNet(obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers).to(cfg.train_device)
        self.infer_model = PolicyValueNet(obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers).to(cfg.infer_device)

        self._sync_infer_from_train()

        self.optimizer = optim.Adam(self.train_model.parameters(), lr=cfg.lr)
        self.update_steps = 0

    def _sync_infer_from_train(self) -> None:
        """Internal sync, assumes thread safety or called from safe context."""
        self.infer_model.load_state_dict(self.train_model.state_dict())

    def sync_infer_from_train(self) -> None:
        """Sync the inference model weights from the training model."""
        self._sync_infer_from_train()

    def act(self, obs: np.ndarray, turn_std: float) -> Tuple[float, float, float, float]:
        """
        Returns turn, boost, logp, value.
        """
        o = torch.tensor(obs, dtype=torch.float32, device=self.cfg.infer_device).unsqueeze(0)
        turn_mean, boost_logit, value = self.infer_model(o)
        turn_mean = turn_mean.item()
        boost_prob = torch.sigmoid(boost_logit).item()
        value = value.item()

        # Simple stochasticity: Normal around mean for turn, Bernoulli for boost.
        turn_sample = np.random.normal(loc=turn_mean, scale=turn_std)
        turn = float(np.clip(turn_sample, -1.0, 1.0))
        boost = 1.0 if (np.random.rand() < boost_prob) else 0.0

        # Approx logp
        normal_logp = -0.5 * (((turn_sample - turn_mean) / turn_std) ** 2) - math.log(turn_std) - 0.5 * math.log(2 * math.pi)
        bern_logp = math.log(boost_prob + 1e-8) if boost > 0.5 else math.log(1.0 - boost_prob + 1e-8)
        logp = float(normal_logp + bern_logp)

        return turn, boost, logp, value

    def _ppo_update_sync(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Internal synchronous PPO update. To be called via run_in_executor/to_thread.
        """
        obs = batch["obs"]
        act_turn = batch["act_turn"]
        act_boost = batch["act_boost"]
        old_logp = batch["old_logp"]
        returns = batch["returns"]
        adv = batch["adv"]

        turn_mean, boost_logit, value = self.train_model(obs)
        boost_prob = torch.sigmoid(boost_logit)

        # Recompute approximate logp under current policy.
        turn_std = torch.tensor(self.cfg.turn_std, device=obs.device)
        normal = torch.distributions.Normal(turn_mean, turn_std)
        logp_turn = normal.log_prob(act_turn)

        bern = torch.distributions.Bernoulli(probs=boost_prob)
        logp_boost = bern.log_prob(act_boost)

        logp = logp_turn + logp_boost
        ratio = torch.exp(logp - old_logp)

        clipped = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
        policy_loss = -(torch.min(ratio * adv, clipped * adv)).mean()

        value_loss = ((returns - value) ** 2).mean()

        entropy = (normal.entropy() + bern.entropy()).mean()

        loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.optimizer.step()

        self.update_steps += 1
        # NOTE: We do NOT sync here because we are in a thread.
        # Syncing must happen on the main thread or safely.

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    async def ppo_update_async(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Async wrapper for PPO update. Runs the heavy lifting in a separate thread.
        This prevents blocking the asyncio event loop.
        """
        # Run the training step in a default executor (thread pool)
        metrics = await asyncio.to_thread(self._ppo_update_sync, batch)
        
        # After thread returns, we are back on the main event loop thread (usually).
        # We can now safely sync the inference model for actors to use.
        if self.update_steps % 1 == 0:
             self._sync_infer_from_train()
             
        return metrics


def gae(rollout: List[Transition], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    t_len = len(rollout)
    adv = np.zeros(t_len, dtype=np.float32)
    ret = np.zeros(t_len, dtype=np.float32)

    last_gae = 0.0
    last_value = 0.0

    for t in reversed(range(t_len)):
        r = rollout[t].reward
        v = rollout[t].value
        d = rollout[t].done
        next_v = last_value if t == t_len - 1 else rollout[t + 1].value
        delta = r + gamma * (1.0 - d) * next_v - v
        last_gae = delta + gamma * lam * (1.0 - d) * last_gae
        adv[t] = last_gae

    ret = adv + np.asarray([tr.value for tr in rollout], dtype=np.float32)
    return adv, ret


def collate_rollouts(rollouts: List[List[Transition]], cfg: Config, device: str) -> Dict[str, torch.Tensor]:
    obs = np.concatenate([np.stack([tr.obs for tr in ro], axis=0) for ro in rollouts], axis=0)
    act_turn = np.concatenate([np.asarray([tr.action_turn for tr in ro], dtype=np.float32) for ro in rollouts], axis=0)
    act_boost = np.concatenate([np.asarray([tr.action_boost for tr in ro], dtype=np.float32) for ro in rollouts], axis=0)
    old_logp = np.concatenate([np.asarray([tr.logp for tr in ro], dtype=np.float32) for ro in rollouts], axis=0)

    advs = []
    rets = []
    for ro in rollouts:
        a, r = gae(ro, cfg.gamma, cfg.gae_lambda)
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

    numbered = ckpt_dir / f"ckpt_h{cfg.net_hidden}_l{cfg.net_layers}_{step:08d}.pt"
    torch.save(payload, numbered)

    _rotate_ckpts(cfg.ckpt_dir, cfg.keep_last)


async def learner_loop(cfg: Config, shared_state: SharedState, experience_q: asyncio.Queue) -> None:
    last_log = time.time()
    updates = 0
    total_steps = 0

    pending_rollouts: List[List[Transition]] = []

    while True:
        _actor_id, rollout = await experience_q.get()
        pending_rollouts.append(rollout)
        total_steps += len(rollout)

        # Train when we have enough samples.
        if sum(len(r) for r in pending_rollouts) < cfg.minibatch:
            continue

        batch = collate_rollouts(pending_rollouts, cfg, device=cfg.train_device)
        pending_rollouts = []

        metrics_accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(cfg.epochs):
            # *** CRITICAL FIX: Use async update to avoid blocking ***
            metrics = await shared_state.ppo_update_async(batch)
            for k in metrics_accum:
                metrics_accum[k] += metrics[k]

        updates += 1

        if cfg.save_every_updates > 0 and (updates % cfg.save_every_updates) == 0:
            try:
                save_checkpoint(cfg, shared_state)
            except Exception as e:
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
    # pylint: disable=too-many-return-statements,too-many-branches
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
    except Exception:
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
        except Exception:
            return None

    shared_state.train_model.load_state_dict(model_sd)
    shared_state.optimizer.load_state_dict(opt_sd)
    shared_state.update_steps = int(payload.get("update_steps", 0))
    shared_state.sync_infer_from_train()
    return cand
