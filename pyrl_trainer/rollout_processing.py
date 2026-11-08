"""Pure rollout segmentation, generalized advantages and PPO batch collation."""

from typing import Dict, List, Tuple

import numpy as np
import torch

from .agent import Transition
from .config import Config


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
        transition = rollout[t]
        reward = transition.reward
        value = transition.value
        done = transition.done
        next_value = last_value if t == t_len - 1 else rollout[t + 1].value
        if transition.bootstrap_override is not None:
            next_value = transition.bootstrap_override
            last_gae = 0.0
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
