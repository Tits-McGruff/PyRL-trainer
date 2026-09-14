"""Utility helpers for the trainer."""

import math
import os
from typing import Dict, List, Optional

import numpy as np

from .config import Config


REWARD_COMPONENT_KEYS = ("growth", "food_approach", "survival", "wall", "hazard")


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a float between lo and hi."""
    return float(max(lo, min(hi, x)))


def build_index(order: List[str]) -> Dict[str, int]:
    """Map sensor labels to index positions."""
    return {label: i for i, label in enumerate(order)}


def compute_stride(tick_rate: int, max_actions_per_second: int) -> int:
    """Compute the action stride to respect maxActionsPerSecond."""
    if max_actions_per_second <= 0:
        return 1
    return max(1, math.ceil(tick_rate / max_actions_per_second))


def default_reward_components(
    prev_obs: Optional[np.ndarray],
    obs: np.ndarray,
    idx: Dict[str, int],
    cfg: Config,
) -> Dict[str, float]:
    """Return the shaped reward split into stable diagnostic components."""
    parts = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    if prev_obs is None:
        return parts

    if "points_delta_norm" in idx:
        parts["growth"] = cfg.reward_growth_scale * float(
            obs[idx["points_delta_norm"]]
        )

    if "nearest_food_dist_norm" in idx:
        food_idx = idx["nearest_food_dist_norm"]
        # Protocol 2 names this as a distance, but its value increases with
        # closeness: roughly -1 at the far/no-food limit and +1 at zero distance.
        curr_closeness = float(obs[food_idx])
        prev_closeness = float(prev_obs[food_idx])
        delta = clamp(
            curr_closeness - prev_closeness,
            -cfg.reward_food_delta_clip,
            cfg.reward_food_delta_clip,
        )
        parts["food_approach"] = cfg.reward_food_approach_scale * delta

    parts["survival"] = cfg.reward_survival_bonus

    if "wall_dist_norm" in idx:
        wall = float(obs[idx["wall_dist_norm"]])
        if wall < cfg.reward_wall_threshold:
            parts["wall"] = -cfg.reward_wall_penalty_scale * ((-wall) ** 2)

    hazard_labels = [key for key in idx if key.startswith("hazard_")]
    if hazard_labels:
        n_bins = len(hazard_labels)
        center_bin = n_bins // 2
        start_bin = max(0, center_bin - 1)
        end_bin = min(n_bins, center_bin + 2)
        front_hazards = hazard_labels[start_bin:end_bin]
        avg_clearance = np.mean([obs[idx[label]] for label in front_hazards])
        if avg_clearance < cfg.reward_hazard_threshold:
            parts["hazard"] = -cfg.reward_hazard_penalty_magnitude

    return parts


def default_reward(
    prev_obs: Optional[np.ndarray],
    obs: np.ndarray,
    idx: Dict[str, int],
    cfg: Config,
) -> float:
    """Return the total shaped reward."""
    return float(sum(default_reward_components(prev_obs, obs, idx, cfg).values()))


def ensure_dir(p: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(p, exist_ok=True)
