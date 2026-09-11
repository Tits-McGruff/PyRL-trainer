"""Utility helpers for the trainer."""

import math
import os
from typing import Dict, List, Optional

import numpy as np


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
) -> Dict[str, float]:
    """Return the shaped reward split into stable diagnostic components."""
    parts = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    if prev_obs is None:
        return parts

    if "points_delta_norm" in idx:
        parts["growth"] = 10.0 * float(obs[idx["points_delta_norm"]])

    if "nearest_food_dist_norm" in idx:
        f_idx = idx["nearest_food_dist_norm"]
        curr_dist = float(obs[f_idx])
        prev_dist = float(prev_obs[f_idx])
        delta = clamp(curr_dist - prev_dist, -0.1, 0.1)
        parts["food_approach"] = 0.5 * delta

    parts["survival"] = 0.0001

    if "wall_dist_norm" in idx:
        wall = float(obs[idx["wall_dist_norm"]])
        if wall < -0.5:
            parts["wall"] = -0.05 * ((-wall) ** 2)

    hazard_labels = [key for key in idx if key.startswith("hazard_")]
    if hazard_labels:
        n_bins = len(hazard_labels)
        center_bin = n_bins // 2
        start_bin = max(0, center_bin - 1)
        end_bin = min(n_bins, center_bin + 2)
        front_hazards = hazard_labels[start_bin:end_bin]
        avg_clearance = np.mean([obs[idx[label]] for label in front_hazards])
        if avg_clearance < -0.5:
            parts["hazard"] = -0.1

    return parts


def default_reward(
    prev_obs: Optional[np.ndarray],
    obs: np.ndarray,
    idx: Dict[str, int],
) -> float:
    """Return the total shaped reward."""
    return float(sum(default_reward_components(prev_obs, obs, idx).values()))


def ensure_dir(p: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(p, exist_ok=True)
