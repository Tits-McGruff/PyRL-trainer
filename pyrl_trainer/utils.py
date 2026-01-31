import math
import os
import numpy as np
from typing import Dict, List, Optional


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def build_index(order: List[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(order)}


def compute_stride(tick_rate: int, max_actions_per_second: int) -> int:
    """
    Ensures we never exceed maxActionsPerSecond.
    If tick_rate <= max APS, stride is 1, send each tick.
    If tick_rate > max APS, stride > 1, send every stride ticks, last action held by server.
    """
    if max_actions_per_second <= 0:
        return 1
    return max(1, math.ceil(tick_rate / max_actions_per_second))


def default_reward(prev_obs: Optional[np.ndarray],
                   obs: np.ndarray,
                   idx: Dict[str, int]) -> float:
    """
    Placeholder reward, purely to make the training loop runnable.
    Replace with your real reward shaping.
    """
    alive_bonus = 0.001
    if prev_obs is None:
        return alive_bonus

    r = alive_bonus

    if "size_norm" in idx:
        r += 0.05 * float(obs[idx["size_norm"]] - prev_obs[idx["size_norm"]])

    if "points_pct" in idx:
        r += 0.02 * float(obs[idx["points_pct"]] - prev_obs[idx["points_pct"]])

    # Mild safety shaping
    hazard_labels = [k for k in idx.keys() if k.startswith("hazard_")]
    wall_labels = [k for k in idx.keys() if k.startswith("wall_")]
    if hazard_labels:
        h = np.mean([obs[idx[k]] for k in hazard_labels])
        # hazard is clearance (-1 close, 1 far). Higher is better.
        r += 0.01 * float(h)
    if wall_labels:
        w = np.mean([obs[idx[k]] for k in wall_labels])
        r += 0.01 * float(w)

    return float(r)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
