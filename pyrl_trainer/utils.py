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
    Shaped reward function:
    - Growth: points_delta_norm (eating +, boosting -)
    - Survival: constant bonus
    - Safety: penalties for wall proximity and frontal hazards
    """
    if prev_obs is None:
        return 0.0

    r = 0.0

    # 1. Growth (Points change: eating +, boosting -)
    # Scale up points_delta massively to prioritize eating over just surviving.
    if "points_delta_norm" in idx:
        r += 10.0 * float(obs[idx["points_delta_norm"]])

    # 1b. Food Approach (Dense Shaping)
    # Give a small reward for moving towards food (value decreasing towards -1).
    # This helps break the "circling" local optimum by providing a gradient.
    if "nearest_food_dist_norm" in idx:
        f_idx = idx["nearest_food_dist_norm"]
        # Assuming -1 is close and 1 is far (like wall_dist).
        # We want to encourage decreasing value (getting closer).
        curr_dist = float(obs[f_idx])
        prev_dist = float(prev_obs[f_idx])
        # Delta: Positive if we got closer (prev > curr)
        delta = prev_dist - curr_dist
        # Clamp delta to avoid massive spikes when target switches
        delta = clamp(delta, -0.1, 0.1)
        r += 0.5 * delta

    # 2. Survival
    # Reduced significantly to prevent "safe circling" local optima.
    r += 0.0001

    # 3. Wall Safety
    # wall_dist_norm: 1.0 (center) -> -1.0 (wall)
    if "wall_dist_norm" in idx:
        wall = float(obs[idx["wall_dist_norm"]])
        if wall < -0.5:
             # Penalize exponentially as we get closer to wall
             # at -0.5 -> penalty 0
             # at -1.0 -> penalty -0.05 * (1.0)^2 = -0.05
             r -= 0.05 * ((-wall) ** 2)

    # 4. Frontal Collision Safety
    # Front bins are usually in the middle of the array if sorted by angle.
    # API: Bin 0 is back (-pi), Bin N/2 is front (0).
    hazard_labels = [k for k in idx.keys() if k.startswith("hazard_")]
    if hazard_labels:
        n_bins = len(hazard_labels)
        center_bin = n_bins // 2
        # Look at 3 center bins (front)
        # Bins are indexed 0..N-1.
        # e.g., 16 bins. center=8. indices 7, 8, 9.
        start_bin = max(0, center_bin - 1)
        end_bin = min(n_bins, center_bin + 2)
        
        front_hazards = [hazard_labels[i] for i in range(start_bin, end_bin)]
        
        # Hazard bin: -1 (blocked/close) to 1 (clear).
        avg_clearance = np.mean([obs[idx[h]] for h in front_hazards])
        
        # If avg_clearance < -0.5 (very close), penalize.
        if avg_clearance < -0.5:
            r -= 0.1

    return float(r)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
