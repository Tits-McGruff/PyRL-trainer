"""Unit tests for utility helpers."""

# pylint: disable=import-error

import numpy as np
import pytest

from pyrl_trainer.utils import (
    compute_stride,
    default_reward,
    default_reward_components,
)

pytestmark = pytest.mark.unit


def test_compute_stride():
    """Compute stride respects max action rates."""
    assert compute_stride(60, 120) == 1
    assert compute_stride(240, 120) == 2
    assert compute_stride(240, 60) == 4
    assert compute_stride(60, 0) == 1


def test_default_reward_points_and_food():
    """Reward increases with points and food proximity."""
    idx = {
        "points_delta_norm": 0,
        "nearest_food_dist_norm": 1,
    }
    prev = np.array([0.0, 0.4], dtype=np.float32)
    curr = np.array([0.1, 0.5], dtype=np.float32)

    reward = default_reward(prev, curr, idx)

    expected = 1.0
    expected += 0.05
    expected += 0.0001

    assert reward == pytest.approx(expected, abs=1e-6)


def test_reward_components_sum_to_default_reward():
    """Diagnostic reward components preserve the exact training reward."""
    idx = {
        "points_delta_norm": 0,
        "nearest_food_dist_norm": 1,
        "wall_dist_norm": 2,
        "hazard_0": 3,
        "hazard_1": 4,
        "hazard_2": 5,
    }
    prev = np.array([0.0, 0.0, -0.6, 1.0, 1.0, 1.0], dtype=np.float32)
    curr = np.array([0.05, 0.1, -0.8, -1.0, -1.0, -1.0], dtype=np.float32)

    parts = default_reward_components(prev, curr, idx)

    assert sum(parts.values()) == pytest.approx(
        default_reward(prev, curr, idx), abs=1e-7
    )
    assert parts["growth"] > 0.0
    assert parts["food_approach"] > 0.0
    assert parts["wall"] < 0.0
    assert parts["hazard"] < 0.0
