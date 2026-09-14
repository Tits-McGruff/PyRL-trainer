"""Unit tests for utility helpers."""

# pylint: disable=import-error

import numpy as np
import pytest

from pyrl_trainer.config import Config
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


def test_default_reward_matches_legacy_vector():
    """Default reward settings reproduce the pre-remediation reward exactly."""
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
    cfg = Config()

    parts = default_reward_components(prev, curr, idx, cfg)

    assert parts["growth"] == pytest.approx(0.5)
    assert parts["food_approach"] == pytest.approx(0.05)
    assert parts["survival"] == pytest.approx(0.0001)
    assert parts["wall"] == pytest.approx(-0.032)
    assert parts["hazard"] == pytest.approx(-0.1)
    assert sum(parts.values()) == pytest.approx(0.4181, abs=1e-7)
    assert default_reward(prev, curr, idx, cfg) == pytest.approx(0.4181, abs=1e-7)


def test_food_closeness_direction_and_clamp():
    """Increasing closeness rewards approach and configured clipping bounds it."""
    idx = {"nearest_food_dist_norm": 0}
    cfg = Config(reward_food_approach_scale=0.5, reward_food_delta_clip=0.1)

    closer = default_reward_components(
        np.array([-0.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        idx,
        cfg,
    )
    farther = default_reward_components(
        np.array([0.4], dtype=np.float32),
        np.array([0.3], dtype=np.float32),
        idx,
        cfg,
    )

    assert closer["food_approach"] == pytest.approx(0.05)
    assert farther["food_approach"] == pytest.approx(-0.05)


def test_reward_threshold_boundaries_remain_strict():
    """Wall and hazard penalties activate only below their configured thresholds."""
    cfg = Config()
    idx = {
        "wall_dist_norm": 0,
        "hazard_0": 1,
        "hazard_1": 2,
        "hazard_2": 3,
    }
    previous = np.zeros(4, dtype=np.float32)
    current = np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32)

    parts = default_reward_components(previous, current, idx, cfg)

    assert parts["wall"] == 0.0
    assert parts["hazard"] == 0.0


def test_first_observation_returns_zero_components():
    """No shaping is emitted before a previous observation exists."""
    cfg = Config()
    idx = {"points_delta_norm": 0}
    parts = default_reward_components(
        None, np.array([1.0], dtype=np.float32), idx, cfg
    )
    assert all(value == 0.0 for value in parts.values())
