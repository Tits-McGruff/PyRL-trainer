import numpy as np
import pytest

from pyrl_trainer.utils import compute_stride, default_reward

pytestmark = pytest.mark.unit


def test_compute_stride():
    assert compute_stride(60, 120) == 1
    assert compute_stride(240, 120) == 2
    assert compute_stride(240, 60) == 4
    assert compute_stride(60, 0) == 1


def test_default_reward_points_and_food():
    idx = {
        "points_delta_norm": 0,
        "nearest_food_dist_norm": 1,
    }
    prev = np.array([0.0, 0.5], dtype=np.float32)
    curr = np.array([0.1, 0.4], dtype=np.float32)

    r = default_reward(prev, curr, idx)

    expected = 1.0  # 10.0 * points_delta_norm
    expected += 0.05  # 0.5 * clamp(0.1)
    expected += 0.0001  # survival bonus

    assert abs(r - expected) < 1e-6
