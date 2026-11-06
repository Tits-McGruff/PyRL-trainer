"""Raw event warnings remain independent of objective weights and log drains."""

from copy import deepcopy
from importlib.resources import files
import json
import logging

import pytest

from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState, format_reward_exposure
from pyrl_trainer.reward_contract import parse_reward

pytestmark = pytest.mark.unit


def interval(*, alive_seconds=0.01):
    """Valid known-shape input after stream admission, without task events."""
    payload = json.loads(files("slither_contracts").joinpath(
        "external-reward-v1-fixture.json").read_text(encoding="utf-8"))[0]
    payload.update(transitionKind="live", reason=None)
    payload["deltas"] = {name: 0 for name in payload["deltas"]}
    payload["deltas"].update(elapsedSeconds=alive_seconds,
                             aliveSecondsDelta=alive_seconds)
    return payload


def state():
    """Use the actual diagnostics owner with a small ordinary policy."""
    return SharedState(1, Config(net_hidden=4, net_layers=1,
                                  train_device="cpu", infer_device="cpu"))


def test_absence_warning_requires_both_exposure_thresholds_and_survives_log_drain(caplog):
    """Many short intervals cannot trigger the warning before sufficient live time."""
    shared = state()
    reward = interval(alive_seconds=0.001)
    with caplog.at_level(logging.WARNING, logger="pyrl_trainer.learner"):
        for _ in range(10_000):
            shared.record_received_reward(reward)
        assert not caplog.records
        shared.drain_runtime_stats()
        shared.record_received_reward(interval(alive_seconds=51))
    assert len(caplog.records) == 1
    assert "10001 received intervals" in caplog.text
    assert "61.000 credited live seconds" in caplog.text
    shared.record_received_reward(reward)
    assert len(caplog.records) == 1


def test_raw_task_event_resets_warning_even_when_its_reward_weights_are_zero(caplog):
    """The warning detects event absence, not a zero-valued reward configuration."""
    shared = state()
    shared.cfg.reward_food_value_scale = 0
    shared.cfg.reward_task_points_scale = 0
    reward = interval()
    with caplog.at_level(logging.WARNING, logger="pyrl_trainer.learner"):
        for _ in range(9_999):
            shared.record_received_reward(reward)
        food = deepcopy(reward)
        food["deltas"].update(foodValueDelta=1, foodCountDelta=1,
                              foodPointsDelta=20, pointsDelta=20)
        shared.record_received_reward(food)
        shared.record_received_reward(reward)
    assert not caplog.records
    stats = shared.drain_runtime_stats()
    assert stats["received_intervals"] == 10_001
    assert stats["received_foodCountDelta"] == 1
    assert stats["reward_food"] == 0
    assert "food_count=1" in format_reward_exposure(stats)


def test_received_cumulative_totals_survive_drain_without_repeating_window_totals():
    """Log snapshots reset interval totals while keeping constant-size run totals."""
    shared = state()
    shared.record_received_reward(interval(alive_seconds=2))
    first = shared.drain_runtime_stats()
    second = shared.drain_runtime_stats()
    assert first["received_aliveSecondsDelta"] == 2
    assert second["received_aliveSecondsDelta"] == 0
    assert second["received_total_aliveSecondsDelta"] == 2
    assert second["received_interval_count"] == 1


def test_survival_subtraction_roundoff_cannot_suppress_absence_warning(caplog):
    """Accepted balance roundoff is not a task event when gross task counters are zero."""
    shared = state()
    reward = interval()
    reward["deltas"].update(survivalPointsDelta=1, pointsDelta=1 + 1e-12)
    reward = parse_reward(reward)
    with caplog.at_level(logging.WARNING, logger="pyrl_trainer.learner"):
        for _ in range(10_000):
            shared.record_received_reward(reward)
    assert len(caplog.records) == 1
    assert "No authoritative task events" in caplog.text
