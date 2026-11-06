"""Pair only complete authoritative scores from the declared scenario set."""

import pytest

from scripts.summarize_ppo_qualification import summarize


MANIFEST = {
    "trainingSeeds": [17],
    "evaluationWorldSeeds": [1001, 1002],
    "turnReference": {"turn": 1, "boost": 0},
    "straightReference": {"turn": 0, "boost": 0},
}


@pytest.fixture(name="records")
def fixture_records():
    """A baseline and candidate with both controls on two matched environments."""
    rows = []
    for scenario in MANIFEST["evaluationWorldSeeds"]:
        for label, score, reference, controls in (
            ("seed17-initial", 2, None, None),
            ("seed17-final", 5 if scenario == 1001 else 3, None, None),
            ("full-lock", 1, "full-lock", [1, 0]),
            ("straight", 0, "straight", [0, 0]),
        ):
            rows.append({
                "policy_label": label, "world_seed": scenario, "policy_sha256": label,
                "policy_version": 1 if label.endswith("-final") else 0,
                "config_hash": "test-config", "sim_speed": 1, "tick_rate": 60,
                "max_actions_per_second": 120, "duration_limit_seconds": 240,
                "sampling_seed": 0, "reward_config": {"food": 0.1},
                "reference": reference, "fixed_controls": controls, "integrity": True,
                "transition_kind": "terminal", "reason": "death", "episode_return": score,
                "reward_components": {
                    "survival": 0.01, "food": score, "growth": 0, "points": 0, "kills": 0,
                    "death": -0.01,
                },
                "raw_deltas": {
                    "foodValueDelta": score, "boostPointsSpentDelta": 0, "killsDelta": 0,
                },
                "food_value_per_alive_second": score / 10, "retained_growth": score,
                "task_net_points": score * 10, "alive_seconds": 10, "elapsed_seconds": 10,
                "sampled_turn_saturation": float(reference == "full-lock"),
                "decisions_per_alive_second": 50, "learned_turn_std": None if reference else 0.5,
            })
    return rows


def test_pair_returns_on_common_scenarios(records):
    """The report preserves both individual differences and their spread."""
    result = summarize(MANIFEST, records)
    assert result["complete"]
    assert result["matched_world_seeds"] == [1001, 1002]
    comparison = result["paired"][17]["initial"]
    assert [row["return"] for row in comparison["scenarios"]] == [3, 1]
    assert comparison["differences"]["return"]["mean"] == 2
    assert comparison["differences"]["task_reward"]["mean"] == 2
    assert comparison["differences"]["survival_reward"]["mean"] == 0
    assert comparison["differences"]["return"]["sample_stddev"] == pytest.approx(2**0.5)
    assert result["policies"]["seed17-final"]["end_reasons"] == {"death": 2}


def test_exclude_incomplete_and_missing_records_from_all_pairs(records):
    """A scenario with an invalid policy episode is excluded consistently."""
    records.pop()
    records[-1].update(integrity=False, reason="transportGap", transition_kind="incomplete")
    result = summarize(MANIFEST, records)
    assert not result["complete"]
    assert result["matched_world_seeds"] == [1001]
    assert result["excluded_world_seeds"] == {1002: {"straight": "missing",
                                                    "full-lock": "transportGap"}}
    assert result["paired"][17]["initial"]["differences"]["return"]["mean"] == 3


@pytest.mark.parametrize(("field", "value", "error"), [
    ("policy_sha256", "different", "changed policy"),
    ("sim_speed", 2, "settings differ"),
    ("reward_config", {"food": 0.2}, "settings differ"),
    ("fixed_controls", [1, 0], "controls do not match"),
])
def test_report_rejects_changed_comparison(records, field, value, error):
    """A mislabeled or incomparable record cannot silently change a comparison."""
    records[-1][field] = value
    with pytest.raises(ValueError, match=error):
        summarize(MANIFEST, records)


def test_report_rejects_duplicate_record(records):
    """Retries must be resolved explicitly before summarization."""
    with pytest.raises(ValueError, match="duplicate"):
        summarize(MANIFEST, records + [records[-1]])
