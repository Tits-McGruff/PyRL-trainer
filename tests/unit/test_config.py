"""Configuration validation tests."""

# pylint: disable=import-error

import math

import pytest

from pyrl_trainer.config import (
    Config,
    load_or_create_config,
    reward_config_record,
    validate_config,
)

pytestmark = pytest.mark.unit


def test_config_rejects_zero_actors(tmp_path):
    """Invalid actor counts fail instead of being silently clamped."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[connection]\nactors = 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="actors"):
        load_or_create_config(str(config_path))


def test_config_rejects_invalid_environment_number(tmp_path, monkeypatch):
    """Malformed environment overrides fail instead of falling back to defaults."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("SLITHER_LR", "not-a-number")

    with pytest.raises(ValueError, match="SLITHER_LR"):
        load_or_create_config(str(config_path))


def test_config_rejects_malformed_toml(tmp_path):
    """A broken config file is reported instead of silently replacing it."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[training\ngamma = 0.99\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to load config"):
        load_or_create_config(str(config_path))


def test_generated_config_contains_rewards_section(tmp_path):
    """Generated TOML exposes every reward parameter with canonical defaults."""
    config_path = tmp_path / "config.toml"
    cfg = load_or_create_config(str(config_path))
    text = config_path.read_text(encoding="utf-8")

    assert "[rewards]" in text
    assert "growth_scale = 10.0" in text
    assert "death_penalty_magnitude = 0.5" in text
    assert cfg.reward_growth_scale == pytest.approx(10.0)


def test_reward_toml_and_environment_override(tmp_path, monkeypatch):
    """Reward settings load from TOML and environment overrides take precedence."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[rewards]\n"
        "growth_scale = 12.0\n"
        "food_approach_scale = 0.25\n"
        "food_delta_clip = 0.2\n"
        "survival_bonus = 0.0002\n"
        "wall_threshold = -0.4\n"
        "wall_penalty_scale = 0.07\n"
        "hazard_threshold = -0.3\n"
        "hazard_penalty_magnitude = 0.2\n"
        "death_penalty_magnitude = 0.9\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SLITHER_REWARD_GROWTH_SCALE", "15.0")
    monkeypatch.setenv("SLITHER_REWARD_DEATH_PENALTY", "1.25")

    cfg = load_or_create_config(str(config_path))

    assert cfg.reward_growth_scale == pytest.approx(15.0)
    assert cfg.reward_food_approach_scale == pytest.approx(0.25)
    assert cfg.reward_death_penalty_magnitude == pytest.approx(1.25)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reward_growth_scale", -0.1),
        ("reward_food_delta_clip", 0.0),
        ("reward_survival_bonus", -0.1),
        ("reward_wall_threshold", -1.1),
        ("reward_hazard_threshold", 1.1),
        ("reward_death_penalty_magnitude", -0.1),
        ("reward_growth_scale", math.inf),
    ],
)
def test_reward_validation_rejects_invalid_values(field, value):
    """Invalid reward settings fail before runtime tasks are started."""
    cfg = Config(net_hidden=8, net_layers=1, train_device="cpu")
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=field):
        validate_config(cfg)


def test_reward_config_record_has_normalized_schema_keys():
    """Checkpoint reward metadata uses the complete stable prefixed field set."""
    record = reward_config_record(Config())

    assert set(record) == {
        "reward_growth_scale",
        "reward_food_approach_scale",
        "reward_food_delta_clip",
        "reward_survival_bonus",
        "reward_wall_threshold",
        "reward_wall_penalty_scale",
        "reward_hazard_threshold",
        "reward_hazard_penalty_magnitude",
        "reward_death_penalty_magnitude",
    }
    assert all(isinstance(value, float) for value in record.values())
