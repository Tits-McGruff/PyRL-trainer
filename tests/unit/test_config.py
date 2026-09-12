"""Configuration validation tests."""

# pylint: disable=import-error

import pytest

from pyrl_trainer.config import load_or_create_config

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
