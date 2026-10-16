"""Trainer-mode selection without actor, model or configuration side effects."""

# pylint: disable=import-error

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pyrl_trainer import __main__ as entry
from pyrl_trainer import distillation_run

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arguments", [[], ["--mode", "ppo"]])
def test_ppo_remains_the_default_owner(monkeypatch, arguments):
    """Default and explicit PPO both call the unchanged asynchronous main."""
    called = []

    async def ppo_main():
        called.append("ppo")

    monkeypatch.setattr(entry, "main", ppo_main)
    assert entry.run(arguments) is None
    assert called == ["ppo"]


def test_distillation_delegates_exact_remaining_arguments_without_ppo(monkeypatch):
    """A canonical job never constructs actors or touches the PPO config owner."""
    called = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("PPO startup must not run in Distillation mode")

    def canonical(arguments):
        called.append(arguments)
        return {"stopReason": "update_limit"}

    monkeypatch.setattr(entry, "main", forbidden)
    monkeypatch.setattr(distillation_run, "run", canonical)
    arguments = ["--resume", "latest.pt", "--job-directory", "jobs", "--device", "cuda:0"]
    assert entry.run(["--mode", "distillation", *arguments]) == {"stopReason": "update_limit"}
    assert called == [arguments]


@pytest.mark.parametrize("mode", ["ppo", "distillation"])
def test_mode_help_has_no_startup_side_effects(monkeypatch, capsys, mode):
    """Each owner's help is available before config creation or network traffic."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("help must not start a trainer")

    monkeypatch.setattr(entry, "main", forbidden)
    with pytest.raises(SystemExit) as stopped:
        entry.run(["--mode", mode, "--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert ("--manifest" if mode == "distillation" else "--mode") in output
    assert ("--job-directory" if mode == "distillation" else "SLITHER_CONFIG") in output


@pytest.mark.parametrize("arguments", [
    ["--mode", "unknown"], ["--mode", "ppo", "--updates", "1"],
    ["--mo", "distillation"],
])
def test_invalid_mode_or_owner_arguments_reject_before_startup(monkeypatch, arguments):
    """Unknown options cannot accidentally start PPO or create its config."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid arguments must not start a trainer")

    monkeypatch.setattr(entry, "main", forbidden)
    with pytest.raises(SystemExit) as stopped:
        entry.run(arguments)
    assert stopped.value.code == 2


def test_distillation_help_never_imports_ppo_or_reads_its_environment(tmp_path):
    """A fresh process ignores malformed PPO settings before any actor module imports."""
    config = tmp_path / "must-not-create.toml"
    script = '''
import json,sys
from pyrl_trainer.__main__ import run
try:
    run(["--mode", "distillation", "--help"])
except SystemExit as stopped:
    if stopped.code != 0:
        raise
print(json.dumps({"ppoModules": [name for name in
    ["pyrl_trainer.agent", "pyrl_trainer.learner", "pyrl_trainer.config"]
    if name in sys.modules]}))
'''
    environment = dict(os.environ, SLITHER_CONFIG=str(config),
                       SLITHER_WS_MAX_MESSAGE="invalid-ppo-value")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
        env=environment, check=True, capture_output=True, text=True, timeout=30,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {"ppoModules": []}
    assert not config.exists()
