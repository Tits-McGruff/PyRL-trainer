"""Verify the matrix binds evaluation to the observed training configuration."""

# pylint: disable=protected-access

import json

import pytest

from pyrl_trainer.config import AUTHORITATIVE_REWARD_CONFIG
from scripts.run_ppo_qualification_matrix import _completed_policies, _effective_evaluation_settings


def test_matrix_uses_effective_startup_overrides(tmp_path):
    """Generated TOML defaults can differ from the CPU policy used in training."""
    (tmp_path / "trainer.toml").write_text(
        'train_device = "cuda"\nnet_hidden = 512\nnet_layers = 4\n', encoding="utf-8"
    )
    (tmp_path / "training.log").write_text(
        "[main] ws=ws://localhost, train_device=cpu, infer_device=cpu, network=256x2\n"
        f"[main] reward_config_version=2, reward_coefficients="
        f"{json.dumps(AUTHORITATIVE_REWARD_CONFIG)}, policy_parameterization=2, "
        "max_actions_per_second=120, max_actions_per_tick=1, "
        "evaluation_max_seconds=240, evaluation_seed=0\n",
        encoding="utf-8",
    )
    settings = _effective_evaluation_settings(tmp_path)
    assert settings == {
        **AUTHORITATIVE_REWARD_CONFIG,
        "train_device": "cpu", "infer_device": "cpu", "net_hidden": 256, "net_layers": 2,
        "max_actions_per_second": 120, "max_actions_per_tick": 1,
        "evaluation_max_seconds": 240.0, "evaluation_seed": 0,
    }


def test_matrix_requires_final_budget_completion_even_with_checkpoint(tmp_path):
    """A periodic latest checkpoint cannot stand in for the final candidate."""
    manifest = {
        "trainingSeed": 17, "trainingSeeds": [17], "evaluationWorldSeeds": [1001],
        "pyrlCommit": "trainer", "slitherCommit": "slither", "configHash": "settings",
        "mathBackend": "rust-sse2-v1", "worldSeed": 42,
        "decisionBudget": 10, "candidateRule": "final complete budget barrier",
    }
    (tmp_path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "exit-code.txt").write_text("0", encoding="utf-8")
    initial = tmp_path / "checkpoints/initial/latest.pt"
    final = tmp_path / "checkpoints/latest.pt"
    initial.parent.mkdir(parents=True)
    initial.touch()
    final.touch()
    log = tmp_path / "training.log"
    log.write_text("[learner] updates=1, steps=8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="budget completion"):
        _completed_policies(manifest, [tmp_path])
    log.write_text(
        "[qualification] admitted_decisions=12, decision_budget=10, "
        "alive_seconds=0.200000, updates=2, policy_v=8\n",
        encoding="utf-8",
    )
    assert _completed_policies(manifest, [tmp_path]) == {
        "seed17-initial": (initial, 0), "seed17-final": (final, 8),
    }
