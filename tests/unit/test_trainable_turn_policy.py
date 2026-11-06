"""Squashed steering gradients, saturation, and exact distribution persistence."""

# pylint: disable=import-error,protected-access

import math

import numpy as np
import pytest
import torch

from pyrl_trainer.checkpointing import (
    load_checkpoint_if_present, save_best_checkpoint, save_checkpoint,
)
from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState
from pyrl_trainer.network import MAX_TURN_LOG_STD, MIN_TURN_LOG_STD
from pyrl_trainer.sensor_contract import SensorContract

pytestmark = pytest.mark.unit


def _state(tmp_path=None):
    cfg = Config(net_hidden=8, net_layers=1, train_device="cpu", infer_device="cpu",
                 turn_std=0.5, lr=1e-3, ent_coef=0.01,
                 ckpt_dir=str(tmp_path) if tmp_path else "unused")
    return SharedState(2, cfg, sensor_contract=SensorContract("v3", ("a", "b")))


def test_turn_parameter_is_optimized_and_receives_entropy_and_likelihood_gradients():
    """Both the PPO likelihood and declared entropy surrogate train the scale."""
    state = _state()
    parameter = state.train_model.turn_log_std
    assert isinstance(parameter, torch.nn.Parameter) and parameter.requires_grad
    assert any(parameter is item for group in state.optimizer.param_groups
               for item in group["params"])
    mean, boost, _value = state.train_model(torch.zeros(4, 2))
    _action, logp, entropy, _boost_entropy = state._joint_logp_entropy(
        mean, boost, torch.tensor([0.2, 0.4, 0.6, 0.8]), torch.zeros(4),
        state.train_model.steering_std(),
    )
    entropy_gradient = torch.autograd.grad(entropy.mean(), parameter, retain_graph=True)[0]
    likelihood_gradient = torch.autograd.grad(logp.mean(), parameter)[0]
    assert float(entropy_gradient) == pytest.approx(1.0)
    assert torch.isfinite(likelihood_gradient) and abs(float(likelihood_gradient)) > 1e-5


def test_real_ppo_entropy_step_changes_turn_log_std():
    """An exercised optimizer step updates variance rather than a config constant."""
    torch.manual_seed(81)
    state = _state()
    obs = np.zeros(2, dtype=np.float32)
    actions = [state.act(obs) for _ in range(8)]
    before = state.train_model.state_dict()["turn_log_std"].clone()
    metrics = state._ppo_minibatch_sync({
        "obs": torch.zeros(8, 2),
        "act_turn_latent": torch.tensor([action[4] for action in actions]),
        "act_boost": torch.tensor([action[1] for action in actions]),
        "old_logp": torch.tensor([action[2] for action in actions]),
        "returns": torch.tensor([action[3] for action in actions]),
        "adv": torch.zeros(8),
    })
    assert metrics["optimizer_step"] == 1
    assert state.train_model.turn_log_std > before
    assert math.isfinite(metrics["loss"])


def test_projection_keeps_variance_inside_a_differentiable_numerical_guard():
    """A large optimizer proposal cannot strand raw log std beyond the clamp."""
    state = _state()
    for proposed, expected in [(-20.0, MIN_TURN_LOG_STD), (20.0, MAX_TURN_LOG_STD)]:
        with torch.no_grad():
            state.train_model.turn_log_std.fill_(proposed)
        state.train_model.constrain_turn_std()
        actual = float(state.train_model.state_dict()["turn_log_std"])
        assert actual == pytest.approx(expected, abs=1e-6)
        assert MIN_TURN_LOG_STD < actual < MAX_TURN_LOG_STD
        gradient = torch.autograd.grad(state.train_model.steering_std(),
                                       state.train_model.turn_log_std)[0]
        assert float(gradient) > 0


def test_large_latent_actions_keep_bounded_turns_and_finite_jacobian():
    """Stable latent-space correction survives float32 tanh saturation."""
    state = _state()
    latent = torch.tensor([-1000.0, -50.0, 50.0, 1000.0])
    turn, logp, entropy, _boost_entropy = state._joint_logp_entropy(
        torch.zeros(4), torch.zeros(4), latent, torch.zeros(4),
        state.train_model.steering_std(),
    )
    assert torch.all(turn >= -1) and torch.all(turn <= 1)
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()


def test_saturated_means_report_executed_distribution_and_learned_dispersion():
    """Actual act() samples expose lost exploration even when latent entropy is high."""
    state = _state()
    with torch.no_grad():
        for parameter in state.train_model.parameters():
            parameter.zero_()
        state.train_model.turn_head.bias.fill_(3.0)
    reports = []
    for std in (0.05, 2.0):
        with torch.no_grad():
            state.train_model.turn_log_std.fill_(math.log(std))
        state.sync_infer_from_train()
        torch.manual_seed(42)
        actions = [state.act(np.zeros(2, dtype=np.float32)) for _ in range(512)]
        reports.append(state._post_update_diagnostics_sync({
            "obs": torch.zeros(512, 2), "returns": torch.zeros(512),
            "act_turn_latent": torch.tensor([action[4] for action in actions]),
        }))
    narrow, broad = reports[0], reports[1]
    assert narrow["turn_saturated_fraction"] == 1
    assert narrow["turn_sample_saturated_fraction"] > 0.99
    assert narrow["turn_sample_sign_disagreement"] == 0
    assert broad["turn_sample_sign_disagreement"] > 0.02
    assert broad["turn_std"] == pytest.approx(2.0)
    assert broad["entropy_turn_gaussian"] > narrow["entropy_turn_gaussian"]
    assert broad["turn_sample_abs_mean"] != pytest.approx(narrow["turn_sample_abs_mean"])


def test_checkpoint_resume_and_frozen_best_preserve_exact_learned_variance(tmp_path):
    """Resume and evaluator snapshots load the learned scale without reinitializing it."""
    state = _state(tmp_path)
    with torch.no_grad():
        state.train_model.turn_log_std.fill_(math.log(0.75))
    state.update_steps = 3
    state.sync_infer_from_train()
    snapshot, evaluator = state.begin_evaluation_policy()
    frozen_log_std = evaluator.state_dict()["turn_log_std"].clone()
    with torch.no_grad():
        state.train_model.turn_log_std.fill_(math.log(0.2))
    state.update_steps = 4
    state.sync_infer_from_train()
    state.record_episode(-0.1, 60, policy_snapshot=snapshot, include_runtime_stats=False)
    best = save_best_checkpoint(state.cfg, state)
    payload = torch.load(best, map_location="cpu")
    assert torch.equal(payload["model"]["turn_log_std"], frozen_log_std)
    assert torch.equal(evaluator.turn_log_std, frozen_log_std)
    assert payload["policy_parameterization_version"] == 2
    assert payload["reward_contract_version"] == 1
    save_checkpoint(state.cfg, state)
    restored = _state(tmp_path)
    assert load_checkpoint_if_present(restored.cfg, restored) is not None
    assert restored.update_steps == 4
    assert torch.equal(restored.train_model.turn_log_std, state.train_model.turn_log_std)
    assert float(restored.infer_model.steering_std().detach()) == pytest.approx(0.2)
