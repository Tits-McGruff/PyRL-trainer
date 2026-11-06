"""Evaluator work must not consume the training policy or minibatch RNG stream."""

# pylint: disable=protected-access

import numpy as np
import pytest
import torch

from pyrl_trainer.config import Config
from pyrl_trainer.evaluation import policy_snapshot_sha256
from pyrl_trainer.learner import SharedState

pytestmark = pytest.mark.unit


def test_frozen_model_sampling_and_optimizer_snapshot_leave_training_rng_unchanged(monkeypatch):
    """Construction and all evaluation draws use isolated RNG ownership."""
    cfg = Config(net_hidden=8, net_layers=1, train_device="cpu", infer_device="cpu")
    state = SharedState(2, cfg)
    def forbidden_rng_fork(*_args, **_kwargs):
        pytest.fail("global RNG save/restore would race the learner thread")
    monkeypatch.setattr(torch.random, "fork_rng", forbidden_rng_fork)
    torch.manual_seed(92)
    before = torch.get_rng_state().clone()
    snapshot, model = state.begin_evaluation_policy()
    generator = torch.Generator(device="cpu").manual_seed(73)
    actions = [state.act(np.zeros(2, dtype=np.float32), model, generator) for _ in range(32)]
    state.fresh_optimizer_state_dict()
    assert torch.equal(before, torch.get_rng_state())

    replay = torch.Generator(device="cpu").manual_seed(73)
    assert actions == [state.act(np.zeros(2, dtype=np.float32), model, replay) for _ in range(32)]
    digest = policy_snapshot_sha256(snapshot)
    with torch.no_grad():
        state.train_model.turn_log_std.add_(0.1)
    state.sync_infer_from_train()
    assert policy_snapshot_sha256(snapshot) == digest
    assert policy_snapshot_sha256(state._published_policy_snapshot) != digest
