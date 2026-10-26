"""Bounded canonical window, reset/preserve, gap and burn-in regressions."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraph, CanonicalGraphManifest, CanonicalGraphError
from pyrl_trainer.canonical_graph import distillation_loss
from pyrl_trainer.sequence_distillation import (
    CanonicalSequenceTrainer,
    SequenceBatch,
    SequenceBoundary,
    SequenceTrainingConfig,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "canonical_graph"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _model(device="cpu", life_rule="preserve"):
    """Build a fixture graph with an explicit deployed life rule."""
    raw = json.loads((ROOT / "gru/nonzero/manifest.json").read_text(encoding="utf-8"))
    raw["deployment"]["lifeStateRule"] = life_rule
    manifest = CanonicalGraphManifest.from_bytes(json.dumps(raw).encode())
    return CanonicalGraph(manifest, (ROOT / "gru/nonzero/weights.f32").read_bytes(), device)


def _batch(model, rows=8, batch_size=2):
    """Create bounded synthetic windows without a world or controller."""
    values = np.frombuffer((ROOT / "gru/observations.f32").read_bytes(), dtype="<f4").copy()
    values = torch.from_numpy(values.reshape(256, model.manifest.input_size)[:rows])
    observations = values.unsqueeze(1).repeat(1, batch_size, 1)
    return SequenceBatch(
        observations,
        observations[..., 0] * 0.2,
        observations[..., 1] > 0,
        torch.ones(rows, batch_size),
        torch.ones(rows, batch_size, dtype=torch.bool),
        torch.zeros(rows, batch_size, dtype=torch.int64),
        torch.full((batch_size, model.manifest.total_state_size), 0.17),
    )


def _reference(model, batch):
    """Evaluate declared boundaries directly before updating weights."""
    state = batch.initial_state.to(model.flat_parameters.device)
    with torch.no_grad():
        for index in range(batch.observations.shape[0]):
            active = batch.active[index].to(state.device)
            boundaries = batch.boundaries[index].to(state.device)
            reset = (
                (boundaries == SequenceBoundary.GENERATION)
                | (boundaries == SequenceBoundary.GAP)
                | (boundaries == SequenceBoundary.CAPTURE_START)
            )
            if model.manifest.life_state_rule == "reset":
                reset = reset | (boundaries == SequenceBoundary.LIFE)
            before = torch.where((reset & active).unsqueeze(-1), torch.zeros_like(state), state)
            _output, after = model(batch.observations[index].to(state.device), before)
            state = torch.where(active.unsqueeze(-1), after, state)
    return state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("life_rule", ["reset", "preserve"])
def test_boundaries_and_padding_follow_deployment_state_rules(device, life_rule):
    """Reset/preserve, gaps and padding match a direct recurrence reference."""
    model = _model(device, life_rule)
    batch = _batch(model)
    batch.boundaries[2, 0] = SequenceBoundary.LIFE
    batch.boundaries[4, 0] = SequenceBoundary.GAP
    batch.boundaries[5, 1] = SequenceBoundary.GENERATION
    batch.boundaries[0, 1] = SequenceBoundary.CAPTURE_START
    batch.active[6:, 1] = False
    expected = _reference(model, batch)
    weights = model.export_weights()
    update = CanonicalSequenceTrainer(
        model, SequenceTrainingConfig(batch_size=2, burn_in_steps=2, bptt_steps=6)
    ).update(batch)
    torch.testing.assert_close(update.next_state, expected, atol=1e-6, rtol=1e-5)
    assert update.next_state.grad_fn is None
    assert update.training_samples == 10 and update.burn_in_samples == 4
    assert update.unknown_state_starts == 2
    assert update.optimizer_updated and model.export_weights() != weights
    assert 0 <= update.boost_disagreement <= 1


@pytest.mark.parametrize("device", DEVICES)
def test_burnin_and_truncation_do_not_backpropagate_to_prior_history(device):
    """Caller observations and carried history stay outside the new backward graph."""
    model = _model(device)
    batch = _batch(model)
    batch.observations.requires_grad_()
    batch.initial_state.requires_grad_()
    trainer = CanonicalSequenceTrainer(
        model, SequenceTrainingConfig(batch_size=2, burn_in_steps=2, bptt_steps=6)
    )
    first = trainer.update(batch)
    assert batch.observations.grad is None and batch.initial_state.grad is None
    assert first.next_state.grad_fn is None
    second_batch = replace(batch, initial_state=first.next_state.cpu())
    second = trainer.update(second_batch)
    assert second.next_state.grad_fn is None and trainer.completed_updates == 2
    assert torch.isfinite(second.next_state).all()


@pytest.mark.parametrize(
    "fault", ["interior-padding", "nan", "boost", "weight", "boundary", "empty", "precision"]
)
def test_malformed_batches_fail_before_model_or_optimizer_mutation(fault):
    """Malformed windows cannot change model or optimizer continuation."""
    model = _model()
    batch = _batch(model)
    if fault == "interior-padding":
        batch.active[2, 0] = False
    elif fault == "nan":
        batch.observations[0, 0, 0] = float("nan")
    elif fault == "boost":
        batch = replace(batch, teacher_boost_request=torch.full((8, 2), 0.5))
    elif fault == "weight":
        batch.event_weight[0, 0] = 5.0
    elif fault == "boundary":
        batch.boundaries[0, 0] = 7
    elif fault == "empty":
        batch.active[:] = False
    else:
        batch = replace(batch, teacher_turn=batch.teacher_turn.double())
    trainer = CanonicalSequenceTrainer(
        model, SequenceTrainingConfig(batch_size=2, burn_in_steps=2, bptt_steps=6)
    )
    before = model.export_weights()
    with pytest.raises(CanonicalGraphError):
        trainer.update(batch)
    assert model.export_weights() == before
    assert not trainer.optimizer.state and trainer.completed_updates == 0


def test_default_batch_and_exact_burnin_training_counts():
    """The declared 32/32/64 defaults count delivered samples exactly."""
    model = _model()
    trainer = CanonicalSequenceTrainer(model)
    update = trainer.update(_batch(model, rows=96, batch_size=32))
    assert update.burn_in_samples == 32 * 32
    assert update.training_samples == 64 * 32
    assert trainer.config.learning_rate == 3e-4 and trainer.config.gradient_clip == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bptt_steps": 257},
        {"burn_in_steps": 256},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"gradient_clip": 0.0},
    ],
)
def test_configuration_bounds_reject_unbounded_or_nonfinite_training(kwargs):
    """Nonfinite and unbounded configuration rejects before allocation."""
    with pytest.raises(CanonicalGraphError):
        SequenceTrainingConfig(**kwargs)


def test_burnin_gradient_matches_independent_bounded_objective():
    """Burn-in contributes state but neither its labels nor its backward history."""

    model, reference = _model(), _model()
    batch = _batch(model)
    trainer = CanonicalSequenceTrainer(
        model, SequenceTrainingConfig(batch_size=2, burn_in_steps=2, bptt_steps=6)
    )
    state = batch.initial_state
    with torch.no_grad():
        for row in batch.observations[:2]:
            _output, state = reference(row, state)
    losses = []
    for index in range(2, 8):
        output, state = reference(batch.observations[index], state)
        losses.append(
            distillation_loss(
                output,
                batch.teacher_turn[index],
                batch.teacher_boost_request[index],
                batch.event_weight[index],
            )
        )
    expected = torch.stack(losses).mean()
    expected.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    actual = trainer.update(batch)
    torch.testing.assert_close(
        model.flat_parameters.grad, reference.flat_parameters.grad, atol=1e-7, rtol=1e-6
    )
    assert actual.loss == pytest.approx(expected.item(), abs=1e-7)


def test_parameter_free_routing_graph_reports_metrics_without_optimizer_update():
    """Canonical routing graphs stay supported even when there is nothing to learn."""
    root = ROOT / "routing/zero"
    model = CanonicalGraph(CanonicalGraphManifest.read(root / "manifest.json"), b"")
    observations = torch.zeros(4, 1, model.manifest.input_size)
    batch = SequenceBatch(
        observations,
        torch.zeros(4, 1),
        torch.zeros(4, 1),
        torch.ones(4, 1),
        torch.ones(4, 1, dtype=torch.bool),
        torch.zeros(4, 1, dtype=torch.int64),
        model.initial_state(1),
    )
    trainer = CanonicalSequenceTrainer(
        model, SequenceTrainingConfig(batch_size=1, burn_in_steps=0, bptt_steps=4)
    )
    update = trainer.update(batch)
    assert not update.optimizer_updated and trainer.completed_updates == 0
    assert update.training_samples == 4 and update.next_state.shape == (1, 0)
