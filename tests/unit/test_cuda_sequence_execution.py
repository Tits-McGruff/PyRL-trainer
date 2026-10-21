"""CUDA whole-window equivalence, immutable continuations and bounded dispatch."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraph, CanonicalGraphManifest, CanonicalGraphError
from pyrl_trainer.sequence_distillation import (
    CanonicalSequenceTrainer, SequenceBatch, SequenceTrainingConfig, _WindowExecution,
    _sequence_forward, _before_delivery,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures/canonical_graph"
CONFIG = SequenceTrainingConfig(batch_size=2, burn_in_steps=2, bptt_steps=6)


def _model(name, device):
    """Load native-provided graph layouts without substituting recurrent modules."""
    root = ROOT / name
    return CanonicalGraph(CanonicalGraphManifest.read(root / "manifest.json"),
                          (root / "weights.f32").read_bytes(), device)


def _batch(model):
    """Use boundaries, nonzero recurrence and padded lanes on caller-owned CPU storage."""
    width = model.manifest.input_size
    observations = torch.linspace(-0.9, 0.9, 8 * 2 * width).reshape(8, 2, width)
    active = torch.ones(8, 2, dtype=torch.bool)
    active[6:, 1] = False
    boundaries = torch.zeros(8, 2, dtype=torch.int64)
    boundaries[0, 0], boundaries[2, 0] = 4, 2
    boundaries[4, 0], boundaries[5, 1] = 3, 1
    return SequenceBatch(observations, observations[..., 0] * 0.3,
                         observations[..., 1] > 0, torch.ones(8, 2), active, boundaries,
                         torch.full((2, model.manifest.total_state_size), 0.17))


def test_cpu_rejects_explicit_cuda_execution():
    """An acceleration request never silently chooses another device."""
    with pytest.raises(CanonicalGraphError, match="requires a CUDA model"):
        CanonicalSequenceTrainer(_model("gru/nonzero", "cpu"), CONFIG, cuda_graphs=True)
    with pytest.raises(CanonicalGraphError, match="explicit boolean"):
        CanonicalSequenceTrainer(_model("gru/nonzero", "cpu"), CONFIG, cuda_graphs="yes")


def test_dispatch_compiles_at_most_one_signature_and_never_resets_global_cache():
    """Shape changes and oversized work remain eager without growing compiler caches."""
    model = _model("gru/nonzero", "cpu")
    data = _batch(model)
    execute = _WindowExecution(True, CONFIG)
    with patch("torch.compile", side_effect=lambda function, **_kwargs: function) as compile_call, \
            patch("torch.compiler.cudagraph_mark_step_begin"):
        execute(model, data, 2)
        execute(model, data, 2)
        execute(model, data, 1)
        execute(model, replace(data, observations=data.observations.repeat(13, 1, 1),
                               active=data.active.repeat(13, 1),
                               boundaries=data.boundaries.repeat(13, 1)), 2)
    assert compile_call.call_count == 1
    assert compile_call.call_args.args[0].__code__ is not \
        _sequence_forward.__code__
    assert execute.statistics() == {"requested": "cudagraphs", "compiledWindowCalls": 2,
                                    "eagerWindowCalls": 2, "compiledSignatures": 1}


@pytest.mark.parametrize("life_rule", ["preserve", "reset"])
def test_compiler_preserves_integer_boundary_comparisons_on_cpu(life_rule):
    """CPU CI reproduces the enum-folding defect without requiring a CUDA runner."""
    state = torch.full((5, 3), 0.17)
    boundaries = torch.tensor([1, 2, 3, 4, 3], dtype=torch.int64)
    active = torch.tensor([True, True, True, True, False])
    expected = state.clone()
    expected[[0, 2, 3]] = 0
    if life_rule == "reset":
        expected[1] = 0
    compiled = torch.compile(_before_delivery, backend="eager", fullgraph=True)
    actual = compiled(state, boundaries, active, life_rule)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an actual CUDA device")
@pytest.mark.parametrize("name", ["dense/zero", "mlp/zero", "gru/nonzero", "lstm/nonzero",
                                  "rru/nonzero", "mixed/nonzero", "routing/zero"])
@pytest.mark.parametrize("life_rule", ["preserve", "reset"])
def test_cuda_window_matches_eager_updates_and_fresh_optimizer_restore(name, life_rule):
    """All operators retain gradients, masks, RNG and Adam continuation across replay."""
    eager_model = _model(name, "cuda")
    # Manifests are frozen dataclasses; only the explicitly deployed life rule changes.
    eager_model.manifest = replace(eager_model.manifest, life_state_rule=life_rule)
    fast_model = CanonicalGraph(eager_model.manifest, eager_model.export_weights(), "cuda")
    batch = _batch(eager_model)
    eager = CanonicalSequenceTrainer(eager_model, CONFIG)
    fast = CanonicalSequenceTrainer(fast_model, CONFIG, cuda_graphs=True)
    previous = None
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    for index in range(4):
        # Fresh input contents must reach the captured graph, not its warmup buffers.
        current = replace(batch, teacher_turn=batch.teacher_turn + index * 0.03,
                          observations=batch.observations * (1 - index * 0.1))
        burn_in = (2, 1, 0, 2)[index]
        eager.config = fast.config = replace(CONFIG, burn_in_steps=burn_in)
        length = burn_in + CONFIG.bptt_steps
        current = SequenceBatch(*(getattr(current, field)[:length] for field in (
            "observations", "teacher_turn", "teacher_boost_request", "event_weight", "active",
            "boundaries")), current.initial_state)
        expected, actual = eager.update(current), fast.update(current)
        for field in ("loss", "turn_mse", "turn_mae", "gradient_norm"):
            assert getattr(actual, field) == pytest.approx(
                getattr(expected, field), abs=1e-6), (index, field)
        for field in ("training_samples", "burn_in_samples", "unknown_state_starts",
                      "boost_disagreement", "optimizer_updated"):
            assert getattr(actual, field) == getattr(expected, field)
        torch.testing.assert_close(actual.next_state, expected.next_state, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(fast_model.flat_parameters, eager_model.flat_parameters,
                                   atol=1e-6, rtol=1e-5)
        if fast_model.flat_parameters.grad is not None:
            torch.testing.assert_close(fast_model.flat_parameters.grad,
                                       eager_model.flat_parameters.grad, atol=1e-6, rtol=1e-5)
        eager_adam, fast_adam = eager.optimizer.state_dict(), fast.optimizer.state_dict()
        assert eager_adam["param_groups"] == fast_adam["param_groups"]
        assert eager_adam["state"].keys() == fast_adam["state"].keys()
        for parameter, values in eager_adam["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, fast_adam["state"][parameter][key],
                                           atol=1e-6, rtol=1e-5)
        if previous is not None:
            assert torch.equal(previous[0], previous[1])
        previous = actual.next_state, actual.next_state.clone()
        assert actual.next_state.grad_fn is None
        if index == 2:
            # Compiler buffers are derived state: a new owner restores only ordinary
            # canonical parameters and Adam tensors, then compiles its next window.
            restored_model = CanonicalGraph(
                fast_model.manifest, fast_model.export_weights(), "cuda")
            restored = CanonicalSequenceTrainer(restored_model, CONFIG, cuda_graphs=True)
            restored.optimizer.load_state_dict(deepcopy(fast.optimizer.state_dict()))
            restored.completed_updates = fast.completed_updates
            fast, fast_model = restored, restored_model
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert fast.completed_updates == eager.completed_updates
    assert fast.window_execution.statistics()["compiledSignatures"] == 1
    weights = fast_model.export_weights()
    count = fast.completed_updates
    bad = replace(batch, observations=batch.observations.clone())
    bad.observations[0, 0, 0] = float("nan")
    with pytest.raises(CanonicalGraphError, match="finite"):
        fast.update(bad)
    assert fast.completed_updates == count and fast_model.export_weights() == weights
