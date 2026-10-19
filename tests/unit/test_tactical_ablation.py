"""Verify declared feature ports, paired data ownership and actual-history windows."""

import gzip
import json
import time
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraphError
from pyrl_trainer.sequence_distillation import SequenceBoundary, SequenceTrainingConfig
from pyrl_trainer.tactical_ablation import (VARIANTS, TacticalWindow, ablation_graph,
                                          evaluate_agreement, paired_windows, window_batch)
from pyrl_trainer.tactical_corpus import load_tactical_corpus
from pyrl_trainer.tactical_ablation_gate import AblationStudy

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parents[1] / "fixtures/tactical_corpus_v1"


@pytest.fixture(name="corpus")
def native_corpus(tmp_path):
    """Use exact native fixture deliveries without synthesizing a teacher or world."""
    for name in ("corpus.json", "observations.f32", "actions.f32", "steps.u32"):
        (tmp_path / name).write_bytes(gzip.decompress((FIXTURE / (name + ".gz")).read_bytes()))
    return load_tactical_corpus(tmp_path)


def manifest(corpus):
    """Only the batch adapter's validated shape/identity view; no packing is inferred."""
    return SimpleNamespace(input_size=264, total_state_size=1,
                           sensor_sha256=corpus.sensor.structural_sha256,
                           control_interval_seconds=corpus.control_interval_seconds)


@pytest.mark.parametrize("variant,prefixes,width", [
    ("no-bearing", ("head_bearing_residual_",), 248),
    ("no-heading", ("head_heading_sin_", "head_heading_cos_"), 232),
    ("no-velocity", ("head_velocity_forward_", "head_velocity_lateral_"), 232),
    ("no-urgency", ("head_collision_urgency_",), 248),
    ("no-density", ("head_density_",), 248),
])
def test_ablation_ports_remove_exact_native_contract_labels(corpus, variant, prefixes, width):
    """Ground port indices in Rust's ordered labels, catching channel off-by-one errors."""
    graph = ablation_graph(variant)
    ports = [edge["fromPort"] for edge in graph["edges"] if edge["to"] == "selected"]
    omitted = [name for port in range(1, 16) if port not in ports
               for name in corpus.sensor.order[24 + (port - 1) * 16:24 + port * 16]]
    expected = [name for name in corpus.sensor.order if name.startswith(prefixes)]
    assert omitted == expected
    assert len(omitted) == 264 - width
    assert next(node for node in graph["nodes"] if node["id"] == "encode")["inputSize"] == width
    assert [edge["toPort"] for edge in graph["edges"] if edge["to"] == "selected"] == list(
        range(len(ports)))
    assert graph["nodes"][0]["outputSize"] == 264
    assert graph["outputs"] == [{"nodeId": "motor"}]


def test_full_and_core_graphs_keep_scalar_prefix_and_canonical_motor():
    """The experiment preserves the full sensor contract and changes only graph routing."""
    assert len(VARIANTS) == 7
    for variant, width in [("full", 264), ("core-only", 120)]:
        graph = ablation_graph(variant)
        assert next(node for node in graph["nodes"] if node["id"] == "encode")["inputSize"] == width
        assert graph["nodes"][-1] == {"id": "motor", "type": "Dense", "inputSize": 16,
                                      "outputSize": 2}
    with pytest.raises(CanonicalGraphError):
        ablation_graph("unknown")


def test_paired_schedule_uses_only_training_seeds_and_real_prefixes(corpus):
    """Short lives remain eligible and temporal blocks use the same independent RNG schedule."""
    training, validation = corpus.split_by_seed({2})
    config = SequenceTrainingConfig(batch_size=4, bptt_steps=1, burn_in_steps=1)
    before = np.random.get_state()[1].copy()
    first = list(paired_windows(corpus, training, config, 12, 9))
    second = list(paired_windows(corpus, training, config, 12, 9))
    assert first == second
    np.testing.assert_array_equal(np.random.get_state()[1], before)
    assert first != list(paired_windows(corpus, training, config, 12, 10))
    assert all(corpus.sequences[window.sequence] not in validation
               for selected in first for window in selected)
    assert all(window.burn_in == min(1, window.training_start)
               for selected in first for window in selected)
    assert any(window.training_start > window.burn_in for selected in first for window in selected)
    original = corpus.observations.tobytes()
    for selected in first:
        batch, actual = window_batch(corpus, manifest(corpus), selected, config)
        batch.validate(SimpleNamespace(manifest=manifest(corpus)), actual)
        for column, window in enumerate(selected):
            sequence = corpus.sequences[window.sequence]
            begin = sequence.row_offset + window.training_start - window.burn_in
            np.testing.assert_array_equal(batch.observations[0, column].numpy(),
                                          corpus.observations[begin])
            expected = SequenceBoundary.GENERATION if window.training_start == window.burn_in \
                else SequenceBoundary.GAP
            assert batch.boundaries[0, column].item() == expected
        batch.observations.fill_(0)
    assert corpus.observations.tobytes() == original


@pytest.mark.parametrize("window", [None, TacticalWindow(-1, 0, 0), TacticalWindow(0, 99, 0),
                                   TacticalWindow(0, 0, 1), TacticalWindow(True, 0, 0)])
def test_invalid_window_rejects_before_tensor_construction(corpus, window):
    """Malformed indices, history and bool identities cannot enter a canonical update."""
    with pytest.raises(CanonicalGraphError):
        window_batch(corpus, manifest(corpus), [window], SequenceTrainingConfig())


class ConstantModel(torch.nn.Module):
    """Test-only constant outputs with a counted state to check metric/life boundaries."""

    def __init__(self, corpus):
        super().__init__()
        self.manifest = manifest(corpus)
        self.flat_parameters = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def initial_state(self, batch):
        """Start each physical sequence at zero."""
        return torch.zeros((batch, 1))

    def forward(self, observation, state):
        """Count calls; this stub does not implement a canonical neural operation."""
        self.seen.append(state.item())
        return observation.new_tensor([[0, 0.5]]), state + 1


def test_agreement_uses_all_held_out_rows_and_resets_only_at_sequence_boundaries(corpus):
    """Metrics compare request labels at the deployed threshold with carried per-life state."""
    _, validation = corpus.split_by_seed({2})
    model = ConstantModel(corpus)
    result = evaluate_agreement(model, corpus, validation)
    indices = [index for sequence in validation
               for index in range(sequence.row_offset, sequence.row_offset + sequence.rows)]
    assert result["rows"] == len(indices)
    assert result["turnMse"] == pytest.approx(np.square(corpus.actions[indices, 0]).mean())
    assert result["boostDisagreement"] == pytest.approx((corpus.actions[indices, 1] == 0).mean())
    expected = [index for sequence in validation for index in range(sequence.rows)]
    assert model.seen == expected
    assert result["loss"] >= 0 and result["torchEvaluationWallMs"] > 0
    model.manifest.control_interval_seconds = 0.03
    with pytest.raises(CanonicalGraphError):
        evaluate_agreement(model, corpus, validation)



def test_failed_study_retains_incomplete_outcome(tmp_path):
    """An interrupted variant must not publish a successful measurement marker."""
    study = AblationStudy.__new__(AblationStudy)
    study.args = SimpleNamespace(variant=["full"], directory=tmp_path)
    study.started = time.perf_counter()
    study.report = {"measurementComplete": False, "releaseQualified": False, "variants": []}
    study.variant = Mock(side_effect=RuntimeError("native rejection"))
    with pytest.raises(RuntimeError, match="native rejection"):
        study.run()
    result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["measurementComplete"] is False
    assert result["releaseQualified"] is False
    assert result["variants"] == []


def test_changed_runner_is_rejected_before_launch(tmp_path, monkeypatch):
    """A changed binary cannot inherit the fingerprint of the measured tool."""
    runner = tmp_path / "runner"
    runner.write_bytes(b"changed executable")
    study = AblationStudy.__new__(AblationStudy)
    study.runners = {"canonical": runner}
    study.runner_hashes = {"canonical": "0" * 64}
    launch = Mock()
    monkeypatch.setattr("pyrl_trainer.tactical_ablation_gate.subprocess.run", launch)
    with pytest.raises(ValueError, match="runner changed"):
        study._invoke("canonical", [])  # pylint: disable=protected-access
    launch.assert_not_called()
