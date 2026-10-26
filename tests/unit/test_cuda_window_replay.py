"""Retained CUDA measurements and CPU replay of the actual trained native prefix."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pyrl_trainer.brain_artifact import HEADER, read_raw_brain_artifact
from pyrl_trainer.canonical_graph import CanonicalGraph
from pyrl_trainer.canonical_parity import _sequence, _compare

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures/cuda_window_replay_v1"


def test_cuda_trained_brain_replays_all_native_outputs_and_states():
    """The deliberate golden fixture needs no historical experiment/source bundle."""
    fixture = json.loads((ROOT / "fixture.json").read_bytes())
    files = {}
    for name, record in fixture["files"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        files[name] = data
    evidence = fixture["purpose"]
    artifact, manifest, _ = read_raw_brain_artifact(files["trained.slither-brain"])
    receipt = json.loads(files["native-parity/receipt.json"])
    assert receipt["sourceSha256"] == evidence["nativeSourceSha256"]
    assert artifact.weights_sha256 == receipt["weightsSha256"]
    assert manifest.layout_sha256 == receipt["layoutSha256"]
    weights = artifact.data[HEADER.size + HEADER.unpack_from(artifact.data)[1]:]
    model = CanonicalGraph(manifest, weights)
    values = []
    expected_files = (("parity-observations.f32", "observationsSha256", manifest.input_size),
                      ("native-parity/outputs.f32", "outputsSha256", 2),
                      ("native-parity/states.f32", "statesSha256", manifest.total_state_size))
    for name, digest, width in expected_files:
        data = files["" + name]
        assert hashlib.sha256(data).hexdigest() == receipt[digest]
        values.append(torch.from_numpy(np.frombuffer(data, dtype="<f4").copy().reshape(256, width)))
    actual, states = _sequence(model, values[0], model.initial_state(1))
    _compare(actual, values[1], "cpu")
    _compare(states, values[2], "cpu")
