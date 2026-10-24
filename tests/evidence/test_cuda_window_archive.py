"""Retained CUDA measurements and CPU replay of the actual trained native prefix."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pyrl_trainer.evidence_store import required_root
from pyrl_trainer.brain_artifact import HEADER, read_raw_brain_artifact
from pyrl_trainer.canonical_graph import CanonicalGraph
from pyrl_trainer.canonical_parity import _sequence, _compare

pytestmark = pytest.mark.evidence
ROOT = (required_root(Path(__file__).resolve().parents[2])
        / "docs/performance/canonical_graph/cuda-window-windows-v1")


def test_retained_cuda_capsule_hashes_dispatch_and_native_continuation():
    """CI checks bounded archived bytes and replays the trained GPU artifact against Rust."""
    evidence = json.loads((ROOT / "evidence.json").read_text(encoding="utf-8"))
    files = {}
    total = 0
    for name, receipt in evidence["files"].items():
        encoded = (ROOT / (name + ".gz")).read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == receipt["gzipSha256"]
        with gzip.open(ROOT / (name + ".gz"), "rb") as stream:
            data = stream.read(1024**2 + 1)
        total += len(data)
        assert len(data) == receipt["bytes"] <= 1024**2 and total <= 4 * 1024**2
        assert hashlib.sha256(data).hexdigest() == receipt["sha256"]
        files[name] = data
    paired = json.loads(files["paired-updates.json"])
    assert paired["trainerSourceSha256"] == evidence["trainerSourceSha256"]
    assert all(value == 0 for value in paired["maxAbs"].values())
    for mode, run in paired["runs"].items():
        assert len(run["updateWallMs"]) == 32
        assert run["dispatch"]["compiledWindowCalls"] == (32 if mode == "cudagraphs" else 0)
    for name, digest in paired["sources"].items():
        assert hashlib.sha256(files["sources/" + name]).hexdigest() == digest
    functional = json.loads(files["native-roundtrip.json"])
    assert functional["measurementComplete"] and functional["status"] == "measured"
    training = functional["variants"][0]["training"]
    assert training["updates"] == 100
    assert training["windowExecution"]["compiledWindowCalls"] == 100
    assert training["windowExecution"]["eagerWindowCalls"] == 0
    artifact, manifest, _ = read_raw_brain_artifact(files["roundtrip/trained.slither-brain"])
    receipt = json.loads(files["roundtrip/native-parity/receipt.json"])
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
        data = files["roundtrip/" + name]
        assert hashlib.sha256(data).hexdigest() == receipt[digest]
        values.append(torch.from_numpy(np.frombuffer(data, dtype="<f4").copy().reshape(256, width)))
    actual, states = _sequence(model, values[0], model.initial_state(1))
    _compare(actual, values[1], "cpu")
    _compare(states, values[2], "cpu")
