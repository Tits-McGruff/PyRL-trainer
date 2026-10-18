"""Capacity qualification uses production admission and real canonical updates."""

import json
from pathlib import Path
import sys
import subprocess

import pytest

from pyrl_trainer import distillation_capacity_gate as gate

pytestmark = pytest.mark.unit
GRAPH = Path(__file__).resolve().parents[1] / "fixtures/distillation_graph_v1"


def arguments(output, *choices):
    """Use an independently Rust-compiled graph, without HTTP or synthetic codecs."""
    return ["capacity", "--manifest", str(GRAPH / "manifest.json"),
            "--weights", str(GRAPH / "weights.f32"), "--output", str(output),
            "--device", "cpu", "--updates", "1", *choices]


def test_capacity_runs_real_optimizer_and_preserves_create_new_report(tmp_path, monkeypatch):
    """A bounded full window updates weights and records the exact admitted source."""
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", arguments(
        output, "--batch-sequences", "2", "--sequence-length", "16", "--burn-in", "0"))
    report = gate.run()
    assert report["status"] == "passed" and report["qualificationComplete"]
    assert report["initialWeightsSha256"] != report["trainedWeightsSha256"]
    assert report["updates"][0]["training_samples"] == 32
    assert report["updates"][0]["optimizer_updated"]
    assert json.loads(output.read_text(encoding="utf-8"))["trainerSourceSha256"]
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        gate.run()
    assert output.read_bytes() == original


def test_capacity_rejects_over_budget_before_device_or_tensor_allocation(tmp_path, monkeypatch):
    """Rejected work records an honest rejection without touching CUDA or the model."""
    output = tmp_path / "rejected.json"
    monkeypatch.setattr(sys, "argv", arguments(
        output, "--batch-sequences", "256", "--sequence-length", "256", "--burn-in", "0",
        "--device-memory-mib", "512"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("rejected work reached device/model allocation")

    monkeypatch.setattr(gate, "admit_job_device", forbidden)
    monkeypatch.setattr(gate, "CanonicalGraph", forbidden)
    report = gate.run()
    assert report["status"] == "rejected"
    assert not report["admitted"] and not report["qualificationComplete"]
    assert not report["updates"] and "exceeds device admission" in report["rejection"]


@pytest.mark.parametrize("module", ["distillation_capacity_gate",
                                    "distillation_replay_capacity_gate"])
def test_capacity_cannot_qualify_with_disabled_assertions(tmp_path, module):
    """An optimized interpreter rejects before inputs/output or device admission."""
    result = subprocess.run(
        [sys.executable, "-B", "-O", "-m", "pyrl_trainer." + module],
        capture_output=True, text=True, check=False, timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode != 0
    assert "qualification requires Python assertions enabled" in result.stderr
    assert not list(tmp_path.iterdir())
