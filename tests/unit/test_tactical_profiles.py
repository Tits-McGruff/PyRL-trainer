"""Bind variable tactical widths to real Rust corpus and graph receipts."""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraph, CanonicalGraphError, CanonicalGraphManifest
from pyrl_trainer.sequence_distillation import CanonicalSequenceTrainer, SequenceTrainingConfig
from pyrl_trainer.tactical_ablation import (
    ablation_graph, evaluate_agreement, paired_windows, window_batch,
)
from pyrl_trainer.tactical_ablation_gate import AblationStudy
from pyrl_trainer.tactical_contract import policies_for_profile, variants_for_profile
from pyrl_trainer.tactical_corpus import TacticalCorpusError, load_tactical_corpus

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parents[1] / "fixtures/tactical_profiles_v1"


@pytest.fixture(name="profile_case", params=["basic-b8", "motion-b32"])
def native_profile_case(request, tmp_path):
    """Verify exact native files before exposing one isolated mutation workspace."""
    case = request.param
    receipt = json.loads((FIXTURE / "receipt.json").read_bytes())["cases"][case]
    for name, expected in receipt["files"].items():
        path = FIXTURE / case / (name + ".gz")
        assert path.stat().st_size <= 2 * 1024**2
        encoded = path.read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == expected["gzipSha256"]
        assert 0 < expected["bytes"] <= 2 * 1024**2
        with gzip.open(path, "rb") as stream:
            raw = stream.read(expected["bytes"] + 1)
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return tmp_path, receipt


def test_actual_native_contract_controls_arrays_graph_and_whole_seed_ownership(profile_case):
    """Both profile widths use their own real labels without changing the legacy fixture."""
    root, receipt = profile_case
    corpus = load_tactical_corpus(root)
    manifest = CanonicalGraphManifest.from_bytes(
        (root / "initialized/manifest.json").read_bytes())
    assert corpus.sensor.profile == receipt["profile"]
    assert corpus.observations.shape == (receipt["rows"], receipt["width"])
    assert manifest.input_size == receipt["width"]
    assert manifest.sensor_sha256 == corpus.sensor.structural_sha256
    assert corpus.source_sha256 == receipt["nativeSourceSha256"]
    assert not corpus.observations.flags.writeable
    training, validation = corpus.split_by_seed({2})
    assert len(training) == len(validation) == 28
    assert {row.policy for row in corpus.sequences} == set(
        policies_for_profile(corpus.sensor.profile))
    assert {row.seed for row in training} == {1}
    assert {row.seed for row in validation} == {2}


def test_real_profile_windows_update_canonical_parameters_and_evaluate_actual_labels(profile_case):
    """An actual Adam/TBPTT update exercises the dynamic first affine input and recurrence."""
    root, _receipt = profile_case
    corpus = load_tactical_corpus(root)
    manifest = CanonicalGraphManifest.from_bytes(
        (root / "initialized/manifest.json").read_bytes())
    model = CanonicalGraph(manifest, (root / "initialized/weights.f32").read_bytes(), "cpu")
    config = SequenceTrainingConfig(batch_size=2, bptt_steps=16, burn_in_steps=0)
    training, validation = corpus.split_by_seed({2})
    windows = next(paired_windows(corpus, training, config, 1, 9))
    batch, actual = window_batch(corpus, manifest, windows, config)
    assert batch.observations.shape == (16, 2, corpus.sensor.sensor_count)
    for column, window in enumerate(windows):
        sequence = corpus.sequences[window.sequence]
        count = min(16, sequence.rows)
        np.testing.assert_array_equal(
            batch.observations[:count, column].numpy().view(np.uint32),
            corpus.observations[sequence.row_offset:sequence.row_offset + count].view(np.uint32),
        )
    before = model.export_weights()
    trainer = CanonicalSequenceTrainer(model, actual)
    result = trainer.update(batch)
    assert result.optimizer_updated and result.training_samples > 0
    assert trainer.completed_updates == 1
    assert model.export_weights() != before
    agreement = evaluate_agreement(model, corpus, validation)
    assert agreement["rows"] == sum(sequence.rows for sequence in validation)
    assert set(agreement["perPolicy"]) == set(policies_for_profile(corpus.sensor.profile))


def test_profile_or_input_reinterpretation_rejects_before_tensor_exposure(profile_case):
    """A self-hashed changed profile still cannot reinterpret another Rust contract."""
    root, _receipt = profile_case
    metadata = json.loads((root / "corpus.json").read_bytes())
    canonical = json.loads(metadata["sensorCanonicalJson"])
    canonical["profile"] = "motion" if canonical["profile"] == "basic" else "basic"
    metadata["sensorCanonicalJson"] = json.dumps(canonical, separators=(",", ":"))
    metadata["sensorSha256"] = hashlib.sha256(
        b"slither.sensor-contract.v1" + bytes((0,)) + metadata["sensorCanonicalJson"].encode(),
    ).hexdigest()
    (root / "corpus.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(TacticalCorpusError, match="sensor"):
        load_tactical_corpus(root)


def test_motion_b32_ablation_ports_match_exact_native_channel_labels(profile_case):
    """Port routing retains Rust's complete channel order at the maximum width."""
    root, receipt = profile_case
    corpus = load_tactical_corpus(root)
    if receipt["profile"] == "basic":
        assert variants_for_profile("basic") == ("full",)
        for variant in ("no-bearing", "no-heading", "core-only"):
            with pytest.raises(CanonicalGraphError):
                ablation_graph(variant, bins=8, profile="basic")
        return
    prefixes = {
        "no-bearing": ("head_bearing_residual_",),
        "no-heading": ("head_heading_sin_", "head_heading_cos_"),
        "no-velocity": ("head_velocity_forward_", "head_velocity_lateral_"),
        "no-urgency": ("head_collision_urgency_",),
        "no-density": ("head_density_",),
    }
    for variant, omitted_prefixes in prefixes.items():
        graph = ablation_graph(variant, bins=corpus.sensor.bins, profile="motion")
        ports = [edge["fromPort"] for edge in graph["edges"] if edge["to"] == "selected"]
        actual = [name for port in range(1, 16) if port not in ports
                  for name in corpus.sensor.order[24 + (port - 1) * 32:24 + port * 32]]
        assert actual == [name for name in corpus.sensor.order if name.startswith(omitted_prefixes)]
        assert next(node for node in graph["nodes"] if node["id"] == "encode")["inputSize"] == (
            corpus.sensor.sensor_count - len(actual))
        assert graph["nodes"][0]["outputSize"] == 504


@pytest.mark.parametrize("profile", ["basic", "motion"])
@pytest.mark.parametrize("bins", range(8, 33))
def test_declared_full_graph_width_range_and_basic_missing_groups(profile, bins):
    """Every admitted count keeps the complete canonical profile and two-output motor."""
    graph = ablation_graph("full", bins=bins, profile=profile)
    width = 24 + (6 if profile == "basic" else 15) * bins
    assert graph["nodes"][0]["outputSize"] == width
    assert next(node for node in graph["nodes"] if node["id"] == "encode")["inputSize"] == width
    assert graph["nodes"][-1]["outputSize"] == 2
    assert all(node["type"] != "Split" for node in graph["nodes"])


@pytest.mark.parametrize("bins", [True, 7, 33, 16.0, None])
def test_invalid_graph_widths_reject(bins):
    """The graph helper cannot silently coerce a different sensor contract."""
    with pytest.raises(CanonicalGraphError):
        ablation_graph("full", bins=bins)


def test_basic_study_rejects_nonexistent_motion_groups_before_device_or_output(profile_case,
                                                                              monkeypatch):
    """Unavailable ablations never allocate a trainer or claim a study directory."""
    root, receipt = profile_case
    if receipt["profile"] != "basic":
        return
    def forbidden(_job):
        """Expose accidental allocation before profile/variant admission."""
        raise AssertionError("device admission reached")
    monkeypatch.setattr("pyrl_trainer.tactical_ablation_gate.admit_job_device", forbidden)
    args = SimpleNamespace(updates=1, validation_update=(), corpus=root, variant=["no-heading"])
    with pytest.raises(ValueError, match="unavailable"):
        AblationStudy(args)



def test_profile_study_report_binds_native_contract_and_default_variant_set(profile_case,
                                                                         monkeypatch):
    """CI and downstream readers receive the exact declared profile/width in each report."""
    root, receipt = profile_case
    canonical = root / "canonical-runner"
    tactical = root / "tactical-runner"
    canonical.write_bytes(b"fixture runner identity only")
    tactical.write_bytes(b"fixture tactical identity only")
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    args = SimpleNamespace(
        updates=2, validation_update=(1,), cuda_graphs=False, corpus=root, variant=None,
        allow_dirty=True,
        validation_seed=2, batch_sequences=2, sequence_length=16, burn_in=0,
        seed=1, device="cpu", cpu_threads=1, device_memory_mib=512,
        canonical_runner=canonical, tactical_runner=tactical, directory=root / "study",
        native_steps=12, native_workers=1, deadline_seconds=60,
    )
    study = AblationStudy(args)
    config = study.report["config"]
    assert config["sensorProfile"] == receipt["profile"]
    assert config["sensorBins"] == study.corpus.sensor.bins
    assert tuple(args.variant) == variants_for_profile(receipt["profile"])
    assert study.report["measurementComplete"] is False
    assert study.report["releaseQualified"] is False
