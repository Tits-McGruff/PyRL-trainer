"""Portable brain bytes and actual 250-update candidate snapshots."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import os

import pytest

from pyrl_trainer.brain_artifact import encode_brain_artifact, read_raw_brain_artifact
from pyrl_trainer.canonical_graph import CanonicalGraphManifest
from pyrl_trainer.distillation_checkpoint import load_job_checkpoint
from pyrl_trainer.distillation_job import DistillationJob, DistillationJobConfig
from pyrl_trainer.distillation_transport import DistillationContractError

SPEC = importlib.util.spec_from_file_location(
    "candidate_fixture_helpers", Path(__file__).with_name("test_distillation_sampling.py")
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
pytestmark = pytest.mark.unit


def artifact():
    """Export the actual native-reference graph and canonical Float32 weights."""
    manifest = CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json")
    weights = (HELPERS.GRAPH / "weights.f32").read_bytes()
    return encode_brain_artifact(
        manifest,
        weights,
        {"schemaVersion": 1, "sourceKind": "manual", "details": {"label": "native reference"}},
    )


def test_portable_raw_weights_and_identity_survive_exact_byte_roundtrip():
    """No Python serialization or sigmoid/value head enters the portable artifact."""
    source = artifact()
    decoded, manifest, provenance = read_raw_brain_artifact(source.data)
    assert decoded == source
    assert decoded.sha256 == hashlib.sha256(source.data).hexdigest()
    assert manifest.input_size == 72 and manifest.control_interval_seconds == 0.06
    assert provenance["sourceKind"] == "manual"
    assert source.data[:8] == b"SLTHBRN1"


def test_artifact_boundaries_reserved_fields_digest_and_finite_values_reject():
    """Truncated/ambiguous metadata and repacked nonfinite weights cannot be handed off."""
    source = artifact()
    for size in (0, 7, 87, 88, len(source.data) - 1):
        with pytest.raises(DistillationContractError):
            read_raw_brain_artifact(source.data[:size])
    for offset in (0, 12, 13, 16, 20, 24, 56, len(source.data) - 1):
        damaged = bytearray(source.data)
        damaged[offset] ^= 255
        with pytest.raises(DistillationContractError):
            read_raw_brain_artifact(bytes(damaged))
    manifest = CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json")
    weights = bytearray((HELPERS.GRAPH / "weights.f32").read_bytes())
    struct.pack_into("<f", weights, 0, float("nan"))
    with pytest.raises(DistillationContractError, match="finite"):
        encode_brain_artifact(
            manifest, bytes(weights), {"schemaVersion": 1, "sourceKind": "manual", "details": {}}
        )


def test_real_250th_optimizer_update_freezes_and_checkpoints_an_immutable_candidate(tmp_path):
    """Exercise the actual cadence, rather than assigning a fake update counter."""
    manifest = CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json")
    job = DistillationJob(
        manifest,
        (HELPERS.GRAPH / "weights.f32").read_bytes(),
        HELPERS.replay_fixture(),
        config=DistillationJobConfig(device="cpu"),
        sequence=HELPERS.config(),
    )
    for _step in range(249):
        assert job.update().optimizer_updated
    assert not job.pending_candidates
    job.update()
    assert job.trainer.completed_updates == 250 and len(job.pending_candidates) == 1
    frozen = job.pending_candidates[0]
    _candidate, _manifest, provenance = read_raw_brain_artifact(frozen.data)
    assert provenance["details"]["updates"] == "00000000000000fa"
    assert int(provenance["details"]["trainingSamples"], 16) == job.training_samples
    job.update()
    assert job.candidate_due and job.pending_candidates[0] == frozen
    assert (
        read_raw_brain_artifact(frozen.data)[0].weights_sha256
        != hashlib.sha256(job.model.export_weights()).hexdigest()
    )
    restored = load_job_checkpoint(job.save(tmp_path))
    assert restored.pending_candidates == job.pending_candidates
    assert restored.training_samples == job.training_samples
    assert restored.trained_generation_min == job.trained_generation_min
    assert restored.acknowledge_candidate(frozen.sha256)
    assert not restored.candidate_due
    assert not restored.acknowledge_candidate(frozen.sha256)
    if os.environ.get("PYRL_BRAIN_ARTIFACT_RUNNER"):
        path = tmp_path / "trained.slither-brain"
        path.write_bytes(frozen.data)
        subprocess.run(
            [os.environ["PYRL_BRAIN_ARTIFACT_RUNNER"], str(path), str(tmp_path / "native-trained")],
            check=True,
        )
        admitted = json.loads(
            (tmp_path / "native-trained/descriptor.json").read_text(encoding="utf-8")
        )
        assert admitted["descriptor"]["artifactSha256"] == frozen.sha256
        assert admitted["descriptor"]["weightsSha256"] == frozen.weights_sha256


@pytest.mark.skipif(
    not os.environ.get("PYRL_BRAIN_ARTIFACT_RUNNER"),
    reason="native artifact tool is required in its CI job",
)
def test_actual_rust_compiler_admits_portable_bytes_and_rejects_a_forged_packing_contract(tmp_path):
    """Rust recompiles the graph and rejects metadata despite repaired header digests."""
    source = artifact()
    path = tmp_path / "reference.slither-brain"
    path.write_bytes(source.data)
    subprocess.run(
        [os.environ["PYRL_BRAIN_ARTIFACT_RUNNER"], str(path), str(tmp_path / "native")], check=True
    )
    native = json.loads((tmp_path / "native/descriptor.json").read_text(encoding="utf-8"))
    assert native["descriptor"]["artifactSha256"] == source.sha256
    assert (tmp_path / "native/weights.f32").read_bytes() == (
        HELPERS.GRAPH / "weights.f32"
    ).read_bytes()
    json_size = int.from_bytes(source.data[8:12], "little")
    metadata = json.loads(source.data[88 : 88 + json_size])
    metadata["graphManifest"]["nodes"][1]["parameterBlocks"][0]["offset"] += 1
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    repaired = bytearray(source.data[:88])
    repaired[8:12] = len(encoded).to_bytes(4, "little")
    bad = tmp_path / "forged.slither-brain"
    bad.write_bytes(bytes(repaired) + encoded + source.data[88 + json_size :])
    rejected = subprocess.run(
        [os.environ["PYRL_BRAIN_ARTIFACT_RUNNER"], str(bad), str(tmp_path / "rejected")],
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0 and not (tmp_path / "rejected").exists()
