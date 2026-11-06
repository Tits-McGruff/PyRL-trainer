"""Production-sized canonical graph, portable handoff and full Adam restart."""

import copy
import hashlib
import json
import os
import subprocess
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from slither_contracts import CONSTANTS
from test_distillation_job import HELPERS, assert_optimizer_equal

from pyrl_trainer.brain_artifact import encode_brain_artifact, read_raw_brain_artifact
from pyrl_trainer.canonical_graph import CanonicalGraphError, CanonicalGraphManifest
from pyrl_trainer.distillation_checkpoint import MEMBER_LIMITS, load_job_checkpoint
from pyrl_trainer.distillation_client import DistillationClient, DistillationFetch
from pyrl_trainer.distillation_job import DistillationJob, DistillationJobConfig
from pyrl_trainer.distillation_replay import TeacherReplay
from pyrl_trainer.distillation_transport import DatasetManifest, DistillationContractError
from pyrl_trainer.sequence_distillation import SequenceTrainingConfig

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/production-large-manifest.json"


def large_manifest():
    """Small metadata emitted by the production Rust compiler, without stored weights."""
    return CanonicalGraphManifest.read(FIXTURE)


def large_job():
    """A matching empty diagnostic replay and synthetic finite weights, no teacher recapture."""
    manifest = large_manifest()
    raw = copy.deepcopy(HELPERS.replay_fixture().manifest.raw)
    raw["contract"].update(
        sensorSha256=manifest.sensor_sha256, inputSize=264, controlIntervalSeconds=1 / 60
    )
    raw["sensorContractJson"] = json.loads(manifest.canonical_json)["sensorCanonicalJson"]
    for key in ("productionWatermark", "storageWatermark", "resolvedThrough", "retainedBytes"):
        raw[key] = "0" * 16
    raw["retainedChunks"] = 0
    replay = TeacherReplay()
    replay.admit(DistillationFetch(DatasetManifest.from_object(raw), 0, 0, (), ()))
    weights = (
        (np.arange(manifest.total_parameters, dtype=np.int32) % 257 - 128).astype("<f4") / 65536
    ).tobytes()
    return DistillationJob(
        manifest,
        weights,
        replay,
        config=DistillationJobConfig(device="cpu"),
        sequence=SequenceTrainingConfig(batch_size=1, bptt_steps=1, burn_in_steps=0),
    )


def test_large_manifest_portable_bytes_digests_and_receipt(monkeypatch):
    """Native graph identities and all 6,984,416 weight bytes survive the portable path."""
    manifest = large_manifest()
    assert (manifest.total_parameters, manifest.total_state_size, manifest.input_size) == (
        1746104,
        792,
        264,
    )
    assert (
        manifest.sensor_sha256 == "fa525c20998440a3eec3a3f1ed944760ed1c26650d8819608efa0cd44f83893e"
    )
    weights = bytes(manifest.total_parameters * 4)
    artifact = encode_brain_artifact(
        manifest, weights, {"schemaVersion": 1, "sourceKind": "distilled", "details": {}}
    )
    decoded, restored, _ = read_raw_brain_artifact(artifact.data)
    assert decoded == artifact and restored == manifest
    assert artifact.weights_sha256 == hashlib.sha256(weights).hexdigest()
    graph = json.loads(manifest.canonical_json)
    descriptor = {
        "schemaVersion": 1,
        "artifactSha256": artifact.sha256,
        "weightsSha256": artifact.weights_sha256,
        "architectureKey": graph["architectureKey"],
        "layoutSha256": manifest.layout_sha256,
        "sensorSha256": manifest.sensor_sha256,
        "inputSize": 264,
        "parameterCount": 1746104,
        "stateSize": 792,
        "controlIntervalSeconds": 1 / 60,
        "lifeStateRule": "reset",
        "sourceKind": "distilled",
    }
    client = DistillationClient("http://localhost", "11" * 16)
    monkeypatch.setattr(
        client,
        "_http",
        lambda *_args, **_kwargs: (
            json.dumps({"descriptor": descriptor, "byteCount": len(artifact.data)}).encode(),
            {},
        ),
    )
    assert client.upload_candidate(artifact) == descriptor
    for field in ("parameterCount", "sensorSha256", "controlIntervalSeconds"):
        changed = descriptor.copy()
        changed[field] = 1 if field != "sensorSha256" else "00" * 32
        monkeypatch.setattr(
            client,
            "_http",
            lambda *_args, descriptor=changed, **_kwargs: (
                json.dumps({"descriptor": descriptor, "byteCount": len(artifact.data)}).encode(),
                {},
            ),
        )
        with pytest.raises(DistillationContractError):
            client.upload_candidate(artifact)
    for offset in (24, 56, len(artifact.data) - 1):
        bad = bytearray(artifact.data)
        bad[offset] ^= 1
        with pytest.raises(DistillationContractError):
            read_raw_brain_artifact(bytes(bad))
    with pytest.raises(DistillationContractError):
        read_raw_brain_artifact(artifact.data[:-1])
    bad = bytearray(artifact.data)
    struct.pack_into("<I", bad, 16, CONSTANTS["maxBrainParameters"] + 1)
    with pytest.raises(DistillationContractError):
        read_raw_brain_artifact(bytes(bad))
    bad = bytearray(weights)
    struct.pack_into("<f", bad, 0, float("nan"))
    with pytest.raises(DistillationContractError):
        encode_brain_artifact(
            manifest, bytes(bad), {"schemaVersion": 1, "sourceKind": "manual", "details": {}}
        )


def test_large_checkpoint_preserves_full_adam_rng_replay_and_pending_candidate(tmp_path):
    """Synthetic gradients populate real full-size Adam moments and continue bit exactly."""
    job = large_job()
    parameter = job.model.flat_parameters
    parameter.grad = torch.full_like(parameter, 0.0001)
    job.trainer.optimizer.step()
    job.trainer.completed_updates = 250
    job.attempted_updates = 250
    job.pending_candidates = [
        encode_brain_artifact(
            job.model.manifest,
            job.model.export_weights(),
            {
                "schemaVersion": 1,
                "sourceKind": "distilled",
                "details": {"jobId": job.job_id, "updates": "00000000000000fa"},
            },
        )
    ]
    job.candidate_due = True
    torch.rand(9, generator=job.cpu_rng)
    job.python_rng.random()
    saved = job.save(tmp_path)
    restored = load_job_checkpoint(saved)
    assert restored.model.export_weights() == job.model.export_weights()
    assert_optimizer_equal(
        job.trainer.optimizer.state_dict(), restored.trainer.optimizer.state_dict()
    )
    assert restored.pending_candidates == job.pending_candidates
    assert restored.sampler.checkpoint_fields() == job.sampler.checkpoint_fields()
    assert restored.python_rng.random() == job.python_rng.random()
    assert torch.equal(
        torch.rand(9, generator=restored.cpu_rng), torch.rand(9, generator=job.cpu_rng)
    )
    for owner in (job, restored):
        owner.model.flat_parameters.grad = torch.full_like(parameter, -0.0002)
        owner.trainer.optimizer.step()
    assert restored.model.export_weights() == job.model.export_weights()
    assert_optimizer_equal(
        job.trainer.optimizer.state_dict(), restored.trainer.optimizer.state_dict()
    )


def test_large_parser_rejects_count_and_shape_before_unsafe_allocation():
    """Ceilings apply to totals, individual tensor dimensions and strided addressing."""
    raw = json.loads(FIXTURE.read_bytes())
    for field, value in (("totalParameters", 2000001), ("totalStateSize", 16385)):
        changed = copy.deepcopy(raw)
        changed[field] = value
        with pytest.raises(CanonicalGraphError):
            CanonicalGraphManifest.from_bytes(json.dumps(changed).encode())
    changed = copy.deepcopy(raw)
    next(node for node in changed["nodes"] if node["parameterBlocks"])["parameterBlocks"][0][
        "shape"
    ][0] = 4097
    with pytest.raises(CanonicalGraphError):
        CanonicalGraphManifest.from_bytes(json.dumps(changed).encode())


def test_exact_two_million_parameter_manifest_and_portable_roundtrip():
    """A real Rust-compiled graph reaches the ceiling, rather than faking a header count."""
    manifest = CanonicalGraphManifest.read(FIXTURE.with_name("ceiling-brain-manifest.json"))
    assert manifest.total_parameters == CONSTANTS["maxBrainParameters"]
    weights = bytes(CONSTANTS["maxBrainWeightBytes"])
    artifact = encode_brain_artifact(
        manifest, weights, {"schemaVersion": 1, "sourceKind": "manual", "details": {}}
    )
    assert read_raw_brain_artifact(artifact.data)[0] == artifact
    changed = json.loads(manifest.canonical_json)
    changed["totalParameters"] += 1
    with pytest.raises(CanonicalGraphError):
        CanonicalGraphManifest.from_bytes(json.dumps(changed).encode())


def test_checkpoint_role_budgets_cover_full_graph_and_sequence_dimensions():
    """Raw weights, both Adam moments and each admitted window state fit their roles."""
    config = SequenceTrainingConfig(batch_size=256)
    assert MEMBER_LIMITS["weights.f32"] == 8_000_000
    assert MEMBER_LIMITS["optimizer.pt"] >= 2 * 8_000_000 + 1024**2
    assert MEMBER_LIMITS["state.f32"] >= config.batch_size * CONSTANTS["maxBrainStateFloats"] * 4


@pytest.mark.skipif(
    not os.environ.get("PYRL_BRAIN_ARTIFACT_RUNNER"),
    reason="current native artifact tool runs in interoperability CI",
)
@pytest.mark.parametrize("name", ["production-large-manifest.json", "ceiling-brain-manifest.json"])
def test_large_python_export_is_independently_recompiled_by_rust(tmp_path, name):
    """Both production and ceiling artifacts roundtrip exact bits through native validation."""
    manifest = CanonicalGraphManifest.read(FIXTURE.with_name(name))
    weights = (
        (np.arange(manifest.total_parameters, dtype=np.int32) % 257 - 128).astype("<f4") / 65536
    ).tobytes()
    artifact = encode_brain_artifact(
        manifest,
        weights,
        {"schemaVersion": 1, "sourceKind": "distilled", "details": {"fixture": name}},
    )
    source = tmp_path / "large.slither-brain"
    source.write_bytes(artifact.data)
    target = tmp_path / "native"
    subprocess.run([os.environ["PYRL_BRAIN_ARTIFACT_RUNNER"], str(source), str(target)], check=True)
    receipt = json.loads((target / "descriptor.json").read_bytes())["descriptor"]
    assert receipt["artifactSha256"] == artifact.sha256
    assert receipt["weightsSha256"] == artifact.weights_sha256
    assert receipt["parameterCount"] == manifest.total_parameters
    assert (target / "weights.f32").read_bytes() == weights
