"""Self-contained canonical job restarts, atomic publication and isolation."""

from pathlib import Path
import math
import copy
import importlib.util
import random
import zipfile

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraphManifest
from pyrl_trainer.distillation_checkpoint import load_job_checkpoint
from pyrl_trainer.distillation_client import (
    DistillationClient,
    DistillationFetch,
    DistillationHTTPError,
)
from pyrl_trainer.distillation_job import (
    DistillationJob,
    DistillationJobConfig,
    trainer_source_sha256,
)
from pyrl_trainer.distillation_transport import DistillationContractError

# Load test-only fixture helpers explicitly; no production module imports a test codec.
SPEC = importlib.util.spec_from_file_location(
    "sampling_fixture_helpers", Path(__file__).with_name("test_distillation_sampling.py")
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
pytestmark = pytest.mark.unit
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def make_job(device="cpu", **kwargs):
    """Construct an actual native-reference GRU job with isolated deterministic seed."""
    manifest = CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json")
    return DistillationJob(
        manifest,
        (HELPERS.GRAPH / "weights.f32").read_bytes(),
        HELPERS.replay_fixture(),
        config=DistillationJobConfig(device=device),
        sequence=HELPERS.config(),
        **kwargs,
    )


def assert_optimizer_equal(left, right):
    """Check every Adam moment and scalar, rather than comparing only canonical weights."""
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for identity, state in left["state"].items():
        for name, value in state.items():
            other = right["state"][identity][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)
            else:
                assert value == other


@pytest.mark.parametrize("device", DEVICES)
def test_exact_job_restart_reproduces_optimizer_rng_next_sample_and_update(tmp_path, device):
    """One restored owner reproduces the very next update, including Adam and recurrence bits."""
    job = make_job(device)
    job.update()
    job.validate()
    job.python_rng.random()
    torch.rand(7, generator=job.cpu_rng)
    if job.cuda_rng is not None:
        torch.rand(7, generator=job.cuda_rng, device=job.device)
    saved = job.save(tmp_path)
    expected = job.update()
    expected_weights = job.model.export_weights()
    expected_optimizer = copy.deepcopy(job.trainer.optimizer.state_dict())
    expected_sampler = job.sampler.checkpoint_fields()
    expected_python = job.python_rng.random()
    expected_cpu = torch.rand(5, generator=job.cpu_rng)
    expected_cuda = (
        None if job.cuda_rng is None else torch.rand(5, generator=job.cuda_rng, device=job.device)
    )
    restored = load_job_checkpoint(saved)
    actual = restored.update()
    assert restored.model.export_weights() == expected_weights
    assert_optimizer_equal(restored.trainer.optimizer.state_dict(), expected_optimizer)
    assert restored.sampler.checkpoint_fields() == expected_sampler
    assert restored.python_rng.random() == expected_python
    assert torch.equal(torch.rand(5, generator=restored.cpu_rng), expected_cpu)
    if expected_cuda is not None:
        assert torch.equal(
            torch.rand(5, generator=restored.cuda_rng, device=restored.device), expected_cuda
        )
    assert torch.equal(actual.next_state, expected.next_state)
    assert restored.last_windows == job.last_windows
    assert restored.trainer.completed_updates == job.trainer.completed_updates == 2
    assert actual.loss == expected.loss
    assert restored.last_validation == job.last_validation
    assert restored.peak_device_bytes <= restored.config.device_bytes


def test_checkpoints_rotate_three_valid_and_ignore_incomplete_temp_files(tmp_path):
    """Retention follows complete valid files, with interrupted publications never counted."""
    job = make_job()
    paths = []
    for _index in range(4):
        job.update()
        paths.append(job.save(tmp_path))
    assert not paths[0].exists()
    assert all(path.exists() for path in paths[1:])
    scratch = paths[-1].parent / ".job-interrupted.tmp"
    scratch.write_bytes(b"partial")
    job.save(tmp_path)
    assert scratch.read_bytes() == b"partial"
    assert len(list(paths[-1].parent.glob("job-*.distill-job"))) == 3


def test_missing_required_replay_fails_offline_restart_explicitly(tmp_path):
    """A valid-looking snapshot cannot reconstruct or silently skip absent teacher observations."""
    job = make_job()
    job.update()
    saved = job.save(tmp_path)
    broken = tmp_path / "missing-data.distill-job"
    with zipfile.ZipFile(saved) as source, zipfile.ZipFile(broken, "w") as target:
        skipped = False
        for item in source.infolist():
            if item.filename.startswith("chunks/") and not skipped:
                skipped = True
                continue
            target.writestr(item.filename, source.read(item.filename))
    with pytest.raises(DistillationContractError, match="count|missing"):
        load_job_checkpoint(broken)
    assert job.trainer.completed_updates == 1


def test_failed_publication_preserves_last_valid_checkpoint(tmp_path, monkeypatch):
    """A disk/publication failure never overwrites the prior durable optimizer/RNG snapshot."""
    job = make_job()
    saved = job.save(tmp_path)
    frozen = saved.read_bytes()

    def fail_link(*_args):
        raise OSError("injected exclusive publication failure")

    monkeypatch.setattr("pyrl_trainer.distillation_checkpoint.os.link", fail_link)
    with pytest.raises(OSError, match="publication failure"):
        job.save(tmp_path)
    assert saved.read_bytes() == frozen
    assert not list(saved.parent.glob(".job-*.tmp"))
    assert load_job_checkpoint(saved).trainer.completed_updates == 0


def test_job_rng_isolation_waiting_and_failure_pause(tmp_path, monkeypatch):
    """Owned RNG draws preserve global states; device failures pause this owner."""
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    job = make_job()
    job.update()
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert numpy_state[0] == current[0] and np.array_equal(numpy_state[1], current[1])
    assert numpy_state[2:] == current[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    saved = job.save(tmp_path)

    def fail_update(_batch):
        raise RuntimeError("injected device allocation failure")

    monkeypatch.setattr(job.trainer, "update", fail_update)
    with pytest.raises(RuntimeError, match="allocation failure"):
        job.update()
    assert job.status == "paused" and not job.checkpoint_allowed
    with pytest.raises(DistillationContractError, match="failed trainer"):
        job.save(tmp_path)
    assert load_job_checkpoint(saved).status == "running"


def test_checkpoint_schedule_is_wall_time_or_successful_update_count():
    """The declared 60-second/1,000-update cadence remains independent of game time."""
    job = make_job()
    # This monotonic value reproduces subtraction rounding below 60 on Linux.
    job.last_save_time = 123.539807243
    deadline = job.last_save_time + 60
    assert not job.checkpoint_due(math.nextafter(deadline, -math.inf))
    assert not job.checkpoint_due(job.last_save_time + 59)
    assert job.checkpoint_due(job.last_save_time + 60)
    job.trainer.completed_updates = 1000
    assert job.checkpoint_due(job.last_save_time)


def test_waiting_for_more_history_keeps_job_and_sampler_resumable():
    """Short live buffers wait without advancing draws or disabling future training."""
    manifest = CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json")
    job = DistillationJob(
        manifest,
        (HELPERS.GRAPH / "weights.f32").read_bytes(),
        HELPERS.replay_fixture((1,), chunks_per_stream=1),
        config=DistillationJobConfig(device="cpu"),
        sequence=HELPERS.config(),
    )
    before = job.sampler.checkpoint_fields()
    assert job.update() is None
    assert job.status == "waiting" and job.checkpoint_allowed
    assert job.sampler.checkpoint_fields() == before
    assert job.trainer.completed_updates == 0


def test_http_configuration_restarts_locally_without_join_fetch_or_actions(tmp_path):
    """A checkpoint preserves transport admission settings and needs no remote data to load."""

    replay = HELPERS.replay_fixture()
    client = DistillationClient(
        "http://127.0.0.1:9222",
        replay.manifest.contract.capture_epoch,
        page_size=7,
        owned_bytes=3 * 1024**2,
        timeout=2.0,
        lease_ms=12345,
    )
    client._accept_manifest(replay.manifest.raw)  # pylint: disable=protected-access
    client.cursor = replay.cursor
    job = make_job(client=client)
    job.update()
    restored = load_job_checkpoint(job.save(tmp_path))
    assert restored.client.cursor == restored.replay.cursor == client.cursor
    assert (
        restored.client.page_size,
        restored.client.owned_bytes,
        restored.client.timeout,
        restored.client.lease_ms,
    ) == (7, 3 * 1024**2, 2.0, 12345)


def test_offline_cli_runs_actual_updates_and_publishes_a_restartable_boundary(tmp_path, capsys):
    """The standalone job entry point never requires an actor or live server for retained data."""
    from pyrl_trainer.distillation_run import run  # pylint: disable=import-outside-toplevel

    saved = make_job().save(tmp_path)
    result = run(
        [
            "--resume",
            str(saved),
            "--offline",
            "--updates",
            "2",
            "--job-directory",
            str(tmp_path),
            "--max-wall-seconds",
            "10",
        ]
    )
    assert result["updates"] == 2 and result["reason"] == "update_limit"
    assert load_job_checkpoint(result["checkpoint"]).trainer.completed_updates == 2
    assert '"device": "cpu"' in capsys.readouterr().out


def test_waiting_owner_resumes_after_more_committed_data_arrives():
    """Normal short-history waiting does not permanently disable the live job."""

    manifest = CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json")
    job = DistillationJob(
        manifest,
        (HELPERS.GRAPH / "weights.f32").read_bytes(),
        HELPERS.replay_fixture((1,), chunks_per_stream=1),
        config=DistillationJobConfig(device="cpu"),
        sequence=HELPERS.config(),
    )
    assert job.update() is None
    full = HELPERS.replay_fixture((1,))
    job.ingest(
        DistillationFetch(full.manifest, job.replay.cursor, full.cursor, full.chunks[1:], ())
    )
    assert job.update().optimizer_updated
    assert job.status == "running" and job.trainer.completed_updates == 1


def test_validation_preserves_canonical_weights_optimizer_and_existing_gradients():
    """Validation advances its sampler only; no hidden optimizer update occurs."""
    job = make_job()
    job.update()
    weights = job.model.export_weights()
    optimizer = copy.deepcopy(job.trainer.optimizer.state_dict())
    gradient = job.model.flat_parameters.grad.clone()
    assert job.validate()["samples"] > 0
    assert job.model.export_weights() == weights
    assert_optimizer_equal(job.trainer.optimizer.state_dict(), optimizer)
    assert torch.equal(job.model.flat_parameters.grad, gradient)
    assert job.trainer.completed_updates == 1


def test_checkpoint_source_environment_and_compression_are_independently_admitted(tmp_path):
    """A repaired ZIP CRC cannot override exact-source/environment or payload encoding checks."""
    import json  # pylint: disable=import-outside-toplevel

    saved = make_job().save(tmp_path)
    for change in ("source", "environment", "compression"):
        changed = tmp_path / f"changed-{change}.distill-job"
        compression = zipfile.ZIP_DEFLATED if change == "compression" else zipfile.ZIP_STORED
        with zipfile.ZipFile(saved) as source, zipfile.ZipFile(
            changed, "w", compression=compression
        ) as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "job.json" and change != "compression":
                    record = json.loads(data)
                    if change == "source":
                        record["trainerSourceSha256"] = "00" * 32
                    else:
                        record["environment"]["numpy"] = "unsupported-build"
                    data = json.dumps(record).encode("utf-8")
                target.writestr(item.filename, data)
        with pytest.raises(DistillationContractError, match="source|environment|encoding"):
            load_job_checkpoint(changed)


def test_cli_rejects_changed_resume_seed_and_sequence_settings(tmp_path):
    """Explicit overrides cannot silently alter or be ignored during an exact job restart."""
    from pyrl_trainer.distillation_run import run  # pylint: disable=import-outside-toplevel

    saved = make_job().save(tmp_path)
    for options in (("--seed", "99"), ("--batch-size", "1"), ("--device", "cuda:0")):
        with pytest.raises(DistillationContractError, match="cannot change"):
            run(
                [
                    "--resume",
                    str(saved),
                    "--offline",
                    "--updates",
                    "1",
                    "--job-directory",
                    str(tmp_path),
                    *options,
                ]
            )


def test_loaded_source_identity_does_not_relabel_a_live_process(monkeypatch):
    """A running job stays bound to its loaded implementation after filesystem/source changes."""
    original = trainer_source_sha256()
    monkeypatch.setattr("pyrl_trainer.distillation_job._compute_source_sha256", lambda: "00" * 32)
    assert trainer_source_sha256() == original


@pytest.mark.parametrize("device", DEVICES)
def test_shadow_request_is_saved_before_activation_and_lost_reply_restart_reuses_exact_segment(
    tmp_path, monkeypatch, device
):
    """Real updates freeze bytes; a synthetic HTTP reply loss cannot reset the native segment."""

    job = make_job(device)
    client = DistillationClient("http://localhost:8765", job.replay.manifest.contract.capture_epoch)
    client.manifest, client.cursor = job.replay.manifest, job.replay.cursor
    job.client = client
    job.configure_shadow(0, 1, "0000000000000001")
    for _ in range(250):
        assert job.update().optimizer_updated
    artifact = job.pending_candidates[0]
    requests, receipts = [], {}
    metric = {"schemaVersion": 1, "processEpoch": "13" * 16, "slots": [None] * 4}
    fail_once = [True]
    monkeypatch.setattr(client, "upload_candidate", lambda _artifact: {})

    def json_reply(path, **kwargs):
        """Fake only HTTP outcomes; checkpoint/optimizer/RNG/canonical export remain real."""
        if path == "shadows":
            return copy.deepcopy(metric)
        assert path == "shadows/0/activate"
        payload = kwargs["payload"]
        requests.append(copy.deepcopy(payload))
        receipts.setdefault(
            payload["requestId"],
            {
                "schemaVersion": 1,
                "processEpoch": "13" * 16,
                "requestId": payload["requestId"],
                "commandSequence": "0000000000000002",
                "activationSequence": "0000000000000002",
                "previousActivationSequence": payload["expectedActivationSequence"],
                "runId": payload["expectedRunId"],
                "configRevision": payload["expectedConfigRevision"],
                "captureEpoch": payload["captureEpoch"],
                "artifactSha256": payload["artifactSha256"],
                "shadowSlot": 0,
                "baselineSlot": 1,
                "effectiveStep": "000000000000000c",
                "enabled": True,
                "acknowledgedAt": 1000,
            },
        )
        if fail_once[0]:
            fail_once[0] = False
            raise DistillationHTTPError(0, "injected lost native receipt")
        return copy.deepcopy(receipts[payload["requestId"]])

    monkeypatch.setattr(client, "_json", json_reply)
    assert job.stage_pending_shadow()
    pending = copy.deepcopy(job.online_metrics["pending"])
    with pytest.raises(DistillationContractError, match="committed.*checkpoint"):
        job.activate_pending_shadow()
    assert not requests
    checkpoint = job.save(tmp_path)
    job.online_metrics["pending"]["expectedActivationSequence"] = "0000000000000001"
    with pytest.raises(DistillationContractError, match="committed.*checkpoint"):
        job.activate_pending_shadow()
    job.online_metrics["pending"]["expectedActivationSequence"] = pending[
        "expectedActivationSequence"
    ]
    with pytest.raises(DistillationHTTPError, match="lost native"):
        job.activate_pending_shadow()
    assert job.online_metrics["pending"] == pending
    assert job.pending_candidates[0] == artifact
    restored = load_job_checkpoint(checkpoint)
    restored.client = client
    received = restored.activate_pending_shadow()
    assert received["requestId"] == pending["requestId"]
    assert requests[0] == requests[1] and len(receipts) == 1
    assert not restored.pending_candidates and restored.online_metrics["pending"] is None
    assert restored.model.export_weights() == job.model.export_weights()
    assert_optimizer_equal(
        restored.trainer.optimizer.state_dict(), job.trainer.optimizer.state_dict()
    )
    saved = restored.save(tmp_path)
    again = load_job_checkpoint(saved)
    assert again.online_metrics == restored.online_metrics
    assert again.trainer.completed_updates == 250
    assert again.update().optimizer_updated
    expected = restored.update()
    assert (
        expected.optimizer_updated
        and again.model.export_weights() == restored.model.export_weights()
    )


def test_shadow_metrics_bind_immutable_segment_and_preserve_exact_large_counters(monkeypatch):
    """Diagnostic polling cannot round u64 identities or consume training/dataset state."""
    job = make_job()
    client = DistillationClient("http://localhost:8765", job.replay.manifest.contract.capture_epoch)
    client.manifest, client.cursor = job.replay.manifest, job.replay.cursor
    job.client = client
    job.configure_shadow(0, 1, "0000000000000001")
    receipt = {
        "schemaVersion": 1,
        "processEpoch": "13" * 16,
        "requestId": "17" * 16,
        "commandSequence": "0000000000000002",
        "activationSequence": "0000000000000002",
        "previousActivationSequence": "0" * 16,
        "runId": client.manifest.contract.run_id,
        "configRevision": "0000000000000001",
        "captureEpoch": client.capture_epoch,
        "artifactSha256": "23" * 32,
        "shadowSlot": 0,
        "baselineSlot": 1,
        "effectiveStep": "000000000000000c",
        "enabled": True,
        "acknowledgedAt": 1000,
    }
    job.online_metrics["receipt"] = receipt
    total = {
        "samples": "0020000000000001",
        "turnAbsoluteError": 0,
        "turnSquaredError": 0,
        "boostDisagreements": "0" * 16,
        "turnMae": 0,
        "turnMse": 0,
        "boostDisagreementRate": 0,
    }
    snapshot = {
        "shadowSlot": 0,
        "artifactSha256": receipt["artifactSha256"],
        "captureEpoch": client.capture_epoch,
        "baselineSlot": 1,
        "activationSequence": receipt["activationSequence"],
        "activationStep": receipt["effectiveStep"],
        "completedStep": "0020000000000003",
        "lastSampleStep": "0020000000000002",
        "generation": "0000000000000001",
        "physicalSnakeId": "8000000000000001",
        "enabled": True,
        "error": None,
        "rawOutputs": [0.1, 0.2],
        "totals": total,
        "strategies": [total] * 4,
        "lifeStages": [total] * 3,
        "strata": [total] * 6,
        "cruising": total,
        "preDeath": total,
        "deaths": "0" * 16,
    }
    metrics = {"schemaVersion": 1, "processEpoch": "13" * 16, "slots": [snapshot, None, None, None]}
    monkeypatch.setattr(client, "_json", lambda *_args, **_kwargs: copy.deepcopy(metrics))
    weights, cursor = job.model.export_weights(), client.cursor
    assert job.poll_shadow_metrics()["totals"]["samples"] == "0020000000000001"
    assert job.model.export_weights() == weights and client.cursor == cursor
    snapshot["totals"] = {**total, "samples": "0000000000000001"}
    with pytest.raises(DistillationContractError, match="regressed"):
        job.poll_shadow_metrics()
    metrics["processEpoch"] = "19" * 16
    with pytest.raises(DistillationContractError, match="replaced/interrupted"):
        job.poll_shadow_metrics()
