"""Public canonical-job controls, resource admission and exact restart ownership."""

import copy
import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import struct
from types import SimpleNamespace
import zipfile

import pytest
import torch

from pyrl_trainer import distillation_checkpoint as snapshots
from pyrl_trainer import distillation_job as owner
from pyrl_trainer import distillation_run as runner
from pyrl_trainer.distillation_checkpoint import load_job_checkpoint, _preflight
from pyrl_trainer.distillation_client import DistillationFetch
from pyrl_trainer.distillation_replay import TeacherReplay
from pyrl_trainer.distillation_transport import DistillationContractError

# Reuse only test fixture construction; production never imports this synthetic codec.
SPEC = importlib.util.spec_from_file_location(
    "controls_sampling_fixture", Path(__file__).with_name("test_distillation_sampling.py")
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
pytestmark = pytest.mark.unit
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
MIB = 1024**2


@pytest.fixture(name="local_source")
def local_source_fixture(monkeypatch):
    """Replace only HTTP intake with validated native-format rows; use the real client cursor."""
    replay = HELPERS.replay_fixture()

    def fetch(client):
        client.refresh_storage_policy()
        # A real fetch binds the manifest before producing its bounded immutable proposal.
        manifest = client._accept_manifest(replay.manifest.raw)  # pylint: disable=protected-access
        return DistillationFetch(
            manifest, client.cursor, replay.cursor,
            tuple(chunk for chunk in replay.chunks if chunk.sequence > client.cursor), (),
        )

    monkeypatch.setattr(runner.DistillationClient, "_json", lambda _client, _path, **_kwargs: {
        "schemaVersion": 1, "scope": "producer-global", "maxBytes": "0000000100000000",
        "maxChunks": 65536, "maxDatasets": 256, "maxLeases": 64, "maxPinnedChunks": 1024,
        "maxPinnedBytes": "0000000020000000", "maxLeaseMs": 300000,
        "diskReserveBytes": "0000000040000000",
    })
    monkeypatch.setattr(runner.DistillationClient, "fetch", fetch)
    return replay


def new_arguments(directory, device="cpu"):
    """Pass explicit graph/source identities through the public runner entry point."""
    return [
        "--manifest", str(HELPERS.GRAPH / "manifest.json"),
        "--weights", str(HELPERS.GRAPH / "weights.f32"),
        "--origin", "http://127.0.0.1:1",
        "--capture-epoch", HELPERS.REFERENCE["contract"]["captureEpoch"],
        "--job-directory", str(directory), "--device", device,
        "--updates", "1", "--max-wall-seconds", "30",
    ]


@pytest.mark.parametrize("process_cpus", [1, 3])
def test_new_cli_defaults_are_normalized_and_reported(tmp_path, monkeypatch, local_source,
                                                      process_cpus):
    """Fresh public defaults use seed one and at most two available CPUs, then checkpoint them."""
    monkeypatch.setattr(owner.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(owner.os, "process_cpu_count", lambda: process_cpus, raising=False)
    monkeypatch.setattr(owner.os, "sched_getaffinity", lambda _pid: set(range(process_cpus)),
                        raising=False)
    result = runner.run(new_arguments(tmp_path))
    job = load_job_checkpoint(result["checkpoint"])
    assert result["updates"] == 1 and job.trainer.completed_updates == 1
    assert result["settings"] == {
        "spoolGiB": 4.0,
        "producerStoragePolicy": job.client.producer_storage.to_object(),
        "seed": 1, "cpuThreads": min(2, process_cpus), "deviceMemoryMiB": 8192,
        "replayRamMiB": 256, "learningRate": 3e-4, "batchSequences": 32,
        "sequenceLength": 64, "burnIn": 32,
    }
    assert job.config.cpu_threads == min(2, process_cpus)
    assert job.replay.cursor == local_source.cursor
    # Library/checkpoint defaults retain their recorded semantics outside this public runner.
    assert owner.DistillationJobConfig().seed == 42


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_maximum_batch_and_replay_controls_train_and_exactly_restart(
    tmp_path, local_source, device
):
    """Upper admission choices do real CPU/CUDA updates without preallocating replay capacity."""
    result = runner.run(new_arguments(tmp_path, device) + [
        "--seed", "4294967295", "--cpu-threads", "1", "--device-memory-mib", "512",
        "--replay-ram-mib", "2048", "--learning-rate", "0.01", "--batch-size", "256",
        "--burn-in", "0", "--bptt", "16",
    ])
    saved = Path(result["checkpoint"])
    assert saved.stat().st_size < MIB
    left, right = load_job_checkpoint(saved), load_job_checkpoint(saved)
    assert left.config.seed == (1 << 32) - 1
    assert left.config.cpu_threads == 1 and left.config.device_bytes == 512 * MIB
    assert left.replay.max_bytes == 2048 * MIB and left.replay.owned_bytes < MIB
    assert left.replay.cursor == local_source.cursor
    assert left.sequence.batch_size == 256 and left.sequence.learning_rate == 0.01
    assert left.sequence.burn_in_steps == 0 and left.sequence.bptt_steps == 16
    expected = left.update()
    expected_optimizer = copy.deepcopy(left.trainer.optimizer.state_dict())
    actual = right.update()
    assert expected.optimizer_updated and actual.optimizer_updated
    assert left.model.export_weights() == right.model.export_weights()
    assert torch.equal(expected.next_state, actual.next_state)
    assert left.sampler.checkpoint_fields() == right.sampler.checkpoint_fields()
    assert expected.loss == actual.loss
    optimizer = right.trainer.optimizer.state_dict()
    assert expected_optimizer["param_groups"] == optimizer["param_groups"]
    for identity, values in expected_optimizer["state"].items():
        for name, value in values.items():
            other = optimizer["state"][identity][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert left.peak_device_bytes <= left.config.device_bytes


@pytest.mark.parametrize("options", [
    ["--seed", "-1"], ["--seed", "4294967296"],
    ["--cpu-threads", "0"], ["--cpu-threads", "3"],
    ["--device-memory-mib", "511"], ["--device-memory-mib", "8193"],
    ["--replay-ram-mib", "63"], ["--replay-ram-mib", "2049"],
    ["--learning-rate", "nan"], ["--learning-rate", "inf"],
    ["--learning-rate", "0.0000009"], ["--learning-rate", "0.010001"],
    ["--batch-size", "0"], ["--batch-size", "257"],
    ["--burn-in", "-1"], ["--burn-in", "129"],
    ["--bptt", "15"], ["--bptt", "257"], ["--burn-in", "128", "--bptt", "129"],
])
def test_invalid_public_settings_reject_before_opening_any_owner(tmp_path, monkeypatch, options):
    """Bad resource/range/combined-window choices cannot touch files, HTTP or CUDA."""
    monkeypatch.setattr(runner, "available_trainer_cpus", lambda: 2)

    def fail_open(_args):
        raise AssertionError("invalid settings cannot reach checkpoint/source/device admission")

    monkeypatch.setattr(runner, "_open_job", fail_open)
    with pytest.raises(SystemExit) as caught:
        runner.run(["--resume", str(tmp_path / "absent"), "--job-directory", str(tmp_path),
                    *options])
    assert caught.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [
    ["--cpu-threads", "2"], ["--device-memory-mib", "1024"],
    ["--replay-ram-mib", "128"], ["--learning-rate", "0.001"],
])
def test_exact_resume_rejects_changed_resource_and_optimizer_settings(
    tmp_path, monkeypatch, local_source, options
):
    """New public flags cannot resize or retune an exact checkpoint by accident."""
    monkeypatch.setattr(runner, "available_trainer_cpus", lambda: 2)
    result = runner.run(new_arguments(tmp_path) + [
        "--cpu-threads", "1", "--device-memory-mib", "512", "--replay-ram-mib", "64",
    ])
    saved = Path(result["checkpoint"])
    original = saved.read_bytes()
    with pytest.raises(DistillationContractError, match="cannot change"):
        runner.run(["--resume", str(saved), "--offline", "--job-directory", str(tmp_path),
                    *options])
    assert saved.read_bytes() == original and local_source.cursor > 0


def test_matching_explicit_controls_resume_and_new_defaults_do_not_override_saved_choices(
    tmp_path, local_source
):
    """All public controls survive an actual offline CLI restart with exact matching flags."""
    controls = ["--seed", "99", "--cpu-threads", "1", "--device-memory-mib", "512",
                "--replay-ram-mib", "64", "--learning-rate", "0.001", "--batch-size", "2",
                "--burn-in", "0", "--bptt", "16"]
    first = runner.run(new_arguments(tmp_path) + controls)
    second = runner.run(["--resume", first["checkpoint"], "--offline", "--updates", "1",
                         "--job-directory", str(tmp_path), "--device", "cpu", *controls])
    third = runner.run(["--resume", second["checkpoint"], "--offline", "--updates", "1",
                        "--job-directory", str(tmp_path)])
    assert first["settings"] == second["settings"] == third["settings"]
    assert third["updates"] == 3
    assert load_job_checkpoint(third["checkpoint"]).replay.cursor == local_source.cursor


def test_cpu_availability_uses_known_process_limits_and_has_safe_fallback(monkeypatch):
    """Missing counts cannot mask a known affinity, and inaccessible affinity is optional."""
    monkeypatch.setattr(owner, "_windows_affinity_cpus", lambda: None)
    monkeypatch.setattr(owner.os, "cpu_count", lambda: None)
    monkeypatch.setattr(owner.os, "process_cpu_count", lambda: 6, raising=False)
    monkeypatch.setattr(owner.os, "sched_getaffinity", lambda _pid: {0, 2, 4}, raising=False)
    assert owner.available_trainer_cpus() == 3
    monkeypatch.setattr(owner.os, "process_cpu_count", lambda: None)

    def denied(_pid):
        raise OSError("no affinity information")

    monkeypatch.setattr(owner.os, "sched_getaffinity", denied)
    assert owner.available_trainer_cpus() == 1


def test_cuda_admission_requires_availability_and_physical_capacity(monkeypatch):
    """An unavailable or too-small GPU is explicit; a ceiling does not reserve free VRAM."""
    config = owner.DistillationJobConfig()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DistillationContractError, match="unavailable"):
        owner.admit_job_device(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _device: SimpleNamespace(total_memory=4 * 1024**3))
    with pytest.raises(DistillationContractError, match="physical CUDA capacity"):
        owner.admit_job_device(config)
    smaller = owner.DistillationJobConfig(device_bytes=512 * MIB)
    assert owner.admit_job_device(smaller) == torch.device("cuda:0")


def test_replay_ceiling_is_explicit_without_an_eager_capacity_allocation():
    """Larger quotas retain the same fixed empty ownership and reject an out-of-plan quota."""
    small, large = TeacherReplay(max_bytes=64 * MIB), TeacherReplay(max_bytes=2048 * MIB)
    assert small.owned_bytes == large.owned_bytes and large.retained_chunks == 0
    with pytest.raises(DistillationContractError):
        TeacherReplay(max_bytes=2048 * MIB + 1)


@pytest.mark.parametrize("change", [None, "locator", "count", "central"])
def test_zip64_size_directory_with_small_member_count_is_admitted_and_strictly_checked(
    tmp_path, monkeypatch, local_source, change
):
    """Exercise ZIP64-by-size without writing a multi-gigabyte test fixture."""
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 128)
    result = runner.run(new_arguments(tmp_path) + ["--burn-in", "0", "--bptt", "16"])
    original = Path(result["checkpoint"])
    data = bytearray(original.read_bytes())
    assert struct.unpack_from("<H", data, len(data) - 12)[0] < 65535
    record, locator = data.rfind(b"PK\x06\x06"), len(data) - 42
    assert record > 0 and data[locator:locator + 4] == b"PK\x06\x07"
    if change is None:
        restored = load_job_checkpoint(original)
        assert restored.replay.cursor == local_source.cursor
        assert restored.update().optimizer_updated
        return
    if change == "locator":
        struct.pack_into("<Q", data, locator + 8, record + 1)
    elif change == "count":
        count = struct.unpack_from("<Q", data, record + 32)[0]
        struct.pack_into("<QQ", data, record + 24, count - 1, count - 1)
    else:
        struct.pack_into("<Q", data, record + 40, 8 * MIB + 1)
        struct.pack_into("<I", data, len(data) - 10, 0xFFFFFFFF)
    changed = tmp_path / "changed.distill-job"
    changed.write_bytes(data)
    with pytest.raises(DistillationContractError, match="ZIP64|central directory"):
        load_job_checkpoint(changed)


@pytest.mark.parametrize("burn_in,bptt", [(0, 256), (128, 128)])
@pytest.mark.parametrize("device", DEVICES)
def test_public_window_endpoints_use_admitted_tensors_and_real_training(
    tmp_path, monkeypatch, device, burn_in, bptt
):
    """A full 256-delivery window supports either the BPTT or burn-in public endpoint."""
    replay = HELPERS.replay_fixture((1, 2), chunks_per_stream=80)

    def fetch(client):
        manifest = client._accept_manifest(replay.manifest.raw)  # pylint: disable=protected-access
        return DistillationFetch(manifest, 0, replay.cursor, replay.chunks, ())

    monkeypatch.setattr(runner.DistillationClient, "fetch", fetch)
    result = runner.run(new_arguments(tmp_path, device) + [
        "--batch-size", "256", "--burn-in", str(burn_in), "--bptt", str(bptt),
        "--cpu-threads", "1", "--device-memory-mib", "2048",
    ])
    restored = load_job_checkpoint(result["checkpoint"])
    assert restored.sequence.bptt_steps + restored.sequence.burn_in_steps == 256
    assert restored.sampler.batch_bytes <= restored.sampler.tensor_bytes
    assert restored.last_training["training_samples"] > 0 and result["updates"] == 1
    assert restored.last_training["burn_in_samples"] == burn_in * 256
    assert restored.peak_device_bytes <= restored.config.device_bytes


@pytest.mark.parametrize("device", DEVICES)
def test_device_admission_uses_estimate_instead_of_requiring_free_ceiling(monkeypatch, device):
    """A modest job can use an eight-GiB ceiling when reported free VRAM is only 256 MiB."""
    if device != "cpu":
        total = torch.cuda.get_device_properties(0).total_memory
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (256 * MIB, total))
    job = owner.DistillationJob(
        HELPERS.model().manifest, (HELPERS.GRAPH / "weights.f32").read_bytes(),
        HELPERS.replay_fixture(), config=owner.DistillationJobConfig(device=device),
        sequence=HELPERS.config(),
    )
    assert job.device_estimate < 256 * MIB < job.config.device_bytes
    assert job.update().optimizer_updated


def test_unavailable_cuda_rejects_before_source_files_or_http(tmp_path, monkeypatch):
    """An explicit unavailable CUDA request fails without reaching even graph file reads."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    arguments = new_arguments(tmp_path, "cuda:0")
    arguments[arguments.index("--manifest") + 1] = str(tmp_path / "absent-manifest")
    with pytest.raises(DistillationContractError, match="CUDA device unavailable"):
        runner.run(arguments)
    assert not list(tmp_path.iterdir())


def test_classic_zip_with_exactly_65535_members_does_not_require_a_zip64_locator(tmp_path):
    """The classic count maximum remains a valid count when size/offset fit classic ZIP."""
    path = tmp_path / "classic-count-limit.zip"
    with zipfile.ZipFile(path, "w", allowZip64=False) as archive:
        for index in range(65535):
            archive.writestr(f"part-{index:05}", b"")
    assert path.stat().st_size < 8 * MIB
    assert _preflight(path) == 65535


@pytest.mark.parametrize("change", ["metadata-shape", "raw-budget"])
def test_checkpoint_replay_admission_rejects_before_any_chunk_read(
    tmp_path, monkeypatch, local_source, change
):
    """A repaired index digest cannot request replay I/O above the recorded RAM admission."""
    result = runner.run(new_arguments(tmp_path))
    original = Path(result["checkpoint"])
    with zipfile.ZipFile(original) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(files["job.json"])
    if change == "metadata-shape":
        metadata["replay"] = None
    else:
        metadata["replay"]["maxBytes"] = 2 * MIB
        index = bytearray(files["chunks.index"])
        for position in range(0, len(index), snapshots.INDEX_RECORD.size):
            struct.pack_into("<I", index, position + 8, MIB)
        files["chunks.index"] = bytes(index)
        metadata["digests"]["chunks.index"] = hashlib.sha256(index).hexdigest()
    files["job.json"] = json.dumps(metadata).encode("utf-8")
    changed = tmp_path / "unadmitted-replay.distill-job"
    with zipfile.ZipFile(changed, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    read = snapshots._read  # pylint: disable=protected-access

    def checked_read(archive, name, maximum):
        if name.startswith("chunks/"):
            raise AssertionError("unadmitted raw replay must be rejected before chunk reads")
        return read(archive, name, maximum)

    monkeypatch.setattr(snapshots, "_read", checked_read)
    with pytest.raises(DistillationContractError, match="metadata fields|raw replay exceeds"):
        load_job_checkpoint(changed)
    assert local_source.cursor > 0


@pytest.mark.skipif(os.name != "nt" and not hasattr(os, "sched_setaffinity"),
                    reason="supported affinity qualification needs Windows or Linux")
def test_owned_subprocess_respects_actual_single_cpu_affinity():
    """An inherited one-CPU restriction defeats even an explicit Python CPU-count override."""
    code = r"""
import os
if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    current = kernel.GetCurrentProcess
    current.argtypes, current.restype = [], wintypes.HANDLE
    query = kernel.GetProcessAffinityMask
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                      ctypes.POINTER(ctypes.c_size_t)]
    query.restype = wintypes.BOOL
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not query(current(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    mask = process_mask.value & -process_mask.value
    if not mask:
        raise RuntimeError("no primary-group CPU available for qualification")
    change = kernel.SetProcessAffinityMask
    change.argtypes, change.restype = [wintypes.HANDLE, ctypes.c_size_t], wintypes.BOOL
    if not change(current(), mask):
        raise ctypes.WinError(ctypes.get_last_error())
else:
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(allowed)})
from pyrl_trainer.distillation_job import available_trainer_cpus
assert available_trainer_cpus() == 1
print("available=1")
"""
    environment = dict(os.environ, PYTHON_CPU_COUNT="16")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
        timeout=45, cwd=Path(__file__).resolve().parents[2], env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "available=1"


@pytest.mark.parametrize("quota", [0, 65])
def test_public_spool_range_rejects_before_owner_or_files(tmp_path, monkeypatch, quota):
    """A bad operator resource choice never reaches the source/device/checkpoint owner."""
    monkeypatch.setattr(runner, "_open_job", lambda _args: pytest.fail("owner opened"))
    with pytest.raises(SystemExit):
        runner.run(new_arguments(tmp_path / "missing") + ["--spool-gib", str(quota)])
    assert not list(tmp_path.iterdir())


def test_requested_policy_is_recorded_and_exact_resume_cannot_change_it(tmp_path, local_source):
    """Observed source policy survives a real update/save/load without fetching on restore."""
    result = runner.run(new_arguments(tmp_path) + ["--spool-gib", "4"])
    job = load_job_checkpoint(result["checkpoint"])
    assert job.client.producer_storage.max_bytes == 4 * 1024**3
    assert (job.client.checkpoint_fields()["storagePolicy"]
            == result["settings"]["producerStoragePolicy"])
    with pytest.raises(DistillationContractError, match="exact resume"):
        runner.run(["--resume", result["checkpoint"], "--job-directory", str(tmp_path),
                    "--offline", "--spool-gib", "8", "--updates", "1"])
    assert job.replay.cursor == local_source.cursor


def test_requested_quota_mismatch_never_creates_a_job(tmp_path, local_source):
    """Producer admission must match an explicit choice before initial snapshot publication."""
    with pytest.raises(DistillationContractError, match="requested spool"):
        runner.run(new_arguments(tmp_path / "unpublished") + ["--spool-gib", "8"])
    assert not (tmp_path / "unpublished").exists()
    assert local_source.cursor > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quota", [1, 64])
def test_disk_budget_endpoints_survive_update_checkpoint_and_offline_resume(
    tmp_path, monkeypatch, local_source, device, quota
):
    """Record observed disk admission on CPU/CUDA without reserving that disk capacity."""
    policy = {"schemaVersion": 1, "scope": "producer-global", "maxBytes": f"{quota * 1024**3:016x}",
              "maxChunks": 65536, "maxDatasets": 256, "maxLeases": 64, "maxPinnedChunks": 1024,
              "maxPinnedBytes": "0000000020000000", "maxLeaseMs": 300000,
              "diskReserveBytes": "0000000040000000"}
    monkeypatch.setattr(runner.DistillationClient, "_json", lambda *_args, **_kwargs: policy)
    result = runner.run(new_arguments(tmp_path, device) + ["--spool-gib", str(quota)])
    saved = Path(result["checkpoint"])
    assert saved.stat().st_size < MIB
    restored = load_job_checkpoint(saved)
    assert restored.client.producer_storage.to_object() == policy
    assert restored.replay.cursor == local_source.cursor
    monkeypatch.setattr(runner.DistillationClient, "_json",
                        lambda *_args, **_kwargs: pytest.fail("offline resume made HTTP request"))
    resumed = runner.run(["--resume", str(saved), "--job-directory", str(tmp_path),
                          "--offline", "--spool-gib", str(quota), "--updates", "1"])
    assert resumed["updates"] == result["updates"] + 1
    assert resumed["settings"]["producerStoragePolicy"] == policy
