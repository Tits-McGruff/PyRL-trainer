"""Self-contained DistillationJobV1 snapshots, independent of PPO checkpoint schema.

All immutable replay bytes are retained locally. Only local trainer optimizer/RNG
payloads use Torch serialization; deployed candidate brains never use this format.
"""

# This module is the persistence owner of the job's private committed-request witness.
# pylint: disable=protected-access

from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
import zipfile

# Owner/serializer entry points resolve each other only inside save/load calls.
# pylint: disable=cyclic-import

import numpy as np
import torch

from .distillation_shadows import online_state
from .brain_artifact import MAX_ARTIFACT_BYTES, read_raw_brain_artifact
from .canonical_graph import CanonicalGraphManifest
from .distillation_client import DistillationClient
from .distillation_replay import TeacherReplay
from .distillation_sampling import WindowIdentity
from .distillation_transport import (
    DatasetManifest,
    DistillationContractError,
    TeacherChunk,
    bounded_json,
    exact_keys,
    hex_identity,
    integer,
)
from .sequence_distillation import SequenceTrainingConfig

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
DISK_RESERVE_BYTES = 1024**3
MAX_CENTRAL_BYTES = 8 * 1024 * 1024
INDEX_RECORD = struct.Struct("<QI32s")
BASE_FILES = {
    "job.json",
    "graph.json",
    "weights.f32",
    "optimizer.pt",
    "rng.pt",
    "state.f32",
    "chunks.index",
}
MEMBER_LIMITS = {
    "job.json": 1024**2,
    "graph.json": 1024**2,
    "weights.f32": 4 * 1024**2,
    "optimizer.pt": 32 * 1024**2,
    "rng.pt": 256 * 1024,
    "state.f32": 8 * 1024**2,
    "chunks.index": 65536 * INDEX_RECORD.size,
}


def _sha(data):
    """Digest exact portable/local bytes, independently of ZIP CRC checks."""
    return hashlib.sha256(data).hexdigest()


def _json(data):
    """Serialize finite bounded metadata with deterministic key order."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if len(encoded) > 1024**2:
        raise DistillationContractError("job metadata exceeds its byte admission")
    return encoded


def _torch_bytes(value, maximum):
    """Serialize bounded local trainer state; no deployed neural artifact is produced."""
    stream = io.BytesIO()
    torch.save(value, stream)
    if stream.tell() > maximum:
        raise DistillationContractError("trainer state exceeds checkpoint admission")
    return stream.getvalue()


def _cpu_state(value):
    """Snapshot detached optimizer tensors once while preserving ordinary nested scalar settings."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _directory_sync(directory):
    """Flush directory publication on POSIX; Windows flushes complete writable file handles."""
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _preflight(path):
    """Bound ZIP directory allocation before constructing Python metadata objects."""
    path = Path(path)
    size = path.stat().st_size
    if not path.is_file() or not 22 <= size <= MAX_CHECKPOINT_BYTES:
        raise DistillationContractError("job checkpoint file admission")
    with path.open("rb") as stream:
        stream.seek(-22, os.SEEK_END)
        signature, disk, start_disk, on_disk, count, central_size, offset, comment = struct.unpack(
            "<4s4H2IH", stream.read(22)
        )
        if signature != b"PK\x05\x06" or disk or start_disk or comment or on_disk != count:
            raise DistillationContractError("unsupported job checkpoint ZIP directory")
        boundary = size - 22
        if count == 65535:
            stream.seek(size - 42)
            sig, zip_disk, zip_offset, disks = struct.unpack("<4sIQI", stream.read(20))
            if sig != b"PK\x06\x07" or zip_disk or disks != 1 or zip_offset > size - 98:
                raise DistillationContractError("unsupported job ZIP64 locator")
            stream.seek(zip_offset)
            values = struct.unpack("<4sQ2H2I4Q", stream.read(56))
            if (
                values[0] != b"PK\x06\x06"
                or values[1] != 44
                or values[4]
                or values[5]
                or values[6] != values[7]
            ):
                raise DistillationContractError("unsupported job ZIP64 directory")
            count, central_size, offset, boundary = values[7], values[8], values[9], zip_offset
        if (
            not 7 <= count <= 65547
            or central_size > MAX_CENTRAL_BYTES
            or offset + central_size != boundary
        ):
            raise DistillationContractError("job central directory exceeds admission")
    return count


def _read(archive, name, maximum):
    """Read one admitted uncompressed member with an independent actual-byte ceiling."""
    item = archive.getinfo(name)
    if item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1 or item.file_size > maximum:
        raise DistillationContractError("job member compression/size admission")
    with archive.open(item) as stream:
        data = stream.read(maximum + 1)
    if len(data) != item.file_size or len(data) > maximum:
        raise DistillationContractError("job member actual-byte mismatch")
    return data


def _manifest(path):
    # Validate the fixed inventory as one complete admission transaction.
    # pylint: disable=too-many-branches,too-many-statements
    """Verify fixed roles and immutable payloads before accepting a valid snapshot."""
    count = _preflight(path)
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != count or len(set(names)) != count or not BASE_FILES <= set(names):
            raise DistillationContractError("job checkpoint missing/duplicate roles")
        for item in archive.infolist():
            if item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1:
                raise DistillationContractError("unsupported job member encoding")
        metadata = bounded_json(_read(archive, "job.json", MEMBER_LIMITS["job.json"]))
        exact_keys(
            metadata,
            "schemaVersion purpose jobId serial trainerSourceSha256 environment config sequence "
            "dataset replay client sampler updates attemptedUpdates status lastError "
            "training validation "
            "onlineMetrics lastWindows peakDeviceBytes candidateDue digests chunkCount "
            "candidates droppedCandidates trainingSamples burnInSamples "
            "generationStart generationEnd",
        )
        integer(metadata["schemaVersion"], 1, 1)
        if metadata["purpose"] != "distillation-job":
            raise DistillationContractError("checkpoint is not a distillation job")
        hex_identity(metadata["jobId"], 32, True)
        integer(metadata["serial"], 1, (1 << 64) - 1)
        chunk_count = integer(metadata["chunkCount"], 0, 65536)
        candidates = metadata["candidates"]
        if not isinstance(candidates, list) or len(candidates) > 4:
            raise DistillationContractError("job pending candidate count admission")
        if count != len(BASE_FILES) + chunk_count + len(candidates):
            raise DistillationContractError("job replay/candidate member count mismatch")
        exact_keys(
            metadata["digests"], "graph.json weights.f32 optimizer.pt rng.pt state.f32 chunks.index"
        )
        payloads = {}
        for name in sorted(BASE_FILES - {"job.json"}):
            data = _read(archive, name, MEMBER_LIMITS[name])
            if _sha(data) != hex_identity(metadata["digests"][name], 64):
                raise DistillationContractError("job immutable role digest mismatch")
            payloads[name] = data
        index = payloads["chunks.index"]
        if len(index) != chunk_count * INDEX_RECORD.size:
            raise DistillationContractError("job replay index byte count mismatch")
        expected, previous = set(BASE_FILES), 0
        for position in range(0, len(index), INDEX_RECORD.size):
            sequence, size, digest = INDEX_RECORD.unpack_from(index, position)
            if sequence <= previous or not 1 <= size <= 1024**2:
                raise DistillationContractError("job replay index identity/size admission")
            name = f"chunks/{sequence:016x}.bin"
            expected.add(name)
            data = _read(archive, name, size)
            if len(data) != size or hashlib.sha256(data).digest() != digest:
                raise DistillationContractError("required replay bytes missing/corrupt")
            previous = sequence
        for candidate in candidates:
            exact_keys(candidate, "sha256 bytes")
            identity = hex_identity(candidate["sha256"], 64)
            size = integer(candidate["bytes"], 88, MAX_ARTIFACT_BYTES)
            name = f"candidates/{identity}.slither-brain"
            if name in expected:
                raise DistillationContractError("duplicate pending candidate identity")
            expected.add(name)
            data = _read(archive, name, size)
            if len(data) != size or _sha(data) != identity:
                raise DistillationContractError("required pending candidate missing/corrupt")
        if set(names) != expected:
            raise DistillationContractError("unexpected job archive roles")
    return metadata, payloads


def save_job_checkpoint(job, directory):
    # Metadata and all immutable files are published together, after complete file flush.
    # pylint: disable=too-many-statements,too-many-branches
    """Publish an exclusive complete ZIP snapshot; a failed save never replaces a valid file."""
    from .distillation_job import trainer_source_sha256  # pylint: disable=import-outside-toplevel

    if not job.checkpoint_allowed:
        raise DistillationContractError("failed trainer state cannot become an exact checkpoint")
    root = Path(directory) / job.job_id
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise DistillationContractError("job checkpoint directory cannot be a symlink")
    chunks = job.replay.chunks
    index = b"".join(
        INDEX_RECORD.pack(
            item.sequence, len(item.raw_bytes), hashlib.sha256(item.raw_bytes).digest()
        )
        for item in chunks
    )
    payloads = {
        "graph.json": job.model.manifest.canonical_json,
        "weights.f32": job.model.export_weights(),
        "optimizer.pt": _torch_bytes(
            _cpu_state(job.trainer.optimizer.state_dict()), MEMBER_LIMITS["optimizer.pt"]
        ),
        "rng.pt": _torch_bytes(
            {
                "python": job.python_rng.getstate(),
                "cpu": job.cpu_rng.get_state(),
                "cuda": None if job.cuda_rng is None else job.cuda_rng.get_state(),
            },
            MEMBER_LIMITS["rng.pt"],
        ),
        "state.f32": job.last_state.detach().cpu().numpy().astype("<f4").tobytes(),
        "chunks.index": index,
    }
    for name, data in payloads.items():
        if len(data) > MEMBER_LIMITS[name]:
            raise DistillationContractError("job role exceeds checkpoint admission")
    existing = [
        int(path.stem[4:], 16)
        for path in root.glob("job-*.distill-job")
        if re.fullmatch(r"job-[0-9a-f]{16}", path.stem)
    ]
    serial = max([job.serial, *existing]) + 1
    integer(serial, 1, (1 << 64) - 1)
    metadata = {
        "schemaVersion": 1,
        "purpose": "distillation-job",
        "jobId": job.job_id,
        "serial": serial,
        "trainerSourceSha256": trainer_source_sha256(),
        "environment": job.environment,
        "config": asdict(job.config),
        "sequence": asdict(job.sequence),
        "dataset": job.replay.manifest.raw,
        "replay": job.replay.checkpoint_fields(),
        "client": (
            None
            if job.client is None
            else {
                "checkpoint": job.client.checkpoint_fields(),
                "leases": [],
                "options": {
                    "page_size": job.client.page_size,
                    "owned_bytes": job.client.owned_bytes,
                    "timeout": job.client.timeout,
                    "lease_ms": job.client.lease_ms,
                },
            }
        ),
        "sampler": job.sampler.checkpoint_fields(),
        "updates": job.trainer.completed_updates,
        "attemptedUpdates": job.attempted_updates,
        "status": job.status,
        "lastError": job.last_error,
        "training": job.last_training,
        "validation": job.last_validation,
        "onlineMetrics": job.online_metrics,
        "lastWindows": [asdict(item) for item in job.last_windows],
        "peakDeviceBytes": job.peak_device_bytes,
        "candidateDue": job.candidate_due,
        "candidates": [
            {"sha256": artifact.sha256, "bytes": len(artifact.data)}
            for artifact in job.pending_candidates
        ],
        "droppedCandidates": f"{job.dropped_candidates:016x}",
        "trainingSamples": f"{job.training_samples:016x}",
        "burnInSamples": f"{job.burn_in_samples:016x}",
        "generationStart": (
            None if job.trained_generation_min is None else f"{job.trained_generation_min:016x}"
        ),
        "generationEnd": (
            None if job.trained_generation_max is None else f"{job.trained_generation_max:016x}"
        ),
        "digests": {name: _sha(data) for name, data in payloads.items()},
        "chunkCount": len(chunks),
    }
    encoded = _json(metadata)
    estimated = sum(map(len, payloads.values())) + sum(len(item.raw_bytes) for item in chunks)
    estimated += sum(len(artifact.data) for artifact in job.pending_candidates)
    estimated += len(encoded) + 1024 * (len(chunks) + len(BASE_FILES) + len(job.pending_candidates))
    if (
        estimated > MAX_CHECKPOINT_BYTES
        or shutil.disk_usage(root).free < estimated + DISK_RESERVE_BYTES
    ):
        raise DistillationContractError("job checkpoint disk/byte admission refused")
    descriptor, temp_name = tempfile.mkstemp(prefix=".job-", suffix=".tmp", dir=root)
    temp = Path(temp_name)
    final = root / f"job-{serial:016x}.distill-job"
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                archive.writestr("job.json", encoded)
                for name, data in payloads.items():
                    archive.writestr(name, data)
                for item in chunks:
                    archive.writestr(f"chunks/{item.sequence:016x}.bin", item.raw_bytes)
                for artifact in job.pending_candidates:
                    archive.writestr(f"candidates/{artifact.sha256}.slither-brain", artifact.data)
            if stream.tell() > MAX_CHECKPOINT_BYTES:
                raise DistillationContractError("actual job checkpoint exceeds byte admission")
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive hard-link publication gives readers only the complete flushed inode.
        os.link(temp, final)
        with final.open("r+b") as stream:
            os.fsync(stream.fileno())
        _directory_sync(root)
        job.serial = serial
        job.last_save_time, job.last_save_update = time.monotonic(), job.trainer.completed_updates
        valid = []
        for path in sorted(root.glob("job-*.distill-job"), reverse=True):
            if not re.fullmatch(r"job-[0-9a-f]{16}\.distill-job", path.name):
                continue
            try:
                record, _payloads = _manifest(path)
                if record["jobId"] == job.job_id:
                    valid.append(path)
            except (OSError, ValueError, KeyError, struct.error, zipfile.BadZipFile):
                continue
        for path in valid[3:]:
            path.unlink()
        _directory_sync(root)
        # The persistence owner grants this witness only after complete durable publication.
        job._shadow_checkpoint_request = (
            job.shadow_request_digest()
        )  # pylint: disable=protected-access
        return final
    finally:
        temp.unlink(missing_ok=True)


def _optimizer(job, encoded, updates):
    """Admit only method-v1 Adam state over the one canonical flat Float32 parameter."""
    state = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    exact_keys(state, "state param_groups")
    expected = job.trainer.optimizer.state_dict()["param_groups"]
    if state["param_groups"] != expected or not isinstance(state["state"], dict):
        raise DistillationContractError("job optimizer method/settings mismatch")
    if set(state["state"]) not in (set(), {0}) or (updates and not state["state"]):
        raise DistillationContractError("job optimizer parameter ownership mismatch")
    if state["state"]:
        moments = state["state"][0]
        exact_keys(moments, "step exp_avg exp_avg_sq")
        for name in ("exp_avg", "exp_avg_sq"):
            value = moments[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.shape != (job.model.manifest.total_parameters,)
                or not torch.isfinite(value).all()
            ):
                raise DistillationContractError("invalid Adam canonical moment tensor")
        step = moments["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or step.dtype != torch.float32
            or not torch.isfinite(step).all()
        ):
            raise DistillationContractError("invalid Adam update step tensor")
        if not 0 < step.item() <= updates or not float(step.item()).is_integer():
            raise DistillationContractError("invalid Adam update step")
    return state


def _metrics(training, validation, sequence):
    """Admit finite bounded metric records before any optimizer/RNG publication."""
    if training is not None:
        exact_keys(
            training,
            "loss turn_mse turn_mae boost_disagreement training_samples "
            "burn_in_samples unknown_state_starts gradient_norm optimizer_updated",
        )
        for key in ("loss", "turn_mse", "turn_mae", "boost_disagreement", "gradient_norm"):
            value = training[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise DistillationContractError("invalid training metric")
        if training["boost_disagreement"] > 1 or not isinstance(
            training["optimizer_updated"], bool
        ):
            raise DistillationContractError("invalid training disagreement/update state")
        integer(training["training_samples"], 1, sequence.batch_size * sequence.bptt_steps)
        integer(training["burn_in_samples"], 0, sequence.batch_size * sequence.burn_in_steps)
        integer(
            training["unknown_state_starts"],
            0,
            sequence.batch_size * (sequence.bptt_steps + sequence.burn_in_steps),
        )
    if validation is not None:
        exact_keys(validation, "turn_mse turn_mae boost_disagreement samples")
        for key in ("turn_mse", "turn_mae", "boost_disagreement"):
            value = validation[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise DistillationContractError("invalid validation metric")
        if validation["boost_disagreement"] > 1:
            raise DistillationContractError("invalid validation disagreement")
        integer(validation["samples"], 1, sequence.batch_size * sequence.bptt_steps)


def load_job_checkpoint(path):
    # Construct/validate a fresh private owner before returning any restored state.
    # pylint: disable=too-many-statements,too-many-branches,import-outside-toplevel
    """Exact same-source/environment restart; every required local replay object must exist."""
    from .distillation_job import (
        DistillationJob,
        DistillationJobConfig,
        trainer_source_sha256,
    )  # pylint: disable=import-outside-toplevel

    metadata, payloads = _manifest(path)
    if metadata["trainerSourceSha256"] != trainer_source_sha256():
        raise DistillationContractError("exact job restart requires matching trainer source")
    exact_keys(metadata["config"], "seed device cpu_threads device_bytes")
    exact_keys(
        metadata["sequence"], "batch_size bptt_steps burn_in_steps learning_rate gradient_clip"
    )
    config, sequence = DistillationJobConfig(**metadata["config"]), SequenceTrainingConfig(
        **metadata["sequence"]
    )
    manifest = CanonicalGraphManifest.from_bytes(payloads["graph.json"])
    dataset = DatasetManifest.from_object(metadata["dataset"])
    records, owned = [], 256 * 1024
    with zipfile.ZipFile(path, "r") as archive:
        index = payloads["chunks.index"]
        maximum = integer(metadata["replay"]["maxBytes"], 2 * 1024**2, 512 * 1024**2)
        for position in range(0, len(index), INDEX_RECORD.size):
            identity, size, digest = INDEX_RECORD.unpack_from(index, position)
            data = _read(archive, f"chunks/{identity:016x}.bin", size)
            if hashlib.sha256(data).digest() != digest:
                raise DistillationContractError("required replay bytes changed during restart")
            chunk = TeacherChunk.decode(data)
            if chunk.sequence != identity:
                raise DistillationContractError("job raw replay/index identity mismatch")
            owned += TeacherReplay._cost(chunk)  # pylint: disable=protected-access
            if owned > maximum:
                raise DistillationContractError("job replay exceeds restored RAM admission")
            records.append(chunk)
    replay = TeacherReplay.restore_snapshot(dataset, records, metadata["replay"])
    client = None
    if metadata["client"] is not None:
        source = metadata["client"]
        exact_keys(source, "checkpoint leases options")
        if source["leases"] != []:
            raise DistillationContractError("unsupported pending HTTP storage pins")
        exact_keys(source["options"], "page_size owned_bytes timeout lease_ms")
        saved = source["checkpoint"]
        suffix = "/api/distillation/v1/"
        if not isinstance(saved.get("origin"), str) or not saved["origin"].endswith(suffix):
            raise DistillationContractError("job HTTP source identity mismatch")
        client = DistillationClient(
            saved["origin"][: -len(suffix)],
            dataset.contract.capture_epoch,
            checkpoint=saved,
            **source["options"],
        )
        client._accept_manifest(dataset.raw)  # pylint: disable=protected-access
    job = DistillationJob(
        manifest, payloads["weights.f32"], replay, config=config, sequence=sequence, client=client
    )
    if job.environment != metadata["environment"]:
        raise DistillationContractError("exact job restart requires matching numerical environment")
    updates = integer(metadata["updates"], 0, (1 << 64) - 1)
    attempts = integer(metadata["attemptedUpdates"], updates, (1 << 64) - 1)
    optimizer = _optimizer(job, payloads["optimizer.pt"], updates)
    rng = torch.load(io.BytesIO(payloads["rng.pt"]), map_location="cpu", weights_only=True)
    exact_keys(rng, "python cpu cuda")
    python = rng["python"]
    if (
        not isinstance(python, tuple)
        or len(python) != 3
        or python[0] != 3
        or not isinstance(python[1], tuple)
        or len(python[1]) != 625
    ):
        raise DistillationContractError("job Python generator admission")
    for value in python[1][:-1]:
        integer(value, 0, (1 << 32) - 1)
    integer(python[1][-1], 0, 624)
    if python[2] is not None and (not isinstance(python[2], float) or not math.isfinite(python[2])):
        raise DistillationContractError("job Python Gaussian state admission")
    for name, generator in (("cpu", job.cpu_rng), ("cuda", job.cuda_rng)):
        value = rng[name]
        if generator is None:
            if value is not None:
                raise DistillationContractError("foreign CUDA generator state")
        elif (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.shape != generator.get_state().shape
        ):
            raise DistillationContractError("job Torch generator admission")
    rows, columns = sequence.batch_size, manifest.total_state_size
    raw_state = payloads["state.f32"]
    if len(raw_state) != rows * columns * 4:
        raise DistillationContractError("job detached recurrent state byte count")
    state = np.frombuffer(raw_state, dtype="<f4").copy().reshape(rows, columns)
    if not np.isfinite(state).all():
        raise DistillationContractError("nonfinite job detached recurrent state")
    if not isinstance(metadata["lastWindows"], list) or len(metadata["lastWindows"]) not in (
        0,
        rows,
    ):
        raise DistillationContractError("job sampled-window cursor admission")
    windows = []
    for item in metadata["lastWindows"]:
        exact_keys(item, "baseline_slot generation first_sample first_training_sample last_sample")
        integer(item["baseline_slot"], 0, 119)
        integer(item["generation"], 1, (1 << 64) - 1)
        first = integer(item["first_sample"], 1, (1 << 64) - 1)
        train = integer(item["first_training_sample"], first, (1 << 64) - 1)
        last = integer(item["last_sample"], train, (1 << 64) - 1)
        if train - first != sequence.burn_in_steps or last - first >= 256:
            raise DistillationContractError("job sampled-window chronology admission")
        windows.append(WindowIdentity(**item))
    if metadata["status"] not in ("running", "waiting", "paused"):
        raise DistillationContractError("unsupported job status")
    if not isinstance(metadata["candidateDue"], bool):
        raise DistillationContractError("invalid candidate publication state")
    if metadata["lastError"] is not None and (
        not isinstance(metadata["lastError"], str) or len(metadata["lastError"]) > 1024
    ):
        raise DistillationContractError("job error admission")
    _metrics(metadata["training"], metadata["validation"], sequence)
    online_state(metadata["onlineMetrics"])
    candidates = []
    with zipfile.ZipFile(path, "r") as archive:
        for record in metadata["candidates"]:
            data = _read(archive, f"candidates/{record['sha256']}.slither-brain", record["bytes"])
            artifact, candidate_manifest, provenance = read_raw_brain_artifact(data)
            actual_identity = (
                artifact.sha256,
                candidate_manifest.layout_sha256,
                candidate_manifest.sensor_sha256,
                candidate_manifest.life_state_rule,
                candidate_manifest.control_interval_seconds,
                provenance["sourceKind"],
                provenance["details"].get("jobId"),
            )
            expected_identity = (
                record["sha256"],
                manifest.layout_sha256,
                manifest.sensor_sha256,
                manifest.life_state_rule,
                manifest.control_interval_seconds,
                "distilled",
                metadata["jobId"],
            )
            if actual_identity != expected_identity:
                raise DistillationContractError(
                    "job pending candidate identity/deployment mismatch"
                )
            candidate_update = int(hex_identity(provenance["details"].get("updates"), 16), 16)
            if not 0 < candidate_update <= updates or candidate_update % 250:
                raise DistillationContractError("job candidate update cadence mismatch")
            candidates.append(artifact)
    shadow = metadata["onlineMetrics"]
    if shadow:
        receipt = shadow["receipt"]
        if receipt is not None and (
            receipt["runId"] != replay.manifest.contract.run_id
            or receipt["captureEpoch"] != replay.manifest.contract.capture_epoch
        ):
            raise DistillationContractError("job online shadow dataset mismatch")
        pending = shadow["pending"]
        if pending is not None and pending["artifactSha256"] not in {
            candidate.sha256 for candidate in candidates
        }:
            raise DistillationContractError("pending shadow candidate bytes missing")
    if metadata["candidateDue"] != bool(candidates):
        raise DistillationContractError("job pending candidate flag mismatch")
    for key in ("generationStart", "generationEnd"):
        if metadata[key] is not None:
            hex_identity(metadata[key], 16, True)
    if (metadata["generationStart"] is None) != (metadata["generationEnd"] is None):
        raise DistillationContractError("job training generation extent mismatch")
    if (
        metadata["generationStart"] is not None
        and metadata["generationStart"] > metadata["generationEnd"]
    ):
        raise DistillationContractError("job training generation order mismatch")
    job.sampler.restore(metadata["sampler"])
    job.trainer.optimizer.load_state_dict(optimizer)
    job.trainer.completed_updates, job.attempted_updates = updates, attempts
    job.python_rng.setstate(python)
    job.cpu_rng.set_state(rng["cpu"])
    if job.cuda_rng is not None:
        job.cuda_rng.set_state(rng["cuda"])
    job.last_state = torch.from_numpy(state).to(job.device)
    job.last_windows = tuple(windows)
    job.job_id, job.serial, job.status = metadata["jobId"], metadata["serial"], metadata["status"]
    job.last_error, job.candidate_due = metadata["lastError"], metadata["candidateDue"]
    job.last_training, job.last_validation = metadata["training"], metadata["validation"]
    job.online_metrics = metadata["onlineMetrics"]
    job._shadow_checkpoint_request = job.shadow_request_digest()  # pylint: disable=protected-access
    job.peak_device_bytes = integer(metadata["peakDeviceBytes"], 0, config.device_bytes)
    job.pending_candidates = candidates
    job.dropped_candidates = int(hex_identity(metadata["droppedCandidates"], 16), 16)
    job.training_samples = int(hex_identity(metadata["trainingSamples"], 16), 16)
    job.burn_in_samples = int(hex_identity(metadata["burnInSamples"], 16), 16)
    job.trained_generation_min = (
        None if metadata["generationStart"] is None else int(metadata["generationStart"], 16)
    )
    job.trained_generation_max = (
        None if metadata["generationEnd"] is None else int(metadata["generationEnd"], 16)
    )
    job.last_save_update = updates
    return job
