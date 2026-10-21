"""Qualify live committed teacher intake, exact process restart and native handoff.

The Slither qualification host owns the disposable authority and HTTP endpoint.
This process uses only DistillationClient; it never constructs an actor/client lease.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

from .brain_artifact import HEADER, read_raw_brain_artifact
from .distillation_checkpoint import load_job_checkpoint
from .distillation_client import DistillationClient
from .distillation_job import DistillationJob, DistillationJobConfig, trainer_source_sha256
from .distillation_replay import TeacherReplay
from .sequence_distillation import SequenceTrainingConfig


def _digest(value):
    """Canonical witness for numeric optimizer/RNG state, independent of pickle bytes."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"[" + str(len(item)).encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        elif isinstance(item, float):
            digest.update(b"f" + struct.pack("<d", item))
        else:
            digest.update(type(item).__name__.encode() + repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def _witness(job, updated):
    """Bind the next real update, optimizer, recurrence, sampler and all private RNG."""
    return {
        "weights": hashlib.sha256(job.model.export_weights()).hexdigest(),
        "optimizer": _digest(job.trainer.optimizer.state_dict()),
        "state": _digest(updated.next_state),
        "sampler": _digest(job.sampler.checkpoint_fields()),
        "windows": _digest([asdict(window) for window in job.last_windows]),
        "pythonRng": _digest(job.python_rng.getstate()),
        "cpuRng": _digest(job.cpu_rng.get_state()),
        "cudaRng": None if job.cuda_rng is None else _digest(job.cuda_rng.get_state()),
        "training": job.last_training,
    }


def _peak_rss():
    """Peak process working set on supported Windows/Linux qualification hosts."""
    if sys.platform == "win32":
        import ctypes  # pylint: disable=import-outside-toplevel
        from ctypes import wintypes  # pylint: disable=import-outside-toplevel

        class Counters(ctypes.Structure):  # pylint: disable=too-few-public-methods
            """Native PROCESS_MEMORY_COUNTERS layout; byte counts use pointer width."""
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile")]

        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        memory = ctypes.windll.psapi.GetProcessMemoryInfo
        memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters(ctypes.sizeof(Counters))
        if not memory(process(), ctypes.byref(counters), counters.cb):
            raise OSError("could not read qualification process working set")
        return counters.peak_working
    import resource  # pylint: disable=import-outside-toplevel,import-error
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _ready_replay(client, deadline):
    """Require both held-out and training streams with the complete default window width."""
    replay = TeacherReplay()
    while time.monotonic() < deadline:
        fetched = client.fetch()
        replay.admit(fetched)
        client.commit(fetched)
        counts = {}
        for segment in replay.segments():
            key = segment.partition, segment.chunk.baseline_slot, segment.chunk.generation
            counts[key] = counts.get(key, 0) + len(segment.chunk.steps)
        if all(any(key[0] == partition and count >= 96 for key, count in counts.items())
               for partition in ("training", "validation")):
            return replay
        time.sleep(0.1)
    raise TimeoutError("live source did not retain full default training and validation windows")


def _update_to(job, target, deadline):
    """Bound real optimizer work; no synthetic rows or fabricated update counters."""
    updated = None
    while job.trainer.completed_updates < target:
        if time.monotonic() >= deadline:
            raise TimeoutError("live distillation qualification update deadline")
        updated = job.update()
        if updated is None or not updated.optimizer_updated:
            raise RuntimeError("retained live samples did not produce an optimizer update")
    return updated


def _train_phase(args, deadline):
    """Train on real retained rows and save the exact next-update oracle."""
    metadata_path = args.directory / "restart.json"
    artifact_data = args.artifact.read_bytes()
    _artifact, manifest, _provenance = read_raw_brain_artifact(artifact_data)
    metadata_bytes = HEADER.unpack_from(artifact_data)[1]
    weights = artifact_data[HEADER.size + metadata_bytes:]
    client = DistillationClient(args.origin, args.capture_epoch,
                                expected_spool_bytes=args.spool_gib * 1024**3)
    replay = _ready_replay(client, deadline)
    job = DistillationJob(manifest, weights, replay, client=client,
                          config=DistillationJobConfig(device=args.device),
                          sequence=SequenceTrainingConfig())
    _update_to(job, 125, deadline)
    checkpoint = job.save(args.directory / "checkpoints")
    updated = _update_to(job, 126, deadline)
    metadata_path.write_text(json.dumps({"checkpoint": str(checkpoint),
                                        "witness": _witness(job, updated),
                                        "storagePolicy": client.producer_storage.to_object()},
                                       sort_keys=True),
                             encoding="utf-8")
    report = {"phase": "train", "updates": job.trainer.completed_updates,
              "retainedChunks": job.replay.retained_chunks,
              "sourceRows": sum(len(chunk.steps) for chunk in job.replay.chunks)}
    return job, report


def _resume_phase(args, deadline):
    """Verify a new process reproduces its saved update, then activate its candidate."""
    metadata_path = args.directory / "restart.json"
    expected = json.loads(metadata_path.read_text(encoding="utf-8"))
    job = load_job_checkpoint(Path(expected["checkpoint"]))
    if (str(job.device) != args.device
            or job.client.capture_epoch != args.capture_epoch):
        raise RuntimeError("qualification restart source/device mismatch")
    if (job.client.producer_storage is None
            or job.client.producer_storage.max_bytes != args.spool_gib * 1024**3
            or job.client.producer_storage.to_object() != expected["storagePolicy"]):
        raise RuntimeError("qualification restart producer storage policy mismatch")
    updated = _update_to(job, 126, deadline)
    if _witness(job, updated) != expected["witness"]:
        raise RuntimeError("cross-process next optimizer/RNG/sampler/recurrence witness differs")
    _update_to(job, 250, deadline)
    if len(job.pending_candidates) != 1:
        raise RuntimeError("update 250 did not freeze exactly one immutable candidate")
    candidate = job.pending_candidates[0]
    candidate_path = args.directory / "candidate.slither-brain"
    candidate_path.write_bytes(candidate.data)
    job.configure_shadow(0, 0, "0000000000000001")
    if not job.stage_pending_shadow():
        raise RuntimeError("candidate did not stage a durable shadow request")
    job.save(args.directory / "checkpoints")
    receipt = job.activate_pending_shadow()
    metrics = None
    while time.monotonic() < deadline:
        metrics = job.poll_shadow_metrics()
        if metrics is not None and int(metrics["totals"]["samples"], 16) > 0:
            break
        time.sleep(0.1)
    if metrics is None or int(metrics["totals"]["samples"], 16) == 0:
        raise RuntimeError("trained candidate produced no actual Rust shadow evaluations")
    checkpoint = job.save(args.directory / "checkpoints")
    (args.directory / "sustain-checkpoint.json").write_text(
        json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8")
    report = {"phase": "resume", "updates": job.trainer.completed_updates,
              "restartExact": True, "candidateSha256": candidate.sha256,
              "candidateWeightsSha256": candidate.weights_sha256,
              "candidatePath": str(candidate_path), "receipt": receipt,
              "metrics": metrics,
              "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "validation": job.last_validation}
    return job, report


def _sustain_phase(args, deadline):
    """Consume new live chunks while actual optimizer work contends with four native shadows."""
    pointer = json.loads((args.directory / "sustain-checkpoint.json").read_text(encoding="utf-8"))
    job = load_job_checkpoint(Path(pointer["checkpoint"]))
    if (str(job.device) != args.device or job.client.capture_epoch != args.capture_epoch
            or job.client.producer_storage.max_bytes != args.spool_gib * 1024**3):
        raise RuntimeError("sustained trainer source/device/storage mismatch")
    started = time.monotonic()
    initial_updates = job.trainer.completed_updates
    initial_cursor = job.client.cursor
    initial_samples = job.training_samples
    next_fetch = started
    fetches = 0
    peak_replay = job.replay.owned_bytes
    peak_reserved = 0
    while time.monotonic() - started < args.seconds:
        if time.monotonic() >= deadline:
            raise TimeoutError("sustained trainer exceeded its outer deadline")
        if time.monotonic() >= next_fetch:
            job.ingest(job.client.fetch())
            fetches += 1
            next_fetch = time.monotonic() + 1
            peak_replay = max(peak_replay, job.replay.owned_bytes)
        updated = job.update()
        if updated is None:
            time.sleep(0.01)
        elif not updated.optimizer_updated:
            raise RuntimeError("sustained trainer did not update trainable outputs")
        if job.device.type == "cuda":
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(job.device))
    if job.trainer.completed_updates <= initial_updates or job.client.cursor <= initial_cursor:
        raise RuntimeError("sustained trainer requires both optimizer and live intake progress")
    report = {"phase": "sustain", "updates": job.trainer.completed_updates,
              "initialUpdates": initial_updates,
              "optimizerUpdates": job.trainer.completed_updates - initial_updates,
              "trainingSamplesInWindow": job.training_samples - initial_samples,
              "initialCursor": initial_cursor, "finalCursor": job.client.cursor,
              "fetches": fetches, "requestedSeconds": args.seconds,
              "activeWallSeconds": time.monotonic() - started,
              "peakReplayBytes": peak_replay,
              "pendingCandidates": len(job.pending_candidates),
              "droppedCandidates": job.dropped_candidates,
              "peakReservedInWindowBytes": peak_reserved}
    return job, report


def run(argv=None):
    """First process saves update 125; the second proves update 126 then hands off 250."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "resume", "sustain"), required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--capture-epoch", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--spool-gib", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--max-wall-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_wall_seconds) or not 1 <= args.max_wall_seconds <= 600:
        parser.error("wall deadline must be finite and between one and 600 seconds")
    if not 60 <= args.seconds <= 300:
        parser.error("sustained window must be 60..300 seconds")
    if args.phase == "sustain" and args.max_wall_seconds < args.seconds + 30:
        parser.error("sustained window requires at least 30 seconds of outer startup allowance")
    if not 1 <= args.spool_gib <= 64:
        parser.error("spool budget must be 1..64 GiB")
    args.directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    phase = {"train": _train_phase, "resume": _resume_phase, "sustain": _sustain_phase}[args.phase]
    job, report = phase(args, deadline)
    report.update({"schemaVersion": 1, "device": str(job.device), "environment": job.environment,
                   "producerStoragePolicy": job.client.producer_storage.to_object(),
                   "sequence": asdict(job.sequence), "replayBytes": job.replay.owned_bytes,
                   "peakDeviceBytes": job.peak_device_bytes, "peakRssBytes": _peak_rss(),
                   "trainerSourceSha256": trainer_source_sha256(),
                   "trainingSamples": job.training_samples, "burnInSamples": job.burn_in_samples,
                   "peakReservedDeviceBytes": (0 if job.device.type == "cpu"
                                               else torch.cuda.max_memory_reserved(job.device)),
                   "elapsedWallSeconds": time.monotonic() - started})
    (args.directory / (args.phase + "-report.json")).write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "phase", "device", "updates", "elapsedWallSeconds")}))
    return report


if __name__ == "__main__":
    run()
