"""Measure loaded replay with real Adam, durable job save and exact next-update restore.

Rust supplies immutable synthetic chunks and the initialized canonical graph.
This is an offline resource gate; no HTTP, game contention or soak is qualified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from .canonical_graph import CanonicalGraphManifest
from .distillation_checkpoint import DISK_RESERVE_BYTES, load_job_checkpoint
from .distillation_client import DistillationFetch
from .distillation_job import (
    DistillationJob, DistillationJobConfig, available_trainer_cpus, trainer_source_sha256,
)
from .distillation_replay import TeacherReplay, REPLAY_FIXED_BYTES, REPLAY_METADATA_BYTES
from .distillation_replay_capacity_gate import _chunks, _witness as replay_witness
from .distillation_transport import DatasetManifest, TeacherChunk
from .live_distillation_gate import _peak_rss, _witness as update_witness
from .sequence_distillation import SequenceTrainingConfig


def _arguments():
    """Validate inputs and exclusive output/workspace before constructing any mutable owner."""
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.flags.optimize:
        parser.error("qualification requires Python assertions enabled")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 64 <= args.ram_mib <= 2048 or not 1 <= args.deadline_seconds <= 1800:
        parser.error("RAM 64..2048 MiB and deadline 1..1800 seconds required")
    if args.workspace.exists() or args.output.exists():
        parser.error("create-new workspace and output required")
    if not args.workspace.parent.is_dir() or not args.output.parent.is_dir():
        parser.error("existing workspace and output parents required")
    workspace = args.workspace.resolve()
    if args.output.resolve().is_relative_to(workspace):
        parser.error("report must be outside disposable workspace")
    for path, maximum in [(args.fixture / "reference.json", 1024**2),
                          (args.manifest, 1024**2), (args.weights, 4_000_000)]:
        if not path.is_file() or path.stat().st_size > maximum:
            parser.error("immutable input exceeds admission")
    if shutil.disk_usage(workspace.parent).free < args.ram_mib * 1024**2 + DISK_RESERVE_BYTES:
        parser.error("checkpoint load must preserve physical disk reserve")
    return args


def exact_next_update(job, checkpoint):
    """Return independent restored ownership only after exact optimizer/state/RNG continuation."""
    expected = job.update()
    assert expected is not None and expected.optimizer_updated
    wanted = update_witness(job, expected)
    restored = load_job_checkpoint(checkpoint)
    assert restored.replay.chunks[0] is not job.replay.chunks[0]
    assert replay_witness(restored.replay.chunks) == replay_witness(job.replay.chunks)
    assert restored.replay.checkpoint_fields() == job.replay.checkpoint_fields()
    actual = restored.update()
    assert actual is not None and actual.optimizer_updated
    received = update_witness(restored, actual)
    assert wanted == received, "durable checkpoint did not reproduce the exact next update"
    assert restored.training_samples == job.training_samples
    assert restored.burn_in_samples == job.burn_in_samples
    return restored, received


def _save_report(path, report):
    """Publish a complete exclusive report and fsync before disposable checkpoint cleanup."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run():
    # This host records successive actual ownership peaks in one bounded case.
    # pylint: disable=too-many-statements,too-many-locals
    """Fill through real retention, train, save, restore and verify the very next update."""
    args = _arguments()
    reference_bytes = (args.fixture / "reference.json").read_bytes()
    reference = json.loads(reference_bytes)
    if reference["schemaVersion"] != 1 or not 64 <= reference["chunks"] <= 16384:
        raise ValueError("unsupported replay fixture")
    dataset = DatasetManifest.from_object(reference["dataset"])
    manifest_bytes, weights = args.manifest.read_bytes(), args.weights.read_bytes()
    manifest = CanonicalGraphManifest.from_bytes(manifest_bytes)
    bundle = args.fixture / "chunks.bin"
    before = bundle.stat()
    assert before.st_size == reference["bundleBytes"]
    replay = TeacherReplay(max_bytes=args.ram_mib * 1024**2)
    args.workspace.mkdir()
    identity = args.workspace.stat()
    started, cpu = time.perf_counter(), time.process_time()
    report = {
        "schemaVersion": 1, "gate": "canonical-durable-checkpoint-capacity-v1",
        "scope": ("Rust-encoded synthetic teacher chunks and canonical initialized graph; "
                  "actual loaded replay, Adam, fsynced job checkpoint and exact next update. "
                  "Original and restored jobs coexist. No HTTP, live game contention, "
                  "candidate backlog, checkpoint rotation or integrated soak is qualified."),
        "hostSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trainerSourceSha256": trainer_source_sha256(),
        "referenceSha256": hashlib.sha256(reference_bytes).hexdigest(),
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "initialWeightsSha256": hashlib.sha256(weights).hexdigest(),
        "nativeSourceSha256": reference["sourceSha256"],
        "fixtureHostCanonicalLfSha256": reference["hostCanonicalLfSha256"],
        "bundleSha256": reference["bundleSha256"],
        "options": {"ramMiB": args.ram_mib, "device": args.device},
        "parameters": manifest.total_parameters, "inputSize": manifest.input_size,
        "baselinePeakRssBytes": _peak_rss(), "status": "failed",
        "qualificationComplete": False, "workspaceRemoved": False,
    }
    failure = None
    try:
        def deadline():
            """Keep each bounded phase inside one total monotonic deadline."""
            if time.perf_counter() - started > args.deadline_seconds:
                raise TimeoutError("durable checkpoint capacity total deadline")

        digest, pending, count = hashlib.sha256(), [], 0
        for prefix, data in _chunks(bundle):
            deadline()
            digest.update(prefix)
            digest.update(data)
            chunk = TeacherChunk.decode(data)
            count += 1
            assert chunk.sequence == count and chunk.contract == dataset.contract
            pending.append(chunk)
            if len(pending) == 8 or count == reference["chunks"]:
                proposal = DistillationFetch(dataset, replay.cursor, count, tuple(pending), ())
                assert replay.admit(proposal)
                pending.clear()
        assert count == reference["chunks"] and digest.hexdigest() == reference["bundleSha256"]
        assert replay.evicted_chunks > 0 and replay.retained_chunks > 0
        assert (replay.max_bytes - replay.owned_bytes
                < replay.chunks[0].byte_cost + REPLAY_METADATA_BYTES)
        assert replay.owned_bytes > REPLAY_FIXED_BYTES
        report.update({"decodedChunks": count, "retainedChunks": replay.retained_chunks,
                       "ownedReplayBytes": replay.owned_bytes,
                       "evictedChunks": replay.evicted_chunks,
                       "fillMs": (time.perf_counter() - started) * 1000,
                       "filledPeakRssBytes": _peak_rss()})
        config = DistillationJobConfig(device=args.device,
                                       cpu_threads=min(2, available_trainer_cpus()))
        job = DistillationJob(manifest, weights, replay, config=config,
                              sequence=SequenceTrainingConfig())
        report["environment"] = job.environment
        if job.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(job.device)
        phase = time.perf_counter()
        first = job.update()
        assert first is not None and first.optimizer_updated
        assert job.trainer.optimizer.state_dict()["state"], "Adam moments must be populated"
        # Non-default private streams must be serialized, even though training uses sampler RNG.
        job.python_rng.random()
        torch.rand(7, generator=job.cpu_rng)
        if job.cuda_rng is not None:
            torch.rand(7, generator=job.cuda_rng, device=job.device)
        deadline()
        report.update({"firstUpdateMs": (time.perf_counter() - phase) * 1000,
                       "trainedPeakRssBytes": _peak_rss()})
        phase = time.perf_counter()
        checkpoint = job.save(args.workspace)
        deadline()
        report.update({"checkpointBytes": checkpoint.stat().st_size,
                       "saveMs": (time.perf_counter() - phase) * 1000,
                       "savedPeakRssBytes": _peak_rss()})
        phase = time.perf_counter()
        restored, witness = exact_next_update(job, checkpoint)
        deadline()
        report.update({"restoreAndNextUpdateMs": (time.perf_counter() - phase) * 1000,
                       "nextUpdateExact": True, "independentReplayOwners": True,
                       "nextUpdateWitness": witness,
                       "completedUpdates": restored.trainer.completed_updates,
                       "trainingSamples": restored.training_samples,
                       "burnInSamples": restored.burn_in_samples,
                       "restoredPeakRssBytes": _peak_rss()})
        if job.device.type == "cuda":
            report["peakAllocatedDeviceBytes"] = torch.cuda.max_memory_allocated(job.device)
            report["peakReservedDeviceBytes"] = torch.cuda.max_memory_reserved(job.device)
            assert report["peakAllocatedDeviceBytes"] <= config.device_bytes
        after = bundle.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
            after.st_size, after.st_mtime_ns, after.st_ino)
        assert shutil.disk_usage(args.workspace).free >= DISK_RESERVE_BYTES
        report.update({"status": "checks-passed-cleanup-pending"})
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure = error
        report["failure"] = type(error).__name__ + ": " + str(error)
    report.update({"elapsedMs": (time.perf_counter() - started) * 1000,
                   "processCpuMs": (time.process_time() - cpu) * 1000, "peakRssBytes": _peak_rss()})
    _save_report(args.output.with_suffix(args.output.suffix + ".progress.json"), report)
    current = args.workspace.stat()
    assert not args.workspace.is_symlink()
    assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)
    if failure is None:
        shutil.rmtree(args.workspace)
    report.update({"status": "failed" if failure else "passed",
                   "qualificationComplete": failure is None,
                   "workspaceRemoved": not args.workspace.exists()})
    _save_report(args.output, report)
    print("[checkpoint-capacity]", report["status"], flush=True)
    if failure:
        raise failure


if __name__ == "__main__":
    run()
