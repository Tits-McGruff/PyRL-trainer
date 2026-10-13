"""Run a bounded canonical distillation job, independently of PPO actor processes."""

import argparse
import json
import math
from pathlib import Path
import time

from .canonical_graph import CanonicalGraphManifest
from .distillation_checkpoint import load_job_checkpoint
from .distillation_client import DistillationClient
from .distillation_job import DistillationJob, DistillationJobConfig
from .distillation_replay import TeacherReplay
from .distillation_transport import DistillationContractError, integer
from .sequence_distillation import SequenceTrainingConfig


def _arguments(argv):
    """Require explicit source files/epoch for new jobs and an exact checkpoint for resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--origin")
    parser.add_argument("--capture-epoch")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--job-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", *(f"cuda:{index}" for index in range(8))))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--bptt", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--poll-ms", type=int, default=1000)
    args = parser.parse_args(argv)
    integer(args.updates, 1, 1000000)
    integer(args.poll_ms, 100, 60000)
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 86400:
        parser.error("max wall seconds must be finite and at most one day")
    if args.resume:
        if any((args.manifest, args.weights, args.origin, args.capture_epoch)):
            parser.error("resume uses the saved source/graph; new-source flags require a new job")
    elif args.offline or not all((args.manifest, args.weights, args.origin, args.capture_epoch)):
        parser.error("new jobs require manifest, weights, origin and capture epoch")
    return args


def _open_job(args):
    """Construct a new admitted owner or restore its exact configuration."""
    if args.resume:
        job = load_job_checkpoint(args.resume)
        for supplied, expected in (
            (args.device, job.config.device),
            (args.seed, job.config.seed),
            (args.batch_size, job.sequence.batch_size),
            (args.burn_in, job.sequence.burn_in_steps),
            (args.bptt, job.sequence.bptt_steps),
        ):
            if supplied is not None and supplied != expected:
                raise DistillationContractError("exact resume cannot change job/device settings")
    else:
        manifest = CanonicalGraphManifest.read(args.manifest)
        with args.weights.open("rb") as stream:
            weights = stream.read(4 * manifest.total_parameters + 1)
        client = DistillationClient(args.origin, args.capture_epoch)
        fetched = client.fetch()
        replay = TeacherReplay()
        replay.admit(fetched)
        client.commit(fetched)
        sequence = SequenceTrainingConfig(
            batch_size=32 if args.batch_size is None else args.batch_size,
            burn_in_steps=32 if args.burn_in is None else args.burn_in,
            bptt_steps=64 if args.bptt is None else args.bptt,
        )
        job = DistillationJob(
            manifest,
            weights,
            replay,
            client=client,
            sequence=sequence,
            config=DistillationJobConfig(
                seed=42 if args.seed is None else args.seed, device=args.device or "cuda:0"
            ),
        )
    return job


def run(argv=None):
    """Train, periodically checkpoint and stop within explicit update/wall limits."""
    args = _arguments(argv)
    job = _open_job(args)
    target = job.trainer.completed_updates + args.updates
    started, next_fetch = time.monotonic(), time.monotonic() + args.poll_ms / 1000
    # Establish a valid durable recovery boundary before the first optimizer operation.
    latest = job.save(args.job_directory)
    stop_reason = "update_limit"
    try:
        while job.trainer.completed_updates < target:
            now = time.monotonic()
            if now - started >= args.max_wall_seconds:
                stop_reason = "wall_limit"
                break
            if job.client is not None and not args.offline and now >= next_fetch:
                job.ingest(job.client.fetch())
                next_fetch = time.monotonic() + args.poll_ms / 1000
            updated = job.update()
            if updated is None:
                if args.offline or job.replay.manifest.state != "active":
                    stop_reason = "insufficient_retained_history"
                    break
                time.sleep(max(0.0, min(0.1, args.max_wall_seconds - (time.monotonic() - started))))
            elif not updated.optimizer_updated:
                stop_reason = "no_trainable_output"
                break
            elif job.trainer.completed_updates % 50 == 0:
                job.validate()
            if job.pending_candidates and job.client is not None and not args.offline:
                # Publish the immutable pending handoff before attempting a network receipt.
                latest = job.save(args.job_directory)
                job.upload_pending_candidates()
            if job.checkpoint_due():
                latest = job.save(args.job_directory)
        latest = job.save(args.job_directory)
    except (RuntimeError, MemoryError, DistillationContractError) as error:
        job.status, job.last_error = "paused", str(error)[:1024]
        # Failed partial optimizer state is never published; the last valid boundary survives.
        result = {
            "schemaVersion": 1,
            "jobId": job.job_id,
            "status": job.status,
            "reason": "job_failure",
            "error": job.last_error,
            "checkpoint": str(latest),
            "updates": job.trainer.completed_updates,
        }
        print(json.dumps(result, sort_keys=True))
        raise
    result = {
        "schemaVersion": 1,
        "jobId": job.job_id,
        "status": job.status,
        "reason": stop_reason,
        "checkpoint": str(latest),
        "updates": job.trainer.completed_updates,
        "datasetCursor": f"{job.replay.cursor:016x}",
        "device": str(job.device),
        "peakDeviceBytes": job.peak_device_bytes,
        "training": job.last_training,
        "validation": job.last_validation,
        "candidateDue": job.candidate_due,
    }
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
