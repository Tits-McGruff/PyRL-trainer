"""Run a bounded canonical distillation job, independently of PPO actor processes."""

import argparse
import json
import math
from pathlib import Path
import time

# The standalone entry applies its scheduling policy before heavy Torch imports.
# Library callers retain ownership of their process policy.
# pylint: disable=wrong-import-position,wrong-import-order
from .distillation_process import configure_cli_priority, current_process_priority
if __name__ == "__main__":
    configure_cli_priority()

from .canonical_graph import CanonicalGraphManifest
from .distillation_checkpoint import load_job_checkpoint
from .distillation_client import DistillationClient
from .distillation_job import (
    DistillationJob,
    DistillationJobConfig,
    admit_job_device,
    available_trainer_cpus,
    explicit_trainer_device,
)
from .distillation_replay import TeacherReplay
from .distillation_transport import DistillationContractError, integer
from .sequence_distillation import SequenceTrainingConfig


MIB = 1024**2


def _validate_job_controls(parser, args):
    """Reject public settings before any source, device or checkpoint I/O."""
    for name, low, high in (
        ("seed", 0, (1 << 32) - 1),
        ("cpu_threads", 1, available_trainer_cpus()),
        ("device_memory_mib", 512, ((1 << 63) - 1) // MIB),
        ("replay_ram_mib", 64, 2048),
        ("spool_gib", 1, 64),
        ("batch_size", 1, 256),
        ("burn_in", 0, 128),
        ("bptt", 16, 256),
    ):
        value = getattr(args, name)
        if value is not None and not low <= value <= high:
            parser.error(f"--{name.replace('_', '-')} requires {low}..{high}")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or not 1e-6 <= args.learning_rate <= 1e-2
    ):
        parser.error("--learning-rate requires finite 1e-6..1e-2")
    burn_in = args.burn_in if args.burn_in is not None else (0 if args.resume else 32)
    bptt = args.bptt if args.bptt is not None else (0 if args.resume else 64)
    if burn_in + bptt > 256:
        parser.error("burn-in plus BPTT must not exceed 256 delivered observations")


def _arguments(argv):
    """Require explicit source files/epoch for new jobs and an exact checkpoint for resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--origin")
    parser.add_argument("--capture-epoch")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--shadow-slot", type=int)
    parser.add_argument("--shadow-baseline-slot", type=int)
    parser.add_argument("--shadow-config-revision")
    parser.add_argument("--job-directory", type=Path, required=True)
    parser.add_argument(
        "--device", type=explicit_trainer_device,
        help="cpu or explicit cuda:N; checked against available devices",
    )
    parser.add_argument("--seed", type=int, help="new-job UInt32 seed (default: 1)")
    parser.add_argument(
        "--cpu-threads", type=int,
        help="1..available process CPUs (default: min(2, available))",
    )
    parser.add_argument(
        "--device-memory-mib", type=int,
        help="512 MiB through physical CUDA capacity; ceiling, not reservation (default: 8192)",
    )
    parser.add_argument(
        "--replay-ram-mib", type=int, help="64..2048 MiB owned replay ceiling (default: 256)",
    )
    parser.add_argument(
        "--spool-gib", type=int,
        help="require the producer's admitted global disk quota, 1..64 GiB (default: observe)",
    )
    parser.add_argument(
        "--learning-rate", type=float, help="finite Adam rate 1e-6..1e-2 (default: 3e-4)",
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, help="1..256 sequence windows (default: 32)")
    parser.add_argument(
        "--burn-in", type=int, help="0..128 deliveries (default: 32; total window <=256)",
    )
    parser.add_argument(
        "--bptt", type=int, help="16..256 training deliveries (default: 64; total window <=256)",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--poll-ms", type=int, default=1000)
    args = parser.parse_args(argv)
    _validate_job_controls(parser, args)
    shadow_fields = (args.shadow_slot, args.shadow_baseline_slot, args.shadow_config_revision)
    if any(value is not None for value in shadow_fields) and (
        args.offline or any(value is None for value in shadow_fields)
    ):
        parser.error(
            "shadow opt-in requires slot, baseline slot, config revision and an online source"
        )
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
        if args.spool_gib is not None:
            policy = None if job.client is None else job.client.producer_storage
            if policy is None or policy.max_bytes != args.spool_gib * 1024**3:
                raise DistillationContractError(
                    "exact resume cannot change or infer producer spool budget"
                )
        for supplied, expected in (
            (args.device, job.config.device),
            (args.seed, job.config.seed),
            (args.cpu_threads, job.config.cpu_threads),
            (None if args.device_memory_mib is None else args.device_memory_mib * MIB,
             job.config.device_bytes),
            (None if args.replay_ram_mib is None else args.replay_ram_mib * MIB,
             job.replay.max_bytes),
            (args.learning_rate, job.sequence.learning_rate),
            (args.batch_size, job.sequence.batch_size),
            (args.burn_in, job.sequence.burn_in_steps),
            (args.bptt, job.sequence.bptt_steps),
        ):
            if supplied is not None and supplied != expected:
                raise DistillationContractError("exact resume cannot change job/device settings")
    else:
        config = DistillationJobConfig(
            seed=1 if args.seed is None else args.seed,
            device=args.device or "cuda:0",
            cpu_threads=(min(2, available_trainer_cpus())
                         if args.cpu_threads is None else args.cpu_threads),
            device_bytes=(8192 if args.device_memory_mib is None else args.device_memory_mib) * MIB,
        )
        sequence = SequenceTrainingConfig(
            batch_size=32 if args.batch_size is None else args.batch_size,
            burn_in_steps=32 if args.burn_in is None else args.burn_in,
            bptt_steps=64 if args.bptt is None else args.bptt,
            learning_rate=3e-4 if args.learning_rate is None else args.learning_rate,
        )
        admit_job_device(config)
        manifest = CanonicalGraphManifest.read(args.manifest)
        with args.weights.open("rb") as stream:
            weights = stream.read(4 * manifest.total_parameters + 1)
        client = DistillationClient(
            args.origin, args.capture_epoch,
            expected_spool_bytes=None if args.spool_gib is None else args.spool_gib * 1024**3,
        )
        fetched = client.fetch()
        replay = TeacherReplay(
            max_bytes=(256 if args.replay_ram_mib is None else args.replay_ram_mib) * MIB
        )
        replay.admit(fetched)
        client.commit(fetched)
        job = DistillationJob(
            manifest,
            weights,
            replay,
            client=client,
            sequence=sequence,
            config=config,
        )
    if args.shadow_slot is not None:
        configuration = {
            "shadowSlot": args.shadow_slot,
            "baselineSlot": args.shadow_baseline_slot,
            "configRevision": args.shadow_config_revision,
        }
        if args.resume:
            if job.online_metrics.get("configuration") != configuration:
                raise DistillationContractError("exact resume cannot change shadow configuration")
        else:
            job.configure_shadow(
                args.shadow_slot, args.shadow_baseline_slot, args.shadow_config_revision
            )
    return job


def _candidate_handoff(job, directory):
    """Persist an immutable diagnostic request before its ordered native activation."""
    if job.online_metrics:
        job.stage_pending_shadow()
        job.save(directory)
        job.activate_pending_shadow()
        return job.save(directory)
    checkpoint = job.save(directory)
    job.upload_pending_candidates()
    return checkpoint


def _job_settings(job):
    """Report effective admitted choices, including unchanged historical resume settings."""
    policy = None if job.client is None else job.client.producer_storage
    return {
        "spoolGiB": None if policy is None else policy.max_bytes / 1024**3,
        "producerStoragePolicy": None if policy is None else policy.to_object(),
        "seed": job.config.seed,
        "cpuThreads": job.environment["threads"],
        "deviceMemoryMiB": job.config.device_bytes // MIB,
        "replayRamMiB": job.replay.max_bytes // MIB,
        "learningRate": job.sequence.learning_rate,
        "batchSequences": job.sequence.batch_size,
        "sequenceLength": job.sequence.bptt_steps,
        "burnIn": job.sequence.burn_in_steps,
    }


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
                job.poll_shadow_metrics()
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
                latest = _candidate_handoff(job, args.job_directory)
            if job.checkpoint_due():
                latest = job.save(args.job_directory)
        latest = job.save(args.job_directory)
    except (RuntimeError, MemoryError, DistillationContractError) as error:
        job.status, job.last_error = "paused", str(error)[:1024]
        if job.checkpoint_allowed:
            latest = job.save(args.job_directory)
        # Failed partial optimizer state is never published; the last valid boundary survives.
        result = {
            "schemaVersion": 1,
            "jobId": job.job_id,
            "status": job.status,
            "reason": "job_failure",
            "error": job.last_error,
            "checkpoint": str(latest),
            "updates": job.trainer.completed_updates,
            "settings": _job_settings(job),
            "windowExecution": job.trainer.window_execution.statistics(),
            "processPriority": current_process_priority(),
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
        "settings": _job_settings(job),
        "peakDeviceBytes": job.peak_device_bytes,
        "windowExecution": job.trainer.window_execution.statistics(),
        "processPriority": current_process_priority(),
        "training": job.last_training,
        "validation": job.last_validation,
        "candidateDue": job.candidate_due,
    }
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
