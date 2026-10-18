"""Measure admitted canonical TBPTT allocation using explicit synthetic Float32 windows.

Rust supplies the manifest and initialized flat weights. This host does not encode
teacher chunks, join a game, change a deployment contract, or qualify replay/HTTP.
"""

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from .canonical_graph import CanonicalGraph, CanonicalGraphManifest
from .distillation_job import (
    DistillationJobConfig, admit_job_device, available_trainer_cpus,
    estimate_training_device_bytes, numerical_environment, trainer_source_sha256,
)
from .live_distillation_gate import _peak_rss
from .sequence_distillation import CanonicalSequenceTrainer, SequenceBatch, SequenceTrainingConfig


def _arguments():
    """Require an explicit device/immutable Rust inputs and a create-new report."""
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.flags.optimize:
        parser.error("qualification requires Python assertions enabled")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--device-memory-mib", type=int, default=8192)
    parser.add_argument("--cpu-threads", type=int, default=min(2, available_trainer_cpus()))
    parser.add_argument("--batch-sequences", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    for name, low, high in [("device_memory_mib", 512, 8192),
                            ("cpu_threads", 1, available_trainer_cpus()),
                            ("batch_sequences", 1, 256), ("sequence_length", 16, 256),
                            ("burn_in", 0, 128), ("updates", 1, 20)]:
        if not low <= getattr(args, name) <= high:
            parser.error("out-of-range " + name)
    if args.sequence_length + args.burn_in > 256:
        parser.error("burn-in plus BPTT exceeds 256 deliveries")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("report must be new with an existing parent")
    if args.manifest.stat().st_size > 1024**2 or args.weights.stat().st_size > 4_000_000:
        parser.error("canonical source file exceeds admission")
    return args


def run():
    # pylint: disable=too-many-statements,too-many-branches
    """Exercise actual full windows/Adam updates under the unchanged production estimate."""
    args = _arguments()
    source = args.manifest.read_bytes()
    weights = args.weights.read_bytes()
    manifest = CanonicalGraphManifest.from_bytes(source)
    config = DistillationJobConfig(device=args.device, cpu_threads=args.cpu_threads,
                                  device_bytes=args.device_memory_mib * 1024**2)
    sequence = SequenceTrainingConfig(batch_size=args.batch_sequences,
                                     bptt_steps=args.sequence_length, burn_in_steps=args.burn_in)
    estimate = estimate_training_device_bytes(manifest, sequence)
    report = {
        "schemaVersion": 1, "gate": "canonical-tbptt-capacity-v1",
        "scope": ("Synthetic full active windows through canonical model, loss, Adam and TBPTT. "
                  "No real teacher codec/replay, HTTP, trainer checkpoint, game contention, "
                  "shadows or integrated soak is qualified."),
        "trainerSourceSha256": trainer_source_sha256(),
        "hostSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifestSha256": hashlib.sha256(source).hexdigest(),
        "initialWeightsSha256": hashlib.sha256(weights).hexdigest(),
        "layoutSha256": manifest.layout_sha256, "sensorSha256": manifest.sensor_sha256,
        "inputSize": manifest.input_size, "parameters": manifest.total_parameters,
        "recurrentStateSize": manifest.total_state_size,
        "options": {"device": args.device, "deviceMemoryMiB": args.device_memory_mib,
                    "cpuThreads": args.cpu_threads, "updates": args.updates, **asdict(sequence)},
        "deviceEstimateBytes": estimate, "status": "rejected", "admitted": False,
        "qualificationComplete": False, "updates": [],
    }
    started = time.perf_counter()
    cpu = time.process_time()
    failure = None
    if estimate > config.device_bytes:
        report["rejection"] = "canonical training estimate exceeds device admission"
    else:
        try:
            device = admit_job_device(config)
            torch.set_num_threads(config.cpu_threads)
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info(device)
                report["physicalDeviceBytes"] = total
                report["freeDeviceBytesBefore"] = free
                if free < estimate:
                    raise RuntimeError("insufficient free CUDA memory for admitted job")
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            model = CanonicalGraph(manifest, weights, device)
            trainer = CanonicalSequenceTrainer(model, sequence)
            # Private CPU generator never consumes the trainer's or game's RNG streams.
            generator = torch.Generator(device="cpu").manual_seed(1)
            shape = (sequence.burn_in_steps + sequence.bptt_steps, sequence.batch_size)
            observations = torch.rand(
                (*shape, manifest.input_size), generator=generator) * 2 - 1
            boundary = torch.zeros(shape, dtype=torch.int64)
            boundary[0] = 1
            batch = SequenceBatch(observations, torch.rand(shape, generator=generator) * 2 - 1,
                                  torch.rand(shape, generator=generator) > 0.5,
                                  torch.full(shape, 4.0),
                                  torch.ones(shape, dtype=torch.bool), boundary,

                                  torch.zeros((shape[1], manifest.total_state_size)))
            report["environment"] = numerical_environment(device)
            report["admitted"] = True
            for update in range(args.updates):
                if time.perf_counter() - started > 900:
                    raise TimeoutError("canonical capacity total deadline")
                before = time.perf_counter()
                result = trainer.update(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    assert torch.cuda.max_memory_allocated(device) <= config.device_bytes
                assert torch.isfinite(model.flat_parameters).all()
                assert result.optimizer_updated
                assert result.training_samples == shape[1] * sequence.bptt_steps
                assert result.burn_in_samples == shape[1] * sequence.burn_in_steps
                item = {field.name: getattr(result, field.name)
                        for field in fields(result) if field.name != "next_state"}
                item["milliseconds"] = (time.perf_counter() - before) * 1000
                report["updates"].append(item)
                print("[canonical-capacity] update", update + 1, flush=True)
            if device.type == "cuda":
                report["peakAllocatedDeviceBytes"] = torch.cuda.max_memory_allocated(device)
                report["peakReservedDeviceBytes"] = torch.cuda.max_memory_reserved(device)
                assert report["peakAllocatedDeviceBytes"] <= config.device_bytes
                assert report["peakReservedDeviceBytes"] <= config.device_bytes
            report["trainedWeightsSha256"] = hashlib.sha256(model.export_weights()).hexdigest()
            report["status"] = "passed"
            report["qualificationComplete"] = True
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Keep the concrete failure; never silently fall back to CPU.
            failure = error
            report["status"] = "failed"
            report["failure"] = type(error).__name__ + ": " + str(error)
    report["elapsedMs"] = (time.perf_counter() - started) * 1000
    report["processCpuMs"] = (time.process_time() - cpu) * 1000
    report["peakRssBytes"] = _peak_rss()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if failure:
        raise failure
    return report


if __name__ == "__main__":
    run()
