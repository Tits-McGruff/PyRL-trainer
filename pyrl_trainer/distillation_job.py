"""Independent canonical distillation job owner, outside PPO and all game control."""

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import time
import uuid

import numpy as np
import torch

from .brain_artifact import encode_brain_artifact
from .canonical_graph import CanonicalGraph, CanonicalGraphError
from .distillation_replay import TeacherReplay
from .distillation_sampling import ReplayDataUnavailable, ReplayWindowSampler
from .distillation_transport import DistillationContractError, integer, hex_identity
from .distillation_shadows import online_state
from .sequence_distillation import CanonicalSequenceTrainer, SequenceTrainingConfig

SOURCE_FILES = (
    "canonical_graph.py",
    "sensor_contract.py",
    "sequence_distillation.py",
    "distillation_transport.py",
    "distillation_client.py",
    "distillation_storage.py",
    "distillation_replay.py",
    "distillation_sampling.py",
    "distillation_job.py",
    "distillation_checkpoint.py",
    "distillation_run.py",
    "brain_artifact.py",
    "distillation_shadows.py",
)


def _compute_source_sha256():
    """Bind exact restart to the actual canonical/job implementation, independently of Git state."""
    digest = hashlib.sha256(b"slither.distillation-trainer-source.v1\0")
    for name in SOURCE_FILES:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


# Capture loaded source identity once; later filesystem edits cannot relabel a live job.
_LOADED_SOURCE_SHA256 = _compute_source_sha256()


def trainer_source_sha256():
    """Return the source identity captured when this process loaded the trainer owner."""
    return _LOADED_SOURCE_SHA256


def _windows_affinity_cpus():
    """Read a conservative process-affinity ceiling on Windows, including older Python."""
    if os.name != "nt":
        return None
    # Local read-only Win32 query; no process affinity is changed by the job owner.
    # https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-getprocessaffinitymask
    import ctypes  # pylint: disable=import-outside-toplevel
    from ctypes import wintypes  # pylint: disable=import-outside-toplevel

    kernel = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    current = kernel.GetCurrentProcess
    current.argtypes, current.restype = [], wintypes.HANDLE
    query = kernel.GetProcessAffinityMask
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                      ctypes.POINTER(ctypes.c_size_t)]
    query.restype = wintypes.BOOL
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not query(current(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        return None
    # This is the primary-group ceiling on large multi-group systems; use it conservatively.
    return process_mask.value.bit_count() or None


def available_trainer_cpus():
    """Respect the process's CPU affinity and Python's process-aware count where available."""
    logical_count = os.cpu_count()
    counts = [logical_count] if logical_count else []
    windows_count = _windows_affinity_cpus()
    if windows_count:
        counts.append(windows_count)
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        count = process_count()
        if count:
            counts.append(count)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
            if count:
                counts.append(count)
        except (OSError, NotImplementedError):
            pass
    return max(1, min(counts or [1]))


@dataclass(frozen=True)
class DistillationJobConfig:
    """Isolated process settings; explicitly requested CUDA never falls back to CPU."""

    seed: int = 42
    device: str = "cuda:0"
    cpu_threads: int = 2
    device_bytes: int = 8 * 1024**3

    def __post_init__(self):
        integer(self.seed, 0, (1 << 64) - 1)
        integer(self.cpu_threads, 1, max(256, available_trainer_cpus()))
        integer(self.device_bytes, 128 * 1024**2, 8 * 1024**3)
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda:[0-7]", self.device):
            raise DistillationContractError("unsupported explicit trainer device")


def admit_job_device(config):
    """Admit a CUDA device and its budget without reserving the whole ceiling."""
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise DistillationContractError("requested CUDA device unavailable")
        if config.device_bytes > torch.cuda.get_device_properties(device).total_memory:
            raise DistillationContractError(
                "requested device budget exceeds physical CUDA capacity"
            )
    return device


def numerical_environment(device):
    """Record numerical build, thread and device facts needed for same-environment restart."""
    result = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cudaRuntime": torch.version.cuda,
        "torchBuildSha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "threads": torch.get_num_threads(),
        "device": str(device),
        "float32": True,
        "tf32": False,
        "autocast": False,
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministicWarnOnly": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmulPrecision": torch.get_float32_matmul_precision(),
        "cudnnDeterministic": torch.backends.cudnn.deterministic,
        "cudnnBenchmark": torch.backends.cudnn.benchmark,
        "cpuCapability": torch.backends.cpu.get_cpu_capability(),
        "cpuProcessor": platform.processor(),
        "cublasWorkspaceConfig": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "atenCpuCapability": os.environ.get("ATEN_CPU_CAPABILITY"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
            "totalBytes": properties.total_memory,
        }
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            versions = sorted(set(driver.stdout.strip().splitlines()))
            if not versions or len(driver.stdout) > 4096:
                raise DistillationContractError("invalid CUDA driver identity")
            result["cudaDriver"] = versions
        except (OSError, subprocess.SubprocessError) as error:
            raise DistillationContractError("CUDA driver provenance unavailable") from error
    return result


def estimate_training_device_bytes(manifest, sequence):
    """Price canonical Adam/TBPTT storage before model or sequence allocation.

    This is the existing conservative job bound, shared by capacity qualification.
    It is not a promise that arbitrary graphs or the whole process fit this budget.
    """
    width = sum(
        sum(node.output_sizes) + sum(node.hidden_sizes) + (node.hidden_size or 0) * 8
        for node in manifest.nodes
    )
    return (
        128 * 1024**2
        + manifest.total_parameters * 48
        + width * sequence.batch_size * sequence.bptt_steps * 96
        + manifest.total_state_size * sequence.batch_size * 16
    )


class DistillationJob:
    """Private RNG, bounded canonical training and independent durable snapshots."""

    def __init__(
        self,
        manifest,
        weights,
        replay,
        *,
        config=DistillationJobConfig(),
        sequence=SequenceTrainingConfig(),
        client=None,
    ):
        # Graph, data, device and deployment contracts are separate admitted inputs.
        # pylint: disable=too-many-arguments
        if not isinstance(replay, TeacherReplay) or replay.manifest is None:
            raise DistillationContractError("job requires an admitted immutable dataset")
        if client is not None and (
            client.cursor != replay.cursor
            or client.manifest is None
            or not client.manifest.compatible(replay.manifest)
        ):
            raise DistillationContractError("job HTTP/replay identity/cursor mismatch")
        self.config, self.sequence, self.replay, self.client = config, sequence, replay, client
        self.device = admit_job_device(config)
        torch.set_num_threads(min(config.cpu_threads, available_trainer_cpus()))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device_estimate = estimate_training_device_bytes(manifest, sequence)
        if self.device_estimate > config.device_bytes:
            raise DistillationContractError("canonical training estimate exceeds device admission")
        if self.device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self.device)
            if free < self.device_estimate:
                raise DistillationContractError("insufficient free CUDA memory for admitted job")
            # One distillation process owns this device policy, including private
            # compiler graph pools; no eager/compiled CUDA path may exceed it.
            torch.cuda.set_per_process_memory_fraction(
                config.device_bytes / torch.cuda.get_device_properties(self.device).total_memory,
                self.device)
        self.model = CanonicalGraph(manifest, weights, self.device)
        self.trainer = CanonicalSequenceTrainer(
            self.model, sequence, cuda_graphs=self.device.type == "cuda")
        self.sampler = ReplayWindowSampler(replay, manifest, sequence, seed=config.seed ^ 0x5A17)
        self.sampler.checkpoint_fields()  # Prove sensor/cadence identity before any training.
        self.python_rng = random.Random(config.seed ^ 0x7171)
        self.cpu_rng = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x435055)
        self.cuda_rng = (
            torch.Generator(device=self.device).manual_seed(config.seed ^ 0x43554441)
            if self.device.type == "cuda"
            else None
        )
        self.environment = numerical_environment(self.device)
        self.environment["sequenceExecution"] = (
            "bounded-cuda-window-v1" if self.device.type == "cuda" else "eager-v1")
        self.job_id = uuid.uuid4().hex
        self.serial = 0
        self.attempted_updates = 0
        self.status = "running"
        self.last_error = None
        self.last_training = None
        self.last_validation = None
        self.online_metrics = {}
        self._shadow_checkpoint_request = None
        self.last_windows = ()
        self.last_state = torch.zeros(
            sequence.batch_size, manifest.total_state_size, dtype=torch.float32, device=self.device
        )
        self.checkpoint_allowed = True
        self.last_save_time = time.monotonic()
        self.last_save_update = 0
        self.candidate_due = False
        self.pending_candidates = []
        self.dropped_candidates = 0
        self.training_samples = 0
        self.burn_in_samples = 0
        self.trained_generation_min = None
        self.trained_generation_max = None
        self.peak_device_bytes = 0

    def ingest(self, fetched):
        """Own one validated proposal before committing its independent HTTP cursor."""
        if self.client is not None and (
            fetched.start_cursor != self.client.cursor
            or fetched.manifest.contract.capture_epoch != self.client.capture_epoch
        ):
            raise DistillationContractError("job intake HTTP/replay source mismatch")
        admitted = self.replay.admit(fetched)
        if self.client is not None:
            self.client.commit(fetched)
        return admitted

    def update(self):
        """Run one bounded update; failures pause this job and preserve valid checkpoints."""
        if self.status not in ("running", "waiting"):
            raise DistillationContractError("distillation job is paused")
        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            sampled = self.sampler.sample()
            updated = self.trainer.update(sampled.batch)
            if not torch.isfinite(self.model.flat_parameters).all():
                raise CanonicalGraphError("nonfinite post-update canonical weights")
            if self.device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(self.device)
                self.peak_device_bytes = max(self.peak_device_bytes, peak)
                if peak > self.config.device_bytes:
                    raise DistillationContractError("CUDA job exceeded admitted device budget")
            self.status = "running"
            self.attempted_updates += 1
            metrics = {
                field.name: getattr(updated, field.name)
                for field in fields(updated)
                if field.name != "next_state"
            }
            self.last_training = metrics
            self.last_state = updated.next_state
            self.last_windows = sampled.windows
            self.training_samples += updated.training_samples
            self.burn_in_samples += updated.burn_in_samples
            generations = [window.generation for window in sampled.windows]
            self.trained_generation_min = min(
                [
                    min(generations),
                    *([] if self.trained_generation_min is None else [self.trained_generation_min]),
                ]
            )
            self.trained_generation_max = max(
                [
                    max(generations),
                    *([] if self.trained_generation_max is None else [self.trained_generation_max]),
                ]
            )
            if updated.optimizer_updated and self.trainer.completed_updates % 250 == 0:
                self._capture_candidate()
            self.candidate_due = bool(self.pending_candidates)
            if not updated.optimizer_updated:
                self.status, self.last_error = "paused", "graph has no trainable output dependency"
            return updated
        except ReplayDataUnavailable:
            self.status = "waiting"
            return None
        except (RuntimeError, MemoryError, CanonicalGraphError, DistillationContractError) as error:
            self.status, self.last_error = "paused", str(error)[:1024]
            # A failed optimizer/device operation may have partially changed trainer state.
            self.checkpoint_allowed = False
            raise

    def _capture_candidate(self):
        """Freeze each 250th update; bounded pressure discards oldest pending exports."""
        dataset = self.replay.manifest
        # Metrics attached to this immutable weight version must evaluate that same version.
        validation = self.validate()
        details = {
            "method": "canonical-distillation-v1",
            "lossRevision": 1,
            "jobId": self.job_id,
            "trainerSourceSha256": trainer_source_sha256(),
            "seed": f"{self.config.seed:016x}",
            "updates": f"{self.trainer.completed_updates:016x}",
            "trainingSamples": f"{self.training_samples:016x}",
            "burnInSamples": f"{self.burn_in_samples:016x}",
            "generationStart": f"{self.trained_generation_min:016x}",
            "generationEnd": f"{self.trained_generation_max:016x}",
            "teacherContract": dataset.raw["contract"],
            "runSeed": dataset.run_seed,
            "sourceCursor": f"{self.replay.cursor:016x}",
            "training": self.last_training,
            "validation": validation,
            "validationModelUpdate": (
                None if validation is None else f"{self.trainer.completed_updates:016x}"
            ),
            "onlineMetrics": self.online_metrics,
            "environment": self.environment,
        }
        artifact = encode_brain_artifact(
            self.model.manifest,
            self.model.export_weights(),
            {"schemaVersion": 1, "sourceKind": "distilled", "details": details},
        )
        if len(self.pending_candidates) >= 4:
            pending = self.online_metrics.get("pending")
            protected = pending["artifactSha256"] if pending else None
            index = next(
                index
                for index, candidate in enumerate(self.pending_candidates)
                if candidate.sha256 != protected
            )
            self.pending_candidates.pop(index)
            self.dropped_candidates += 1
        self.pending_candidates.append(artifact)

    def upload_pending_candidates(self):
        """Release only matching native validation receipts; retry failures retain exact bytes."""
        if self.client is None:
            return 0
        if self.online_metrics:
            raise DistillationContractError("configured job requires checkpointed shadow handoff")
        count = 0
        for artifact in tuple(self.pending_candidates):
            receipt = self.client.upload_candidate(artifact)
            if (
                receipt["layoutSha256"] != self.model.manifest.layout_sha256
                or receipt["sensorSha256"] != self.model.manifest.sensor_sha256
                or receipt["controlIntervalSeconds"] != self.model.manifest.control_interval_seconds
                or receipt["lifeStateRule"] != self.model.manifest.life_state_rule
            ):
                raise DistillationContractError("native candidate deployment receipt mismatch")
            self.acknowledge_candidate(artifact.sha256)
            count += 1
        return count

    def acknowledge_candidate(self, identity):
        """Release only an explicitly confirmed immutable handoff, never a gameplay ACK."""
        for index, artifact in enumerate(self.pending_candidates):
            if artifact.sha256 == identity:
                self.pending_candidates.pop(index)
                self.candidate_due = bool(self.pending_candidates)
                return True
        return False

    def validate(self):
        """Unweighted held-out-stream metrics; model/optimizer weights remain immutable."""
        if self.status not in ("running", "waiting"):
            raise DistillationContractError("distillation job is paused")
        try:
            sampled = self.sampler.sample("validation")
            self.last_validation = self.trainer.evaluate(sampled.batch)
            return self.last_validation
        except ReplayDataUnavailable:
            return None
        except (RuntimeError, MemoryError, CanonicalGraphError, DistillationContractError) as error:
            self.status, self.last_error = "paused", str(error)[:1024]
            self.checkpoint_allowed = False
            raise

    def checkpoint_due(self, now=None):
        """Checkpoint after 60 wall seconds or 1,000 successful optimizer updates."""
        current = time.monotonic() if now is None else now
        return (
            current >= self.last_save_time + 60
            or self.trainer.completed_updates - self.last_save_update >= 1000
        )

    def save(self, directory):
        """Publish one self-contained job checkpoint and retain the newest three valid files."""
        # The reader lazily constructs an owner; local resolution avoids initialization cycles.
        # pylint: disable=import-outside-toplevel
        from .distillation_checkpoint import save_job_checkpoint

        return save_job_checkpoint(self, directory)

    def configure_shadow(self, slot, baseline_slot, config_revision):
        """Explicit opt-in for one diagnostic slot; immutable configuration is checkpoint state."""
        if self.client is None or self.online_metrics:
            raise DistillationContractError(
                "shadow configuration requires a connected unconfigured job"
            )
        self.online_metrics = {
            "schemaVersion": 1,
            "configuration": {
                "shadowSlot": integer(slot, 0, 3),
                "baselineSlot": integer(baseline_slot, 0, 119),
                "configRevision": hex_identity(config_revision, 16, True),
            },
            "pending": None,
            "receipt": None,
            "snapshot": None,
        }
        online_state(self.online_metrics)

    def stage_pending_shadow(self):
        """Upload and freeze one candidate request; caller checkpoints before issuing activation."""
        if not self.online_metrics or self.client is None or not self.pending_candidates:
            return False
        if self.online_metrics["pending"] is not None:
            return True
        artifact = self.pending_candidates[0]
        self.client.upload_candidate(artifact)
        metrics = self.client.shadow_metrics()
        configuration = self.online_metrics["configuration"]
        previous = metrics["slots"][configuration["shadowSlot"]]
        self._shadow_checkpoint_request = None
        self.online_metrics["pending"] = {
            "artifactSha256": artifact.sha256,
            "requestId": uuid.uuid4().hex,
            "processEpoch": metrics["processEpoch"],
            "expectedActivationSequence": previous["activationSequence"] if previous else "0" * 16,
        }
        online_state(self.online_metrics)
        return True

    def shadow_request_digest(self):
        """Bind the complete saved request and configuration, including source identity."""
        if not self.online_metrics or self.online_metrics["pending"] is None:
            return None
        journal = {
            "configuration": self.online_metrics["configuration"],
            "pending": self.online_metrics["pending"],
            "runId": self.replay.manifest.contract.run_id,
            "captureEpoch": self.replay.manifest.contract.capture_epoch,
        }
        return hashlib.sha256(
            json.dumps(journal, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()

    def activate_pending_shadow(self):
        """Submit the saved request; lost replies preserve bytes and request identity."""
        if not self.online_metrics or self.online_metrics["pending"] is None or self.client is None:
            return None
        pending, configuration = (
            self.online_metrics["pending"],
            self.online_metrics["configuration"],
        )
        if self._shadow_checkpoint_request != self.shadow_request_digest():
            raise DistillationContractError(
                "shadow activation requires committed pending-request checkpoint"
            )
        result = self.client.activate_shadow(
            slot=configuration["shadowSlot"],
            baseline_slot=configuration["baselineSlot"],
            config_revision=configuration["configRevision"],
            artifact_sha256=pending["artifactSha256"],
            request_id=pending["requestId"],
            process_epoch=pending["processEpoch"],
            expected_activation_sequence=pending["expectedActivationSequence"],
        )
        (
            self.online_metrics["receipt"],
            self.online_metrics["snapshot"],
            self.online_metrics["pending"],
        ) = (result, None, None)
        self.acknowledge_candidate(pending["artifactSha256"])
        online_state(self.online_metrics)
        return result

    def poll_shadow_metrics(self):
        """Bind online statistics to the recorded activation without changing training state."""
        if not self.online_metrics or self.online_metrics["receipt"] is None or self.client is None:
            return None
        metrics = self.client.shadow_metrics()
        receipt = self.online_metrics["receipt"]
        snapshot = metrics["slots"][receipt["shadowSlot"]]
        old = self.online_metrics["snapshot"]
        if metrics["processEpoch"] == receipt["processEpoch"] and old is None:
            # The reliable command receipt can precede the nonblocking native
            # diagnostic mailbox. Only the exact prior activation may be stale;
            # once this segment is observed, every interruption remains fatal.
            previous = receipt["previousActivationSequence"]
            if (snapshot is None and previous == "0" * 16) or (
                snapshot is not None and snapshot["activationSequence"] == previous
            ):
                return None
        if (
            metrics["processEpoch"] != receipt["processEpoch"]
            or snapshot is None
            or any(
                snapshot[name] != receipt[name]
                for name in ("artifactSha256", "captureEpoch", "activationSequence", "baselineSlot")
            )
        ):
            raise DistillationContractError("online shadow segment replaced/interrupted")
        if old and (
            snapshot["completedStep"] < old["completedStep"]
            or snapshot["totals"]["samples"] < old["totals"]["samples"]
        ):
            raise DistillationContractError("online shadow metrics regressed")
        self.online_metrics["snapshot"] = snapshot
        online_state(self.online_metrics)
        return snapshot
