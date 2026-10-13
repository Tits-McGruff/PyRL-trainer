"""Independent canonical distillation job owner, outside PPO and all game control."""

from dataclasses import dataclass, fields
import hashlib
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

from .canonical_graph import CanonicalGraph, CanonicalGraphError
from .distillation_replay import TeacherReplay
from .distillation_sampling import ReplayDataUnavailable, ReplayWindowSampler
from .distillation_transport import DistillationContractError, integer
from .sequence_distillation import CanonicalSequenceTrainer, SequenceTrainingConfig

SOURCE_FILES = (
    "canonical_graph.py",
    "sequence_distillation.py",
    "distillation_transport.py",
    "distillation_client.py",
    "distillation_replay.py",
    "distillation_sampling.py",
    "distillation_job.py",
    "distillation_checkpoint.py",
    "distillation_run.py",
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


@dataclass(frozen=True)
class DistillationJobConfig:
    """Isolated process settings; explicitly requested CUDA never falls back to CPU."""

    seed: int = 42
    device: str = "cuda:0"
    cpu_threads: int = 2
    device_bytes: int = 8 * 1024**3

    def __post_init__(self):
        integer(self.seed, 0, (1 << 64) - 1)
        integer(self.cpu_threads, 1, 256)
        integer(self.device_bytes, 128 * 1024**2, 8 * 1024**3)
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda:[0-7]", self.device):
            raise DistillationContractError("unsupported explicit trainer device")


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
        client=None
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
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and (
            not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count()
        ):
            raise DistillationContractError("requested CUDA device unavailable")
        torch.set_num_threads(min(config.cpu_threads, os.cpu_count() or 1))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        width = sum(
            sum(node.output_sizes) + sum(node.hidden_sizes) + (node.hidden_size or 0) * 8
            for node in manifest.nodes
        )
        self.device_estimate = (
            128 * 1024**2
            + manifest.total_parameters * 48
            + width * sequence.batch_size * sequence.bptt_steps * 96
            + manifest.total_state_size * sequence.batch_size * 16
        )
        if self.device_estimate > config.device_bytes:
            raise DistillationContractError("canonical training estimate exceeds device admission")
        if self.device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(self.device)
            if free < self.device_estimate:
                raise DistillationContractError("insufficient free CUDA memory for admitted job")
        self.model = CanonicalGraph(manifest, weights, self.device)
        self.trainer = CanonicalSequenceTrainer(self.model, sequence)
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
        self.job_id = uuid.uuid4().hex
        self.serial = 0
        self.attempted_updates = 0
        self.status = "running"
        self.last_error = None
        self.last_training = None
        self.last_validation = None
        self.online_metrics = {}
        self.last_windows = ()
        self.last_state = torch.zeros(
            sequence.batch_size, manifest.total_state_size, dtype=torch.float32, device=self.device
        )
        self.checkpoint_allowed = True
        self.last_save_time = time.monotonic()
        self.last_save_update = 0
        self.candidate_due = False
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
            self.candidate_due = (
                updated.optimizer_updated and self.trainer.completed_updates % 250 == 0
            )
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
            current - self.last_save_time >= 60
            or self.trainer.completed_updates - self.last_save_update >= 1000
        )

    def save(self, directory):
        """Publish one self-contained job checkpoint and retain the newest three valid files."""
        # The reader lazily constructs an owner; local resolution avoids initialization cycles.
        # pylint: disable=import-outside-toplevel
        from .distillation_checkpoint import save_job_checkpoint

        return save_job_checkpoint(self, directory)
