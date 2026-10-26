"""Bounded independent replay windows and checkpointable sampler RNG.

Every window starts with declared fresh/unknown state and burns in its bounded
prefix. No interpolation or unbounded recurrent-history reconstruction occurs.
"""

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import struct

import numpy as np
import torch

from .canonical_graph import CanonicalGraphManifest
from .distillation_replay import PARTITION_REVISION, TeacherReplay
from .distillation_transport import DistillationContractError, exact_keys, integer
from .sequence_distillation import SequenceBatch, SequenceBoundary, SequenceTrainingConfig

MAX_INDEX_BYTES = 64 * 1024 * 1024
INDEX_CHUNK_BYTES = 512
INDEX_STREAM_BYTES = 256
BATCH_METADATA_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class WindowIdentity:
    """One bounded delivered-observation span, independently sampled within one split."""

    baseline_slot: int
    generation: int
    first_sample: int
    first_training_sample: int
    last_sample: int
    sample_count: int


@dataclass(frozen=True)
class SampledWindows:
    """CPU tensors plus exact spans; caller transfers once through the canonical trainer."""

    batch: SequenceBatch
    windows: tuple
    partition: str
    index_sha256: str


@dataclass(frozen=True)
class _Stream:
    """Chunk references and cumulative row extents; no per-observation index objects."""

    segments: tuple
    ends: np.ndarray
    count: int
    partition: str


class ReplayDataUnavailable(DistillationContractError):
    """Normal waiting condition: retained streams do not yet contain a training window."""


class ReplayWindowSampler:
    """Uniform delivered-row windows with separate PCG64 state, beside PPO sampling."""

    def __init__(
        self,
        replay,
        manifest,
        config=SequenceTrainingConfig(),
        *,
        seed=42,
        tensor_bytes=128 * 1024 * 1024
    ):
        # Independent declared tensor admission and RNG seed remain explicit.
        # pylint: disable=too-many-arguments
        if (
            not isinstance(replay, TeacherReplay)
            or not isinstance(config, SequenceTrainingConfig)
            or not isinstance(manifest, CanonicalGraphManifest)
        ):
            raise DistillationContractError("invalid canonical replay sampler owner/configuration")
        self.replay = replay
        self.manifest = manifest
        self.config = config
        integer(seed, 0, (1 << 64) - 1)
        self.tensor_bytes = integer(tensor_bytes, 2 * 1024 * 1024, 256 * 1024 * 1024)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self.drawn_windows = 0
        self._signature = None
        self._streams = ()
        self.index_sha256 = None
        self.index_bytes = 0
        time, batch = config.burn_in_steps + config.bptt_steps, config.batch_size
        # Observations, turns, labels, weights, active mask, int64 boundaries and initial state.
        self.batch_bytes = (
            time * batch * (manifest.input_size * 4 + 4 + 1 + 4 + 1 + 8)
            + batch * manifest.total_state_size * 4
            + BATCH_METADATA_BYTES
        )
        if self.batch_bytes > self.tensor_bytes:
            raise DistillationContractError("complete sampler batch exceeds tensor RAM admission")

    def _refresh(self):
        """Rebuild only on replay progress/retention change; bind exact immutable data identity."""
        replay = self.replay
        dataset = replay.manifest
        if dataset is None:
            raise DistillationContractError("sampler has no validated replay dataset")
        if (
            dataset.contract.sensor_sha256 != self.manifest.sensor_sha256
            or dataset.sensor_json
            != json.loads(self.manifest.canonical_json)["sensorCanonicalJson"]
            or dataset.contract.control_interval_seconds != self.manifest.control_interval_seconds
            or dataset.contract.input_size != self.manifest.input_size
        ):
            raise DistillationContractError("sampler graph/sensor/deployment contract mismatch")
        signature = replay.cursor, replay.evicted_chunks, replay.retained_chunks
        if signature == self._signature:
            return
        if replay.retained_chunks * 2 * INDEX_CHUNK_BYTES > MAX_INDEX_BYTES:
            raise DistillationContractError("sampler stream index exceeds RAM admission")
        groups = {}
        digest = hashlib.sha256(b"slither.distillation-replay-index.v1\0")
        digest.update(bytes.fromhex(dataset.contract.capture_epoch))
        digest.update(struct.pack("<IQ", dataset.run_seed, replay.cursor))
        digest.update(
            json.dumps(asdict(dataset.contract), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(self.manifest.canonical_json)
        digest.update(
            json.dumps(asdict(self.config), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for segment in replay.segments():
            item = segment.chunk
            groups.setdefault((item.baseline_slot, item.generation), []).append(segment)
            if (
                2 * (replay.retained_chunks * INDEX_CHUNK_BYTES + len(groups) * INDEX_STREAM_BYTES)
                > MAX_INDEX_BYTES
            ):
                raise DistillationContractError("sampler stream index exceeds RAM admission")
            digest.update(struct.pack("<Q", item.sequence))
            digest.update(hashlib.sha256(item.raw_bytes).digest())
        index_bytes = 2 * (
            replay.retained_chunks * INDEX_CHUNK_BYTES + len(groups) * INDEX_STREAM_BYTES
        )
        if index_bytes > MAX_INDEX_BYTES:
            raise DistillationContractError("sampler stream index exceeds RAM admission")
        streams = []
        for key in sorted(groups):
            segments = groups[key]
            ends = np.cumsum([len(segment.chunk.steps) for segment in segments], dtype=np.int64)
            ends.flags.writeable = False
            streams.append(_Stream(tuple(segments), ends, int(ends[-1]), segments[0].partition))
        self._streams = tuple(streams)
        self.index_sha256 = digest.hexdigest()
        self.index_bytes = index_bytes
        self._signature = signature

    def sample(self, partition="training"):
        # One bounded allocation/fill transaction precedes RNG publication.
        # pylint: disable=too-many-locals,too-many-statements
        """Sample independent windows; no train/validation crossing or RNG advance on rejection."""
        if partition not in ("training", "validation"):
            raise DistillationContractError("unsupported sampler partition")
        self._refresh()
        if self.drawn_windows + self.config.batch_size > (1 << 64) - 1:
            raise DistillationContractError("sampler counter exhausted")
        burn = self.config.burn_in_steps
        streams = tuple(
            stream
            for stream in self._streams
            if stream.partition == partition and stream.count > burn
        )
        eligible = np.cumsum([stream.count - burn for stream in streams], dtype=np.int64)
        if eligible.size == 0:
            raise ReplayDataUnavailable("no retained stream has training rows after burn-in")
        rng = np.random.Generator(np.random.PCG64(0))
        rng.bit_generator.state = copy.deepcopy(self._rng.bit_generator.state)
        time, batch = burn + self.config.bptt_steps, self.config.batch_size
        observations = np.zeros((time, batch, self.manifest.input_size), dtype=np.float32)
        turns = np.zeros((time, batch), dtype=np.float32)
        labels = np.zeros((time, batch), dtype=np.bool_)
        weights = np.zeros((time, batch), dtype=np.float32)
        active = np.zeros((time, batch), dtype=np.bool_)
        boundaries = np.zeros((time, batch), dtype=np.int64)
        identities = []
        for column in range(batch):
            chosen = int(rng.integers(int(eligible[-1])))
            selected = int(np.searchsorted(eligible, chosen, side="right"))
            stream = streams[selected]
            training = burn + chosen - (int(eligible[selected - 1]) if selected else 0)
            begin, end = training - burn, min(training + self.config.bptt_steps, stream.count)
            position, first_sample, training_sample, last_sample = 0, None, None, None
            while begin < end:
                index = int(np.searchsorted(stream.ends, begin, side="right"))
                segment = stream.segments[index]
                chunk = segment.chunk
                offset = begin - (int(stream.ends[index - 1]) if index else 0)
                count = min(end - begin, len(chunk.steps) - offset)
                target = slice(position, position + count)
                source = slice(offset, offset + count)
                observations[target, column] = chunk.observations[source]
                turns[target, column] = chunk.applied_turn[source]
                labels[target, column] = (chunk.flags[source] & 1) != 0
                weights[target, column] = np.where(chunk.strata[source] != 0, 4, 1)
                active[target, column] = True
                boundary = segment.boundary if offset == 0 else SequenceBoundary.CONTINUE
                if position == 0 and boundary in (SequenceBoundary.CONTINUE, SequenceBoundary.LIFE):
                    boundary = SequenceBoundary.GAP
                boundaries[position, column] = boundary
                if first_sample is None:
                    first_sample = int(chunk.ordinals[offset])
                if position <= burn < position + count:
                    training_sample = int(chunk.ordinals[offset + burn - position])
                last_sample = int(chunk.ordinals[offset + count - 1])
                begin += count
                position += count
            item = stream.segments[0].chunk
            identities.append(
                WindowIdentity(
                    item.baseline_slot, item.generation, first_sample, training_sample, last_sample,
                    position,
                )
            )
        data = SequenceBatch(
            *(
                torch.from_numpy(array)
                for array in (observations, turns, labels, weights, active, boundaries)
            ),
            torch.zeros(batch, self.manifest.total_state_size, dtype=torch.float32)
        )
        # Dimensions/source finite values were already admitted by the decoder/configuration.
        self._rng.bit_generator.state = rng.bit_generator.state
        self.drawn_windows += batch
        return SampledWindows(data, tuple(identities), partition, self.index_sha256)

    def checkpoint_fields(self):
        """Bind RNG/counters to exactly the required retained replay snapshot."""
        self._refresh()
        return {
            "schemaVersion": 1,
            "partitionRevision": PARTITION_REVISION,
            "indexSha256": self.index_sha256,
            "drawnWindows": self.drawn_windows,
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def restore(self, fields):
        """Reject missing/different replay before publishing saved sampler state."""
        exact_keys(fields, "schemaVersion partitionRevision indexSha256 drawnWindows rng")
        integer(fields["schemaVersion"], 1, 1)
        integer(fields["partitionRevision"], PARTITION_REVISION, PARTITION_REVISION)
        count = integer(fields["drawnWindows"], 0, (1 << 64) - 1)
        self._refresh()
        if fields["indexSha256"] != self.index_sha256:
            raise DistillationContractError("sampler required replay snapshot mismatch")
        exact_keys(fields["rng"], "bit_generator state has_uint32 uinteger")
        if fields["rng"]["bit_generator"] != "PCG64":
            raise DistillationContractError("unsupported sampler generator")
        exact_keys(fields["rng"]["state"], "state inc")
        integer(fields["rng"]["state"]["state"], 0, (1 << 128) - 1)
        increment = integer(fields["rng"]["state"]["inc"], 1, (1 << 128) - 1)
        if not increment & 1:
            raise DistillationContractError("invalid PCG64 stream increment")
        integer(fields["rng"]["has_uint32"], 0, 1)
        integer(fields["rng"]["uinteger"], 0, (1 << 32) - 1)
        try:
            proposed = np.random.Generator(np.random.PCG64(0))
            proposed.bit_generator.state = copy.deepcopy(fields["rng"])
        except (TypeError, ValueError, KeyError) as error:
            raise DistillationContractError("invalid sampler PCG64 checkpoint") from error
        self._rng = proposed
        self.drawn_windows = count
