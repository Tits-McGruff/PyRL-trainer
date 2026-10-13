"""Bounded whole-chunk replay and versioned recurrent-stream partitioning.

Replay commits local proposals only. The durable job must checkpoint replay,
HTTP cursor, sampler, optimizer and RNG together; this buffer is not that owner.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from .distillation_client import DistillationFetch
from .distillation_transport import DistillationContractError, TeacherChunk, integer
from .sequence_distillation import SequenceBoundary

PARTITION_REVISION = 1
PARTITION_DOMAIN = b"slither.distillation-partition.v1\0"
REPLAY_METADATA_BYTES = 4096
REPLAY_FIXED_BYTES = 256 * 1024
_RECORDED_RESETS = {
    "captureStart": SequenceBoundary.CAPTURE_START,
    "generationStart": SequenceBoundary.GENERATION,
}
STRATUM_LABELS = ("hazard", "pursuit_escape", "boost", "transition", "saturation", "respawn")


def stream_partition(run_seed, baseline_slot, generation):
    """Assign every physical life in one seed/slot/generation stream to the same 20% split."""
    integer(run_seed, 0, (1 << 32) - 1)
    integer(baseline_slot, 0, 119)
    integer(generation, 1, (1 << 64) - 1)
    digest = hashlib.sha256(
        PARTITION_DOMAIN + struct.pack("<IIQ", run_seed, baseline_slot, generation)
    ).digest()
    return "validation" if int.from_bytes(digest[:8], "little") % 5 == 0 else "training"


def event_weights(chunk):
    """Rare-event weights affect training loss; raw validation/counts remain unchanged."""
    if not isinstance(chunk, TeacherChunk):
        raise DistillationContractError("expected a decoded teacher chunk")
    weights = np.where(chunk.strata != 0, np.float32(4), np.float32(1))
    weights.flags.writeable = False
    return weights


@dataclass(frozen=True)
class ReplaySegment:
    """A delivered chunk and its actual first-row continuity boundary, without row copies."""

    chunk: TeacherChunk
    boundary: SequenceBoundary
    partition: str


def _boundary(previous, chunk):
    """Physical markers cannot reconstruct an evicted or missing recurrent prefix."""
    recorded = _RECORDED_RESETS.get(chunk.start_reason)
    if recorded is not None:
        return recorded
    if chunk.start_reason == "gap" or previous is None:
        return SequenceBoundary.GAP
    if (
        previous.generation != chunk.generation
        or int(chunk.ordinals[0]) != int(previous.ordinals[-1]) + 1
        or int(chunk.steps[0]) <= int(previous.steps[-1])
    ):
        return SequenceBoundary.GAP
    if chunk.start_reason == "physicalLifeStart":
        known_life = (
            previous.physical_snake_id != chunk.physical_snake_id
            and previous.end_reason == "lifeEnded"
        )
        return SequenceBoundary.LIFE if known_life else SequenceBoundary.GAP
    if chunk.physical_snake_id == previous.physical_snake_id and previous.end_reason in (
        "maximumSamples",
        "simulatedSecond",
    ):
        return SequenceBoundary.CONTINUE
    return SequenceBoundary.GAP


class TeacherReplay:
    """FIFO whole immutable chunks under independent RAM/object admission ceilings."""

    def __init__(self, *, max_bytes=256 * 1024 * 1024, max_chunks=65536):
        self.max_bytes = integer(max_bytes, 2 * 1024 * 1024, 512 * 1024 * 1024)
        self.max_chunks = integer(max_chunks, 1, 65536)
        self.manifest = None
        self.cursor = 0
        self.owned_bytes = REPLAY_FIXED_BYTES
        self.evicted_chunks = 0
        self.evicted_samples = 0
        self._chunks = deque()
        self._last_proposal = None

    @staticmethod
    def _cost(chunk):
        """Charge raw/decoded arrays and reserved Python/container metadata headroom."""
        return chunk.byte_cost + REPLAY_METADATA_BYTES

    @property
    def chunks(self):
        """Bounded immutable-order view; no observation or payload copies."""
        return tuple(self._chunks)

    def admit(self, fetched):
        # Validate/stage the entire proposal before replacing the retained replay.
        # pylint: disable=too-many-branches,too-many-statements
        """Commit local ownership atomically; exact retries never resurrect evicted data."""
        if not isinstance(fetched, DistillationFetch):
            raise DistillationContractError("expected a validated HTTP proposal")
        manifest = fetched.manifest
        if self.manifest is not None:
            if not self.manifest.compatible(manifest):
                raise DistillationContractError("replay immutable dataset identity mismatch")
            if (
                manifest.production_watermark < self.manifest.production_watermark
                or manifest.resolved_through < self.manifest.resolved_through
                or manifest.storage_watermark < self.manifest.storage_watermark
            ):
                raise DistillationContractError("replay dataset watermark regression")
        if not 0 <= fetched.start_cursor <= fetched.next_cursor <= manifest.resolved_through:
            raise DistillationContractError("replay proposal exceeds settled availability")
        if len(fetched.chunks) > 256 or len(fetched.gaps) > 257:
            raise DistillationContractError("replay proposal object limit")
        intervals, identities = [], []
        previous = fetched.start_cursor
        for chunk in fetched.chunks:
            if (
                not isinstance(chunk, TeacherChunk)
                or chunk.contract != manifest.contract
                or not previous < chunk.sequence <= fetched.next_cursor
            ):
                raise DistillationContractError("replay proposal chunk identity/order mismatch")
            cost = self._cost(chunk)
            if cost > self.max_bytes - REPLAY_FIXED_BYTES:
                raise DistillationContractError("complete replay object exceeds RAM admission")
            digest = hashlib.sha256(chunk.raw_bytes).hexdigest()
            identities.append((chunk.sequence, digest))
            intervals.append((chunk.sequence, chunk.sequence))
            previous = chunk.sequence
        for gap in fetched.gaps:
            if not isinstance(gap, tuple) or len(gap) != 2:
                raise DistillationContractError("invalid replay unavailable interval")
            low, high = gap
            integer(low, fetched.start_cursor + 1, fetched.next_cursor)
            integer(high, low, fetched.next_cursor)
            intervals.append(gap)
        covered = fetched.start_cursor
        for low, high in sorted(intervals):
            if low != covered + 1:
                raise DistillationContractError("replay proposal missing/overlapping disposition")
            covered = high
        if covered != fetched.next_cursor:
            raise DistillationContractError("replay proposal tail is not settled")
        identity = (fetched.start_cursor, fetched.next_cursor, tuple(identities), fetched.gaps)
        if identity == self._last_proposal:
            self.manifest = manifest
            return False
        if fetched.start_cursor != self.cursor:
            raise DistillationContractError("stale replay proposal cursor")
        planned = deque(self._chunks)
        owned = self.owned_bytes
        dropped = dropped_samples = 0
        for chunk in fetched.chunks:
            cost = self._cost(chunk)
            while planned and (owned + cost > self.max_bytes or len(planned) >= self.max_chunks):
                old = planned.popleft()
                owned -= self._cost(old)
                dropped += 1
                dropped_samples += len(old.steps)
            planned.append(chunk)
            owned += cost
        self._chunks = planned
        self.owned_bytes = owned
        self.evicted_chunks += dropped
        self.evicted_samples += dropped_samples
        self.cursor = fetched.next_cursor
        self.manifest = manifest
        self._last_proposal = identity
        return True

    def segments(self, partition=None):
        """Yield correct boundaries; chunk seals alone do not reset recurrent state."""
        if partition not in (None, "training", "validation"):
            raise DistillationContractError("unsupported replay partition")
        previous = {}
        for chunk in self._chunks:
            selected = stream_partition(
                self.manifest.run_seed, chunk.baseline_slot, chunk.generation
            )
            boundary = _boundary(previous.get(chunk.baseline_slot), chunk)
            previous[chunk.baseline_slot] = chunk
            if partition is None or selected == partition:
                yield ReplaySegment(chunk, boundary, selected)

    def raw_counts(self):
        """Report unweighted retained rows and overlapping rare-event strata without resampling."""
        counts = {label: 0 for label in STRATUM_LABELS}
        counts.update(samples=0, cruising=0, boost_request=0, realized_boost=0, post_step_death=0)
        for chunk in self._chunks:
            counts["samples"] += len(chunk.steps)
            counts["cruising"] += int(np.count_nonzero(chunk.strata == 0))
            for index, label in enumerate(STRATUM_LABELS):
                counts[label] += int(np.count_nonzero(chunk.strata & (1 << index)))
            for index, label in enumerate(("boost_request", "realized_boost", "post_step_death")):
                counts[label] += int(np.count_nonzero(chunk.flags & (1 << index)))
        return counts
