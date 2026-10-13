"""Replay ownership, stable stream splits and unknown-prefix continuity tests."""

from dataclasses import replace
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from pyrl_trainer.distillation_client import DistillationFetch
from pyrl_trainer.distillation_replay import TeacherReplay, event_weights, stream_partition
from pyrl_trainer.distillation_transport import (
    DatasetManifest,
    DistillationContractError,
    TeacherChunk,
)
from pyrl_trainer.sequence_distillation import SequenceBoundary

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "teacher_chunk_v1"
REFERENCE = json.loads((ROOT / "reference.json").read_text(encoding="utf-8"))
DATA = (ROOT / "chunk.bin").read_bytes()


# Named native stream fields remain explicit in this test-only metadata mutation helper.
# pylint: disable=too-many-arguments
def chunk(sequence, *, ordinal=1, step=42, physical=17, start=0, end=1, generation=1):
    """Modify only test reference metadata, then validate through the actual strict decoder."""
    data = bytearray(DATA)
    tail = 25 + data[24] + 96
    struct.pack_into("<QQIQBB", data, tail + 15, sequence, generation, 119, physical, start, end)
    first = tail + 47
    stride = 25 + 72 * 4
    for index in range(3):
        struct.pack_into("<QQ", data, first + index * stride, step + index * 4, ordinal + index)
        flag = data[first + index * stride + 20] & ~12
        if index == 0 and start == 1:
            flag |= 8
        if index == 2 and end == 2:
            flag |= 4
        data[first + index * stride + 20] = flag
        struct.pack_into(
            "<f", data, first + index * stride + 25 + 21 * 4, 1.0 if flag & 8 else -1.0
        )
    return TeacherChunk.decode(bytes(data))


def proposal(start, end, chunks=(), gaps=(), seed=42):
    """One coherent settled proposal over the frozen native immutable identity."""
    manifest = DatasetManifest.from_object(
        {
            "schemaVersion": 1,
            "runSeed": seed,
            "contract": REFERENCE["contract"],
            "sensorContractJson": REFERENCE["sensorContractJson"],
            "state": "active",
            "productionWatermark": f"{end:016x}",
            "storageWatermark": f"{end:016x}",
            "resolvedThrough": f"{end:016x}",
            "retentionFloor": "0000000000000001",
            "nativeDroppedChunks": "0000000000000000",
            "nativeDroppedSamples": "0000000000000000",
            "nativeDroppedBytes": "0000000000000000",
            "spoolDroppedChunks": "0000000000000000",
            "retainedChunks": len(chunks),
            "retainedBytes": f"{sum(len(c.raw_bytes) for c in chunks):016x}",
            "error": None,
        }
    )
    return DistillationFetch(manifest, start, end, tuple(chunks), tuple(gaps))


def test_replay_whole_object_eviction_idempotence_and_unknown_prefix():
    """A fragment surviving eviction starts unknown; a retry never resurrects an old chunk."""
    replay = TeacherReplay(max_chunks=1)
    first = proposal(0, 1, [chunk(1)])
    assert replay.admit(first)
    assert not replay.admit(first)
    second = proposal(1, 2, [chunk(2, ordinal=4, step=54, start=2)])
    assert replay.admit(second)
    assert [item.sequence for item in replay.chunks] == [2]
    assert replay.evicted_chunks == 1 and replay.evicted_samples == 3
    assert replay.owned_bytes <= replay.max_bytes
    assert [segment.boundary for segment in replay.segments()] == [SequenceBoundary.GAP]
    with pytest.raises(DistillationContractError, match="stale|regression"):
        replay.admit(first)


def test_seals_lives_generations_and_transport_loss_have_distinct_boundaries():
    """Ordinary seals continue memory; unknown ordinals require an explicit gap reset."""
    replay = TeacherReplay()
    chunks = [
        chunk(1),
        chunk(2, ordinal=4, step=54, start=2, end=2),
        chunk(3, ordinal=7, step=66, physical=18, start=1),
        chunk(5, ordinal=13, step=90, physical=18, start=2),
        chunk(6, ordinal=16, step=102, physical=18, start=4, generation=2),
    ]
    replay.admit(proposal(0, 6, chunks, [(4, 4)]))
    assert [segment.boundary for segment in replay.segments()] == [
        SequenceBoundary.CAPTURE_START,
        SequenceBoundary.CONTINUE,
        SequenceBoundary.LIFE,
        SequenceBoundary.GAP,
        SequenceBoundary.GENERATION,
    ]
    assert (
        len({segment.partition for segment in replay.segments() if segment.chunk.generation == 1})
        == 1
    )
    assert sum(len(list(replay.segments(name))) for name in ("training", "validation")) == 5


def test_gap_for_other_slots_does_not_destroy_proved_ordinal_continuity():
    """Global sequences multiplex owners; missing other-owner data cannot invent a local gap."""
    replay = TeacherReplay()
    replay.admit(proposal(0, 3, [chunk(1), chunk(3, ordinal=4, step=54, start=2)], [(2, 2)]))
    assert [segment.boundary for segment in replay.segments()][-1] == SequenceBoundary.CONTINUE


def test_replay_rejects_unsettled_missing_overlap_identity_before_mutation():
    """Malformed or foreign proposals cannot partially evict or advance replay ownership."""
    replay = TeacherReplay()
    replay.admit(proposal(0, 1, [chunk(1)]))
    saved = replay.chunks, replay.cursor, replay.owned_bytes
    for bad in (
        proposal(1, 3, [chunk(3)]),
        proposal(1, 3, [chunk(3)], [(2, 3)]),
        proposal(1, 2, [chunk(2)], seed=43),
        replace(proposal(1, 2, [chunk(2)]), next_cursor=3),
    ):
        with pytest.raises(DistillationContractError):
            replay.admit(bad)
        assert (replay.chunks, replay.cursor, replay.owned_bytes) == saved


def test_weights_and_counts_preserve_raw_unweighted_outcomes():
    """Overlapping strata receive one bounded multiplier; actual burn is not a request label."""
    replay = TeacherReplay()
    item = TeacherChunk.decode(DATA)
    replay.admit(proposal(0, 7, [item], [(1, 6)]))
    np.testing.assert_array_equal(event_weights(item), [1, 4, 4])
    counts = replay.raw_counts()
    assert counts["samples"] == 3 and counts["cruising"] == 1
    assert counts["boost_request"] == 2 and counts["realized_boost"] == 1
    assert counts["post_step_death"] == 1
    assert all(
        counts[label] == 2
        for label in ("hazard", "pursuit_escape", "boost", "transition", "saturation", "respawn")
    )


def test_stable_partition_is_seed_slot_generation_scoped_with_twenty_percent_validation():
    """Physical IDs, capture epochs and chunk boundaries cannot leak one stream between splits."""
    choices = [
        stream_partition(42, slot, generation)
        for slot in range(120)
        for generation in range(1, 101)
    ]
    assert 0.18 <= choices.count("validation") / len(choices) <= 0.22
    assert choices == [
        stream_partition(42, slot, generation)
        for slot in range(120)
        for generation in range(1, 101)
    ]
    assert choices != [
        stream_partition(43, slot, generation)
        for slot in range(120)
        for generation in range(1, 101)
    ]
    assert stream_partition(42, 119, (1 << 64) - 1) == "training"
    for args in ((True, 0, 1), (42, 120, 1), (42, 0, 0), (42, 0, 1 << 64)):
        with pytest.raises(DistillationContractError):
            stream_partition(*args)


def test_ram_pressure_rotates_whole_chunks_with_reserved_bookkeeping():
    """The byte ceiling includes raw bytes, decoded arrays and fixed metadata headroom."""
    replay = TeacherReplay(max_bytes=2 * 1024 * 1024)
    replay.admit(
        proposal(
            0,
            256,
            [
                chunk(i, ordinal=1 + 3 * (i - 1), step=42 + 12 * (i - 1), start=0 if i == 1 else 2)
                for i in range(1, 257)
            ],
        )
    )
    replay.admit(
        proposal(
            256,
            384,
            [
                chunk(i, ordinal=1 + 3 * (i - 1), step=42 + 12 * (i - 1), start=2)
                for i in range(257, 385)
            ],
        )
    )
    assert replay.owned_bytes <= replay.max_bytes
    assert replay.evicted_chunks > 0
    assert replay.evicted_samples == 3 * replay.evicted_chunks
    assert len(replay.chunks) + replay.evicted_chunks == 384
    assert replay.chunks[0].sequence == replay.evicted_chunks + 1
    assert next(replay.segments()).boundary == SequenceBoundary.GAP


def test_no_progress_retry_refreshes_closure_without_replaying_rows():
    """Final closure/counters can change without assigning another chunk."""
    replay = TeacherReplay()
    empty = proposal(0, 0)
    assert replay.admit(empty)
    stopped = replace(empty, manifest=replace(empty.manifest, state="stopped"))
    assert not replay.admit(stopped)
    assert replay.manifest.state == "stopped" and replay.cursor == 0
    assert not replay.chunks
