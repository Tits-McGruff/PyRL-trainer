"""Load real Rust-encoded synthetic teacher chunks into bounded production replay.

This is a RAM/decoder/retention and independent-decode restore probe, without
HTTP, trainer optimization, game contention or a durable trainer checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

from .distillation_client import DistillationFetch
from .distillation_job import trainer_source_sha256
from .distillation_replay import TeacherReplay, REPLAY_FIXED_BYTES, REPLAY_METADATA_BYTES
from .distillation_transport import DatasetManifest, TeacherChunk, MAX_CHUNK_BYTES
from .live_distillation_gate import _peak_rss


def _arguments():
    """Bound every selected resource and require immutable input/create-new output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 64 <= args.ram_mib <= 2048 or not 1 <= args.deadline_seconds <= 1800:
        parser.error("RAM 64..2048 MiB and deadline 1..1800 seconds required")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("create-new output with existing parent required")
    if (args.fixture / "reference.json").stat().st_size > 1024**2:
        parser.error("reference exceeds metadata admission")
    return args


def _chunks(path):
    """Read one bounded Rust object at a time; reject truncation before decoding."""
    with path.open("rb") as stream:
        while prefix := stream.read(4):
            if len(prefix) != 4:
                raise ValueError("partial replay fixture length")
            length = struct.unpack("<I", prefix)[0]
            if not 1 <= length <= MAX_CHUNK_BYTES:
                raise ValueError("replay fixture object exceeds codec admission")
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("partial replay fixture object")
            yield prefix, data


def _witness(chunks):
    """Hash exact retained raw identity without copying arrays or collecting reports per row."""
    witness = hashlib.sha256()
    for chunk in chunks:
        witness.update(struct.pack("<Q", chunk.sequence))
        witness.update(hashlib.sha256(chunk.raw_bytes).digest())
    return witness.hexdigest()


def run():
    # pylint: disable=too-many-statements
    """Fill through rotation, then decode the retained suffix independently for restore."""
    args = _arguments()
    reference_bytes = (args.fixture / "reference.json").read_bytes()
    reference = json.loads(reference_bytes)
    if reference["schemaVersion"] != 1 or not 64 <= reference["chunks"] <= 16384:
        raise ValueError("unsupported replay fixture")
    manifest = DatasetManifest.from_object(reference["dataset"])
    bundle = args.fixture / "chunks.bin"
    before_stat = bundle.stat()
    if before_stat.st_size != reference["bundleBytes"]:
        raise ValueError("replay bundle byte identity mismatch")
    replay = TeacherReplay(max_bytes=args.ram_mib * 1024**2)
    started = time.perf_counter()
    cpu_start = time.process_time()
    initial_rss = _peak_rss()
    report = {"schemaVersion": 1, "gate": "canonical-replay-capacity-v1",
              "scope": ("Rust-encoded synthetic teacher chunks, real Python decoder/replay and "
                        "independent-decode in-process restore. The two replay owners coexist "
                        "during restore; process peak includes both. No HTTP, optimizer, durable "
                        "trainer checkpoint, live game contention or integrated "
                        "soak is qualified."),
              "hostSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trainerSourceSha256": trainer_source_sha256(),
              "referenceSha256": hashlib.sha256(reference_bytes).hexdigest(),
              "nativeSourceSha256": reference["sourceSha256"],
              "fixtureHostCanonicalLfSha256":
                  reference["hostCanonicalLfSha256"],
              "bundleSha256": reference["bundleSha256"],
              "options": {"ramMiB": args.ram_mib},
              "baselinePeakRssBytes": initial_rss,
              "status": "failed", "qualificationComplete": False}
    failure = None
    try:
        digest = hashlib.sha256()
        pending = []
        cost = None
        count = 0
        for prefix, data in _chunks(bundle):
            if time.perf_counter() - started > args.deadline_seconds:
                raise TimeoutError("replay capacity total deadline")
            digest.update(prefix)
            digest.update(data)
            chunk = TeacherChunk.decode(data)
            count += 1
            assert chunk.sequence == count and chunk.contract == manifest.contract
            assert len(chunk.steps) == reference["samplesPerChunk"]
            current_cost = chunk.byte_cost + REPLAY_METADATA_BYTES
            if cost is None:
                cost = current_cost
            assert cost == current_cost
            pending.append(chunk)
            if len(pending) < 8 and count < reference["chunks"]:
                continue
            proposal = DistillationFetch(manifest, replay.cursor, count, tuple(pending), ())
            assert replay.admit(proposal)
            assert not replay.admit(proposal)  # Exact retry cannot duplicate or resurrect rows.
            expected = min(count, (replay.max_bytes - REPLAY_FIXED_BYTES) // cost)
            assert replay.retained_chunks == expected
            assert replay.owned_bytes == REPLAY_FIXED_BYTES + expected * cost <= replay.max_bytes
            pending.clear()
            if count % 1024 == 0:
                print("[replay-capacity] decoded", count, flush=True)
        assert count == reference["chunks"] and digest.hexdigest() == reference["bundleSha256"]
        assert replay.evicted_chunks > 0, "fixture must load through the actual RAM ceiling"
        assert replay.evicted_chunks + replay.retained_chunks == count
        assert replay.evicted_samples == replay.evicted_chunks * reference["samplesPerChunk"]
        assert cost is not None
        assert replay.max_bytes - replay.owned_bytes < cost
        fields = replay.checkpoint_fields()
        original = replay.chunks
        witness = _witness(original)
        report.update({"decodedChunks": count, "retainedChunks": replay.retained_chunks,
                       "ownedReplayBytes": replay.owned_bytes, "perChunkChargedBytes": cost,
                       "evictedChunks": replay.evicted_chunks,
                       "evictedSamples": replay.evicted_samples,
                       "fillMs": (time.perf_counter() - started) * 1000,
                       "filledPeakRssBytes": _peak_rss(), "retainedSha256": witness})
        restoring = time.perf_counter()
        restored_chunks = []
        first = original[0].sequence
        for sequence, (_prefix, data) in enumerate(_chunks(bundle), 1):
            if time.perf_counter() - started > args.deadline_seconds:
                raise TimeoutError("replay capacity restore deadline")
            if sequence >= first:
                restored_chunks.append(TeacherChunk.decode(data))
        restored = TeacherReplay.restore_snapshot(manifest, restored_chunks, fields)
        assert restored.chunks[0] is not original[0]
        assert _witness(restored.chunks) == witness
        assert restored.checkpoint_fields() == fields
        assert restored.owned_bytes == replay.owned_bytes
        after_stat = bundle.stat()
        assert (before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_ino) == (
            after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_ino)
        report.update({"restoreMs": (time.perf_counter() - restoring) * 1000,
                       "independentlyDecodedRestoreExact": True, "status": "passed",
                       "qualificationComplete": True})
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure = error
        report["failure"] = type(error).__name__ + ": " + str(error)
    report["elapsedMs"] = (time.perf_counter() - started) * 1000
    report["processCpuMs"] = (time.process_time() - cpu_start) * 1000
    report["peakRssBytes"] = _peak_rss()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if failure:
        raise failure
    return report


if __name__ == "__main__":
    run()
