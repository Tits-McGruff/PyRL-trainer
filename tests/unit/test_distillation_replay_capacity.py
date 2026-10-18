"""Bounded length framing wraps genuine frozen Rust codec bytes without re-encoding them."""

from pathlib import Path
import struct

import pytest

from pyrl_trainer.distillation_replay_capacity_gate import _chunks
from pyrl_trainer.distillation_transport import MAX_CHUNK_BYTES, TeacherChunk

pytestmark = pytest.mark.unit
DATA = (Path(__file__).resolve().parents[1] / "fixtures/teacher_chunk_v1/chunk.bin").read_bytes()


def test_capacity_reader_preserves_actual_rust_objects_without_bulk_read(tmp_path):
    """Length framing remains opaque; the real decoder proves ownership and numeric bits."""
    path = tmp_path / "chunks.bin"
    prefix = struct.pack("<I", len(DATA))
    path.write_bytes((prefix + DATA) * 2)
    values = list(_chunks(path))
    assert values == [(prefix, DATA), (prefix, DATA)]
    chunk = TeacherChunk.decode(values[0][1])
    assert chunk.sequence == 7 and chunk.observations.shape == (3, 72)
    assert chunk.raw_bytes == DATA and not chunk.observations.flags.writeable


@pytest.mark.parametrize("encoded,reason", [
    (b"xx", "partial replay fixture length"),
    (struct.pack("<I", 0), "exceeds codec admission"),
    (struct.pack("<I", MAX_CHUNK_BYTES + 1), "exceeds codec admission"),
    (struct.pack("<I", len(DATA)) + DATA[:-1], "partial replay fixture object"),
])
def test_capacity_reader_rejects_unbounded_and_partial_objects(tmp_path, encoded, reason):
    """Corrupt framing cannot allocate the declared oversize or accept an incomplete object."""
    path = tmp_path / "chunks.bin"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match=reason):
        list(_chunks(path))
