"""Exercise actual native corpus bytes and reject malformed ownership before training."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyrl_trainer.tactical_corpus import TacticalCorpusError, load_tactical_corpus

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parents[1] / "fixtures" / "tactical_corpus_v1"


@pytest.fixture(name="corpus_root")
def native_corpus_root(tmp_path):
    """Unpack exact native bytes into a private mutation workspace."""
    receipt = json.loads((FIXTURE / "receipt.json").read_text(encoding="utf-8"))
    for name, expected in receipt["files"].items():
        compressed = (FIXTURE / (name + ".gz")).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == expected["gzipSha256"]
        value = gzip.decompress(compressed)
        assert hashlib.sha256(value).hexdigest() == expected["sha256"]
        (tmp_path / name).write_bytes(value)
    return tmp_path


def metadata(root):
    """Read the small completion manifest for an explicit rejection case."""
    return json.loads((root / "corpus.json").read_text(encoding="utf-8"))


def publish(root, value):
    """Replace only a test-owned manifest."""
    (root / "corpus.json").write_text(json.dumps(value), encoding="utf-8")


def test_native_rows_are_read_only_and_seed_splits_preserve_whole_trajectories(corpus_root):
    """Retain complete life windows and hold out distinct native seeds."""
    corpus = load_tactical_corpus(corpus_root)
    receipt = json.loads((FIXTURE / "receipt.json").read_text(encoding="utf-8"))
    assert corpus.source_sha256 == receipt["nativeSourceSha256"]
    assert corpus.observations.shape == (receipt["rows"], 264)
    assert corpus.actions.shape == (receipt["rows"], 2)
    assert len(corpus.sequences) == 56
    assert corpus.sensor.sensor_count == 264
    assert corpus.control_interval_seconds == 0.06
    assert not corpus.observations.flags.writeable
    assert not corpus.actions.flags.writeable
    assert not corpus.steps.flags.writeable
    train, validation = corpus.split_by_seed({2})
    assert len(train) == len(validation) == 28
    assert {sequence.seed for sequence in train} == {1}
    assert {sequence.seed for sequence in validation} == {2}
    assert sum(sequence.rows for sequence in train + validation) == receipt["rows"]
    assert (np.diff(corpus.steps[:corpus.sequences[0].rows].astype(np.int64)) > 1).all()
    for invalid in [set(), {1, 2}, {3}, {True}]:
        with pytest.raises(TacticalCorpusError):
            corpus.split_by_seed(invalid)


@pytest.mark.parametrize("field,value", [
    ("schemaVersion", True), ("schemaVersion", 2), ("inputSize", 263),
    ("rows", 999999999), ("rows", True), ("seedCount", 0),
    ("maximumSteps", 601), ("controlIntervalSeconds", 0.1),
    ("sourceSha256", "x" * 64), ("sensorSha256", "0" * 64),
])
def test_invalid_header_rejects(corpus_root, field, value):
    """Reject unadmitted headers before exposing tensors."""
    value_metadata = metadata(corpus_root)
    value_metadata[field] = value
    publish(corpus_root, value_metadata)
    with pytest.raises(ValueError):
        load_tactical_corpus(corpus_root)


@pytest.mark.parametrize("change", [
    "path", "duplicate", "length", "matrix", "identity", "overlap", "unowned",
])
def test_object_and_sequence_ownership_rejects(corpus_root, change):
    """Reject path traversal, overlaps, duplicates and incomplete matrices."""
    value = metadata(corpus_root)
    if change == "path":
        value["objects"][0]["file"] = "../observations.f32"
    elif change == "duplicate":
        value["objects"][1] = value["objects"][0]
    elif change == "length":
        value["objects"][0]["bytes"] += 4
    elif change == "matrix":
        value["sequences"].pop()
    elif change == "identity":
        value["sequences"][1]["seed"] = True
    elif change == "overlap":
        value["sequences"][1]["rowOffset"] -= 1
    elif change == "unowned":
        value["sequences"][-1]["rows"] -= 1
    publish(corpus_root, value)
    with pytest.raises(TacticalCorpusError):
        load_tactical_corpus(corpus_root)


@pytest.mark.parametrize("name,index,value", [
    ("observations.f32", 0, float("nan")), ("observations.f32", 0, 1.1),
    ("observations.f32", 21, -1), ("actions.f32", 0, 2),
    ("actions.f32", 1, 0.5), ("steps.u32", 0, 2), ("steps.u32", 1, 1),
])
def test_rehashed_invalid_values_do_not_bypass_semantic_validation(corpus_root, name, index, value):
    """Validate numerical and chronology semantics even with matching hashes."""
    dtype = "<u4" if name.endswith("u32") else "<f4"
    values = np.frombuffer((corpus_root / name).read_bytes(), dtype=dtype).copy()
    values[index] = value
    data = values.tobytes()
    (corpus_root / name).write_bytes(data)
    manifest = metadata(corpus_root)
    receipt = next(item for item in manifest["objects"] if item["file"] == name)
    receipt["sha256"] = hashlib.sha256(data).hexdigest()
    publish(corpus_root, manifest)
    with pytest.raises(TacticalCorpusError):
        load_tactical_corpus(corpus_root)


def test_partial_changed_oversized_and_duplicate_metadata_are_rejected(corpus_root):
    """Reject incomplete or changed file publication."""
    actions = corpus_root / "actions.f32"
    original = actions.read_bytes()
    actions.write_bytes(original + b"x")
    with pytest.raises(TacticalCorpusError):
        load_tactical_corpus(corpus_root)
    actions.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(TacticalCorpusError, match="hash mismatch"):
        load_tactical_corpus(corpus_root)
    actions.write_bytes(original)
    manifest = corpus_root / "corpus.json"
    original_manifest = manifest.read_bytes()
    manifest.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(TacticalCorpusError, match="admission"):
        load_tactical_corpus(corpus_root)
    manifest.write_bytes(b'{"schemaVersion":1,' + original_manifest[1:])
    with pytest.raises(TacticalCorpusError, match="JSON"):
        load_tactical_corpus(corpus_root)
