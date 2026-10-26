"""Python decoding against immutable bytes emitted by Rust's actual codec."""

from dataclasses import replace
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from pyrl_trainer.distillation_transport import (
    DatasetManifest,
    DistillationContractError,
    TeacherChunk,
    TeacherContract,
    TeacherDescriptor,
    bounded_json,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "teacher_chunk_v1"
REFERENCE = json.loads((ROOT / "reference.json").read_text(encoding="utf-8"))
DATA = (ROOT / "chunk.bin").read_bytes()
CONTRACT = TeacherContract.from_object(REFERENCE["contract"])
DESCRIPTOR = TeacherDescriptor.from_object(REFERENCE["descriptor"], CONTRACT)


def manifest(**updates):
    """Build bounded protocol metadata while retaining exact native canonical bytes."""
    result = {
        "schemaVersion": 1,
        "runSeed": 42,
        "contract": REFERENCE["contract"],
        "sensorContractJson": REFERENCE["sensorContractJson"],
        "state": "active",
        "productionWatermark": "0000000000000009",
        "storageWatermark": "0000000000000007",
        "retentionFloor": "0000000000000007",
        "resolvedThrough": "0000000000000009",
        "nativeDroppedChunks": "0000000000000008",
        "nativeDroppedSamples": "0000000000000000",
        "nativeDroppedBytes": "0000000000000000",
        "spoolDroppedChunks": "0000000000000000",
        "retainedChunks": 1,
        "retainedBytes": f"{len(DATA):016x}",
        "error": None,
    }
    result.update(updates)
    return result


def test_rust_reference_preserves_identity_float_bits_flags_and_readonly_ownership():
    """Full u64 identities and signed-zero observations survive the portable decoder."""
    chunk = TeacherChunk.decode(DATA, DESCRIPTOR)
    assert chunk.contract.run_id == "codec-\u03b1"
    assert chunk.generation == chunk.physical_snake_id == (1 << 64) - 1
    assert chunk.sequence == 7 and chunk.baseline_slot == 119
    assert chunk.steps.tolist() == [42, 46, 50]
    assert chunk.ordinals.tolist() == [1, 2, 3]
    assert chunk.observations.shape == (3, 72)
    assert chunk.observations[:, 0].view(np.uint32).tolist() == [0x80000000] * 3
    assert chunk.applied_turn.view(np.uint32)[0] == 0x80000000
    assert chunk.boost_request.tolist() == [True, False, True]
    assert chunk.flags.tolist() == [1, 0, 7]
    assert chunk.strategy.tolist() == chunk.life_stage.tolist() == [0, 1, 2]
    assert chunk.strata.tolist() == [0, 63, 63]
    assert chunk.start_reason == "captureStart" and chunk.end_reason == "lifeEnded"
    assert chunk.raw_bytes == DATA and chunk.byte_cost < 2 * len(DATA)
    with pytest.raises(ValueError):
        chunk.observations[0, 0] = 1


def test_every_truncated_prefix_trailing_byte_and_unsupported_version_reject():
    """Dimensions are proven before allocation and no partial object is accepted."""
    for end in range(len(DATA)):
        with pytest.raises(DistillationContractError):
            TeacherChunk.decode(DATA[:end])
    for invalid in (DATA + b"\0", b"OTHER\0\0\0" + DATA[8:], DATA * 1024):
        with pytest.raises(DistillationContractError):
            TeacherChunk.decode(invalid)


@pytest.mark.parametrize(
    "change",
    [
        "reserved",
        "nan-turn",
        "nan-input",
        "bad-policy",
        "bad-stage",
        "bad-stratum",
        "early-death",
        "duplicate-step",
        "ordinal-gap",
        "late-first",
        "fake-birth",
    ],
)
def test_invalid_native_boundaries_finite_values_and_chronology_reject(change):
    """Reject malformed flags, physical boundaries and interior missing observations."""
    data = bytearray(DATA)
    first = 168 + DATA[24]
    stride = 25 + 72 * 4
    if change == "reserved":
        data[first + 20] = 128
    elif change == "nan-turn":
        struct.pack_into("<I", data, first + 16, 0x7FC00000)
    elif change == "nan-input":
        struct.pack_into("<I", data, first + 25, 0x7FC00000)
    elif change == "bad-policy":
        data[first + 21] = 4
    elif change == "bad-stage":
        data[first + 22] = 3
    elif change == "bad-stratum":
        struct.pack_into("<H", data, first + 23, 64)
    elif change == "early-death":
        data[first + 20] |= 4
    elif change == "duplicate-step":
        struct.pack_into("<Q", data, first + stride, 42)
    elif change == "ordinal-gap":
        struct.pack_into("<Q", data, first + stride + 8, 7)
    elif change == "late-first":
        data[first + stride + 20] |= 8
    else:
        data[first - 4] = 1  # PhysicalLifeStart lacks an actual first delivery.
    with pytest.raises(DistillationContractError):
        TeacherChunk.decode(bytes(data))


def test_binary_descriptor_identity_digest_and_width_cannot_be_substituted():
    """A same-width alternate sensor digest remains an incompatible observation contract."""
    for descriptor in (
        replace(DESCRIPTOR, sha256="00" * 32),
        replace(DESCRIPTOR, sequence=8),
        replace(DESCRIPTOR, contract=replace(CONTRACT, sensor_sha256="00" * 32)),
    ):
        with pytest.raises(DistillationContractError):
            TeacherChunk.decode(DATA, descriptor)


def test_manifest_requires_seed_exact_canonical_bytes_and_coherent_watermarks():
    """No seed or structural identity may be guessed for an exact restart/split."""
    parsed = DatasetManifest.from_object(manifest())
    assert parsed.run_seed == 42 and parsed.contract == CONTRACT
    for updates in (
        {"runSeed": None},
        {"runSeed": True},
        {"runSeed": 1 << 32},
        {"sensorContractJson": REFERENCE["sensorContractJson"] + " "},
        {"resolvedThrough": "000000000000000a"},
        {"storageWatermark": "000000000000000a"},
        {"retentionFloor": "0000000000000000"},
    ):
        with pytest.raises(DistillationContractError):
            DatasetManifest.from_object(manifest(**updates))
    for payload in (
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b"\xff",
        b"{",
        b" " * (1024 * 1024 + 1),
    ):
        with pytest.raises(DistillationContractError):
            bounded_json(payload)


@pytest.mark.skipif(
    not os.environ.get("PYRL_TEACHER_CHUNK_REFERENCE"),
    reason="current Rust encoder runs in the canonical native CI job",
)
def test_current_rust_encoder_matches_frozen_sample_semantics():
    """Current native builds must retain the portable codec, including exact float bits."""
    root = Path(os.environ["PYRL_TEACHER_CHUNK_REFERENCE"])
    reference = json.loads((root / "reference.json").read_text(encoding="utf-8"))
    contract = TeacherContract.from_object(reference["contract"])
    descriptor = TeacherDescriptor.from_object(reference["descriptor"], contract)
    current = TeacherChunk.decode((root / "chunk.bin").read_bytes(), descriptor)
    frozen = TeacherChunk.decode(DATA, DESCRIPTOR)
    assert reference["scope"] == REFERENCE["scope"]
    assert reference["runSeed"] == REFERENCE["runSeed"]
    assert reference["sensorContractJson"] == REFERENCE["sensorContractJson"]
    assert replace(contract, teacher_sha256=CONTRACT.teacher_sha256) == CONTRACT
    assert replace(descriptor, contract=CONTRACT, sha256=DESCRIPTOR.sha256) == DESCRIPTOR
    for field in (
        "steps",
        "ordinals",
        "observations",
        "applied_turn",
        "flags",
        "strategy",
        "life_stage",
        "strata",
    ):
        assert getattr(current, field).tobytes() == getattr(frozen, field).tobytes()


@pytest.mark.parametrize("retained", [4 * 1024**3 + 1, 64 * 1024**3])
def test_manifest_accepts_the_protocol_storage_ceiling_as_metadata_only(retained):
    """Large advertised inventories remain bounded metadata, without allocating their bytes."""
    value = manifest(retainedChunks=65536, retainedBytes=f"{retained:016x}")
    assert DatasetManifest.from_object(value).raw["retainedBytes"] == f"{retained:016x}"
    with pytest.raises(DistillationContractError, match="watermarks"):
        DatasetManifest.from_object(manifest(retainedBytes=f"{64 * 1024**3 + 1:016x}"))
