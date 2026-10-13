"""Strict Rust TeacherSampleV1 byte decoding and immutable dataset identities.

Only the trainer decodes observation rows. Rust owns encoding and committed
publication; successful decoding alone is not proof that gameplay committed.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct

import numpy as np

MAX_CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
START_REASONS = ("captureStart", "physicalLifeStart", "continuation", "gap", "generationStart")
END_REASONS = (
    "maximumSamples",
    "simulatedSecond",
    "lifeEnded",
    "generationEnded",
    "contractChanged",
    "captureStopped",
    "interrupted",
)
HEADER_TAIL = struct.Struct("<IBdHQQIQBBH")
SAMPLE = struct.Struct("<QQfBBBH")


class DistillationContractError(ValueError):
    """Malformed, missing, incompatible or nonfinite diagnostic/training data."""


def integer(value, low, high):
    """Reject bool/float coercion and dimensions outside declared admission."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise DistillationContractError("invalid bounded distillation integer")
    return value


def hex_identity(value, width, nonzero=False):
    """Keep long identities exact; never round them through a floating point value."""
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is None:
        raise DistillationContractError("invalid distillation hexadecimal identity")
    if nonzero and int(value, 16) == 0:
        raise DistillationContractError("zero distillation identity")
    return value


def exact_keys(value, keys):
    """Require the complete versioned metadata object with no alternate fields."""
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise DistillationContractError("unsupported distillation metadata fields")


def _object_pairs(pairs):
    """Duplicate JSON keys cannot silently replace previously validated identities."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise DistillationContractError("duplicate distillation JSON field")
        result[key] = value
    return result


def _json_float(value):
    """Finite lexical numbers only; exponent overflow must not create Infinity."""
    number = float(value)
    if not math.isfinite(number):
        raise DistillationContractError("nonfinite JSON number")
    return number


def bounded_json(data):
    """Decode bounded UTF-8 with duplicate/nonfinite-number rejection."""
    if not isinstance(data, bytes) or len(data) > MAX_JSON_BYTES:
        raise DistillationContractError("distillation JSON byte limit")
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_float=_json_float,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DistillationContractError("nonfinite JSON number")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DistillationContractError("malformed distillation JSON") from exc


@dataclass(frozen=True)
class TeacherContract:
    """Exact native run/config/sensor/teacher/cadence identity; no model packing."""

    capture_epoch: str
    run_id: str
    config_sha256: str
    sensor_sha256: str
    teacher_sha256: str
    teacher_revision: int
    teacher_mode: int
    control_interval_seconds: float
    input_size: int

    def __post_init__(self):
        hex_identity(self.capture_epoch, 32, True)
        for digest in (self.config_sha256, self.sensor_sha256, self.teacher_sha256):
            hex_identity(digest, 64)
        if not isinstance(self.run_id, str) or not 1 <= len(self.run_id.encode("utf-8")) <= 128:
            raise DistillationContractError("invalid teacher run UTF-8")
        integer(self.teacher_revision, 2, 2)
        integer(self.teacher_mode, 0, 1)
        integer(self.input_size, 1, 504)
        cadence = self.control_interval_seconds
        if (
            isinstance(cadence, bool)
            or not isinstance(cadence, (int, float))
            or not math.isfinite(cadence)
            or not 0.008 <= cadence <= 0.06
        ):
            raise DistillationContractError("invalid teacher cadence")

    @classmethod
    def from_object(cls, raw):
        """Require every native structural/deployment identity on the HTTP boundary."""
        exact_keys(
            raw,
            "captureEpoch runId configSha256 sensorSha256 teacherSha256 "
            "teacherRevision teacherMode controlIntervalSeconds inputSize",
        )
        return cls(
            raw["captureEpoch"],
            raw["runId"],
            raw["configSha256"],
            raw["sensorSha256"],
            raw["teacherSha256"],
            raw["teacherRevision"],
            raw["teacherMode"],
            raw["controlIntervalSeconds"],
            raw["inputSize"],
        )


@dataclass(frozen=True)
class TeacherDescriptor:
    """One immutable object's identity, stream extent, boundaries and exact digest."""

    contract: TeacherContract
    sequence: int
    generation: int
    baseline_slot: int
    physical_snake_id: int
    first_sample: int
    last_sample: int
    first_step: int
    last_step: int
    count: int
    start_reason: str
    end_reason: str
    byte_count: int
    sha256: str

    @classmethod
    def from_object(cls, raw, contract):
        """Validate complete metadata without treating width alone as compatibility."""
        exact_keys(
            raw,
            "captureEpoch sequence generation baselineSlot physicalSnakeId "
            "firstSample lastSample firstStep lastStep count inputSize configSha256 "
            "sensorSha256 teacherSha256 startReason endReason byteCount sha256",
        )
        for key, expected in (
            ("captureEpoch", contract.capture_epoch),
            ("configSha256", contract.config_sha256),
            ("sensorSha256", contract.sensor_sha256),
            ("teacherSha256", contract.teacher_sha256),
            ("inputSize", contract.input_size),
        ):
            if raw[key] != expected:
                raise DistillationContractError("descriptor contract mismatch")

        def word(key):
            """Decode one exact positive native counter."""
            return int(hex_identity(raw[key], 16, True), 16)

        count = integer(raw["count"], 1, 256)
        first, last = word("firstSample"), word("lastSample")
        start, end = word("firstStep"), word("lastStep")
        if last - first != count - 1 or not count - 1 <= end - start <= 60:
            raise DistillationContractError("descriptor chronology mismatch")
        if raw["startReason"] not in START_REASONS or raw["endReason"] not in END_REASONS:
            raise DistillationContractError("invalid teacher boundary reasons")
        return cls(
            contract,
            word("sequence"),
            word("generation"),
            integer(raw["baselineSlot"], 0, 119),
            word("physicalSnakeId"),
            first,
            last,
            start,
            end,
            count,
            raw["startReason"],
            raw["endReason"],
            integer(raw["byteCount"], 1, MAX_CHUNK_BYTES),
            hex_identity(raw["sha256"], 64),
        )


@dataclass(frozen=True)
class TeacherChunk:
    """Bounded immutable decoded rows, preserving exact Float32/sign-zero bits."""

    contract: TeacherContract
    sequence: int
    generation: int
    baseline_slot: int
    physical_snake_id: int
    start_reason: str
    end_reason: str
    steps: np.ndarray
    ordinals: np.ndarray
    observations: np.ndarray
    applied_turn: np.ndarray
    flags: np.ndarray
    strategy: np.ndarray
    life_stage: np.ndarray
    strata: np.ndarray
    raw_bytes: bytes

    @property
    def byte_cost(self):
        """Charge raw backing and all owned decoded arrays to replay admission."""
        return len(self.raw_bytes) + sum(
            array.nbytes
            for array in (
                self.steps,
                self.ordinals,
                self.observations,
                self.applied_turn,
                self.flags,
                self.strategy,
                self.life_stage,
                self.strata,
            )
        )

    @property
    def boost_request(self):
        """Committed action labels are distinct from realized physical boost."""
        return (self.flags & 1) != 0

    @classmethod
    def decode(cls, data, descriptor=None):
        # The fixed wire record is validated together before any object is published.
        # pylint: disable=too-many-statements,too-many-boolean-expressions
        # pylint: disable=too-many-branches
        """Prove complete dimensions before allocating any row vectors."""
        if not isinstance(data, bytes) or not 168 <= len(data) <= MAX_CHUNK_BYTES:
            raise DistillationContractError("teacher chunk byte limit")
        if data[:8] != b"SLTHTCH1":
            raise DistillationContractError("unsupported teacher binary version")
        epoch, run_size = data[8:24].hex(), data[24]
        integer(run_size, 1, 128)
        position = 25 + run_size
        if position + 96 + HEADER_TAIL.size > len(data):
            raise DistillationContractError("truncated teacher header")
        try:
            run_id = data[25:position].decode("utf-8")
        except UnicodeError as exc:
            raise DistillationContractError("invalid teacher run UTF-8") from exc
        digests = [data[position + offset : position + offset + 32].hex() for offset in (0, 32, 64)]
        position += 96
        revision, mode, cadence, width, sequence, generation, slot, physical, start, end, count = (
            HEADER_TAIL.unpack_from(data, position)
        )
        position += HEADER_TAIL.size
        contract = TeacherContract(epoch, run_id, *digests, revision, mode, cadence, width)
        integer(count, 1, 256)
        integer(sequence, 1, (1 << 64) - 1)
        integer(generation, 1, (1 << 64) - 1)
        integer(physical, 1, (1 << 64) - 1)
        integer(slot, 0, 119)
        if (
            start >= len(START_REASONS)
            or end >= len(END_REASONS)
            or len(data) - position != count * (SAMPLE.size + width * 4)
        ):
            raise DistillationContractError("teacher dimensions/trailing data/boundary mismatch")
        steps, ordinals = np.empty(count, dtype=np.uint64), np.empty(count, dtype=np.uint64)
        observations, turn = np.empty((count, width), dtype=np.float32), np.empty(
            count, dtype=np.float32
        )
        flags, strategy, stages = (np.empty(count, dtype=np.uint8) for _ in range(3))
        strata = np.empty(count, dtype=np.uint16)
        previous_step = previous_ordinal = 0
        previous_dead = False
        for index in range(count):
            step, ordinal, action, flag, policy, stage, stratum = SAMPLE.unpack_from(data, position)
            position += SAMPLE.size
            if step == 0 or ordinal == 0 or flag & ~15 or policy > 3 or stage > 2 or stratum & ~63:
                raise DistillationContractError("invalid teacher sample metadata")
            if not math.isfinite(action) or abs(action) > 1:
                raise DistillationContractError("invalid teacher action")
            if index and (
                step <= previous_step
                or ordinal != previous_ordinal + 1
                or previous_dead
                or flag & 8
            ):
                raise DistillationContractError("invalid teacher stream chronology")
            row = np.frombuffer(data, dtype="<f4", count=width, offset=position)
            if not np.isfinite(row).all() or (np.abs(row) > 1).any():
                raise DistillationContractError("invalid normalized Float32 observation")
            observations[index] = row
            steps[index], ordinals[index], turn[index] = step, ordinal, action
            flags[index], strategy[index], stages[index], strata[index] = (
                flag,
                policy,
                stage,
                stratum,
            )
            previous_step, previous_ordinal, previous_dead = step, ordinal, bool(flag & 4)
            position += width * 4
        if (
            int(steps[-1]) - int(steps[0]) > 60
            or (start == 1 and not flags[0] & 8)
            or (start == 2 and flags[0] & 8)
        ):
            raise DistillationContractError("invalid teacher physical/time boundary")
        if descriptor is not None:
            if not isinstance(descriptor, TeacherDescriptor) or descriptor.contract != contract:
                raise DistillationContractError("binary teacher contract mismatch")
            actual = (
                sequence,
                generation,
                slot,
                physical,
                int(ordinals[0]),
                int(ordinals[-1]),
                int(steps[0]),
                int(steps[-1]),
                count,
                START_REASONS[start],
                END_REASONS[end],
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            expected = (
                descriptor.sequence,
                descriptor.generation,
                descriptor.baseline_slot,
                descriptor.physical_snake_id,
                descriptor.first_sample,
                descriptor.last_sample,
                descriptor.first_step,
                descriptor.last_step,
                descriptor.count,
                descriptor.start_reason,
                descriptor.end_reason,
                descriptor.byte_count,
                descriptor.sha256,
            )
            if actual != expected:
                raise DistillationContractError("binary teacher descriptor/digest mismatch")
        for array in (steps, ordinals, observations, turn, flags, strategy, stages, strata):
            array.flags.writeable = False
        return cls(
            contract,
            sequence,
            generation,
            slot,
            physical,
            START_REASONS[start],
            END_REASONS[end],
            steps,
            ordinals,
            observations,
            turn,
            flags,
            strategy,
            stages,
            strata,
            data,
        )


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable native identity and coherent durable availability watermarks."""

    contract: TeacherContract
    run_seed: int
    sensor_json: str
    state: str
    production_watermark: int
    storage_watermark: int
    retention_floor: int
    resolved_through: int
    raw: dict

    @classmethod
    def from_object(cls, raw):
        # Every structural condition must be checked before row allocation.
        # pylint: disable=too-many-boolean-expressions
        """Require a recorded seed; unknown legacy seeds cannot support exact partitioning."""
        exact_keys(
            raw,
            "schemaVersion runSeed contract sensorContractJson state productionWatermark "
            "storageWatermark retentionFloor resolvedThrough nativeDroppedChunks "
            "nativeDroppedSamples nativeDroppedBytes spoolDroppedChunks "
            "retainedChunks retainedBytes error",
        )
        integer(raw["schemaVersion"], 1, 1)
        seed = integer(raw["runSeed"], 0, (1 << 32) - 1)
        contract = TeacherContract.from_object(raw["contract"])
        sensor_json = raw["sensorContractJson"]
        if not isinstance(sensor_json, str) or len(sensor_json.encode("utf-8")) > 65536:
            raise DistillationContractError("sensor metadata byte limit")
        sensor_bytes = sensor_json.encode("utf-8")
        if (
            hashlib.sha256(b"slither.sensor-contract.v1\0" + sensor_bytes).hexdigest()
            != contract.sensor_sha256
        ):
            raise DistillationContractError("sensor canonical digest mismatch")
        sensor = bounded_json(sensor_bytes)
        exact_keys(
            sensor,
            "schemaVersion layoutVersion profile bins inputSize scalarCount "
            "normalizationRevision geometryRevision releaseStatus scalarEncoding "
            "angularGeometry validityEncoding scalarSemantics channels order",
        )
        for key, value in (
            ("schemaVersion", 1),
            ("layoutVersion", 4),
            ("scalarCount", 24),
            ("normalizationRevision", 0),
            ("geometryRevision", 0),
        ):
            integer(sensor[key], value, value)
        if (
            sensor.get("schemaVersion") != 1
            or sensor.get("layoutVersion") != 4
            or sensor.get("profile") not in ("basic", "motion")
            or sensor.get("normalizationRevision") != 0
            or sensor.get("geometryRevision") != 0
        ):
            raise DistillationContractError("unsupported structural sensor revision/profile")
        bins = integer(sensor.get("bins"), 8, 32)
        width = 24 + (6 if sensor["profile"] == "basic" else 15) * bins
        order = sensor.get("order")
        if (
            sensor.get("inputSize") != contract.input_size
            or width != contract.input_size
            or sensor.get("scalarCount") != 24
            or not isinstance(order, list)
            or len(order) != width
            or any(
                not isinstance(label, str) or not 1 <= len(label.encode("utf-8")) <= 128
                for label in order
            )
            or len(set(order)) != len(order)
        ):
            raise DistillationContractError("sensor labels/width mismatch")
        if order[21:24] != [
            "first_sample_of_life",
            "pellet_query_saturated",
            "body_or_ray_query_saturated",
        ]:
            raise DistillationContractError("sensor physical/validity channel mismatch")
        if raw["state"] not in ("active", "stopped", "interrupted"):
            raise DistillationContractError("unsupported dataset state")
        words = {
            key: int(hex_identity(raw[key], 16), 16)
            for key in (
                "productionWatermark",
                "storageWatermark",
                "retentionFloor",
                "resolvedThrough",
                "nativeDroppedChunks",
                "nativeDroppedSamples",
                "nativeDroppedBytes",
                "spoolDroppedChunks",
                "retainedBytes",
            )
        }
        if (
            words["storageWatermark"] > words["productionWatermark"]
            or words["resolvedThrough"] > words["productionWatermark"]
            or words["retentionFloor"] == 0
            or words["retainedBytes"] > 4 * 1024**3
        ):
            raise DistillationContractError("incoherent dataset watermarks")
        integer(raw["retainedChunks"], 0, 65536)
        if raw["error"] is not None and (
            not isinstance(raw["error"], str) or len(raw["error"]) > 1024
        ):
            raise DistillationContractError("dataset diagnostic limit")
        return cls(
            contract,
            seed,
            sensor_json,
            raw["state"],
            words["productionWatermark"],
            words["storageWatermark"],
            words["retentionFloor"],
            words["resolvedThrough"],
            dict(raw),
        )

    def compatible(self, other):
        """Bind exact immutable capture identity, including its recorded run seed."""
        return (
            self.contract == other.contract
            and self.run_seed == other.run_seed
            and self.sensor_json == other.sensor_json
        )
