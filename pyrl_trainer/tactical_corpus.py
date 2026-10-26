"""Read bounded native synthetic tactical corpora without simulation or resampling."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from .sensor_contract import SensorContract, SensorContractError
from .tactical_contract import SCENARIOS, policies_for_profile

MAX_BYTES = 256 * 1024 * 1024
# Historical Motion/B16 fixture width; runtime shapes come from validated native metadata.
WIDTH = 264



class TacticalCorpusError(ValueError):
    """An offline corpus is incomplete, corrupted or outside its finite admission."""


def _require(condition, message):
    if not condition:
        raise TacticalCorpusError(message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _integer(value, minimum, maximum):
    _require(_is_int(value) and minimum <= value <= maximum, "invalid bounded corpus integer")
    return value


def _json(bytes_value):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate corpus JSON field")
            result[key] = value
        return result
    try:
        return json.loads(bytes_value, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as error:
        raise TacticalCorpusError("invalid corpus JSON") from error


def _read(path, maximum):
    # Check the opened descriptor too; an advertised size never authorizes a larger read.
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink()
             and not (getattr(info, "st_file_attributes", 0) & 0x400),
             "ordinary corpus file required")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        actual = os.fstat(stream.fileno())
        _require(stat.S_ISREG(actual.st_mode) and actual.st_size <= maximum,
             "corpus file exceeds admission")
        value = stream.read(maximum + 1)
        _require(len(value) <= maximum and len(value) == actual.st_size, "corpus file changed size")
        return value


@dataclass(frozen=True)
class TacticalSequence:
    """One physical teacher life, with explicit rows rather than held-step interpolation."""

    scenario: str
    seed: int
    policy: str
    row_offset: int
    rows: int


@dataclass(frozen=True)
class TacticalCorpus:
    """Read-only little-endian arrays; each sequence starts with zero policy recurrence."""

    sensor: SensorContract
    control_interval_seconds: float
    observations: np.ndarray
    actions: np.ndarray
    steps: np.ndarray
    sequences: tuple[TacticalSequence, ...]
    source_sha256: str

    def split_by_seed(self, validation_seeds):
        """Partition complete trajectories so overlapping windows cannot leak across splits."""
        seeds = frozenset(validation_seeds)
        _require(all(_is_int(seed) for seed in seeds),
             "validation seeds must be integer identities")
        available = {sequence.seed for sequence in self.sequences}
        _require(bool(seeds) and seeds < available,
             "validation requires a proper subset of corpus seeds")
        return (tuple(sequence for sequence in self.sequences if sequence.seed not in seeds),
                tuple(sequence for sequence in self.sequences if sequence.seed in seeds))


def _validate_observations(observations):
    # Bound scratch instead of allocating a second full observation matrix.
    for start in range(0, len(observations), 4096):
        block = observations[start:start + 4096]
        _require(np.isfinite(block).all() and (np.abs(block) <= 1).all(),
                 "invalid normalized corpus observations")


def _sensor(metadata, width):
    """Validate exact Rust canonical metadata before pricing or exposing any tensor."""
    canonical = metadata.get("sensorCanonicalJson")
    _require(isinstance(canonical, str) and len(canonical) <= 1024 * 1024,
             "missing native sensor metadata")
    raw = _json(canonical)
    _require(isinstance(raw, dict), "invalid native sensor metadata")
    try:
        sensor = SensorContract.from_spec({"layoutVersion": "v4", "sensorCount": width,
            "profile": raw.get("profile"), "order": raw.get("order"),
            "contract": {"canonicalJson": canonical, "sha256": metadata.get("sensorSha256")}})
    except SensorContractError as error:
        raise TacticalCorpusError("invalid native sensor contract") from error
    return sensor


def load_tactical_corpus(directory):
    """Verify finite metadata/bytes and complete sequence ownership before exposing arrays."""
    root = Path(directory)
    metadata = _json(_read(root / "corpus.json", 1024 * 1024))
    _require(isinstance(metadata, dict) and _is_int(metadata.get("schemaVersion"))
             and metadata["schemaVersion"] == 1
             and metadata.get("kind") == "native-tactical-corpus-v1", "unsupported tactical corpus")
    width = _integer(metadata.get("inputSize"), 72, 504)
    rows = _integer(metadata.get("rows"), 1, (MAX_BYTES - 1024 * 1024) // ((width + 3) * 4))
    seeds = _integer(metadata.get("seedCount"), 1, 64)
    maximum_steps = _integer(metadata.get("maximumSteps"), 1, 600)
    cadence = metadata.get("controlIntervalSeconds")
    _require(type(cadence) in (float, int) and np.isfinite(cadence) and 0.008 <= cadence <= 0.06,
             "invalid corpus deployment cadence")
    source = metadata.get("sourceSha256")
    _require(isinstance(source, str) and len(source) == 64
             and all(char in "0123456789abcdef" for char in source),
             "invalid native source identity")
    sensor = _sensor(metadata, width)
    policies = policies_for_profile(sensor.profile)
    objects = metadata.get("objects")
    _require(isinstance(objects, list) and len(objects) == 3, "complete corpus objects required")
    expected = {
        "observations.f32": rows * width * 4, "actions.f32": rows * 8, "steps.u32": rows * 4,
    }
    blobs = {}
    for receipt in objects:
        _require(isinstance(receipt, dict), "invalid corpus object receipt")
        name = receipt.get("file")
        _require(isinstance(name, str) and name in expected and name not in blobs,
             "invalid or duplicate corpus object")
        _require(_is_int(receipt.get("bytes")) and receipt["bytes"] == expected[name],
             "corpus object length mismatch")
        value = _read(root / name, expected[name])
        _require(len(value) == expected[name]
                 and hashlib.sha256(value).hexdigest() == receipt.get("sha256"),
                 "corpus object hash mismatch")
        blobs[name] = value
    observations = np.frombuffer(blobs["observations.f32"], dtype="<f4").reshape(rows, width)
    actions = np.frombuffer(blobs["actions.f32"], dtype="<f4").reshape(rows, 2)
    steps = np.frombuffer(blobs["steps.u32"], dtype="<u4")
    _validate_observations(observations)
    _require(np.isfinite(actions).all() and (np.abs(actions[:, 0]) <= 1).all()
             and ((actions[:, 1] == 0) | (actions[:, 1] == 1)).all(),
             "invalid committed corpus actions")
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, list)
             and len(sequences) == len(SCENARIOS) * seeds * len(policies),
             "incomplete tactical sequence matrix")
    result = []
    offset = 0
    expected_matrix = ((scenario, seed, policy) for scenario in SCENARIOS
                       for seed in range(1, seeds + 1) for policy in policies)
    for sequence, (scenario, seed, policy) in zip(sequences, expected_matrix):
        _require(isinstance(sequence, dict) and sequence.get("scenario") == scenario
                 and _is_int(sequence.get("seed")) and sequence["seed"] == seed
                          and sequence.get("policy") == policy,
                 "changed or duplicate tactical sequence identity")
        count = _integer(sequence.get("rows"), 1, maximum_steps)
        _require(_is_int(sequence.get("rowOffset")) and sequence["rowOffset"] == offset
                 and offset + count <= rows, "overlapping or missing corpus sequence rows")
        delivered = steps[offset:offset + count]
        _require(delivered[0] == 1 and delivered[-1] <= maximum_steps
                 and (delivered[1:] > delivered[:-1]).all(), "invalid delivered step chronology")
        _require(observations[offset, 21] == 1, "missing first-life corpus observation")
        result.append(TacticalSequence(scenario, seed, policy, offset, count))
        offset += count
    _require(offset == rows, "unowned trailing corpus rows")
    return TacticalCorpus(
        sensor, float(cadence), observations, actions, steps, tuple(result), source
    )
