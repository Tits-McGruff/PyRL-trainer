"""Exact sensor layout contract shared by trainer components."""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence, Tuple


class SensorContractError(RuntimeError):
    """Raised when a server sensor layout is missing or incompatible."""


@dataclass(frozen=True)
class SensorContract:
    """Immutable sensor layout identity used by models and checkpoints."""

    layout_version: str
    order: Tuple[str, ...]
    profile: str | None = None
    normalization_revision: int | None = None
    structural_sha256: str | None = None
    canonical_json: str | None = None
    bins: int | None = None
    scalar_count: int | None = None
    channel_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout_version, str) or not self.layout_version:
            raise SensorContractError("sensor layoutVersion must be a non-empty string")
        if len(self.order) > 504:
            raise SensorContractError("sensor input width exceeds native admission")
        if not self.order:
            raise SensorContractError("sensor order must contain at least one label")
        if any(not isinstance(label, str) or not label for label in self.order):
            raise SensorContractError("sensor order contains an invalid label")
        if len(set(self.order)) != len(self.order):
            raise SensorContractError("sensor order contains duplicate labels")
        if self.layout_version == "v4":
            _validate_v4(self)

    @property
    def sensor_count(self) -> int:
        """Return the number of ordered model inputs."""
        return len(self.order)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "SensorContract":
        """Parse and validate the Protocol 2 sensorSpec payload."""
        if not isinstance(spec, Mapping):
            raise SensorContractError("server sensorSpec must be an object")

        raw_order = spec.get("order")
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
            raise SensorContractError("server sensorSpec omitted sensor order")
        order = tuple(raw_order)

        layout_version = spec.get("layoutVersion")
        if layout_version == "v4":
            envelope = spec.get("contract")
            if not isinstance(envelope, Mapping):
                raise SensorContractError("v4 welcome requires native structural metadata")
            raw = _canonical(envelope.get("canonicalJson"), envelope.get("sha256"))
            contract = cls(
                layout_version=layout_version,
                order=order,
                profile=spec.get("profile"),
                normalization_revision=raw.get("normalizationRevision"),
                structural_sha256=envelope.get("sha256"),
                canonical_json=envelope.get("canonicalJson"),
                bins=raw.get("bins"),
                scalar_count=raw.get("scalarCount"),
                channel_count=(
                    len(raw["channels"]) if isinstance(raw.get("channels"), list) else None
                ),
            )
        else:
            # Historical v3 checkpoint identity remains exactly version plus ordered labels.
            contract = cls(layout_version=layout_version, order=order)

        sensor_count = _integer(spec.get("sensorCount"), 1, 504)
        if sensor_count != contract.sensor_count:
            raise SensorContractError(
                "server sensorSpec sensorCount does not match its ordered labels"
            )
        return contract

    def checkpoint_fields(self) -> dict[str, Any]:
        """Return the stable checkpoint representation for this contract."""
        record = {"sensor_layout_version": self.layout_version, "sensor_order": list(self.order)}
        if self.layout_version == "v4":
            record.update(
                sensor_profile=self.profile,
                sensor_normalization_revision=self.normalization_revision,
                sensor_structural_sha256=self.structural_sha256,
                sensor_canonical_json=self.canonical_json,
                sensor_input_size=self.sensor_count,
                sensor_bins=self.bins,
                sensor_scalar_count=self.scalar_count,
                sensor_channel_count=self.channel_count,
            )
        return record

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "SensorContract":
        """Read a sensor contract from checkpoint metadata."""
        raw_order = payload.get("sensor_order")
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
            raise SensorContractError("checkpoint omitted sensor_order")
        version = payload.get("sensor_layout_version")
        if version == "v4":
            contract = cls(
                layout_version=version,
                order=tuple(raw_order),
                profile=payload.get("sensor_profile"),
                normalization_revision=payload.get("sensor_normalization_revision"),
                structural_sha256=payload.get("sensor_structural_sha256"),
                canonical_json=payload.get("sensor_canonical_json"),
                bins=payload.get("sensor_bins"),
                scalar_count=payload.get("sensor_scalar_count"),
                channel_count=payload.get("sensor_channel_count"),
            )
            if _integer(payload.get("sensor_input_size"), 1, 504) != contract.sensor_count:
                raise SensorContractError("checkpoint v4 input width mismatch")
            return contract
        return cls(layout_version=version, order=tuple(raw_order))


def _integer(value, minimum, maximum):
    """Reject bool/fraction/string coercion at structural width/revision boundaries."""
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SensorContractError("invalid structural sensor integer")
    return value


def _pairs(pairs):
    """Native canonical fields cannot be replaced by duplicate JSON keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise SensorContractError("duplicate canonical sensor field")
        result[key] = value
    return result


def _canonical(value, digest):
    """Bind exact bounded UTF-8 Rust bytes; do not regenerate or reorder sensor metadata."""
    if not isinstance(value, str) or not isinstance(digest, str):
        raise SensorContractError("missing native canonical sensor bytes/digest")
    try:
        data = value.encode("utf-8")
        if (
            len(data) > 65536
            or hashlib.sha256(b"slither.sensor-contract.v1\0" + data).hexdigest() != digest
        ):
            raise SensorContractError("canonical sensor digest/byte admission mismatch")
        raw = json.loads(
            data,
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                SensorContractError("nonfinite sensor metadata")
            ),
        )
        keys = set(
            "schemaVersion layoutVersion profile bins inputSize scalarCount normalizationRevision "
            "geometryRevision releaseStatus scalarEncoding angularGeometry validityEncoding "
            "scalarSemantics channels order".split()
        )
        if not isinstance(raw, dict) or set(raw) != keys:
            raise SensorContractError("unsupported canonical sensor metadata fields")
        return raw
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SensorContractError("malformed native canonical sensor metadata") from error


def _validate_v4(self):
    """Validate known v4 identity and canonical channel-major metadata without sensing."""
    if (
        self.profile not in ("basic", "motion")
        or self.normalization_revision != 0
        or isinstance(self.normalization_revision, bool)
    ):
        raise SensorContractError("unsupported v4 profile/normalization revision")
    if (
        not isinstance(self.structural_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", self.structural_sha256) is None
    ):
        raise SensorContractError("v4 requires an exact structural digest")
    raw = _canonical(self.canonical_json, self.structural_sha256)
    for name, expected in (
        ("schemaVersion", 1),
        ("layoutVersion", 4),
        ("normalizationRevision", 0),
        ("geometryRevision", 0),
        ("scalarCount", 24),
    ):
        _integer(raw.get(name), expected, expected)
    if raw.get("profile") != self.profile or raw.get("releaseStatus") != "candidate":
        raise SensorContractError("unsupported v4 structural sensor contract")
    bins = _integer(self.bins, 8, 32)
    channels = 6 if self.profile == "basic" else 15
    _integer(self.scalar_count, 24, 24)
    _integer(self.channel_count, channels, channels)
    if (
        self.sensor_count != 24 + channels * bins
        or _integer(raw.get("bins"), 8, 32) != bins
        or _integer(raw.get("inputSize"), 1, 504) != self.sensor_count
        or raw.get("order") != list(self.order)
    ):
        raise SensorContractError("v4 structural width/order mismatch")
    metadata = raw.get("channels")
    if not isinstance(metadata, list) or len(metadata) != channels:
        raise SensorContractError("v4 angular channel shape mismatch")
    for index, channel in enumerate(metadata):
        if (
            not isinstance(channel, dict)
            or channel.get("offset") != 24 + index * bins
            or not isinstance(channel.get("label"), str)
            or self.order[24 + index * bins : 24 + (index + 1) * bins]
            != tuple(f"{channel['label']}_{angular}" for angular in range(bins))
        ):
            raise SensorContractError("v4 channel-major label/offset mismatch")
    scalars = raw.get("scalarSemantics")
    if (
        not isinstance(scalars, list)
        or len(scalars) != 24
        or any(
            not isinstance(value, dict) or value.get("label") != self.order[index]
            for index, value in enumerate(scalars)
        )
    ):
        raise SensorContractError("v4 scalar semantics/order mismatch")
