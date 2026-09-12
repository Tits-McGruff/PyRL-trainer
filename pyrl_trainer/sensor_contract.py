"""Exact sensor layout contract shared by trainer components."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


class SensorContractError(RuntimeError):
    """Raised when a server sensor layout is missing or incompatible."""


@dataclass(frozen=True)
class SensorContract:
    """Immutable sensor layout identity used by models and checkpoints."""

    layout_version: str
    order: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout_version, str) or not self.layout_version:
            raise SensorContractError("sensor layoutVersion must be a non-empty string")
        if not self.order:
            raise SensorContractError("sensor order must contain at least one label")
        if any(not isinstance(label, str) or not label for label in self.order):
            raise SensorContractError("sensor order contains an invalid label")
        if len(set(self.order)) != len(self.order):
            raise SensorContractError("sensor order contains duplicate labels")

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
        contract = cls(layout_version=layout_version, order=order)

        try:
            sensor_count = int(spec.get("sensorCount", -1))
        except (TypeError, ValueError) as exc:
            raise SensorContractError("server sensorSpec has invalid sensorCount") from exc
        if sensor_count != contract.sensor_count:
            raise SensorContractError(
                "server sensorSpec sensorCount does not match its ordered labels"
            )
        return contract

    def checkpoint_fields(self) -> dict[str, Any]:
        """Return the stable checkpoint representation for this contract."""
        return {
            "sensor_layout_version": self.layout_version,
            "sensor_order": list(self.order),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "SensorContract":
        """Read a sensor contract from checkpoint metadata."""
        raw_order = payload.get("sensor_order")
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
            raise SensorContractError("checkpoint omitted sensor_order")
        return cls(
            layout_version=payload.get("sensor_layout_version"),
            order=tuple(raw_order),
        )
