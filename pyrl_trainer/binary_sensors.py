"""Compact binary Protocol 2 sensor observation decoder."""

import struct
from typing import Any, Dict

import numpy as np

BINARY_SENSORS_MAGIC = b"SNS2"
BINARY_SENSORS_VERSION = 1
BINARY_SENSORS_HEADER_BYTES = 44
_HEADER = struct.Struct("<4sBBHQI3d")


class BinarySensorError(ValueError):
    """Raised when a compact sensor packet violates its fixed wire format."""


def decode_binary_sensors(raw: bytes | bytearray | memoryview) -> Dict[str, Any]:
    """Decode one compact observation while keeping its float32 body zero-copy."""
    view = memoryview(raw)
    if len(view) < BINARY_SENSORS_HEADER_BYTES:
        raise BinarySensorError("binary sensor packet is shorter than its header")
    magic, version, flags, count, tick, snake_id, x, y, direction = _HEADER.unpack_from(view)
    if magic != BINARY_SENSORS_MAGIC or version != BINARY_SENSORS_VERSION:
        raise BinarySensorError("binary sensor packet has an unsupported identity")
    if flags & ~1:
        raise BinarySensorError("binary sensor packet has unsupported flags")
    expected = BINARY_SENSORS_HEADER_BYTES + int(count) * 4
    if len(view) != expected:
        raise BinarySensorError("binary sensor packet length does not match its sensor count")
    sensors = np.frombuffer(view, dtype="<f4", count=int(count), offset=BINARY_SENSORS_HEADER_BYTES)
    message: Dict[str, Any] = {
        "type": "sensors",
        "tick": int(tick),
        "snakeId": int(snake_id),
        "sensors": sensors,
    }
    if flags & 1:
        message["meta"] = {"x": x, "y": y, "dir": direction}
    return message
