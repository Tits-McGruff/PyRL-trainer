"""Compact binary Protocol 2 sensor observation decoder."""

import json
import math
import struct
from typing import Any, Dict

import numpy as np
from slither_contracts import CONSTANTS

BINARY_SENSORS_MAGIC = b"SNS2"
BINARY_SENSORS_VERSION = 1
BINARY_SENSORS_HEADER_BYTES = 44
_HEADER = struct.Struct("<4sBBHQI3d")
_LENGTH = struct.Struct("<I")


class BinarySensorError(ValueError):
    """Raised when a compact sensor packet violates its fixed wire format."""


def decode_binary_sensors(raw: bytes | bytearray | memoryview) -> Dict[str, Any]:
    """Decode one compact observation while keeping its float32 body zero-copy."""
    view = memoryview(raw)
    if len(view) < BINARY_SENSORS_HEADER_BYTES:
        raise BinarySensorError("binary sensor packet is shorter than its header")
    magic, version, flags, count, tick, snake_id, x, y, direction = _HEADER.unpack_from(view)
    reward_version = CONSTANTS.get("externalRewardBinaryVersion")
    if magic != BINARY_SENSORS_MAGIC or version not in (BINARY_SENSORS_VERSION, reward_version):
        raise BinarySensorError("binary sensor packet has an unsupported identity")
    if flags & ~1:
        raise BinarySensorError("binary sensor packet has unsupported flags")
    expected = BINARY_SENSORS_HEADER_BYTES + int(count) * 4
    if len(view) < expected or (version == BINARY_SENSORS_VERSION and len(view) != expected):
        raise BinarySensorError("binary sensor packet length does not match its sensor count")
    sensors = np.frombuffer(view, dtype="<f4", count=int(count), offset=BINARY_SENSORS_HEADER_BYTES)
    if tick > 2**53 - 1 or snake_id == 0 or not np.isfinite(sensors).all():
        raise BinarySensorError("binary sensor identity or numeric vector is invalid")
    message: Dict[str, Any] = {
        "type": "sensors",
        "tick": int(tick),
        "snakeId": int(snake_id),
        "sensors": sensors,
    }
    if flags & 1:
        if not all(math.isfinite(value) for value in (x, y, direction)):
            raise BinarySensorError("binary sensor pose is nonfinite")
        message["meta"] = {"x": x, "y": y, "dir": direction}
    if version != BINARY_SENSORS_VERSION:
        message["reward"] = _decode_reward(view, expected, int(tick), int(snake_id))
    return message


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BinarySensorError("duplicate binary reward JSON field")
        result[key] = value
    return result


def _decode_reward(view: memoryview, offset: int, tick: int, snake_id: int) -> dict:
    # Legacy packet decoding remains usable with older installed contract types.
    # pylint: disable=import-outside-toplevel
    from .reward_contract import RewardContractError, parse_reward

    if len(view) < offset + _LENGTH.size:
        raise BinarySensorError("binary reward packet omits its length")
    length = _LENGTH.unpack_from(view, offset)[0]
    limit = CONSTANTS.get("externalRewardMaxEnvelopeBytes", 0)
    if not 0 < length <= limit or len(view) != offset + _LENGTH.size + length:
        raise BinarySensorError("binary reward length is invalid or exceeds its bound")
    try:
        reward = parse_reward(json.loads(
            view[offset + _LENGTH.size:].tobytes(), object_pairs_hook=_unique_object
        ))
    except (ValueError, RewardContractError) as exc:
        raise BinarySensorError(f"invalid binary reward: {exc}") from exc
    if reward["snakeId"] != snake_id or int(reward["toStep"], 16) != tick:
        raise BinarySensorError("binary reward does not match the accompanying observation")
    return reward
