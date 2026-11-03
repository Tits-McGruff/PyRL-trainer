"""Compact Protocol 2 sensor packet tests."""

import struct

import numpy as np
import pytest

from pyrl_trainer.binary_sensors import (
    BINARY_SENSORS_HEADER_BYTES,
    BinarySensorError,
    decode_binary_sensors,
)


def packet(values, *, tick=42, snake_id=7, flags=1):
    """Build one server-compatible fixed header plus little-endian float32 body."""
    body = np.asarray(values, dtype="<f4").tobytes()
    header = struct.pack(
        "<4sBBHQI3d", b"SNS2", 1, flags, len(values), tick, snake_id, 3.5, -4.25, 0.75
    )
    assert len(header) == BINARY_SENSORS_HEADER_BYTES
    return header + body


def test_decode_binary_sensors_keeps_float_body_zero_copy():
    """Binary observations avoid JSON parsing and a list-to-float32 conversion."""
    raw = packet([1.25, -2.5, 0.125])
    message = decode_binary_sensors(raw)
    assert message["type"] == "sensors"
    assert message["tick"] == 42
    assert message["snakeId"] == 7
    assert message["meta"] == {"x": 3.5, "y": -4.25, "dir": 0.75}
    np.testing.assert_array_equal(
        message["sensors"], np.asarray([1.25, -2.5, 0.125], dtype=np.float32)
    )
    assert message["sensors"].flags["OWNDATA"] is False


@pytest.mark.parametrize(
    "raw",
    [
        b"short",
        b"BAD!" + packet([1.0])[4:],
        packet([1.0])[:-1],
        bytearray(packet([1.0], flags=2)),
    ],
)
def test_decode_binary_sensors_rejects_malformed_packets(raw):
    """Malformed binary observations fail instead of being treated as JSON or frames."""
    with pytest.raises(BinarySensorError):
        decode_binary_sensors(raw)
