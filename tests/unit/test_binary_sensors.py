"""Compact Protocol 2 sensor packet tests."""

import json
from importlib.resources import files
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


def reward_packet(*, reward=None, encoded=None):
    """Extend the unchanged packed body using the version-2 suffix."""
    if reward is None:
        reward = json.loads(files("slither_contracts").joinpath(
            "external-reward-v1-fixture.json"
        ).read_text(encoding="utf-8"))[0]
    raw = bytearray(packet([1.25, -2.5], tick=1))
    raw[4] = 2
    body = json.dumps(reward, separators=(",", ":")).encode() if encoded is None else encoded
    return bytes(raw) + struct.pack("<I", len(body)) + body


def test_reward_packet_preserves_native_fixture_and_zero_copy_sensors():
    """Reward bytes decode separately from the compact float body."""
    message = decode_binary_sensors(reward_packet())
    assert message["reward"]["transitionKind"] == "terminal"
    assert message["reward"]["deltas"]["foodValueDelta"] == 1
    assert message["reward"]["snakeId"] == message["snakeId"]
    assert int(message["reward"]["toStep"], 16) == message["tick"]
    assert not message["sensors"].flags["OWNDATA"]
    np.testing.assert_array_equal(message["sensors"], [1.25, -2.5])


@pytest.mark.parametrize("raw", [
    reward_packet()[:-1], reward_packet() + b"extra",
    reward_packet(encoded=b"x" * 4097), reward_packet(encoded=b""),
    reward_packet(encoded=b"\xff"), reward_packet(encoded=b"{}"),
    reward_packet(encoded=b'{"rewardVersion":1,"rewardVersion":1}'),
])
def test_reward_packets_reject_truncation_bounds_and_invalid_json(raw):
    """The length prefix and complete structured contract are mandatory."""
    with pytest.raises(BinarySensorError):
        decode_binary_sensors(raw)


@pytest.mark.parametrize("change", [{"snakeId": 8}, {"toStep": "0000000000000002"},
                                    {"rewardVersion": 2}])
def test_reward_packet_cannot_mix_observation_or_contract_identity(change):
    """A valid standalone reward must still belong to the same observation."""
    reward = json.loads(files("slither_contracts").joinpath(
        "external-reward-v1-fixture.json"
    ).read_text(encoding="utf-8"))[0]
    reward.update(change)
    with pytest.raises(BinarySensorError):
        decode_binary_sensors(reward_packet(reward=reward))
