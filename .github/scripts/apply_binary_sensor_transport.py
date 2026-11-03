from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


Path("pyrl_trainer/binary_sensors.py").write_text(
    '''"""Compact binary Protocol 2 sensor observation decoder."""\n\nimport struct\nfrom typing import Any, Dict\n\nimport numpy as np\n\nBINARY_SENSORS_MAGIC = b"SNS2"\nBINARY_SENSORS_VERSION = 1\nBINARY_SENSORS_HEADER_BYTES = 44\n_HEADER = struct.Struct("<4sBBHQI3d")\n\n\nclass BinarySensorError(ValueError):\n    """Raised when a compact sensor packet violates its fixed wire format."""\n\n\ndef decode_binary_sensors(raw: bytes | bytearray | memoryview) -> Dict[str, Any]:\n    """Decode one compact observation while keeping its float32 body zero-copy."""\n    view = memoryview(raw)\n    if len(view) < BINARY_SENSORS_HEADER_BYTES:\n        raise BinarySensorError("binary sensor packet is shorter than its header")\n    magic, version, flags, count, tick, snake_id, x, y, direction = _HEADER.unpack_from(view)\n    if magic != BINARY_SENSORS_MAGIC or version != BINARY_SENSORS_VERSION:\n        raise BinarySensorError("binary sensor packet has an unsupported identity")\n    if flags & ~1:\n        raise BinarySensorError("binary sensor packet has unsupported flags")\n    expected = BINARY_SENSORS_HEADER_BYTES + int(count) * 4\n    if len(view) != expected:\n        raise BinarySensorError("binary sensor packet length does not match its sensor count")\n    sensors = np.frombuffer(view, dtype="<f4", count=int(count), offset=BINARY_SENSORS_HEADER_BYTES)\n    message: Dict[str, Any] = {\n        "type": "sensors",\n        "tick": int(tick),\n        "snakeId": int(snake_id),\n        "sensors": sensors,\n    }\n    if flags & 1:\n        message["meta"] = {"x": x, "y": y, "dir": direction}\n    return message\n''',
    encoding="utf-8",
)

replace_exact(
    "pyrl_trainer/agent.py",
    """from .config import Config, MAX_WS_MESSAGE_BYTES, PROTOCOL_VERSION\nfrom .sensor_contract import SensorContract, SensorContractError\n""",
    """from .binary_sensors import decode_binary_sensors\nfrom .config import Config, MAX_WS_MESSAGE_BYTES, PROTOCOL_VERSION\nfrom .sensor_contract import SensorContract, SensorContractError\n""",
)
replace_exact(
    "pyrl_trainer/agent.py",
    """        hello = {\n            \"type\": \"hello\",\n            \"clientType\": \"bot\",\n            \"version\": PROTOCOL_VERSION,\n        }\n""",
    """        hello = {\n            \"type\": \"hello\",\n            \"clientType\": \"bot\",\n            \"version\": PROTOCOL_VERSION,\n            \"binarySensors\": True,\n        }\n""",
)
replace_exact(
    "pyrl_trainer/agent.py",
    """        sensors = msg.get(\"sensors\") or []\n        obs = np.asarray(sensors, dtype=np.float32)\n""",
    """        sensors = msg.get(\"sensors\")\n        if sensors is None:\n            sensors = []\n        obs = np.asarray(sensors, dtype=np.float32)\n""",
)
replace_exact(
    "pyrl_trainer/agent.py",
    """        while True:\n            raw = await ws.recv()\n            if isinstance(raw, (bytes, bytearray)):\n                continue\n\n            msg = json.loads(raw)\n            msg_type = msg.get(\"type\")\n""",
    """        while True:\n            raw = await ws.recv()\n            if isinstance(raw, (bytes, bytearray)):\n                await self._handle_sensors(decode_binary_sensors(raw), ws)\n                continue\n\n            msg = json.loads(raw)\n            msg_type = msg.get(\"type\")\n""",
)

Path("tests/unit/test_binary_sensors.py").write_text(
    '''"""Compact Protocol 2 sensor packet tests."""\n\nimport struct\n\nimport numpy as np\nimport pytest\n\nfrom pyrl_trainer.binary_sensors import (\n    BINARY_SENSORS_HEADER_BYTES,\n    BinarySensorError,\n    decode_binary_sensors,\n)\n\n\ndef packet(values, *, tick=42, snake_id=7, flags=1):\n    """Build one server-compatible fixed header plus little-endian float32 body."""\n    body = np.asarray(values, dtype="<f4").tobytes()\n    header = struct.pack(\n        "<4sBBHQI3d", b"SNS2", 1, flags, len(values), tick, snake_id, 3.5, -4.25, 0.75\n    )\n    assert len(header) == BINARY_SENSORS_HEADER_BYTES\n    return header + body\n\n\ndef test_decode_binary_sensors_keeps_float_body_zero_copy():\n    """Binary observations avoid JSON parsing and a list-to-float32 conversion."""\n    raw = packet([1.25, -2.5, 0.125])\n    message = decode_binary_sensors(raw)\n    assert message["type"] == "sensors"\n    assert message["tick"] == 42\n    assert message["snakeId"] == 7\n    assert message["meta"] == {"x": 3.5, "y": -4.25, "dir": 0.75}\n    np.testing.assert_array_equal(\n        message["sensors"], np.asarray([1.25, -2.5, 0.125], dtype=np.float32)\n    )\n    assert message["sensors"].flags["OWNDATA"] is False\n\n\n@pytest.mark.parametrize(\n    "raw",\n    [\n        b"short",\n        b"BAD!" + packet([1.0])[4:],\n        packet([1.0])[:-1],\n        bytearray(packet([1.0], flags=2)),\n    ],\n)\ndef test_decode_binary_sensors_rejects_malformed_packets(raw):\n    """Malformed binary observations fail instead of being treated as JSON or frames."""\n    with pytest.raises(BinarySensorError):\n        decode_binary_sensors(raw)\n''',
    encoding="utf-8",
)

# Keep the API notes aligned with the negotiated additive capability.
docs_path = Path("docs/API-Instructions")
docs = docs_path.read_text(encoding="utf-8")
needle = """3. No auth headers, API keys, allowlists, or session tokens are required or\n   supported for WebSocket connections. The server does not inspect custom\n   headers for WS auth.\n"""
replacement = needle + """   Trainer actors advertise `binarySensors: true` in the Protocol 2 `hello`.\n   Supporting servers then send `sensors` observations as compact `SNS2` binary\n   packets; lifecycle, stats, assignment, replacement, and error traffic stays JSON.\n"""
if docs.count(needle) != 1:
    raise SystemExit("docs/API-Instructions: expected connection note once")
docs_path.write_text(docs.replace(needle, replacement, 1), encoding="utf-8")
