"""Run Slither's real TypeScript encoder against the Python reward decoder.

Set PYRL_SLITHER_ROOT to the matching Slither checkout for this explicit
interoperability test. Node must satisfy that checkout's supported version.
"""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from pyrl_trainer.binary_sensors import decode_binary_sensors
from pyrl_trainer.reward_contract import reward_components

_SLITHER_ROOT = os.environ.get("PYRL_SLITHER_ROOT")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _SLITHER_ROOT, reason="set PYRL_SLITHER_ROOT for real Node encoder"),
]


def test_node_reward_packet_matches_python_and_native_fixture():
    """Real Node bytes retain native event identity and terminal objective value."""
    root = Path(_SLITHER_ROOT).resolve()
    script = """
import fs from 'node:fs';
import { encodeBinarySensors } from './server/binarySensors.ts';
const reward = JSON.parse(fs.readFileSync(
  './contracts/slither_contracts/external-reward-v1-fixture.json', 'utf8'))[0];
process.stdout.write(encodeBinarySensors({
  type: 'sensors', tick: 1, snakeId: 7, sensors: [1.25, -2.5], reward
}, 2));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=root, check=True, capture_output=True, timeout=30,
    )
    message = decode_binary_sensors(result.stdout)
    expected = json.loads((root / "contracts/slither_contracts/external-reward-v1-fixture.json")
                          .read_text(encoding="utf-8"))[0]
    assert message["reward"] == expected
    assert message["tick"] == int(expected["toStep"], 16)
    assert message["snakeId"] == expected["snakeId"]
    assert not message["sensors"].flags["OWNDATA"]
    np.testing.assert_array_equal(message["sensors"], [1.25, -2.5])
    assert sum(reward_components(message["reward"]).values()) == pytest.approx(-0.15)
