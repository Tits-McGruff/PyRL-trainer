"""A minimized real operand row protects the CPU arithmetic used by canonical graphs."""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import _affine_dot

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures/compensated_dot_counterexample_v1"


def test_actual_failing_export_row_uses_accurate_dot_and_analytic_gradient():
    """Execute today's CPU helper against an independent oracle and the original operands."""
    receipt = json.loads((ROOT / "fixture.json").read_bytes())
    values = {}
    for name, record in receipt["files"].items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        values[name] = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(1, 504)
    left, right = values["values.f32"], values["weights.f32"]
    oracle = math.fsum(x * y for x, y in zip(left.flatten().tolist(), right.flatten().tolist()))
    assert oracle == receipt["oracle"]
    old = 0.0
    for x, y in zip(left.flatten().tolist(), right.flatten().tolist()):
        product = struct.unpack("<f", struct.pack("<f", x * y))[0]
        old = struct.unpack("<f", struct.pack("<f", old + product))[0]
    assert old == receipt["legacySequentialFloat32"]
    assert abs(old - oracle) > receipt["maximumAbsoluteError"]
    left.requires_grad_()
    right.requires_grad_()
    actual = _affine_dot(left, right)
    assert abs(actual.item() - oracle) <= receipt["maximumAbsoluteError"]
    actual.sum().backward()
    torch.testing.assert_close(left.grad, right, rtol=0, atol=0)
    torch.testing.assert_close(right.grad, left, rtol=0, atol=0)
