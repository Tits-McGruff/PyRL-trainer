"""Keep optimizer qualification source identity stable across all graph cases."""
from unittest.mock import patch

import pytest
import torch

from pyrl_trainer import canonical_parity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("include_large", [False, True])
def test_source_change_during_optimizer_gate_rejects_report(tmp_path, include_large):
    """Never label already executed models with source bytes edited during the gate."""
    runner = tmp_path / "runner"
    runner.touch()
    with torch.random.fork_rng(devices=[]):
        with patch.object(canonical_parity, "_training_source_sha256",
                          side_effect=["a" * 64, "b" * 64]):
            with patch.object(canonical_parity, "_case", return_value=[]) as cases:
                with pytest.raises(RuntimeError, match="source changed during qualification"):
                    canonical_parity.run_gate(runner, tmp_path, "cpu", include_large)
    assert cases.call_count == (9 if include_large else 6)
