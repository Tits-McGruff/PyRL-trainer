"""Durable capacity uses actual canonical job ownership and exact Adam continuation."""

import subprocess
import sys

import pytest

from test_distillation_job import DEVICES, make_job

from pyrl_trainer.distillation_checkpoint_capacity_gate import exact_next_update, save_checkpoints

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("device", DEVICES)
def test_capacity_witness_requires_real_durable_next_update(tmp_path, device):
    """Actual native-reference graph/codec data produces identical independent restored work."""
    job = make_job(device)
    job.update()
    saved = job.save(tmp_path)
    restored, witness = exact_next_update(job, saved)
    assert restored is not job and restored.replay is not job.replay
    assert restored.trainer.completed_updates == job.trainer.completed_updates == 2
    assert witness["optimizer"] and witness["windows"] and witness["sampler"]
    assert restored.last_training["optimizer_updated"]


def test_capacity_rejects_disabled_assertions_before_input_access():
    """The measured gate cannot silently omit its exactness/capacity checks under Python -O."""
    result = subprocess.run(
        [sys.executable, "-B", "-O", "-m",
         "pyrl_trainer.distillation_checkpoint_capacity_gate"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert "requires Python assertions enabled" in result.stderr


def test_capacity_four_publications_rotate_only_the_oldest_complete_checkpoint(tmp_path):
    """Real publication retains three files and exact latest continuation."""
    job = make_job("cpu")
    job.update()
    latest, rotation = save_checkpoints(job, tmp_path, 4)
    assert [item["serial"] for item in rotation["saves"]] == [1, 2, 3, 4]
    assert rotation["retained"] == 3 and rotation["oldestRotated"]
    assert rotation["retainedBytes"] == sum(
        path.stat().st_size for path in latest.parent.glob("job-*.distill-job")
    )
    restored, _ = exact_next_update(job, latest)
    assert restored.serial == 4 and restored.trainer.completed_updates == 2
