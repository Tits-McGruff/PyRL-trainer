"""Separate-process canonical job restart, including cold CUDA compilation when available."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraphManifest
from pyrl_trainer.distillation_checkpoint import load_job_checkpoint
from pyrl_trainer.distillation_job import DistillationJob, DistillationJobConfig
from pyrl_trainer.live_distillation_gate import _witness

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _worker(phase, device, directory):
    """Use frozen native graph/codec fixtures; these synthetic rows are test data only."""
    directory = Path(directory)
    if phase == "save":
        spec = importlib.util.spec_from_file_location(
            "process_sampling_fixture", ROOT / "tests/unit/test_distillation_sampling.py")
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        job = DistillationJob(CanonicalGraphManifest.read(helpers.GRAPH / "manifest.json"),
                              (helpers.GRAPH / "weights.f32").read_bytes(),
                              helpers.replay_fixture(), sequence=helpers.config(),
                              config=DistillationJobConfig(device=device))
        for _index in range(3):
            assert job.update().optimizer_updated
        job.python_rng.random()
        torch.rand(7, generator=job.cpu_rng)
        if job.cuda_rng is not None:
            torch.rand(7, generator=job.cuda_rng, device=job.device)
        checkpoint = job.save(directory)
        expected = job.update()
        oracle = {"checkpoint": str(checkpoint), "witness": _witness(job, expected),
                  "environment": job.environment,
                  "execution": job.trainer.window_execution.statistics()}
        if device.startswith("cuda"):
            assert oracle["execution"]["compiledWindowCalls"] == 4
            assert oracle["execution"]["eagerWindowCalls"] == 0
        (directory / "oracle.json").write_text(json.dumps(oracle, sort_keys=True),
                                               encoding="utf-8")
    elif phase == "restore":
        oracle = json.loads((directory / "oracle.json").read_text(encoding="utf-8"))
        job = load_job_checkpoint(Path(oracle["checkpoint"]))
        assert str(job.device) == device and job.environment == oracle["environment"]
        assert job.trainer.window_execution.compiled is None
        assert job.trainer.window_execution.signature is None
        actual = job.update()
        assert _witness(job, actual) == oracle["witness"]
        execution = job.trainer.window_execution.statistics()
        if device.startswith("cuda"):
            assert execution["compiledWindowCalls"] == 1 and execution["eagerWindowCalls"] == 0
        else:
            assert execution["eagerWindowCalls"] == 1 and execution["compiledWindowCalls"] == 0
        (directory / "restored.json").write_text(
            json.dumps({"restartExact": True, "device": device, "execution": execution}),
            encoding="utf-8")
    else:
        raise ValueError("unknown test worker phase")


def _run_worker(phase, device, directory):
    """Bound one owned process tree, including a Windows virtual-environment launcher."""
    bootstrap = ("import runpy; runpy.run_path(" + repr(str(Path(__file__).resolve()))
                 + ", run_name='__main__')")
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with subprocess.Popen([sys.executable, "-B", "-c", bootstrap, phase, device, str(directory)],
                          cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
                          creationflags=flags) as child:
        try:
            output, _unused = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               capture_output=True, timeout=10, check=False, creationflags=flags)
            else:
                getattr(os, "killpg")(child.pid, getattr(signal, "SIGKILL"))
            child.communicate(timeout=10)
            raise
        assert child.returncode == 0, output[-16384:].decode("utf-8", errors="replace")


@pytest.mark.parametrize("device", DEVICES)
def test_cold_job_process_restores_exact_next_update_without_compiler_payload(tmp_path, device):
    """Fresh interpreters match optimizer, recurrence, sample schedule and private RNG bits."""
    _run_worker("save", device, tmp_path)
    _run_worker("restore", device, tmp_path)
    result = json.loads((tmp_path / "restored.json").read_text(encoding="utf-8"))
    assert result["restartExact"] and result["device"] == device


if __name__ == "__main__":
    _worker(*sys.argv[1:])
