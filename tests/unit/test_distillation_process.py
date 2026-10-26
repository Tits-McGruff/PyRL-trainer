"""Standalone priority policy is bounded to its child process and preserves lower choices."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pyrl_trainer.distillation_process import current_process_priority

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_cli_lowers_only_its_own_process_before_training_imports():
    """The real OS reports a lower child priority while the test parent stays unchanged."""
    before = current_process_priority()
    output = subprocess.check_output(
        [sys.executable, "-B", "-m", "pyrl_trainer.distillation_process"],
        cwd=ROOT, text=True, timeout=20)
    actual = json.loads(output)
    if os.name == "nt":
        assert actual["priorityClass"] in (0x4000, 0x40)
    else:
        assert actual["nice"] >= max(5, before["nice"])
    assert current_process_priority() == before


def test_existing_lower_priority_is_not_raised():
    """An owner-selected low priority survives a second application of CLI policy."""
    if os.name == "nt":
        setup = ("from pyrl_trainer.distillation_process import _windows_api; "
                 "k=_windows_api(); assert k.SetPriorityClass(k.GetCurrentProcess(),0x40); ")
    else:
        setup = "import os; os.setpriority(os.PRIO_PROCESS,0,10); "
    code = (setup + "import json; from pyrl_trainer.distillation_process import "
            "configure_cli_priority; print(json.dumps(configure_cli_priority()))")
    actual = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code],
                                                cwd=ROOT, text=True, timeout=20))
    assert actual == ({"platform": "windows", "priorityClass": 0x40} if os.name == "nt"
                      else {"platform": "posix", "nice": 10})
