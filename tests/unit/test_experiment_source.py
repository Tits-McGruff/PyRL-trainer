"""Experiments identify Git commits and never snapshot dirty source."""
import subprocess

import pytest

from pyrl_trainer.experiment_source import capture

pytestmark = pytest.mark.unit


def test_clean_commit_and_explicit_dirty_diagnostic(tmp_path):
    """Dirty use requires explicit consent and produces no source artifact."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path)
    git("init")
    git("remote", "add", "origin", "https://example.invalid/experiment.git")
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf8")
    git("add", "module.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-m", "experiment")
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_text("{}", encoding="utf8")
    before = capture(["module.py"], output, tmp_path)
    assert before["commit"] == git("rev-parse", "HEAD").decode().strip()
    assert before["dirty"] is False
    source.write_text("VALUE = 2\n", encoding="utf8")
    with pytest.raises(ValueError, match="WIP"):
        capture(["module.py"], output, tmp_path)
    changed = capture(["module.py"], output, tmp_path, allow_dirty=True)
    assert changed["dirty"] and changed["dirtyPaths"]
    assert changed["versions"] != before["versions"]
    assert [p.name for p in output.iterdir()] == ["result.json"]
