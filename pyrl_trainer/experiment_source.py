"""Git identity for diagnostic runs; source files are never copied into outputs."""
import hashlib
from pathlib import Path
import subprocess
import sys


def capture(names, directory=None, source_root=None, *, allow_dirty=False):
    """Require a commit and explicit opt-in to dirty diagnostic code; retain restart hashes."""
    modules = Path(source_root or Path(__file__).parent).resolve()
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=modules, stderr=subprocess.PIPE
        ).decode("utf-8").rstrip("\r\n")
    repo = Path(git("rev-parse", "--show-toplevel"))
    arguments = ["-C", str(repo), "status", "--porcelain", "--untracked-files=normal", "--", "."]
    if directory is not None:
        output = Path(directory).resolve()
        if output.is_relative_to(repo):
            arguments.append(":(exclude)" + output.relative_to(repo).as_posix())
    dirty = git(*arguments).splitlines()
    if dirty and not allow_dirty:
        raise ValueError(
            "Experiment source is dirty; commit the experiment/WIP first, "
            "or use --allow-dirty for a local diagnostic run"
        )
    versions = {name: {"path": (modules / name).relative_to(repo).as_posix(),
                       "sha256": hashlib.sha256((modules / name).read_bytes()).hexdigest()}
                for name in names}
    dependencies = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                    for name in ("requirements.txt", "constraints.txt",
                                 "requirements-dev.txt", "requirements-contracts.txt")
                    if (repo / name).is_file()}
    return {"repository": git("remote", "get-url", "origin"), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(dirty), "dirtyPaths": dirty, "versions": versions,
            "dependencies": dependencies, "python": sys.version, "command": sys.argv,
            "hashPolicy": "Raw loaded module bytes for restart identity; Git identifies source."}
