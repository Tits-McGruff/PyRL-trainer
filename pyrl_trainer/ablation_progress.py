"""Bounded, checksum-verified completed-variant checkpoints for diagnostic studies."""
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile


def replace_report(path, value):
    """Publish complete JSON atomically; failed prepublication leaves the prior file intact."""
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 8 * 1024**2:
        raise ValueError("bounded ablation checkpoint required")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".ablation-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def variant_inventory(directory):
    """Bind every ordinary output file without retaining another copy of its contents."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("ordinary variant directory required")
    files = {}
    total = 0
    for index, path in enumerate(directory.rglob("*")):
        if (index >= 128 or path.is_symlink()
                or getattr(path.lstat(), "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("bounded ordinary variant outputs required")
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name):
            raise ValueError("invalid variant output path")
        if not path.is_file() or path.stat().st_size > 8 * 1024**2:
            raise ValueError("bounded variant output file required")
        with path.open("rb") as stream:
            raw = stream.read(8 * 1024**2 + 1)
        total += len(raw)
        if len(raw) > 8 * 1024**2 or len(files) >= 64 or total > 64 * 1024**2:
            raise ValueError("variant output admission exceeded")
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if not files:
        raise ValueError("completed variant outputs required")
    return files


@contextmanager
def study_lease(directory):
    """Hold one process-owned advisory lock; process death releases ownership automatically."""
    library = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
    with (directory / ".study.lock").open("a+b") as stream:
        if not stream.seek(0, os.SEEK_END):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                library.locking(stream.fileno(), library.LK_NBLCK, 1)
            else:
                library.flock(stream.fileno(), library.LOCK_EX | library.LOCK_NB)
        except OSError as error:
            raise ValueError("study already owned by another process") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                library.locking(stream.fileno(), library.LK_UNLCK, 1)
            else:
                library.flock(stream.fileno(), library.LOCK_UN)
