"""Reconstruct measured sources from Git provenance and explicit experimental patches.

Runtime trainer hashes continue to bind raw loaded bytes. Canonical LF hashes bind
Git content; an explicit EOL recipe restores the exact measured byte identity.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(data):
    """SHA-256 exact bytes."""
    return hashlib.sha256(data).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def _safe(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("confined source path required")
    return path


def _endings(raw):
    if b"\r\n" not in raw:
        return {"lineEndings": "lf"}
    if raw.count(b"\r\n") == raw.count(b"\n"):
        return {"lineEndings": "crlf"}
    return {"lineEndings": "mixed", "crlfLines": [i for i, line in enumerate(
        raw.splitlines(keepends=True)) if line.endswith(b"\r\n")]}


def capture(names, directory, source_root=None):
    """Record committed base, inventory, dependency identity and only uncommitted diffs."""
    modules = Path(source_root or Path(__file__).parent).resolve()
    repo = Path(_git(modules, "rev-parse", "--show-toplevel").decode().strip())
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    remote = _git(repo, "remote", "get-url", "origin").decode().strip()
    versions = {}
    for name in names:
        path = (modules / _safe(name)).relative_to(repo).as_posix()
        raw = (repo / path).read_bytes()
        canonical = raw.replace(b"\r\n", b"\n")
        try:
            base = _git(repo, "show", commit + ":" + path).replace(b"\r\n", b"\n")
        except subprocess.CalledProcessError:
            base = b""
        record = {"repository": remote, "commit": commit, "path": path,
                  "sha256": digest(raw), "bytes": len(raw),
                  "canonicalLfSha256": digest(canonical), **_endings(raw)}
        if base != canonical:
            if not canonical.endswith(b"\n"):
                raise ValueError("experimental source patch requires a final LF")
            patch = "".join(difflib.unified_diff(
                base.decode().splitlines(keepends=True),
                canonical.decode().splitlines(keepends=True),
                fromfile="a/" + path if base else "/dev/null", tofile="b/" + path)).encode()
            filename = "source-" + digest(canonical) + ".patch"
            (Path(directory) / filename).write_bytes(patch)
            record.update(patch=filename, patchSha256=digest(patch))
        versions[name] = record
    dependencies = {}
    for name in ("requirements.txt", "constraints.txt", "requirements-dev.txt"):
        path = repo / name
        if path.is_file():
            dependencies[name] = digest(path.read_bytes())
    return {"schemaVersion": 1, "repository": remote, "commit": commit,
            "normalizationPolicy": ("Git canonical LF; exact measured EOL recipe "
                                    "and raw SHA-256."),
            "python": sys.version, "dependencies": dependencies,
            "command": ["python", "-m", "pyrl_trainer.tactical_ablation_gate", *sys.argv[1:]],
            "versions": versions}


def restore(record, patch_root, repository=None):  # pylint: disable=too-many-branches
    """Recover and verify one exact source; absent Git objects fail with a fetch instruction."""
    root = Path(repository or os.environ.get("PYRL_SOURCE_REPOSITORY", Path(__file__).parents[1]))
    if "slither_neuroevo" in record["repository"]:
        root = Path(os.environ.get("SLITHER_SOURCE_REPOSITORY", root.parent / "slither_neuroevo"))
    if not re.fullmatch(r"[0-9a-f]{40}", record["commit"]):
        raise ValueError("exact source commit SHA required")
    path = _safe(record["path"])
    try:
        canonical = _git(root, "show", record["commit"] + ":" + path.as_posix())
    except subprocess.CalledProcessError as error:
        if "patch" not in record:
            raise RuntimeError("Required source unavailable; fetch commit " + record["commit"]
                               + " from " + record["repository"] + " into " + str(root)) from error
        # Explicit new-file patches have no base blob; the commit still must exist.
        _git(root, "cat-file", "-e", record["commit"] + "^{commit}")
        canonical = b""
    canonical = canonical.replace(b"\r\n", b"\n")
    if "patch" in record:
        patch = (Path(patch_root) / _safe(record["patch"])).read_bytes().replace(b"\r\n", b"\n")
        if digest(patch) != record["patchSha256"]:
            raise ValueError("experimental source patch checksum mismatch")
        with tempfile.TemporaryDirectory(prefix="measured-source-") as temp:
            temporary = Path(temp)
            headers = [line for line in patch.decode().splitlines() if line.startswith("+++ ")]
            if headers != ["+++ b/" + path.as_posix()]:
                raise ValueError("source patch must affect only its admitted path")
            target = temporary / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if canonical:
                target.write_bytes(canonical)
            result = subprocess.run(["git", "-c", "core.autocrlf=false", "apply",
                                     "--unsafe-paths",
                                     "--directory=" + str(temporary)], input=patch,
                                    capture_output=True, check=False)
            if result.returncode:
                raise ValueError("experimental source patch does not apply")
            canonical = target.read_bytes()
    if digest(canonical) != record["canonicalLfSha256"]:
        raise ValueError("canonical source content mismatch")
    ending = record["lineEndings"]
    if ending == "crlf":
        raw = canonical.replace(b"\n", b"\r\n")
    elif ending == "mixed":
        lines = canonical.splitlines(keepends=True)
        for index in record["crlfLines"]:
            lines[index] = lines[index].replace(b"\n", b"\r\n")
        raw = b"".join(lines)
    elif ending == "lf":
        raw = canonical
    else:
        raise ValueError("unsupported line ending policy")
    if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
        raise ValueError("measured source content mismatch")
    return raw


def measured_sources(report, directory):
    """Verify modern provenance or the explicit historical migration registry."""
    if "sourceProvenance" in report:
        provenance = report["sourceProvenance"]
        versions = provenance["versions"]
        patch_root = directory
        records = {name: versions[name] for name in report["sources"]}
    else:
        filename = os.environ.get("PYRL_HISTORICAL_SOURCE_PROVENANCE")
        registry_path = Path(filename) if filename else (
            Path(__file__).parents[1] / "docs/performance/source-provenance.json")
        registry = json.loads(registry_path.read_bytes())
        patch_root = registry_path.parents[2]
        records = {name: registry["versions"][receipt["sha256"]]
                   for name, receipt in report["sources"].items()}
    result = {}
    for name, record in records.items():
        if record["sha256"] != report["sources"][name]["sha256"]:
            raise ValueError("measured source receipt differs from provenance")
        result[name] = restore(record, patch_root)
    return result


def audit_original_patches(index):
    """Check preserved git-format experiments against their recorded ordinary base commits."""
    index = Path(index).resolve(strict=True)
    root = index.parents[1]
    records = json.loads(index.read_bytes())["patches"]
    seen = set()
    for record in records.values():
        if record["sha256"] in seen:
            continue
        seen.add(record["sha256"])
        patch = (root / _safe(record["patch"])).read_bytes().replace(b"\r\n", b"\n")
        if digest(patch) != record["sha256"]:
            raise ValueError("original experimental patch checksum mismatch")
        if not re.fullmatch(r"[0-9a-f]{40}", record["baseCommit"]):
            raise ValueError("original patch requires an exact ordinary base commit")
        paths = re.findall(r"^--- a/(.+)$", patch.decode(), re.MULTILINE)
        with tempfile.TemporaryDirectory(prefix="historical-patch-") as temp:
            temporary = Path(temp)
            for name in paths:
                target = temporary / _safe(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_git(root, "show", record["baseCommit"] + ":" + name)
                                   .replace(b"\r\n", b"\n"))
            subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--check",
                            "--unsafe-paths", "--directory=" + str(temporary)],
                           input=patch, capture_output=True, check=True)
    return len(seen)


def main():
    """Required source inventory audit; every commit/patch/EOL recipe must restore exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--original-patches", type=Path)
    args = parser.parse_args()
    registry = args.registry.resolve(strict=True)
    records = json.loads(registry.read_bytes())["versions"]
    root = registry.parents[2]
    for record in records.values():
        restore(record, root)
    original = audit_original_patches(args.original_patches) if args.original_patches else 0
    print(json.dumps({"sourceVersionsVerified": len(records), "originalPatchesVerified": original,
                      "experimentalPatches": sum("patch" in item for item in records.values())}))


if __name__ == "__main__":
    main()
