"""Fail CI when generated experiment payloads enter the ordinary Git tree.

Small golden inputs require an explicit path/hash/size/purpose approval. Historical
pre-feature docs/todo/evidence is outside this migration's policy scope.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_EXTENSIONS = (".py", ".rs", ".ts", ".mjs", ".cpp", ".h")
PAYLOAD_EXTENSIONS = (".gz", ".zip", ".tar", ".f32", ".u32", ".slither-brain", ".pt", ".pth")
MAX_PERFORMANCE_BYTES = 3 * 1024**2
MAX_FIXTURE_BYTES = 4 * 1024**2


def canonical(name, data):
    """Allow Git checkout EOL conversion for deliberately textual fixture receipts."""
    return data.replace(b"\r\n", b"\n") if name.endswith((".json", ".md", ".txt")) else data


def check(root, names=None):  # pylint: disable=too-many-branches
    """Inspect tracked and unignored candidates; no automatic allowlist expansion."""
    root = Path(root).resolve(strict=True)
    if names is None:
        names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others",
                                         "--exclude-standard"], cwd=root).decode().split("\0")
    approved = json.loads((root / "evidence-fixtures.json").read_bytes())["files"]
    errors, performance, fixtures = [], 0, 0
    for name in sorted(set(filter(None, names))):
        path = root / name
        if not path.exists():
            continue  # Tracked deletion pending staging.
        if name.startswith("docs/todo/evidence/"):
            continue  # Owner-requested preservation of pre-feature history.
        if path.is_symlink() or not path.is_file():
            errors.append("ordinary file required: " + name)
            continue
        size = path.stat().st_size
        fixture = (name.startswith(("tests/fixtures/", "native/fixtures/",
                                    "scripts/qualification/fixtures/")))
        if fixture:
            fixtures += size
            receipt = approved.get(name)
            data = canonical(name, path.read_bytes())
            if (not receipt or not receipt.get("purpose") or receipt["bytes"] != len(data)
                    or receipt["sha256"] != hashlib.sha256(data).hexdigest()):
                errors.append("fixture needs explicit path/hash/size/purpose approval: " + name)
            if size > 2 * 1024**2:
                errors.append("fixture exceeds two MiB admission: " + name)
        if name.startswith("docs/performance/"):
            performance += size
            if (not name.endswith(".md") and (Path(name).name not in {
                    "catalog.json", "source-provenance.json", "run.json", "summary.json"})):
                errors.append("generated performance payload belongs in artifact storage: "
                              + name)
            if size > 1024**2:
                errors.append("compact evidence manifest exceeds one MiB: " + name)
        if name.endswith(tuple(ext + ".gz" for ext in SOURCE_EXTENSIONS)):
            errors.append("compressed source snapshots are forbidden: " + name)
        if name.endswith(PAYLOAD_EXTENSIONS) and not fixture:
            errors.append("generated payload requires approved fixture or external artifact: "
                          + name)
        if name.endswith(".gz") and size < 2 * 1024**2 and not fixture:
            try:
                decoded = gzip.decompress(path.read_bytes())
                if b"SOURCE_FILES =" in decoded or b"import torch" in decoded:
                    errors.append("opaque compressed source snapshot: " + name)
            except (OSError, EOFError):
                pass
    if performance > MAX_PERFORMANCE_BYTES:
        errors.append("performance metadata exceeds three MiB total")
    if fixtures > MAX_FIXTURE_BYTES:
        errors.append("approved fixtures exceed four MiB total")
    if errors:
        raise ValueError("Evidence promotion rejected:\n" + "\n".join(errors))
    return {"performanceBytes": performance, "fixtureBytes": fixtures}


def main():
    """Guard the selected checkout; the command never writes approvals or stages files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(check(args.root)))


if __name__ == "__main__":
    main()
