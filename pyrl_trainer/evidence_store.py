"""Checksum-addressed historical evidence, separate from the normal Git tree.

Export is explicit. Fetch never trusts tar paths, links, sizes or receipts. A
required audit fails with a fetch command when its verified local cache is absent.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import urllib.request

MAX_ARCHIVE = 512 * 1024**2
MAX_TOTAL = 1024**3
MAX_FILE = 128 * 1024**2


def sha(data):
    """Hash exact bytes; source normalization is a separate provenance field."""
    return hashlib.sha256(data).hexdigest()


def confined(root, name):
    """Admit portable relative ordinary paths, excluding traversal and aliases."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name):
        raise ValueError("invalid evidence path")
    if any(part in (".", "..") for part in PurePosixPath(name).parts):
        raise ValueError("evidence traversal")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("evidence path escapes cache")
    return path


def inventory(catalog):
    """Validate the complete bounded manifest before allocation or extraction."""
    if catalog.get("schemaVersion") != 1 or catalog.get("kind") != "external-evidence-v1":
        raise ValueError("unsupported evidence catalog")
    files = catalog["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 10000:
        raise ValueError("bounded evidence inventory required")
    objects = {}
    total = 0
    for name, record in files.items():
        confined(Path.cwd(), name)
        digest, size = record
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(size, int) or isinstance(size, bool)
                or not 0 <= size <= MAX_FILE):
            raise ValueError("invalid evidence object receipt")
        if digest in objects and objects[digest] != size:
            raise ValueError("conflicting object size")
        objects[digest] = size
        total += size
    if total > MAX_TOTAL:
        raise ValueError("evidence inventory exceeds admission")
    return objects


def export(root, output, url, run=None):
    """Export tracked performance files once, deduplicating identical object bytes."""
    root = Path(root).resolve(strict=True)
    if run is None:
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "docs/performance"], cwd=root).decode().split("\0")
    else:
        run = Path(run).resolve(strict=True)
        if not run.is_relative_to(root) or run == root:
            raise ValueError("completed run must be a dedicated directory inside the repository")
        names = [path.relative_to(root).as_posix() for path in run.rglob("*") if path.is_file()]
        provenance = json.loads((run / "provenance.json").read_bytes())
        required = {"repositories", "sources", "build", "toolchain", "dependencies",
                    "commands", "settings", "seeds", "environment"}
        if not required <= provenance.keys():
            raise ValueError("durable promotion requires complete provenance.json: "
                             + ", ".join(sorted(required)))
        for repository in provenance["repositories"]:
            if (not isinstance(repository.get("repository"), str)
                    or not re.fullmatch(r"[0-9a-f]{40}", repository.get("commit", ""))):
                raise ValueError("exact repository/commit provenance required")
    files, objects = {}, {}
    for name in sorted(filter(None, names)):
        path = confined(root, name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("ordinary tracked evidence required: " + name)
        if run is not None and (name.endswith((".py", ".rs", ".ts", ".mjs"))
                                or name.endswith((".py.gz", ".rs.gz", ".ts.gz", ".mjs.gz"))):
            raise ValueError("record Git provenance instead of source snapshots: " + name)
        data = path.read_bytes()
        digest = sha(data)
        files[name] = [digest, len(data)]
        objects[digest] = data
    catalog = {"schemaVersion": 1, "kind": "external-evidence-v1",
               "repository": subprocess.check_output(
                   ["git", "remote", "get-url", "origin"], cwd=root, text=True).strip(),
               "sourceCommit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "bytePolicy": ("Exact pre-migration working-tree bytes; "
                              "no normalization of payloads."),
               "files": files}
    inventory(catalog)
    with Path(output).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for digest, data in sorted(objects.items()):
                    info = tarfile.TarInfo("objects/" + digest)
                    info.size, info.mtime, info.mode = len(data), 0, 0o444
                    archive.addfile(info, io.BytesIO(data))
    data = Path(output).read_bytes()
    catalog["artifact"] = {"url": url, "sha256": sha(data), "bytes": len(data),
                           "objects": len(objects), "decodedBytes": sum(map(len, objects.values()))}
    return catalog


def verify(root, catalog):
    """Rehash every materialized file; a stamp alone never establishes integrity."""
    inventory(catalog)
    for name, (digest, size) in catalog["files"].items():
        path = confined(root, name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("required evidence missing: " + name)
        if path.stat().st_size != size or sha(path.read_bytes()) != digest:
            raise ValueError("evidence content changed: " + name)
    return root


def _asset_bytes(artifact, archive_path):
    """Read one bounded encoded asset and bind it to its published receipt."""
    if archive_path is not None:
        archive_path = Path(archive_path)
        if (archive_path.is_symlink() or not archive_path.is_file()
                or archive_path.stat().st_size != artifact["bytes"]):
            raise ValueError("release asset size/checksum mismatch")
        data = archive_path.read_bytes()
    else:
        if not artifact["url"].startswith("https://github.com/"):
            raise ValueError("explicit HTTPS release asset required")
        try:
            with urllib.request.urlopen(artifact["url"], timeout=60) as response:
                data = response.read(artifact["bytes"] + 1)
        except OSError as error:
            raise RuntimeError("Required evidence unavailable. Download release asset "
                               + artifact["url"] + " and retry fetch --archive FILE") from error
    if len(data) != artifact["bytes"] or sha(data) != artifact["sha256"]:
        raise ValueError("release asset size/checksum mismatch")
    return data


def fetch(catalog_path, cache, archive_path=None):  # pylint: disable=too-many-branches
    """Download/restore an admitted content-addressed archive, then verify restorability."""
    catalog = json.loads(Path(catalog_path).read_bytes())
    objects = inventory(catalog)
    artifact = catalog["artifact"]
    if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
        raise ValueError("invalid archive identity")
    if (not isinstance(artifact["bytes"], int) or isinstance(artifact["bytes"], bool)
            or not 0 < artifact["bytes"] <= MAX_ARCHIVE):
        raise ValueError("archive exceeds admission")
    root = Path(cache).resolve() / artifact["sha256"]
    if (root / "verified.json").is_file():
        return verify(root, catalog)
    root.mkdir(parents=True, exist_ok=True)
    local = root / "archive.tar.gz"
    data = _asset_bytes(artifact, archive_path)
    local.write_bytes(data)
    seen = set()
    with tarfile.open(local, "r:gz") as archive:
        for member in archive:
            digest = member.name.removeprefix("objects/")
            if (member.name != "objects/" + digest or digest not in objects or digest in seen
                    or not member.isfile() or member.size != objects[digest]):
                raise ValueError("unadmitted archive member")
            seen.add(digest)
            stream = archive.extractfile(member)
            value = stream.read(member.size + 1)
            if len(value) != member.size or sha(value) != digest:
                raise ValueError("archive object checksum mismatch")
            target = confined(root, member.name)
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(value)
    if seen != set(objects):
        raise ValueError("archive is incomplete")
    for name, (digest, _) in catalog["files"].items():
        target = confined(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink():
                raise ValueError("cache link refused")
            target.unlink()
        os.link(root / "objects" / digest, target)
    verify(root, catalog)
    (root / "verified.json").write_text(json.dumps({"archiveSha256": artifact["sha256"]}) + "\n")
    return root


def required_root(repository):
    """Resolve explicitly prepared historical inputs; never silently skip qualification."""
    repository = Path(repository)
    catalog_path = repository / "docs/performance/catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    root = Path(os.environ.get("QUALIFICATION_EVIDENCE_ROOT", str(
        repository / ".cache/qualification-evidence" / catalog["artifact"]["sha256"])))
    if not (root / "verified.json").is_file():
        raise RuntimeError("Required historical evidence is unavailable. Run: python -m "
                           "pyrl_trainer.evidence_store fetch --catalog " + str(catalog_path)
                           + " --cache " + str(repository / ".cache/qualification-evidence"))
    return root


def main():
    """Explicit lifecycle commands; no operation stages files or commits artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "fetch", "verify"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--legacy-migration", action="store_true")
    parser.add_argument("--url")
    args = parser.parse_args()
    if args.action == "export":
        if args.run is None and not args.legacy_migration:
            parser.error("export requires --run DIR (or explicit --legacy-migration)")
        result = export(args.root, args.archive, args.url, args.run)
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        args.catalog.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        print(json.dumps(result["artifact"]))
    elif args.action == "fetch":
        print(fetch(args.catalog, args.cache, args.archive))
    else:
        print(verify(args.root, json.loads(args.catalog.read_bytes())))


if __name__ == "__main__":
    main()
