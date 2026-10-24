"""Artifact lifecycle and promotion policy fail closed without large experiment inputs."""
import copy
import gzip
import hashlib
import io
import json
import tarfile

import pytest

from pyrl_trainer.evidence_store import export, fetch, inventory, required_root, verify
from pyrl_trainer.evidence_policy import check
from pyrl_trainer.source_provenance import capture, restore

pytestmark = pytest.mark.unit


def archive_fixture(root, member="objects/", content=b"small historical output"):
    """Build an inert one-file archive with an explicit bounded inventory."""
    digest = hashlib.sha256(content).hexdigest()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo(member + digest)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    data = stream.getvalue()
    path = root / "archive.tar.gz"
    path.write_bytes(data)
    catalog = {"schemaVersion": 1, "kind": "external-evidence-v1",
               "files": {"docs/performance/run/report.json": [digest, len(content)]},
               "artifact": {"url": "https://github.com/example/repo/releases/download/v1/data",
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
    manifest = root / "catalog.json"
    manifest.write_text(json.dumps(catalog), encoding="utf8")
    return path, manifest, catalog


def test_restore_verifies_every_byte_and_rejects_changed_cache(tmp_path):
    """A valid archive restores exact paths; a prior success stamp cannot hide corruption."""
    archive, manifest, catalog = archive_fixture(tmp_path)
    root = fetch(manifest, tmp_path / "cache", archive)
    assert verify(root, catalog) == root
    (root / "docs/performance/run/report.json").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        fetch(manifest, tmp_path / "cache", archive)


def test_changed_asset_cannot_restore(tmp_path):
    """Release URL or filename alone is insufficient provenance."""
    archive, manifest, _ = archive_fixture(tmp_path)
    archive.write_bytes(b"substituted archive")
    with pytest.raises(ValueError, match="checksum"):
        fetch(manifest, tmp_path / "cache", archive)


@pytest.mark.parametrize("member", ["../", "/", "objects/../", "other/"])
def test_unadmitted_tar_members_reject(tmp_path, member):
    """No traversal, absolute paths or unlisted objects enter the cache."""
    archive, manifest, _ = archive_fixture(tmp_path, member)
    with pytest.raises(ValueError, match="unadmitted"):
        fetch(manifest, tmp_path / "cache", archive)


@pytest.mark.parametrize("name", ["../private", "/absolute", "docs/../../private", "C:/private"])
def test_manifest_traversal_rejects_before_restore(tmp_path, name):
    """Receipt paths are validated independently of tar paths."""
    _, _, catalog = archive_fixture(tmp_path)
    catalog["files"] = {name: next(iter(catalog["files"].values()))}
    with pytest.raises(ValueError):
        inventory(catalog)


def test_missing_required_archive_is_actionable(tmp_path):
    """Qualification never passes or skips when its required archive is absent."""
    _, _, catalog = archive_fixture(tmp_path)
    path = tmp_path / "docs/performance"
    path.mkdir(parents=True)
    (path / "catalog.json").write_text(json.dumps(catalog), encoding="utf8")
    with pytest.raises(RuntimeError, match="evidence_store fetch"):
        required_root(tmp_path)


@pytest.mark.parametrize("name,content", [
    ("docs/performance/run/source.py.gz", gzip.compress(b"import torch")),
    ("docs/performance/run/model.slither-brain", b"model"),
    ("docs/performance/run/timing.json", b"{}"),
    ("tests/fixtures/unapproved.f32", bytes(8)),
    ("docs/performance/run/summary.json", b"x" * (1024**2 + 1)),
], ids=["source-snapshot", "model", "timing-output", "unapproved-fixture", "oversized-summary"])
def test_promotion_rejects_payload_source_and_unapproved_fixture(tmp_path, name, content):
    """Renaming an output directory cannot make a generated payload an approved fixture."""
    (tmp_path / "evidence-fixtures.json").write_text('{"files":{}}', encoding="utf8")
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    with pytest.raises(ValueError, match="Evidence promotion rejected"):
        check(tmp_path, [name])


def test_source_provenance_restores_dirty_source_and_preserves_raw_hash(tmp_path):
    """Runtime hashes still bind exact bytes while Git patches carry only the difference."""
    records = capture(("tactical_ablation_gate.py",), tmp_path)["versions"]
    record = records["tactical_ablation_gate.py"]
    assert hashlib.sha256(restore(record, tmp_path)).hexdigest() == record["sha256"]
    bad = copy.deepcopy(record)
    bad["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="measured source"):
        restore(bad, tmp_path)


def test_durable_promotion_requires_complete_metadata_and_refuses_source(tmp_path):
    """Promoting a completed run never turns tracked source into another artifact snapshot."""
    root = tmp_path / "repository"
    run = root / "ignored-output/run"
    run.mkdir(parents=True)
    (run / "provenance.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="complete provenance"):
        export(root, tmp_path / "out.tar.gz", "https://github.com/example/asset", run)
    provenance = {key: {} for key in ("sources", "build", "toolchain", "dependencies",
                                     "commands", "settings", "seeds", "environment")}
    provenance["repositories"] = [{"repository": "https://github.com/example/repo",
                                    "commit": "a" * 40}]
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf8")
    (run / "source.py.gz").write_bytes(gzip.compress(b"print('source copy')"))
    with pytest.raises(ValueError, match="instead of source snapshots"):
        export(root, tmp_path / "out.tar.gz", "https://github.com/example/asset", run)
