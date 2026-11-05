"""Admission-only migration must explain source and preserve actual restart state."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import torch
from slither_contracts import IDENTITY
from test_distillation_job import assert_optimizer_equal, make_job

from pyrl_trainer.distillation_checkpoint import load_job_checkpoint
from pyrl_trainer.distillation_export import export
from pyrl_trainer.distillation_job import SOURCE_FILES
from pyrl_trainer.distillation_migrate import APPROVAL, _normalized, _sha, _source_identity, migrate
from pyrl_trainer.distillation_transport import DistillationContractError

pytestmark = pytest.mark.unit


@pytest.fixture
def _approved_current_contract(tmp_path, monkeypatch):
    """Exercise migration mechanics without broadening the historical release approval.

    The actual allowlist must reject unrelated contract revisions. Payload and
    optimizer tests instead use an explicit test-only approval of this package.
    """
    approval = json.loads(APPROVAL.read_bytes())
    approval["targetContractIdentities"] = [IDENTITY]
    path = tmp_path / "test-only-migration-approval.json"
    path.write_text(json.dumps(approval), encoding="utf-8")
    monkeypatch.setattr("pyrl_trainer.distillation_migrate.APPROVAL", path)


def old_sources(root):
    """Reverse only this patch's reviewed admissions; no old source snapshot is stored."""
    approval = json.loads(APPROVAL.read_bytes())
    package = Path(__file__).resolve().parents[2] / "pyrl_trainer"
    for name in SOURCE_FILES:
        text = (package / name).read_text(encoding="utf-8")
        if name == "canonical_graph.py":
            text = text.replace(
                "    if (not 1 <= len(output_sizes) <= 16 or "
                "sum(output_sizes) > MAX_TENSOR_WIDTH\n            or len(hidden_sizes) > 8):",
                "    if not 1 <= len(output_sizes) <= 16 or sum(output_sizes) > "
                "MAX_TENSOR_WIDTH or len(hidden_sizes) > 8:",
            )
            text = text.replace(
                'MAX_PARAMETERS = CONSTANTS["maxBrainParameters"]\n'
                'MAX_TENSOR_WIDTH = CONSTANTS["maxBrainTensorWidth"]\n'
                'MAX_STATE_FLOATS = CONSTANTS["maxBrainStateFloats"]\n',
                "",
            )
            text = (
                text.replace("MAX_TENSOR_WIDTH + 1", "513")
                .replace("MAX_TENSOR_WIDTH", "512")
                .replace("MAX_PARAMETERS", "1_000_000")
                .replace("MAX_STATE_FLOATS", "16384")
            )
        if name == "brain_artifact.py":
            text = text.replace('or count > CONSTANTS["maxBrainParameters"]', "or count > 1000000")
            text = text.replace(
                'not 0 <= manifest.total_parameters <= CONSTANTS["maxBrainParameters"]\n'
                "        or len(weights)",
                "len(weights)",
            )
        if name == "distillation_checkpoint.py":
            text = text.replace("from slither_contracts import CONSTANTS\n", "").replace(
                '"weights.f32": CONSTANTS["maxBrainWeightBytes"]', '"weights.f32": 4 * 1024**2'
            )
            text = text.replace(
                "    # Two full Float32 Adam moments (16,000,000 bytes) "
                "plus bounded Torch metadata.\n",
                "",
            ).replace("    # Generator state does not scale with parameter count.\n", "")
            text = text.replace(
                "    # SequenceTrainingConfig permits 256 windows; each owns its detached state.\n",
                "",
            ).replace(
                '"state.f32": 256 * CONSTANTS["maxBrainStateFloats"] * 4',
                '"state.f32": 8 * 1024**2',
            )
        if name == "distillation_client.py":
            text = (
                text.replace("\nfrom slither_contracts import CONSTANTS\n", "")
                .replace('CONSTANTS["maxBrainParameters"]', "1000000")
                .replace('CONSTANTS["maxBrainStateFloats"]', "16384")
            )
        data = text.encode()
        assert _sha(_normalized(data)) in approval["sourceFiles"][name], name
        (root / name).write_bytes(data)
    return {name: (root / name).read_bytes() for name in SOURCE_FILES}


def relabel_test_checkpoint(source, target, identity):
    """Forge only test source metadata to model a real pre-maintenance snapshot."""
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "w") as new:
        for name in old.namelist():
            data = old.read(name)
            if name == "job.json":
                metadata = json.loads(data)
                metadata["trainerSourceSha256"] = identity
                data = json.dumps(metadata).encode()
            new.writestr(name, data)


def test_explicit_migration_retains_payloads_and_next_real_update(
    tmp_path, _approved_current_contract
):
    """The normal loader rejects mismatch; a reviewed successor resumes full Adam/replay/RNG."""
    job = make_job()
    job.update()
    saved = job.save(tmp_path / "saved")
    source = tmp_path / "old-package"
    source.mkdir()
    files = old_sources(source)
    approval = json.loads(APPROVAL.read_bytes())
    identity = _source_identity(files, approval["sourceContractIdentities"][0])
    old = tmp_path / "old.distill-job"
    relabel_test_checkpoint(saved, old, identity)
    original = old.read_bytes()
    with pytest.raises(DistillationContractError, match="matching trainer source"):
        load_job_checkpoint(old)
    successor = tmp_path / "successor.distill-job"
    receipt = migrate(old, successor, source)
    assert old.read_bytes() == original and receipt["updates"] == 1
    with zipfile.ZipFile(old) as left, zipfile.ZipFile(successor) as right:
        assert left.namelist() == right.namelist()
        for name in left.namelist():
            if name != "job.json":
                assert left.read(name) == right.read(name)
    restored = load_job_checkpoint(successor)
    expected = job.update()
    actual = restored.update()
    assert (
        actual.loss == expected.loss
        and restored.model.export_weights() == job.model.export_weights()
    )
    assert_optimizer_equal(
        restored.trainer.optimizer.state_dict(), job.trainer.optimizer.state_dict()
    )
    assert restored.sampler.checkpoint_fields() == job.sampler.checkpoint_fields()
    with pytest.raises(DistillationContractError, match="new output"):
        migrate(old, successor, source)
    # An unreviewed computation edit cannot be hidden by providing its matching source hash.
    path = source / "sequence_distillation.py"
    path.write_bytes(path.read_bytes() + b"\n# unreviewed\n")
    with pytest.raises(DistillationContractError, match="unreviewed source change"):
        migrate(old, tmp_path / "rejected.distill-job", source)
    assert not (tmp_path / "rejected.distill-job").exists()


def test_saved_weight_export_requires_no_source_relabel_or_training(tmp_path):
    """A completed model can be portably extracted even when exact restart is unavailable."""
    job = make_job()
    job.update()
    saved = job.save(tmp_path / "saved")
    old = tmp_path / "foreign.distill-job"
    relabel_test_checkpoint(saved, old, "ab" * 32)
    before = old.read_bytes()
    artifact = export(old, tmp_path / "trained.slither-brain")
    assert artifact.data.endswith(job.model.export_weights()) and old.read_bytes() == before
    with pytest.raises(FileExistsError):
        export(old, tmp_path / "trained.slither-brain")
    with zipfile.ZipFile(old) as archive:
        assert json.loads(archive.read("job.json"))["trainerSourceSha256"] == "ab" * 32


def test_migration_rejects_rehashed_invalid_optimizer_and_live_source_changes(
    tmp_path, monkeypatch, _approved_current_contract
):
    """Valid ZIP/member digests cannot bypass optimizer semantics or loaded-source identity."""
    job = make_job()
    job.update()
    saved = job.save(tmp_path / "saved")
    source = tmp_path / "old-package"
    source.mkdir()
    files = old_sources(source)
    approval = json.loads(APPROVAL.read_bytes())
    old = tmp_path / "old.distill-job"
    relabel_test_checkpoint(
        saved, old, _source_identity(files, approval["sourceContractIdentities"][0])
    )
    bad = tmp_path / "bad.distill-job"
    with zipfile.ZipFile(old) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    optimizer = torch.load(io.BytesIO(entries["optimizer.pt"]), weights_only=True)
    optimizer["state"][0]["exp_avg"] = torch.zeros(1)
    buffer = io.BytesIO()
    torch.save(optimizer, buffer)
    entries["optimizer.pt"] = buffer.getvalue()
    metadata = json.loads(entries["job.json"])
    metadata["digests"]["optimizer.pt"] = hashlib.sha256(entries["optimizer.pt"]).hexdigest()
    entries["job.json"] = json.dumps(metadata).encode()
    with zipfile.ZipFile(bad, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    output = tmp_path / "rejected.distill-job"
    with pytest.raises(DistillationContractError, match="Adam canonical moment"):
        migrate(bad, output, source)
    assert not output.exists()
    monkeypatch.setattr(
        "pyrl_trainer.distillation_migrate.trainer_source_sha256", lambda: "ff" * 32
    )
    with pytest.raises(DistillationContractError, match="changed after process startup"):
        migrate(old, output, source)
    assert not output.exists()


def test_unapproved_contract_revision_cannot_publish_migration(tmp_path, monkeypatch):
    """A test approval fixture never weakens production contract admission."""
    monkeypatch.setattr("pyrl_trainer.distillation_migrate.IDENTITY", "ff" * 32)
    output = tmp_path / "must-not-exist.distill-job"
    with pytest.raises(DistillationContractError, match="target contracts are not approved"):
        migrate(tmp_path / "not-read.distill-job", output, tmp_path / "not-read-source")
    assert not output.exists()
