"""Protect committed study results across failures without replaying completed training."""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyrl_trainer.ablation_progress import replace_report, study_lease, variant_inventory
from pyrl_trainer.tactical_ablation_gate import AblationStudy

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parents[1] / "fixtures/tactical_corpus_v1"


def arguments(root):
    """Bind a real native corpus and finite CPU settings; runners are never executed here."""
    corpus = root / "corpus"
    corpus.mkdir()
    for name in ("corpus.json", "observations.f32", "actions.f32", "steps.u32"):
        (corpus / name).write_bytes(gzip.decompress((FIXTURE / (name + ".gz")).read_bytes()))
    for name in ("canonical", "tactical"):
        (root / name).write_bytes(name.encode())
    return SimpleNamespace(
        updates=2, validation_update=(), cuda_graphs=False, corpus=corpus,
        variant=["full", "no-heading"], validation_seed=2, batch_sequences=2,
        sequence_length=16, burn_in=0, seed=1, device="cpu", cpu_threads=1,
        device_memory_mib=512, canonical_runner=root / "canonical",
        tactical_runner=root / "tactical", directory=root / "study",
        native_steps=12, native_workers=1, deadline_seconds=60, resume=False, allow_dirty=True,
    )


def completed_variant(study, name):
    """Supply deterministic output files at the expensive native/training test boundary."""
    root = study.args.directory / name
    root.mkdir()
    (root / "weights.f32").write_bytes(name.encode())
    value = {"variant": name, "training": {
        "scheduleSha256": "same actual schedule", "trainingRows": 64, "burnInRows": 0}}
    for phase, key in (("before", "nativeBefore"), ("after", "nativeAfter")):
        path = root / ("native-" + phase + ".json")
        path.write_text(json.dumps({"referenceV1": {"food": 7}}), encoding="utf8")
        value[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return value


def interrupted(root):
    """Commit one variant, then retain an explicitly failed partial second variant."""
    args = arguments(root)
    study = AblationStudy(args)

    def run(name):
        if name == "full":
            return completed_variant(study, name)
        (args.directory / name).mkdir()
        (args.directory / name / "partial.f32").write_bytes(b"interrupted optimizer")
        raise RuntimeError("injected process interruption")

    study.variant = run
    with pytest.raises(RuntimeError, match="injected"):
        study.run()
    args.resume = True
    return args, json.loads((args.directory / "progress.json").read_bytes())


def test_resume_reuses_only_verified_completed_variants_and_retains_partial_outputs(tmp_path):
    """A new owner skips completed work and restarts the interrupted variant from initialization."""
    args, prior = interrupted(tmp_path)
    resumed = AblationStudy(args)
    resumed.variant = Mock(side_effect=lambda name: completed_variant(resumed, name))
    resumed.run()
    resumed.variant.assert_called_once_with("no-heading")
    report = json.loads((args.directory / "summary.json").read_bytes())
    assert report["measurementComplete"] is True
    assert report["releaseQualified"] is False
    assert report["variants"][0] == prior["variants"][0]
    assert report["completedFiles"]["full"] == prior["completedFiles"]["full"]
    assert report["attempt"] == 2
    assert report["attemptHistory"][0]["status"] == "failed"
    assert (args.directory / "interrupted-2-no-heading/partial.f32").read_bytes() == (
        b"interrupted optimizer")


@pytest.mark.parametrize("change", ["changed", "missing", "extra"])
def test_changed_completed_files_reject_before_checkpoint_or_training(tmp_path, change):
    """No partial or changed receipt may silently cause reuse or fallback retraining."""
    args, _ = interrupted(tmp_path)
    checkpoint = (args.directory / "progress.json").read_bytes()
    weights = args.directory / "full/weights.f32"
    if change == "changed":
        weights.write_bytes(b"other weights")
    elif change == "missing":
        weights.unlink()
    else:
        (weights.parent / "unexpected.f32").write_bytes(b"extra")
    with pytest.raises(ValueError, match="variant files changed"):
        AblationStudy(args)
    assert (args.directory / "progress.json").read_bytes() == checkpoint


@pytest.mark.parametrize("change", ["config", "runner", "source", "environment", "order"])
def test_resume_rejects_different_experiment_identity(tmp_path, change):
    """Changed experiment identities cannot inherit completed results."""
    args, prior = interrupted(tmp_path)
    if change == "config":
        args.seed += 1
    elif change == "runner":
        args.canonical_runner.write_bytes(b"new native build")
    elif change == "source":
        prior["sources"]["tactical_ablation.py"]["sha256"] = "0" * 64
    elif change == "environment":
        prior["environment"]["torch"] = "different build"
    else:
        args.variant.reverse()
    (args.directory / "progress.json").write_text(json.dumps(prior), encoding="utf8")
    checkpoint = (args.directory / "progress.json").read_bytes()
    with pytest.raises(ValueError, match="resume .*changed"):
        AblationStudy(args)
    assert (args.directory / "progress.json").read_bytes() == checkpoint


def test_unpaired_variant_cannot_be_committed_as_completed_progress(tmp_path):
    """A failed same-window assertion remains mandatory before checkpoint publication."""
    args = arguments(tmp_path)
    study = AblationStudy(args)

    def run(name):
        value = completed_variant(study, name)
        if name == "no-heading":
            value["training"]["scheduleSha256"] = "different"
        return value

    study.variant = run
    with pytest.raises(ValueError, match="different windows"):
        study.run()
    saved = json.loads((args.directory / "progress.json").read_bytes())
    assert [value["variant"] for value in saved["variants"]] == ["full"]
    assert set(saved["completedFiles"]) == {"full"}
    assert saved["measurementComplete"] is False


def test_atomic_checkpoint_failure_preserves_previous_receipt(tmp_path, monkeypatch):
    """An interrupted replacement cannot truncate the last verified checkpoint."""
    path = tmp_path / "progress.json"
    replace_report(path, {"previous": True})
    original = path.read_bytes()
    monkeypatch.setattr("pyrl_trainer.ablation_progress.os.replace",
                        Mock(side_effect=OSError("injected publication failure")))
    with pytest.raises(OSError, match="injected"):
        replace_report(path, {"previous": False})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_inventory_rejects_unbounded_outputs(tmp_path):
    """Receipt construction prices file count before accepting an oversized experiment tree."""
    for index in range(65):
        (tmp_path / (str(index) + ".f32")).write_bytes(b"x")
    with pytest.raises(ValueError, match="admission exceeded"):
        variant_inventory(tmp_path)


def test_live_study_lease_rejects_second_owner_without_mutating_checkpoint(tmp_path):
    """A simultaneous resume cannot rename active work or overwrite another owner's progress."""
    args, _ = interrupted(tmp_path)
    resumed = AblationStudy(args)
    checkpoint = (args.directory / "progress.json").read_bytes()
    with study_lease(args.directory):
        with pytest.raises(ValueError, match="already owned"):
            resumed.run()
    assert (args.directory / "progress.json").read_bytes() == checkpoint
    assert (args.directory / "no-heading/partial.f32").is_file()


def test_stale_resume_owner_rejects_progress_published_after_admission(tmp_path):
    """An owner admitted before another successful continuation must revalidate fresh state."""
    args, prior = interrupted(tmp_path)
    resumed = AblationStudy(args)
    prior["elapsedMs"] += 1
    (args.directory / "progress.json").write_text(json.dumps(prior), encoding="utf8")
    with pytest.raises(ValueError, match="changed since resume admission"):
        resumed.run()


def test_output_admission_failure_preserves_the_previous_completed_prefix(tmp_path):
    """An oversized later variant cannot corrupt the reusable prefix during receipt creation."""
    args = arguments(tmp_path)
    study = AblationStudy(args)

    def run(name):
        value = completed_variant(study, name)
        if name == "no-heading":
            for index in range(65):
                (args.directory / name / (str(index) + ".f32")).write_bytes(b"x")
        return value

    study.variant = run
    with pytest.raises(ValueError, match="admission exceeded"):
        study.run()
    saved = json.loads((args.directory / "progress.json").read_bytes())
    assert [value["variant"] for value in saved["variants"]] == ["full"]
    assert set(saved["completedFiles"]) == {"full"}


def test_completed_study_is_immutable_even_with_explicit_resume(tmp_path):
    """Published measurement receipts cannot be relabelled by another completed attempt."""
    args = arguments(tmp_path)
    study = AblationStudy(args)
    study.variant = lambda name: completed_variant(study, name)
    study.run()
    summary = (args.directory / "summary.json").read_bytes()
    progress = (args.directory / "progress.json").read_bytes()
    args.resume = True
    with pytest.raises(ValueError, match="completed studies are immutable"):
        AblationStudy(args)
    assert (args.directory / "summary.json").read_bytes() == summary
    assert (args.directory / "progress.json").read_bytes() == progress
