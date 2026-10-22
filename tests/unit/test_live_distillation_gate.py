"""Sustained qualification must prove both optimizer work and fresh live intake."""
from types import SimpleNamespace
import json

import pytest

from pyrl_trainer import live_distillation_gate as gate

pytestmark = pytest.mark.unit


@pytest.fixture(name="sustained")
def sustained_fixture(monkeypatch, tmp_path):
    """Bound the wall loop while retaining separate intake, optimizer and sample counters."""
    (tmp_path / "sustain-checkpoint.json").write_text(
        '{"checkpoint":"qualified.job"}', encoding="utf-8")
    clock = [0.0]
    def now():
        clock[0] += 0.25
        return clock[0]
    monkeypatch.setattr(gate.time, "monotonic", now)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)
    job = SimpleNamespace(
        client=SimpleNamespace(capture_epoch="capture", cursor=10,
                               producer_storage=SimpleNamespace(max_bytes=4 * 1024**3)),
        trainer=SimpleNamespace(completed_updates=250), training_samples=100,
        replay=SimpleNamespace(owned_bytes=4096), pending_candidates=[], dropped_candidates=0)
    class Device(str):
        """Torch-compatible device label without a model allocation."""
        type = "cpu"
    job.device = Device("cpu")
    job.client.fetch = lambda: "fresh proposal"
    def ingest(proposal):
        assert proposal == "fresh proposal"
        job.client.cursor += 1
        job.replay.owned_bytes += 128
    def update():
        job.trainer.completed_updates += 1
        job.training_samples += 64
        return SimpleNamespace(optimizer_updated=True)
    job.ingest, job.update = ingest, update
    monkeypatch.setattr(gate, "load_job_checkpoint", lambda _path: job)
    args = SimpleNamespace(directory=tmp_path, device="cpu", capture_epoch="capture",
                           spool_gib=4, seconds=60)
    return args, job


def test_sustained_window_requires_fresh_rows_and_actual_updates(sustained):
    """A saved replay alone cannot stand in for ongoing producer/trainer contention."""
    args, job = sustained
    returned, report = gate._sustain_phase(args, 1000)  # pylint: disable=protected-access
    assert returned is job
    assert report["activeWallSeconds"] >= 60
    assert report["initialCursor"] == 10 < report["finalCursor"]
    assert report["fetches"] == report["finalCursor"] - 10
    assert report["initialUpdates"] == 250 < report["updates"]
    assert report["optimizerUpdates"] == report["updates"] - 250
    assert report["trainingSamplesInWindow"] == report["optimizerUpdates"] * 64
    assert report["peakReplayBytes"] > 4096


def test_sustained_window_rejects_stale_replay_only_work(sustained):
    """Optimizer progress cannot hide a disconnected or non-consuming live source."""
    args, job = sustained
    job.ingest = lambda _proposal: None
    with pytest.raises(RuntimeError, match="both optimizer and live intake"):
        gate._sustain_phase(args, 1000)  # pylint: disable=protected-access


def test_sustained_window_rejects_non_updating_output(sustained):
    """Inference or a frozen graph cannot count as trainer contention."""
    args, job = sustained
    job.update = lambda: SimpleNamespace(optimizer_updated=False)
    with pytest.raises(RuntimeError, match="trainable outputs"):
        gate._sustain_phase(args, 1000)  # pylint: disable=protected-access


def test_sustained_window_revalidates_restored_source_and_deadline(sustained):
    """The measurement cannot silently continue another capture or overrun its outer bound."""
    args, job = sustained
    job.client.capture_epoch = "other"
    with pytest.raises(RuntimeError, match="source/device/storage"):
        gate._sustain_phase(args, 1000)  # pylint: disable=protected-access
    job.client.capture_epoch = args.capture_epoch
    with pytest.raises(TimeoutError, match="outer deadline"):
        gate._sustain_phase(args, 1)  # pylint: disable=protected-access


@pytest.mark.parametrize("phase,seconds,wall,soak", [
    ("sustain", 21600, 21900, None), ("sustain", 300, 21900, None),
    ("train", 21600, 21900, 6), ("sustain", 60, 21900, 6),
    ("sustain", 21600, 21901, 6), ("sustain", 21600, 21900, 5),
    ("sustain", 60, 89, None), ("train", 60, float("nan"), None),
])
def test_long_timing_requires_explicit_exact_sustain_contract(phase, seconds, wall, soak):
    """A larger timeout or seconds flag cannot silently enter six-hour mode."""
    with pytest.raises(ValueError):
        gate.qualification_timing(phase, seconds, wall, soak)


def test_ordinary_and_explicit_six_hour_timing_are_separate():
    """Explicit long admission does not relax ordinary commands."""
    gate.qualification_timing("train", 60, 180)
    gate.qualification_timing("sustain", 300, 600)
    gate.qualification_timing("sustain", 21600, 21900, 6)


def test_six_hour_loop_keeps_bounded_live_progress_and_real_snapshot_calls(sustained):
    """A fake clock exercises the full duration without claiming measured six-hour evidence."""
    args, job = sustained
    args.seconds, args.soak_hours = 21600, 6
    saves = []
    def save(directory):
        saves.append(directory)
        return directory / "actual-durable-save.distill-job"
    job.save = save
    returned, report = gate._sustain_phase(args, 21900)  # pylint: disable=protected-access
    assert returned is job and report["sixHourWindow"]
    assert 1 <= report["durableSnapshots"] == len(saves) <= 36
    assert 170 <= len(report["progressWindows"]) <= 362
    previous_time, previous_update, previous_cursor = 0, 250, 10
    for observation in report["progressWindows"]:
        assert 0 < observation["activeWallSeconds"] - previous_time <= 120
        assert observation["updates"] > previous_update and observation["cursor"] > previous_cursor
        previous_time = observation["activeWallSeconds"]
        previous_update, previous_cursor = observation["updates"], observation["cursor"]
    assert (args.directory / "sustain-progress.json").is_file()
    assert not (args.directory / "sustain-progress.tmp").exists()
    assert "actual-durable-save" in (args.directory / "sustain-checkpoint.json").read_text()


@pytest.mark.parametrize("stalled", ["intake", "optimizer"])
def test_six_hour_window_rejects_stalls_after_initial_valid_progress(sustained, stalled):
    """Positive lifetime totals cannot conceal a later loss of one required workload."""
    args, job = sustained
    args.seconds, args.soak_hours = 21600, 6
    ingest, update = job.ingest, job.update
    if stalled == "intake":
        job.ingest = lambda proposal: (ingest(proposal)
                                       if job.trainer.completed_updates < 350 else None)
    else:
        job.update = lambda: update() if job.trainer.completed_updates < 350 else SimpleNamespace(
            optimizer_updated=True)
    with pytest.raises(RuntimeError, match="every bounded interval"):
        gate._sustain_phase(args, 21900)  # pylint: disable=protected-access


def _six_hour_host_wait(args, job):
    """Use the real bounded progress tracker with a fake durable job and clock."""
    args.phase, args.seconds, args.soak_hours = "sustain", 21600, 6
    args.wait_for_native_work = True
    job.save = lambda directory: directory / "actual-durable-save.distill-job"


def test_native_floor_wait_keeps_training_after_minimum_duration(sustained, monkeypatch):
    """A checkpoint hold can delay native work; waiting keeps both real work streams active."""
    args, job = sustained
    _six_hour_host_wait(args, job)
    attempts = []
    receipt = {"schemaVersion": 1, "captureEpoch": "capture",
               "requiredSteps": "1296000", "completedSteps": "1296001"}
    def native_receipt(_path, epoch):
        assert epoch == "capture"
        attempts.append((job.trainer.completed_updates, job.client.cursor))
        return receipt if len(attempts) == 4 else None
    monkeypatch.setattr(gate, "_native_work_receipt", native_receipt)
    _, report = gate._sustain_phase(args, 21900)  # pylint: disable=protected-access
    assert report["activeWallSeconds"] >= 21600
    assert report["nativeWorkCompletion"] == receipt
    assert len(attempts) == 4
    assert all(right[0] > left[0] for left, right in zip(attempts, attempts[1:]))
    assert report["fetches"] > 0 and report["optimizerUpdates"] > 0


def test_missing_native_floor_cannot_extend_existing_outer_deadline(sustained):
    """Missing host completion fails within the original bound, retaining progress limits."""
    args, job = sustained
    _six_hour_host_wait(args, job)
    with pytest.raises(TimeoutError, match="outer deadline"):
        gate._sustain_phase(args, 21900)  # pylint: disable=protected-access


def test_native_receipt_cannot_shorten_six_hour_training(sustained):
    """An early valid native receipt still requires six actual trainer hours."""
    args, job = sustained
    _six_hour_host_wait(args, job)
    update = job.update
    def publish_then_update():
        (args.directory / "native-work-complete.json").write_text(json.dumps({
            "schemaVersion": 1, "captureEpoch": "capture", "requiredSteps": "1296000",
            "completedSteps": "1296000"}), encoding="utf-8")
        job.update = update
        return update()
    job.update = publish_then_update
    _, report = gate._sustain_phase(args, 21900)  # pylint: disable=protected-access
    assert report["activeWallSeconds"] >= 21600
    assert report["nativeWorkCompletion"]["completedSteps"] == "1296000"


@pytest.mark.parametrize("change", [
    {"schemaVersion": True}, {"captureEpoch": "foreign"}, {"requiredSteps": "1"},
    {"completedSteps": "1295999"}, {"completedSteps": "01296000"},
    {"completedSteps": 1296000}, {"completedSteps": "18446744073709551615"},
    {"completedSteps": "１２９６０００"}, {"unexpected": 1},
])
def test_native_receipt_rejects_identity_and_counter_damage(tmp_path, change):
    """The host receipt cannot claim another capture, a partial floor or saturated counter."""
    receipt = {"schemaVersion": 1, "captureEpoch": "capture",
               "requiredSteps": "1296000", "completedSteps": "1296000", **change}
    path = tmp_path / "native-work-complete.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        gate._native_work_receipt(path, "capture")  # pylint: disable=protected-access


def test_native_receipt_file_admission_and_stale_window(sustained):
    """Only a newly published, bounded ordinary file can release this trainer window."""
    args, job = sustained
    _six_hour_host_wait(args, job)
    path = args.directory / "native-work-complete.json"
    assert gate._native_work_receipt(path, "capture") is None  # pylint: disable=protected-access
    path.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError, match="file"):
        gate._native_work_receipt(path, "capture")  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="predate"):
        gate._native_completion_path(args)  # pylint: disable=protected-access
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="file"):
        gate._native_work_receipt(path, "capture")  # pylint: disable=protected-access
    args.seconds = 60
    with pytest.raises(ValueError, match="six-hour"):
        gate._native_completion_path(args)  # pylint: disable=protected-access
