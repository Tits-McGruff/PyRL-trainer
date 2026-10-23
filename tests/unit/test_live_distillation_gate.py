"""Sustained qualification must prove both optimizer work and fresh live intake."""
from types import SimpleNamespace
import json

import pytest

from test_distillation_job import HELPERS

from pyrl_trainer import live_distillation_gate as gate
from pyrl_trainer.distillation_client import DistillationFetch
from pyrl_trainer.distillation_checkpoint import load_job_checkpoint
from pyrl_trainer.distillation_job import DistillationJob, DistillationJobConfig
from pyrl_trainer.canonical_graph import CanonicalGraphManifest

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
        replay=SimpleNamespace(owned_bytes=4096, max_bytes=256 * 1024**2),
        pending_candidates=[], dropped_candidates=0)
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


@pytest.mark.parametrize("mib", [64, 256, 2048])
def test_restored_live_replay_keeps_requested_capacity(sustained, mib):
    """Use the existing owner under each admitted ceiling; capacity is separate from occupancy."""
    args, job = sustained
    args.replay_mib = mib
    job.replay.max_bytes = mib * 1024**2
    returned, report = gate._sustain_phase(args, 1000)  # pylint: disable=protected-access
    assert returned is job
    assert job.replay.max_bytes == mib * 1024**2
    assert report["peakReplayBytes"] < job.replay.max_bytes
    assert report["optimizerUpdates"] > 0 and report["fetches"] > 0


def test_live_replay_capacity_mismatch_rejects_before_training(sustained):
    """An explicit smaller or larger restart request cannot resize persisted replay implicitly."""
    args, job = sustained
    before = job.trainer.completed_updates
    for mib in (64, 2048):
        args.replay_mib = mib
        with pytest.raises(RuntimeError, match="replay capacity mismatch"):
            gate._sustain_phase(args, 1000)  # pylint: disable=protected-access
        assert job.trainer.completed_updates == before
        assert job.client.cursor == 10


@pytest.mark.parametrize("mib", [True, 63, 2049, 256.5, "256"])
def test_live_replay_capacity_rejects_values_outside_public_envelope(sustained, mib):
    """Qualification cannot bypass ordinary typed public replay admission."""
    args, _job = sustained
    args.replay_mib = mib
    with pytest.raises(ValueError, match="64..2048"):
        gate._replay_capacity_bytes(args)  # pylint: disable=protected-access


@pytest.mark.parametrize("minimum,seconds", [
    (True, None), (-1, None), (256, None), (1.5, None), ("1", None),
    (0, 1800), (1, True), (1, 0), (1, 10801), (1, 2.5),
])
def test_replay_fill_rejects_unbounded_or_unreachable_admission(minimum, seconds):
    """Whole-chunk occupancy must stay below capacity and explicit setup stays bounded."""
    args = SimpleNamespace(replay_min_mib=minimum, replay_fill_seconds=seconds)
    with pytest.raises(ValueError, match="replay"):
        gate._replay_fill_admission(args)  # pylint: disable=protected-access


def test_ready_replay_requires_actual_decoded_ownership_after_both_streams_are_ready(monkeypatch):
    """Real frozen codec chunks cross the requested byte floor only after the second intake."""
    source = HELPERS.replay_fixture(chunks_per_stream=40)
    chunks = source.chunks
    first = DistillationFetch(source.manifest, 0, chunks[-2].sequence, chunks[:-1], ())
    second = DistillationFetch(source.manifest, chunks[-2].sequence,
                               chunks[-1].sequence, chunks[-1:], ())
    proposals = iter((first, second))
    committed = []
    client = SimpleNamespace(fetch=lambda: next(proposals), commit=committed.append)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)
    minimum = source.owned_bytes - 1
    # pylint: disable=protected-access
    replay = gate._ready_replay(client, gate.time.monotonic() + 10, 64 * 1024**2, minimum)
    assert committed == [first, second]
    assert replay.retained_chunks == len(chunks)
    assert replay.owned_bytes == source.owned_bytes >= minimum
    assert replay.chunks == chunks


@pytest.mark.parametrize("fault", ["occupancy", "windows", "late"])
def test_ready_replay_keeps_occupancy_windows_and_absolute_deadline(monkeypatch, fault):
    """Missing bytes, missing complete windows and a fetch that returns late all fail."""
    source = HELPERS.replay_fixture(chunks_per_stream=40 if fault != "windows" else 1)
    clock = [0.0]
    def now():
        clock[0] += 0.25
        return clock[0]
    monkeypatch.setattr(gate.time, "monotonic", now)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)
    cursor = [0]
    def fetch():
        start = cursor[0]
        cursor[0] = source.cursor
        if fault == "late":
            clock[0] += 3
        return DistillationFetch(source.manifest, start, source.cursor,
                                 source.chunks if not start else (), ())
    client = SimpleNamespace(fetch=fetch, commit=lambda _proposal: None)
    minimum = source.owned_bytes + 1 if fault == "occupancy" else 0
    with pytest.raises(TimeoutError, match="requested replay bytes"):
        gate._ready_replay(client, 2, 64 * 1024**2, minimum)  # pylint: disable=protected-access


@pytest.mark.parametrize("minimum", [0, 1])
def test_explicit_fill_preserves_optimizer_work_allowance(monkeypatch, minimum):
    """Explicit setup adds its duration; default qualification retains the old deadline."""
    replay = HELPERS.replay_fixture(chunks_per_stream=40)
    calls = []
    def ready(client, deadline, capacity, target):
        calls.append((client, deadline, capacity, target))
        return replay
    monkeypatch.setattr(gate, "_ready_replay", ready)
    times = iter((100.0, 135.0))
    monkeypatch.setattr(gate.time, "monotonic", lambda: next(times))
    client = SimpleNamespace(capture_epoch="live-epoch")
    result, deadline, receipt = gate._fill_live_replay(  # pylint: disable=protected-access
        client, SimpleNamespace(replay_min_mib=minimum), 600)
    assert result is replay
    assert deadline == (635 if minimum else 600)
    assert calls == [(client, 1900 if minimum else 600, 256 * 1024**2, minimum * 1024**2)]
    assert receipt["elapsedSeconds"] == 35
    assert receipt["captureEpoch"] == "live-epoch"
    assert receipt["ownedBytes"] == replay.owned_bytes
    assert receipt["dataBytes"] == sum(chunk.byte_cost for chunk in replay.chunks)
    assert receipt["encodedBytes"] == sum(len(chunk.raw_bytes) for chunk in replay.chunks)


def test_restart_occupancy_must_reach_requested_floor_before_any_optimizer_work(sustained):
    """A matching capacity alone cannot admit an under-filled checkpoint."""
    args, job = sustained
    args.replay_min_mib = 1
    before = job.trainer.completed_updates
    with pytest.raises(RuntimeError, match="occupancy"):
        gate._sustain_phase(args, 1000)  # pylint: disable=protected-access
    assert job.trainer.completed_updates == before and job.client.cursor == 10


def test_sustained_occupancy_cannot_dip_below_floor_after_live_eviction(sustained):
    """A healthy cursor and optimizer cannot conceal later replay loss."""
    args, job = sustained
    args.replay_min_mib = 1
    job.replay.owned_bytes = 1024**2
    ingest = job.ingest
    def shrink(proposal):
        ingest(proposal)
        job.replay.owned_bytes = 1024**2 - 1
    job.ingest = shrink
    with pytest.raises(RuntimeError, match="occupancy"):
        gate._sustain_phase(args, 1000)  # pylint: disable=protected-access

def test_loaded_real_replay_survives_durable_save_and_exact_cpu_next_update(tmp_path):
    """A small real decoded owner exercises the same occupancy/restart path as the large gate."""
    replay = HELPERS.replay_fixture(chunks_per_stream=40)
    job = DistillationJob(
        CanonicalGraphManifest.read(HELPERS.GRAPH / "manifest.json"),
        (HELPERS.GRAPH / "weights.f32").read_bytes(), replay,
        config=DistillationJobConfig(device="cpu"), sequence=HELPERS.config())
    args = SimpleNamespace(replay_min_mib=1)
    gate._require_replay_capacity(job, args)  # pylint: disable=protected-access
    assert job.update().optimizer_updated
    saved = job.save(tmp_path)
    restored = load_job_checkpoint(saved)
    gate._require_replay_capacity(restored, args)  # pylint: disable=protected-access
    assert restored.replay.owned_bytes == job.replay.owned_bytes >= 1024**2
    expected = gate._witness(job, job.update())  # pylint: disable=protected-access
    actual = gate._witness(restored, restored.update())  # pylint: disable=protected-access
    assert actual == expected

@pytest.mark.parametrize("requested,recorded", [(1, 0), (0, 1)])
def test_resume_cannot_change_the_recorded_occupancy_request(sustained, requested, recorded):
    """Lowering a CLI floor cannot relabel an earlier qualification checkpoint."""
    args, job = sustained
    args.replay_min_mib = requested
    job.replay.owned_bytes = 1024**2
    (args.directory / "restart.json").write_text(json.dumps({
        "checkpoint": "qualified.job", "replayFill": {"minimumOwnedBytes": recorded * 1024**2}}))
    before = job.trainer.completed_updates
    with pytest.raises(RuntimeError, match="occupancy request mismatch"):
        gate._resume_phase(args, 1000)  # pylint: disable=protected-access
    assert job.trainer.completed_updates == before
