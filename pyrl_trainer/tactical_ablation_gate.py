"""Train paired canonical feature ablations and deploy their artifacts in native tactics.

This synthetic study measures agreement and actual student gameplay. It does not
select sensor defaults or qualify representative population performance/soak.
"""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import torch

from .brain_artifact import encode_brain_artifact
from .canonical_graph import (CanonicalGraph, CanonicalGraphManifest,
                              configure_distillation_precision)
from .canonical_parity import _compare, _sequence
from .distillation_job import (SOURCE_FILES, DistillationJobConfig, admit_job_device,
                               available_trainer_cpus, estimate_training_device_bytes,
                               numerical_environment, trainer_source_sha256)
from .sequence_distillation import CanonicalSequenceTrainer, SequenceTrainingConfig
from .tactical_ablation import (VARIANTS, ablation_graph, evaluate_agreement,
                                paired_windows, window_batch)
from .tactical_corpus import load_tactical_corpus
from .tactical_contract import variants_for_profile


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path):
    _require(path.stat().st_size <= 8 * 1024**2, "bounded native report required")
    with path.open("rb") as stream:
        value = stream.read(8 * 1024**2 + 1)
    _require(len(value) <= 8 * 1024**2, "native report grew beyond admission")
    return json.loads(value)


def _write(path, value):
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        import os  # pylint: disable=import-outside-toplevel
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def validation_updates(values, updates):
    """Admit at most sixteen explicit held-out checkpoints, always closing at the final update."""
    _require(isinstance(updates, int) and not isinstance(updates, bool)
             and 1 <= updates <= 10000, "invalid curve update budget")
    values = () if values is None else values
    _require(isinstance(values, (tuple, list)) and len(values) <= 16
             and all(isinstance(value, int) and not isinstance(value, bool)
                     and 1 <= value <= updates for value in values)
             and len(set(values)) == len(values),
             "validation updates require distinct integers within the optimizer budget")
    if not values:
        return ()
    result = tuple(sorted({*values, updates}))
    _require(len(result) <= 16, "validation curve including final update exceeds sixteen points")
    return result


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "canonical-runner", "tactical-runner", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--validation-update", type=int, action="append",
                        help=("Record held-out agreement after this update; "
                              "maximum 16 including the automatic final point"))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Qualify bounded whole-window CUDA compilation; CPU is rejected")
    for name, default in [("updates", 250), ("seed", 1), ("validation-seed", 2),
                          ("batch-sequences", 32), ("sequence-length", 64), ("burn-in", 32),
                          ("cpu-threads", min(2, available_trainer_cpus())),
                          ("device-memory-mib", 8192), ("native-steps", 180),
                          ("native-workers", 1), ("deadline-seconds", 1800)]:
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.cuda_graphs and args.device == "cpu":
        parser.error("--cuda-graphs requires --device cuda:0")
    bounds = {"updates": (1, 10000), "seed": (0, 0xffffffff), "validation_seed": (1, 64),
              "batch_sequences": (1, 256), "sequence_length": (16, 256), "burn_in": (0, 128),
              "cpu_threads": (1, available_trainer_cpus()), "device_memory_mib": (512, 65536),
              "native_steps": (1, 600), "native_workers": (1, available_trainer_cpus()),
              "deadline_seconds": (60, 7200)}
    for name, (low, high) in bounds.items():
        if not low <= getattr(args, name) <= high:
            parser.error("out of range " + name)
    if args.sequence_length + args.burn_in > 256:
        parser.error("burn-in plus BPTT exceeds 256 deliveries")
    try:
        args.validation_update = validation_updates(args.validation_update, args.updates)
    except ValueError as error:
        parser.error(str(error))
    if args.variant is not None and len(set(args.variant)) != len(args.variant):
        parser.error("duplicate ablation variant")
    for name in ("canonical_runner", "tactical_runner", "corpus"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.directory = args.directory.resolve()
    if args.directory.exists() or not args.directory.parent.is_dir():
        parser.error("exclusive output directory with an existing parent required")
    return args


class AblationStudy:
    """One isolated, finite experiment; every variant receives identical native windows."""

    def __init__(self, args):
        args.validation_update = validation_updates(
            getattr(args, "validation_update", ()), args.updates)
        self.cuda_graphs = getattr(args, "cuda_graphs", False)
        self.args = args
        self.started = time.perf_counter()
        self.corpus = load_tactical_corpus(args.corpus)
        supported = variants_for_profile(self.corpus.sensor.profile)
        args.variant = list(supported) if args.variant is None else args.variant
        _require(args.variant and len(set(args.variant)) == len(args.variant)
                 and all(variant in supported for variant in args.variant),
                 "ablation variants unavailable for the declared corpus profile")
        self.training, self.validation = self.corpus.split_by_seed({args.validation_seed})
        self.config = SequenceTrainingConfig(batch_size=args.batch_sequences,
                                            bptt_steps=args.sequence_length,
                                            burn_in_steps=args.burn_in)
        self.job = DistillationJobConfig(seed=args.seed, device=args.device,
                                        cpu_threads=args.cpu_threads,
                                        device_bytes=args.device_memory_mib * 1024**2)
        self.device = admit_job_device(self.job)
        if self.cuda_graphs:
            _require(self.device.type == "cuda", "CUDA graphs require a CUDA device")
            torch.cuda.set_per_process_memory_fraction(
                self.job.device_bytes / torch.cuda.get_device_properties(self.device).total_memory,
                self.device)
        torch.set_num_threads(args.cpu_threads)
        configure_distillation_precision()
        self.runners = {"canonical": args.canonical_runner, "tactical": args.tactical_runner}
        self.runner_hashes = {name: _digest(path) for name, path in self.runners.items()}
        self.corpus_hash = _digest(args.corpus / "corpus.json")
        self.reference_v1 = None
        args.directory.mkdir()
        self.report = {"schemaVersion": 1, "kind": "native-tactical-ablation-v1",
                       "releaseQualified": False, "measurementComplete": False,
                       "status": "running", "corpusManifestSha256": self.corpus_hash,
                       "corpusNativeSourceSha256": self.corpus.source_sha256,
                       "runnerSha256": self.runner_hashes,
                       "trainerSourceSha256": trainer_source_sha256(),
                       "environment": numerical_environment(self.device),
                       "config": {**asdict(self.config), "seed": args.seed,
                                  "updates": args.updates, "validationSeed": args.validation_seed,
                                  "nativeSteps": args.native_steps,
                                  "nativeWorkers": args.native_workers,
                                  "sensorProfile": self.corpus.sensor.profile,
                                  "sensorBins": self.corpus.sensor.bins,
                                  "validationUpdates": list(args.validation_update),
                                  **({"cudaGraphs": True} if self.cuda_graphs else {})},
                       "scope": (
                           'Synthetic native tactical teacher lives; same data/seeds/windows '
                           'across canonical graph-input ablations. Split/Concat removes inputs '
                           'before the first trainable layer while retaining the declared v4 '
                           'sensor contract, so no sensing savings are claimed. Before/after '
                           'portable artifacts drive real native physics on the held-out seed. '
                           'PyTorch evaluation wall time includes transfer/loop/metrics and is '
                           'descriptive, not a native real-time gate. Representative '
                           'populations, sensor-layout freeze, saturation, contention and soak '
                           'remain open. Separate profile corpora do not imply paired '
                            'cross-profile data.'
                       ),
                       "sources": self._sources(), "variants": []}
        _write(args.directory / "initial-observation.f32",
               self.corpus.observations[self.training[0].row_offset].tobytes())

    def _sources(self):
        """Retain exact measured source bytes, including the existing canonical method owner."""
        destination = self.args.directory / "sources"
        destination.mkdir()
        names = dict.fromkeys((*SOURCE_FILES, "canonical_parity.py", "tactical_corpus.py",
                               "tactical_ablation.py", "tactical_ablation_gate.py",
                               "tactical_contract.py"))
        records = {}
        for name in names:
            value = (Path(__file__).parent / name).read_bytes()
            compressed = gzip.compress(value, mtime=0)
            _write(destination / (name + ".gz"), compressed)
            records[name] = {"sha256": hashlib.sha256(value).hexdigest(),
                             "gzipSha256": hashlib.sha256(compressed).hexdigest()}
        return records

    def _remaining(self):
        seconds = self.args.deadline_seconds - (time.perf_counter() - self.started)
        if seconds <= 0:
            raise TimeoutError("tactical ablation total deadline")
        return min(seconds, 900)

    def _invoke(self, name, arguments):
        path = self.runners[name]
        _require(_digest(path) == self.runner_hashes[name], "native runner changed")
        try:
            subprocess.run([str(path), *map(str, arguments)], check=True,
                           timeout=self._remaining(), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", errors="replace")[-8192:]
            raise RuntimeError(name + " native invocation failed: " + detail) from error
        _require(_digest(path) == self.runner_hashes[name], "native runner changed during study")

    def _initialize(self, root, variant):
        graph = ablation_graph(variant, bins=self.corpus.sensor.bins,
                               profile=self.corpus.sensor.profile)
        request = {"schemaVersion": 1, "graphSpec": graph,
                   "sensorProfile": self.corpus.sensor.profile, "bins": self.corpus.sensor.bins,
                   "controlIntervalSeconds": self.corpus.control_interval_seconds,
                   "lifeStateRule": "reset", "mathBackend": "rust-sse2-v1",
                   "initializationSeed": self.args.seed,
                   "observationsPath": str(self.args.directory / "initial-observation.f32")}
        _write_json(root / "initialize.json", request)
        self._invoke("canonical", [root / "initialize.json", root / "initialized"])
        manifest = CanonicalGraphManifest.from_bytes(
            (root / "initialized/manifest.json").read_bytes())
        receipt = _json(root / "initialized/receipt.json")
        _require(receipt["sourceSha256"] == self.corpus.source_sha256,
                 "native initializer source differs from corpus")
        _require(manifest.sensor_sha256 == self.corpus.sensor.structural_sha256
                 and manifest.control_interval_seconds == self.corpus.control_interval_seconds,
                 "native manifest differs from corpus deployment")
        return manifest, (root / "initialized/weights.f32").read_bytes(), receipt, request

    def _artifact(self, root, variant, model, updates):
        artifact = encode_brain_artifact(model.manifest, model.export_weights(), {
            "schemaVersion": 1, "sourceKind": "distilled" if updates else "manual",
            "details": {"study": "native-tactical-ablation-v1", "variant": variant,
                        "updates": updates, "seed": self.args.seed,
                        "corpusManifestSha256": self.corpus_hash}})
        path = root / ("trained.slither-brain" if updates else "initial.slither-brain")
        _write(path, artifact.data)
        return path, artifact

    def _native(self, root, name, artifact_path, artifact, source):
        report = root / (name + ".json")
        self._invoke("tactical", ["--student-artifact", artifact_path,
                                  "--seed-start", self.args.validation_seed, "--seed-count", 1,
                                  "--steps", self.args.native_steps,
                                  "--workers", self.args.native_workers,
                                  "--cadence", self.corpus.control_interval_seconds,
                                  "--bins", self.corpus.sensor.bins,
                                  "--output", report])
        value = _json(report)
        _require(value["kind"] == "native-student-tactics-v1" and value["sourceSha256"] == source,
                 "wrong native student/source")
        _require(value["candidate"]["artifactSha256"] == artifact.sha256
                 and value["candidate"]["weightsSha256"] == artifact.weights_sha256,
                 "native evaluated different student bytes")
        _require(value["firstSeed"] == self.args.validation_seed and value["seedCount"] == 1
                 and len(value["trials"]) == 14, "native held-out cohort mismatch")
        _require(all(trial["seed"] == self.args.validation_seed
                     and trial["observations"] > 0 and trial["sharedBatchBudgetHeld"]
                     for trial in value["trials"]), "native student work did not complete")
        if self.reference_v1 is None:
            self.reference_v1 = value["referenceV1"]
        _require(value["referenceV1"] == self.reference_v1, "paired physical reference changed")
        return {"report": report.name, "sha256": _digest(report),
                "survivingTrials": sum(trial["alive"] for trial in value["trials"]),
                "survivalSeconds": sum(trial["aliveSeconds"] for trial in value["trials"]),
                "foodEarned": sum(trial["foodEarned"] for trial in value["trials"]),
                "kills": sum(trial["kills"] for trial in value["trials"])}

    def _curve_point(self, model, update, agreement=None):
        """Observe complete held-out lives without advancing model parameters or trainer RNGs."""
        self._remaining()
        weights = hashlib.sha256(model.export_weights()).hexdigest()
        cpu_rng = torch.random.get_rng_state().clone()
        device = model.flat_parameters.device
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        agreement = (evaluate_agreement(model, self.corpus, self.validation)
                     if agreement is None else agreement)
        _require(hashlib.sha256(model.export_weights()).hexdigest() == weights,
                 "held-out evaluation changed packed weights")
        _require(torch.equal(cpu_rng, torch.random.get_rng_state()),
                 "held-out evaluation advanced CPU RNG")
        if cuda_rng is not None:
            _require(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)),
                     "held-out evaluation advanced CUDA RNG")
        self._remaining()
        # Sufficient statistics for both teacher policies keep sixteen checkpoints bounded.
        # Full per-scenario strata remain in the initial/final agreement records.
        point = {"update": update, "weightsSha256": weights, **{key: agreement[key] for key in
                 ("rows", "sequences", "turnMse", "turnMae", "boostDisagreement", "loss",
                  "perPolicy", "torchEvaluationWallMs")}}
        return point, agreement

    def _train(self, model, before=None):
        """Train one paired schedule, optionally observing independent held-out lives."""
        checkpoints = validation_updates(
            getattr(self.args, "validation_update", ()), self.args.updates)
        curve, last_validation = [], None
        if checkpoints:
            initial, _ = self._curve_point(model, 0, before)
            curve.append(initial)
        trainer = CanonicalSequenceTrainer(model, self.config, cuda_graphs=self.cuda_graphs)
        schedule = hashlib.sha256(b"native-tactical-window-schedule-v1\0")
        trace, rows, burn_rows, unknown = [], 0, 0, 0
        windows = paired_windows(self.corpus, self.training, self.config,
                                 self.args.updates, self.args.seed)
        for update, selected in enumerate(windows, 1):
            self._remaining()
            for window in selected:
                schedule.update(struct.pack("<III", window.sequence,
                                            window.training_start, window.burn_in))
            batch, actual = window_batch(self.corpus, model.manifest, selected, self.config)
            trainer.config = actual
            result = trainer.update(batch)
            _require(result.optimizer_updated, "no real canonical optimizer update")
            rows += result.training_samples
            burn_rows += result.burn_in_samples
            unknown += result.unknown_state_starts
            if update == 1 or update % 25 == 0 or update == self.args.updates:
                print("[tactical-ablation] update", update, flush=True)
                trace.append({"update": update, "loss": result.loss,
                              "turnMse": result.turn_mse,
                              "boostDisagreement": result.boost_disagreement})
            if update in checkpoints:
                point, last_validation = self._curve_point(model, update)
                curve.append(point)
        return {"updates": trainer.completed_updates, "trainingRows": rows,
                "burnInRows": burn_rows, "unknownStateStarts": unknown,
                "scheduleSha256": schedule.hexdigest(), "trace": trace,
                "validationCurve": curve,
                **({"windowExecution": trainer.window_execution.statistics()}
                   if self.cuda_graphs else {})}, last_validation

    def _parity(self, root, model, request, source):
        sequence = self.validation[0]
        observations = self.corpus.observations[
            sequence.row_offset:sequence.row_offset + min(sequence.rows, 256)].copy()
        _write(root / "parity-observations.f32", observations.tobytes())
        _write(root / "trained-weights.f32", model.export_weights())
        request = {key: value for key, value in request.items() if key != "initializationSeed"}
        request.update(weightsPath=str(root / "trained-weights.f32"),
                       observationsPath=str(root / "parity-observations.f32"))
        _write_json(root / "parity.json", request)
        self._invoke("canonical", [root / "parity.json", root / "native-parity"])
        receipt = _json(root / "native-parity/receipt.json")
        _require(receipt["sourceSha256"] == source, "native parity source changed")
        actual, states = _sequence(model, torch.from_numpy(observations).to(self.device),
                                   model.initial_state(1))
        expected = torch.from_numpy(np.frombuffer(
            (root / "native-parity/outputs.f32").read_bytes(), dtype="<f4").copy().reshape(-1, 2))
        expected_states = torch.from_numpy(np.frombuffer(
            (root / "native-parity/states.f32").read_bytes(), dtype="<f4").copy().reshape(
                len(observations), model.manifest.total_state_size))
        _compare(actual, expected, self.device.type)
        _compare(states, expected_states, self.device.type)
        return {"steps": len(observations), "outputMaxAbs": (actual - expected).abs().max().item(),
                "stateMaxAbs": (states - expected_states).abs().max().item(),
                "scope": "Frozen canonical parity tolerances, actual held-out delivered prefix"}

    def variant(self, variant):
        """Measure one initialized/trained pair without changing data or native opponents."""
        root = self.args.directory / variant
        root.mkdir()
        manifest, weights, receipt, request = self._initialize(root, variant)
        estimate = estimate_training_device_bytes(manifest, self.config)
        _require(estimate <= self.job.device_bytes,
                 "canonical training estimate exceeds device budget")
        model = CanonicalGraph(manifest, weights, self.device)
        before = evaluate_agreement(model, self.corpus, self.validation)
        initial_path, initial = self._artifact(root, variant, model, 0)
        native_before = self._native(
            root, "native-before", initial_path, initial, receipt["sourceSha256"]
        )
        training, after = self._train(model, before)
        if after is None:
            after = evaluate_agreement(model, self.corpus, self.validation)
        parity = self._parity(root, model, request, receipt["sourceSha256"])
        trained_path, trained = self._artifact(root, variant, model, self.args.updates)
        _require(trained.weights_sha256 != initial.weights_sha256,
                 "canonical optimizer did not change packed weights")
        native_after = self._native(
            root, "native-after", trained_path, trained, receipt["sourceSha256"]
        )
        return {"variant": variant, "parameters": manifest.total_parameters,
                "layoutSha256": manifest.layout_sha256,
                "nativeSourceSha256": receipt["sourceSha256"],
                "deviceEstimateBytes": estimate, "initialWeightsSha256": initial.weights_sha256,
                "trainedWeightsSha256": trained.weights_sha256,
                "before": before, "after": after, "training": training, "parity": parity,
                "nativeBefore": native_before, "nativeAfter": native_after}

    def run(self):
        """Keep negative quality outcomes as measurements and retain functional failures."""
        try:
            reference = None
            for variant in self.args.variant:
                print("[tactical-ablation]", variant, flush=True)
                result = self.variant(variant)
                identity = tuple(result["training"][key] for key in
                                 ("scheduleSha256", "trainingRows", "burnInRows"))
                reference = identity if reference is None else reference
                _require(reference == identity, "ablation variants trained on different windows")
                self.report["variants"].append(result)
            _require(_digest(self.args.corpus / "corpus.json") == self.corpus_hash,
                     "corpus metadata changed during study")
            for name, receipt in self.report["sources"].items():
                _require(_digest(Path(__file__).parent / name) == receipt["sha256"],
                         "trainer source changed during study")
            self.report.update(status="measured", measurementComplete=True)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.report.update(status="failed", failure=type(error).__name__ + ": " + str(error))
            raise
        finally:
            self.report["elapsedMs"] = (time.perf_counter() - self.started) * 1000
            _write_json(self.args.directory / "summary.json", self.report)


def main():
    """Run only explicit immutable native tools and one new local experiment directory."""
    AblationStudy(_arguments()).run()


if __name__ == "__main__":
    main()
