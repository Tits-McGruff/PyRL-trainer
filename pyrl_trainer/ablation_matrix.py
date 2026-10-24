"""Verify and summarize a balanced grid of actual native tactical ablation studies.

This reader never trains, simulates, chooses a profile or declares model maturity.
Paired differences keep one initialization/held-out-fold cell as the analysis unit.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import statistics

from .source_provenance import measured_sources
from .tactical_contract import (MAX_ABLATION_UPDATES, SCENARIOS, variants_for_profile,
                                policies_for_profile)

MAX_CELLS = 64
MAX_SUMMARY_BYTES = 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low, high):
    _require(isinstance(value, int) and not isinstance(value, bool) and low <= value <= high,
             "bounded integer required")
    return value


def _number(value):
    _require(isinstance(value, (float, int)) and not isinstance(value, bool)
             and math.isfinite(value) and value >= 0,
             "finite nonnegative measurement required")
    return value


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(char in "0123456789abcdef" for char in value), "SHA-256 identity required")
    return value


def _digest(path, maximum=64 * MAX_SUMMARY_BYTES):
    digest = hashlib.sha256()
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
             "bounded ordinary evidence file required")
    with path.open("rb") as stream:
        owned = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            owned += len(chunk)
            _require(owned <= maximum, "evidence grew beyond digest admission")
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, maximum):
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
             "bounded ordinary JSON evidence required")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    _require(len(data) <= maximum, "evidence grew beyond admission")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate evidence JSON field")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique), hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class MatrixSpec:
    """Explicit initialization identities and complete held-out geometry folds."""
    seeds: tuple
    validation_seeds: tuple
    updates: int

    def __post_init__(self):
        _integer(self.updates, 1, MAX_ABLATION_UPDATES)
        for values, maximum in [(self.seeds, 0xffffffff), (self.validation_seeds, 64)]:
            _require(1 <= len(values) <= 8 and len(set(values)) == len(values),
                     "one to eight distinct seed identities required")
            for value in values:
                _integer(value, 0 if maximum == 0xffffffff else 1, maximum)
        _require(len(self.cells) <= MAX_CELLS, "matrix cell admission")

    @property
    def cells(self):
        """Stable analysis order; geometry folds remain separate from initializer seeds."""
        return tuple((seed, fold) for seed in sorted(self.seeds)
                     for fold in sorted(self.validation_seeds))


def _study_sensor(config):
    """Keep legacy Motion/B16 readable; explicit identities require both profile and width."""
    declared = ("sensorProfile" in config, "sensorBins" in config)
    _require(declared[0] == declared[1], "complete study sensor identity required")
    profile, bins = config.get("sensorProfile", "motion"), config.get("sensorBins", 16)
    policies_for_profile(profile)
    _integer(bins, 8, 32)
    return profile, bins



def _study_variants(records, profile):
    """Require the complete declared profile comparison, including unique graph labels."""
    variants = {value["variant"]: value for value in records}
    _require(len(variants) == len(records)
             and set(variants) == set(variants_for_profile(profile)),
             ("every cell requires the same seven distinct ablations" if profile == "motion"
              else "Basic cells require exactly the Full variant"))
    return variants


def read_study(directory):
    """Bind actual native reports, portable bytes and measured source provenance to one summary."""
    root = Path(directory).resolve(strict=True)
    report, digest = _read_json(root / "summary.json", MAX_SUMMARY_BYTES)
    _require(report.get("kind") == "native-tactical-ablation-v1"
             and report.get("schemaVersion") == 1, "unsupported ablation study")
    _require(report.get("measurementComplete") is True and report.get("status") == "measured"
             and report.get("releaseQualified") is False, "complete diagnostic study required")
    if "resumeSchemaVersion" in report:
        from .ablation_progress import variant_inventory  # pylint: disable=import-outside-toplevel
        _integer(report["resumeSchemaVersion"], 1, 1)
        completed = [value["variant"] for value in report["variants"]]
        _require(completed == report["variantOrder"]
                 and len(set(completed)) == len(completed)
                 and set(report["completedFiles"]) == set(completed),
                 "incomplete study checkpoint inventory")
        for name in completed:
            _require(name in variants_for_profile(_study_sensor(report["config"])[0]),
                     "unknown completed variant")
            _require(variant_inventory(root / name) == report["completedFiles"][name],
                     "completed study files changed")
    profile, bins = _study_sensor(report["config"])
    names = report["sources"]
    _require(1 <= len(names) <= 64 and {"distillation_job.py", "canonical_graph.py",
             "sequence_distillation.py", "tactical_ablation.py", "tactical_ablation_gate.py",
             "tactical_contract.py", "tactical_corpus.py"} <= names.keys(),
             "complete measured sources required")
    for name in names:
        _require(Path(name).name == name and name.endswith(".py"),
                 "confined source basename required")
    source_bytes = measured_sources(report, root)
    _require(all(len(value) <= MAX_SUMMARY_BYTES for value in source_bytes.values()),
             "bounded measured source required")
    declarations = [node.value for node in ast.parse(source_bytes["distillation_job.py"]).body
                    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == "SOURCE_FILES" for target in node.targets)]
    _require(len(declarations) == 1, "measured trainer source inventory missing")
    inventory = ast.literal_eval(declarations[0])
    _require(isinstance(inventory, tuple) and len(set(inventory)) == len(inventory),
             "invalid measured trainer source inventory")
    trainer_hash = hashlib.sha256(b"slither.distillation-trainer-source.v1\0")
    for name in inventory:
        _require(name in source_bytes, "trainer source omitted from measured provenance")
        trainer_hash.update(name.encode() + b"\0" + source_bytes[name])
    _require(trainer_hash.hexdigest() == _sha(report["trainerSourceSha256"]),
             "trainer identity differs from measured source provenance")
    reference = None
    for variant in report["variants"]:
        name = variant["variant"]
        _require(name in variants_for_profile(profile), "unknown graph-input ablation")
        for phase, artifact_name in [("Before", "initial.slither-brain"),
                                     ("After", "trained.slither-brain")]:
            receipt = variant["native" + phase]
            filename = "native-" + phase.lower() + ".json"
            _require(receipt["report"] == filename, "confined native report required")
            path = root / name / filename
            native, native_digest = _read_json(path, 8 * MAX_SUMMARY_BYTES)
            _require(native_digest == _sha(receipt["sha256"]), "native report changed")
            _require(native["kind"] == "native-student-tactics-v1"
                     and native["sourceSha256"] == report["corpusNativeSourceSha256"],
                     "native student/source mismatch")
            _require(native["firstSeed"] == report["config"]["validationSeed"]
                     and native["seedCount"] == 1 and len(native["trials"]) == len(SCENARIOS),
                     "native held-out geometry fold mismatch")
            _require({trial["scenario"] for trial in native["trials"]} == set(SCENARIOS)
                     and all(trial["seed"] == native["firstSeed"]
                             and trial["observations"] > 0 and trial["sharedBatchBudgetHeld"]
                             for trial in native["trials"]), "incomplete native trials")
            _require(_digest(root / name / artifact_name) == native["candidate"]["artifactSha256"],
                     "native evaluated different portable bytes")
            weight_key = "initialWeightsSha256" if phase == "Before" else "trainedWeightsSha256"
            weights = variant[weight_key]
            _require(native["candidate"]["weightsSha256"] == _sha(weights),
                     "native weight identity mismatch")
            _require(native["candidate"]["parameterCount"] == variant["parameters"]
                     and native["candidate"]["layoutSha256"] == variant["layoutSha256"],
                     "native packing/parameter count differs from summary")
            if "sensorProfile" in report["config"]:
                _require(native["candidate"]["inputSize"] == 24 + (
                    6 if profile == "basic" else 15) * bins,
                         "native candidate sensor width differs from study")
            measured = {"survivingTrials": sum(trial["alive"] for trial in native["trials"]),
                        "survivalSeconds": sum(trial["aliveSeconds"] for trial in native["trials"]),
                        "foodEarned": sum(trial["foodEarned"] for trial in native["trials"]),
                        "kills": sum(trial["kills"] for trial in native["trials"])}
            _require(all(receipt[key] == value for key, value in measured.items()),
                     "summary does not match native outcomes")
            reference = native["referenceV1"] if reference is None else reference
            _require(native["referenceV1"] == reference, "paired physical reference changed")
    return report, digest


def _policy_metrics(evaluation, compact=False, profile="motion"):
    groups = evaluation["perPolicy"]
    policies = policies_for_profile(profile)
    _require(set(groups) == set(policies), "both teacher policy strata required")
    if not compact:
        _require(set(evaluation["perScenarioPolicy"]) == {
            scenario + "/" + policy for scenario in SCENARIOS for policy in policies},
            "complete scenario/policy strata required")
    result = {}
    for policy, group in groups.items():
        rows = _integer(group["rows"], 1, 1_000_000)
        disagreements = _integer(group["boostDisagreements"], 0, rows)
        result[policy] = {"turnMae": _number(group["turnAbsoluteError"]) / rows,
                          "turnMse": _number(group["turnSquaredError"]) / rows,
                          "loss": _number(group["lossSum"]) / rows,
                          "boostDisagreement": disagreements / rows}
    _require(sum(group["rows"] for group in groups.values()) == evaluation["rows"],
             "policy rows do not partition held-out data")
    if not compact:
        for policy, group in groups.items():
            selected = [evaluation["perScenarioPolicy"][scenario + "/" + policy]
                        for scenario in SCENARIOS]
            for value in selected:
                size = _integer(value["rows"], 1, 1_000_000)
                _integer(value["boostDisagreements"], 0, size)
            for key in ("rows", "turnAbsoluteError", "turnSquaredError",
                        "boostDisagreements", "lossSum"):
                total = sum(_number(value[key]) for value in selected)
                _require(math.isclose(total, _number(group[key]), rel_tol=1e-9, abs_tol=1e-9),
                         "scenario/policy measurements do not reconcile")
    rows = _integer(evaluation["rows"], 1, 1_000_000)
    absolute = sum(_number(group["turnAbsoluteError"]) for group in groups.values()) / rows
    disagreements = sum(group["boostDisagreements"] for group in groups.values()) / rows
    _require(math.isclose(absolute, _number(evaluation["turnMae"]), rel_tol=1e-9, abs_tol=1e-9)
             and math.isclose(disagreements, _number(evaluation["boostDisagreement"]),
                              rel_tol=1e-9, abs_tol=1e-9), "held-out aggregate/strata mismatch")
    for field, numerator in [("turnMse", "turnSquaredError"), ("loss", "lossSum")]:
        mean = sum(_number(group[numerator]) for group in groups.values()) / rows
        _require(math.isclose(mean, _number(evaluation[field]), rel_tol=1e-9, abs_tol=1e-9),
                 "held-out aggregate/strata mismatch")
    return result


def _measurements(variant, profile="motion"):
    result = {key: _number(variant["after"][key])
              for key in ("turnMae", "turnMse", "loss", "boostDisagreement")}
    for key in ("survivingTrials", "survivalSeconds", "foodEarned", "kills"):
        result[key] = _number(variant["nativeAfter"][key])
    for policy, values in _policy_metrics(variant["after"], profile=profile).items():
        result.update({policy + "/" + key: value for key, value in values.items()})
    return result


def _curve_updates(config):
    """Legacy final-only studies remain readable; enabled curves close at the full budget."""
    updates = config.get("validationUpdates", [])
    _require(isinstance(updates, list) and len(updates) <= 16,
             "bounded validation update declaration required")
    for value in updates:
        _integer(value, 1, config["updates"])
    _require(updates == sorted(set(updates)), "canonical distinct validation updates required")
    if updates:
        _require(updates[-1] == config["updates"], "validation curve omits final update")
    return [0, *updates] if updates else []


def _curve_values(point, profile="motion"):
    """Only aggregate and separate policy agreement enters intermediate curve summaries."""
    values = {key: _number(point[key]) for key in
              ("turnMae", "turnMse", "loss", "boostDisagreement")}
    for policy, metrics in _policy_metrics(point, compact=True, profile=profile).items():
        values.update({policy + "/" + key: value for key, value in metrics.items()})
    return values


def _checked_curve(variant, updates, profile="motion"):
    """Bind every compact point to the same held-out exposure and real endpoint weights."""
    points = variant["training"].get("validationCurve", [])
    _require(isinstance(points, list) and len(points) == len(updates),
             "complete declared validation curve required")
    result = {}
    for expected, point in zip(updates, points):
        _require(_integer(point["update"], 0, variant["training"]["updates"]) == expected,
                 "validation curve chronology changed")
        _sha(point["weightsSha256"])
        _number(point["torchEvaluationWallMs"])
        _require(point["rows"] == variant["after"]["rows"]
                 and _integer(point["sequences"], 1, 1_000_000) == variant["after"]["sequences"],
                 "validation curve held-out exposure changed")
        result[expected] = _curve_values(point, profile)
        if expected in (0, variant["training"]["updates"]):
            initial = expected == 0
            _require(point["weightsSha256"] == variant[
                "initialWeightsSha256" if initial else "trainedWeightsSha256"],
                "validation endpoint weights changed")
            evaluation = variant["before" if initial else "after"]
            _require(all(point[key] == evaluation[key] for key in
                         ("rows", "sequences", "turnMae", "turnMse", "loss",
                          "boostDisagreement", "perPolicy")),
                     "validation endpoint metrics changed")
    return result


def _distribution(values):
    return {"count": len(values), "mean": statistics.fmean(values), "minimum": min(values),
            "maximum": max(values), "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "positive": sum(value > 0 for value in values)}


def _summarize_curves(curves, spec, name):
    """Keep paired cell distributions at each measured optimizer checkpoint."""
    points = []
    for update in curves[spec.cells[0], name]:
        absolute, paired = {}, {}
        for cell in spec.cells:
            values = curves[cell, name][update]
            full = curves[cell, "full"][update]
            for key, value in values.items():
                absolute.setdefault(key, []).append(value)
                paired.setdefault(key, []).append(value - full[key])
        points.append({"update": update,
                       "measurements": {key: _distribution(value)
                                        for key, value in absolute.items()},
                       "pairedDifferenceFromFull": {key: _distribution(value)
                                                    for key, value in paired.items()}})
    return points


def combine_studies(studies, spec):
    """Reject missing/unpaired cells and preserve per-cell differences from the Full graph."""
    _require(len(studies) == len(spec.cells), "complete balanced seed/fold matrix required")
    cells, common, initial, parameters, curves = {}, None, {}, {}, {}
    for report, digest in studies:
        _sha(digest)
        _require(report.get("measurementComplete") is True and report.get("status") == "measured"
                 and report.get("releaseQualified") is False,
                 "incomplete study cannot enter matrix")
        config = report["config"]
        profile, bins = _study_sensor(config)
        cell = (_integer(config["seed"], 0, 0xffffffff),
                _integer(config["validationSeed"], 1, 64))
        _require(cell in spec.cells and cell not in cells, "duplicate or foreign seed/fold cell")
        _require(_integer(config["updates"], 1, MAX_ABLATION_UPDATES) == spec.updates,
                 "optimizer budget differs from requested matrix")
        curve_updates = _curve_updates(config)
        fixed_config = {key: value for key, value in config.items()
                        if key not in ("seed", "validationSeed")}
        identity = (fixed_config, report["corpusManifestSha256"],
                    report["corpusNativeSourceSha256"], report["runnerSha256"],
                    report["trainerSourceSha256"], report["sources"], report["environment"])
        common = identity if common is None else common
        _require(identity == common, "matrix source/environment/hyperparameters changed")
        variants = _study_variants(report["variants"], profile)
        schedule = None
        exposure = None
        for name, value in variants.items():
            training = value["training"]
            curves[cell, name] = _checked_curve(value, curve_updates, profile)
            _require(_integer(training["updates"], 1, MAX_ABLATION_UPDATES) == spec.updates
                     and _integer(training["trainingRows"], 1, 1_000_000_000) > 0,
                     "missing actual optimizer work")
            witness = (_sha(training["scheduleSha256"]), training["trainingRows"],
                       _integer(training["burnInRows"], 0, 1_000_000_000))
            schedule = witness if schedule is None else schedule
            _require(witness == schedule, "ablations used different training windows")
            _require(value["nativeSourceSha256"] == report["corpusNativeSourceSha256"],
                     "variant native source changed")
            weights = _sha(value["initialWeightsSha256"])
            _require(_sha(value["trainedWeightsSha256"]) != weights,
                     "optimizer did not change weights")
            key = name, cell[0]
            initial.setdefault(key, weights)
            _require(initial[key] == weights, "same initialization changed across held-out folds")
            shape = _integer(value["parameters"], 1, 1_000_000), _sha(value["layoutSha256"])
            parameters.setdefault(name, shape)
            _require(parameters[name] == shape, "ablation packing changed across cells")
            _policy_metrics(value["before"], profile=profile)
            _measurements(value, profile)
            for phase in ("before", "after"):
                measured = value[phase]
                seen = (measured["rows"], tuple(sorted((key, group["rows"])
                        for key, group in measured["perScenarioPolicy"].items())))
                exposure = seen if exposure is None else exposure
                _require(exposure == seen, "held-out trajectory exposure changed across ablations")
        cells[cell] = variants, digest
    for name in variants_for_profile(profile):
        _require(len({initial[name, seed] for seed in spec.seeds}) == len(spec.seeds),
                 "different initialization seeds reused identical initial parameters")
    results = {}
    for name in variants_for_profile(profile):
        absolute, paired = {}, {}
        for cell in spec.cells:
            values = _measurements(cells[cell][0][name], profile)
            full = _measurements(cells[cell][0]["full"], profile)
            for key, value in values.items():
                absolute.setdefault(key, []).append(value)
                paired.setdefault(key, []).append(value - full[key])
        results[name] = {"parameters": parameters[name][0], "layoutSha256": parameters[name][1],
                         "measurements": {key: _distribution(value)
                                          for key, value in absolute.items()},
                         "pairedDifferenceFromFull": {key: _distribution(value)
                                                      for key, value in paired.items()}}
        if curves[spec.cells[0], name]:
            results[name]["heldOutLearningCurve"] = _summarize_curves(curves, spec, name)
    return {"schemaVersion": 1, "kind": "paired-ablation-matrix-v1", "measurementComplete": True,
            "releaseQualified": False, "maturityQualified": False, "profileChoice": None,
            "replicatedSeedFoldCoverage": len(spec.seeds) >= 3 and len(spec.validation_seeds) >= 3,
            **({"sensorProfile": profile, "sensorBins": bins}
               if "sensorProfile" in config else {}),
            "updatesPerVariant": spec.updates, "initializationSeeds": sorted(spec.seeds),
            "validationSeeds": sorted(spec.validation_seeds), "cellCount": len(cells),
            "corpusManifestSha256": common[1], "corpusNativeSourceSha256": common[2],
            "runnerSha256": common[3], "trainerSourceSha256": common[4],
            "environment": common[6], "variants": results,
            "cells": [{"seed": seed, "validationSeed": fold, "summarySha256": cells[seed, fold][1]}
                      for seed, fold in spec.cells],
            "scope": ("Balanced initialization/held-out geometry folds of "
                      "native synthetic tactics. "
                      "Metrics are macro means of cells and paired differences from Full, with "
                      "separate teacher-policy strata. Cells sharing data/folds are not claimed "
                      "statistically independent; no confidence interval, winner, model-maturity, "
                      "representative-population, sensing-cost, real-time or "
                      "sensor-default conclusion.")}



def main():
    """Read existing completed studies only; never overwrite an input or existing output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--initialization-seed", type=int, action="append", required=True)
    parser.add_argument("--validation-seed", type=int, action="append", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = MatrixSpec(tuple(args.initialization_seed), tuple(args.validation_seed), args.updates)
    _require(len(args.study) == len(spec.cells), "one study per requested cell required")
    output = args.output.resolve()
    roots = [root.resolve(strict=True) for root in args.study]
    _require(len(set(roots)) == len(roots)
             and all(not output.is_relative_to(root) for root in roots),
             "distinct study roots and outside output required")
    report = combine_studies([read_study(root) for root in roots], spec)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    _require(len(payload.encode()) <= MAX_SUMMARY_BYTES, "matrix report byte admission")
    _require(shutil.disk_usage(output.parent).free >= 1024**3 + MAX_SUMMARY_BYTES,
             "matrix report must preserve one GiB disk reserve")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: report[key] for key in
                      ("kind", "cellCount", "replicatedSeedFoldCoverage")}))


if __name__ == "__main__":
    main()
