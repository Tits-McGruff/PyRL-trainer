"""Reader-only grid checks; these fixture records are not measured training evidence."""
import copy
import hashlib
import gzip
import json

import pytest

from pyrl_trainer.ablation_matrix import MatrixSpec, combine_studies, read_study
from pyrl_trainer.tactical_contract import POLICIES, SCENARIOS, VARIANTS

pytestmark = pytest.mark.unit


def digest(value):
    """Stable distinct fixture identities, without generating models or samples."""
    return hashlib.sha256(str(value).encode()).hexdigest()


def evaluation(mae):
    """Reconciled raw aggregate/stratum sufficient statistics."""
    group = {"rows": 10, "turnAbsoluteError": mae * 10, "turnSquaredError": mae * mae * 10,
             "boostDisagreements": 1, "lossSum": 2.0}
    policies = {policy: {key: value * len(SCENARIOS) for key, value in group.items()}
                for policy in POLICIES}
    strata = {scenario + "/" + policy: group.copy()
              for scenario in SCENARIOS for policy in POLICIES}
    return {"rows": 280, "turnMae": mae, "turnMse": mae * mae, "loss": 0.2,
            "boostDisagreement": 0.1,
            "perPolicy": policies, "perScenarioPolicy": strata}


def study(seed, fold):
    """A small complete metadata fixture with paired variant windows."""
    variants = []
    for offset, name in enumerate(VARIANTS):
        variants.append({"variant": name, "parameters": 100 + offset,
                         "layoutSha256": digest(name), "nativeSourceSha256": digest("native"),
                         "initialWeightsSha256": digest((seed, name)),
                         "trainedWeightsSha256": digest((seed, fold, name)),
                         "training": {"updates": 250, "trainingRows": 1000, "burnInRows": 100,
                                      "scheduleSha256": digest((seed, fold))},
                         "before": evaluation(0.5), "after": evaluation(0.2 + offset * 0.01),
                         "nativeAfter": {"survivingTrials": 9, "survivalSeconds": 30.0,
                                         "foodEarned": 12.0, "kills": 1}})
    return {"measurementComplete": True, "status": "measured", "releaseQualified": False,
            "config": {"seed": seed, "validationSeed": fold, "updates": 250,
                       "batch_size": 32, "bptt_steps": 64, "burn_in_steps": 32},
            "corpusManifestSha256": digest("corpus"), "corpusNativeSourceSha256": digest("native"),
            "runnerSha256": {"canonical": digest("runner")},
            "trainerSourceSha256": digest("trainer"), "sources": {"test-only": {}},
            "environment": {"device": "cpu", "torch": "qualified-test-fixture"},
            "variants": variants}


def records(spec):
    """Construct precisely the requested grid, not multiple copies of one cell."""
    return [(study(seed, fold), digest((seed, fold))) for seed, fold in spec.cells]


def test_complete_grid_preserves_paired_differences_and_policy_strata():
    """Replication widens coverage without manufacturing maturity or a release choice."""
    spec = MatrixSpec((3, 1, 2), (4, 2, 3), 250)
    result = combine_studies(records(spec), spec)
    assert result["cellCount"] == 9 and result["replicatedSeedFoldCoverage"] is True
    assert result["releaseQualified"] is False and result["maturityQualified"] is False
    assert result["profileChoice"] is None
    full = result["variants"]["full"]["pairedDifferenceFromFull"]["turnMae"]
    assert full["mean"] == 0 and full["zero"] == 9
    without = result["variants"]["no-bearing"]
    assert without["pairedDifferenceFromFull"]["turnMae"]["mean"] == pytest.approx(0.01)
    assert without["pairedDifferenceFromFull"][POLICIES[0] + "/turnMae"]["count"] == 9
    assert without["pairedDifferenceFromFull"][POLICIES[1] + "/turnMae"]["positive"] == 9


def test_single_initialization_or_fold_is_not_replicated_coverage():
    """A larger update count cannot substitute for independent identities."""
    spec = MatrixSpec((1,), (2,), 250)
    assert combine_studies(records(spec), spec)["replicatedSeedFoldCoverage"] is False


@pytest.mark.parametrize("seeds,folds,updates", [((1, 1), (2,), 250), ((True,), (2,), 250),
                                                ((1,), (0,), 250), ((1,), (2,), 0)])
def test_invalid_identity_and_budget_reject(seeds, folds, updates):
    """Duplicate, Boolean, unavailable and empty-budget identities are not replicates."""
    with pytest.raises(ValueError):
        MatrixSpec(seeds, folds, updates)


def test_missing_or_duplicate_cell_cannot_complete_matrix():
    """Completeness checks the Cartesian grid rather than only a report count."""
    spec = MatrixSpec((1, 2), (2, 3), 250)
    values = records(spec)
    with pytest.raises(ValueError, match="balanced"):
        combine_studies(values[:-1], spec)
    values[-1] = copy.deepcopy(values[0])
    with pytest.raises(ValueError, match="duplicate"):
        combine_studies(values, spec)


@pytest.mark.parametrize("mutation,reason", [
    (lambda report: report["variants"][1]["training"].update(
        scheduleSha256=digest("other")), "windows"),
    (lambda report: report["variants"][1]["after"]["perPolicy"].pop(POLICIES[1]), "policy strata"),
    (lambda report: report["variants"][1]["after"].update(turnMae=float("nan")), "finite"),
    (lambda report: report["variants"].append(copy.deepcopy(report["variants"][0])),
     "seven distinct"),
    (lambda report: report["variants"][0]["after"]["perScenarioPolicy"].pop(next(iter(
        report["variants"][0]["after"]["perScenarioPolicy"]))), "scenario/policy"),
])
def test_unpaired_or_missing_measurements_reject(mutation, reason):
    """A successful child status does not excuse a corrupted pairing or absent measurement."""
    spec = MatrixSpec((1,), (2,), 250)
    values = records(spec)
    mutation(values[0][0])
    with pytest.raises(ValueError, match=reason):
        combine_studies(values, spec)


def test_changed_source_or_initial_weights_cannot_be_pooled():
    """Cross-fold changes must not masquerade as initialization variance."""
    spec = MatrixSpec((1,), (2, 3), 250)
    values = records(spec)
    values[1][0]["environment"]["torch"] = "other-build"
    with pytest.raises(ValueError, match="environment"):
        combine_studies(values, spec)
    values = records(spec)
    values[1][0]["variants"][0]["initialWeightsSha256"] = digest("changed")
    with pytest.raises(ValueError, match="initialization changed"):
        combine_studies(values, spec)


def write_study(root):
    """Publish test-only receipt files to exercise hash binding, with no simulation claims."""
    report = study(1, 2)
    report.update(schemaVersion=1, kind="native-tactical-ablation-v1")
    source_root = root / "sources"
    source_root.mkdir()
    names = ["distillation_job.py", "canonical_graph.py", "sequence_distillation.py",
             "tactical_ablation.py", "tactical_ablation_gate.py", "tactical_contract.py",
             "tactical_corpus.py"]
    sources = {name: b"# reader unit fixture\n" for name in names}
    sources["distillation_job.py"] = b'SOURCE_FILES = ("distillation_job.py",)\n'
    trainer = hashlib.sha256(b"slither.distillation-trainer-source.v1\0")
    trainer.update(b"distillation_job.py\0" + sources["distillation_job.py"])
    report["trainerSourceSha256"] = trainer.hexdigest()
    report["sources"] = {}
    for name, data in sources.items():
        encoded = gzip.compress(data, mtime=0)
        (source_root / (name + ".gz")).write_bytes(encoded)
        report["sources"][name] = {"sha256": hashlib.sha256(data).hexdigest(),
                                   "gzipSha256": hashlib.sha256(encoded).hexdigest()}
    for variant in report["variants"]:
        directory = root / variant["variant"]
        directory.mkdir()
        for phase, artifact in [("Before", "initial.slither-brain"),
                                ("After", "trained.slither-brain")]:
            original = (artifact + variant["variant"]).encode()
            (directory / artifact).write_bytes(original)
            trials = [{"scenario": name, "seed": 2, "observations": 1,
                       "sharedBatchBudgetHeld": True, "alive": True,
                       "aliveSeconds": 1.0, "foodEarned": 1.0, "kills": 0} for name in SCENARIOS]
            native = {"kind": "native-student-tactics-v1", "sourceSha256": digest("native"),
                      "firstSeed": 2, "seedCount": 1, "trials": trials,
                      "candidate": {"artifactSha256": hashlib.sha256(original).hexdigest(),
                                    "weightsSha256": variant[
                                        "initialWeightsSha256" if phase == "Before"
                                        else "trainedWeightsSha256"],
                                    "parameterCount": variant["parameters"],
                                    "layoutSha256": variant["layoutSha256"]},
                      "referenceV1": {"unitFixture": True}}
            filename = "native-" + phase.lower() + ".json"
            encoded = json.dumps(native).encode()
            (directory / filename).write_bytes(encoded)
            variant["native" + phase] = {"report": filename,
                                        "sha256": hashlib.sha256(encoded).hexdigest(),
                                        "survivingTrials": 14, "survivalSeconds": 14.0,
                                        "foodEarned": 14.0, "kills": 0}
    (root / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_reader_binds_native_reports_originals_and_measured_sources(tmp_path):
    """Changing any referenced bytes cannot retain a completed study's qualification receipts."""
    expected = write_study(tmp_path)
    actual, receipt = read_study(tmp_path)
    assert actual == expected and len(receipt) == 64
    artifact = tmp_path / "full/trained.slither-brain"
    artifact.write_bytes(b"changed portable object")
    with pytest.raises(ValueError, match="portable bytes"):
        read_study(tmp_path)


def test_reader_recomputes_trainer_identity_from_archived_inventory(tmp_path):
    """Matching source filenames alone cannot qualify a different executable trainer source."""
    report = write_study(tmp_path)
    report["trainerSourceSha256"] = digest("different trainer")
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="trainer identity"):
        read_study(tmp_path)


def test_reader_rejects_changed_native_outcome_summary(tmp_path):
    """Aggregate food/survival/kills must equal the hash-bound native trials."""
    report = write_study(tmp_path)
    report["variants"][0]["nativeAfter"]["survivingTrials"] = 13
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="native outcomes"):
        read_study(tmp_path)


def test_boolean_cell_identity_and_changed_holdout_exposure_reject():
    """Seed identity and validation exposure cannot be rounded into an apparently paired cell."""
    spec = MatrixSpec((1,), (2,), 250)
    values = records(spec)
    values[0][0]["config"]["seed"] = True
    with pytest.raises(ValueError, match="integer"):
        combine_studies(values, spec)
    values = records(spec)
    values[0][0]["variants"][1]["after"]["perScenarioPolicy"][
        SCENARIOS[0] + "/" + POLICIES[0]]["rows"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        combine_studies(values, spec)



def with_curves(values):
    """Attach small consistent checkpoint statistics; these remain test-only metadata."""
    for report, _ in values:
        report["config"]["validationUpdates"] = [100, 250]
        for variant in report["variants"]:
            points = []
            variant["before"]["sequences"] = variant["after"]["sequences"] = 28
            for update in (0, 100, 250):
                if update == 0:
                    measured = variant["before"]
                    weights = variant["initialWeightsSha256"]
                elif update == 250:
                    measured = variant["after"]
                    weights = variant["trainedWeightsSha256"]
                else:
                    measured = {**evaluation(variant["after"]["turnMae"] + 0.05),
                                "sequences": 28}
                    weights = digest((report["config"]["seed"], variant["variant"], update))
                points.append({key: copy.deepcopy(value) for key, value in {
                    **measured, "update": update, "weightsSha256": weights,
                    "torchEvaluationWallMs": 1.0}.items() if key != "perScenarioPolicy"})
            variant["training"]["validationCurve"] = points
    return values


def test_complete_curves_preserve_paired_updates_and_both_teacher_strata():
    """Learning efficiency is reported by checkpoint, without manufacturing a maturity pass."""
    spec = MatrixSpec((1, 2, 3), (2, 4, 6), 250)
    result = combine_studies(with_curves(records(spec)), spec)
    assert result["maturityQualified"] is False
    points = result["variants"]["no-bearing"]["heldOutLearningCurve"]
    assert [point["update"] for point in points] == [0, 100, 250]
    assert points[0]["measurements"]["turnMae"]["mean"] == 0.5
    assert points[-1]["pairedDifferenceFromFull"]["turnMae"]["mean"] == pytest.approx(0.01)
    assert points[1]["measurements"][POLICIES[0] + "/turnMae"]["count"] == 9
    assert points[1]["measurements"][POLICIES[1] + "/turnMae"]["count"] == 9
    legacy = combine_studies(records(spec), spec)
    assert "heldOutLearningCurve" not in legacy["variants"]["full"]


@pytest.mark.parametrize("mutate,reason", [
    (lambda value: value["training"]["validationCurve"].pop(1), "complete declared"),
    (lambda value: value["training"]["validationCurve"][1].update(update=99), "chronology"),
    (lambda value: value["training"]["validationCurve"][1].update(update=True), "integer"),
    (lambda value: value["training"]["validationCurve"][1].update(sequences=29), "exposure"),
    (lambda value: value["training"]["validationCurve"][1].update(turnMae=float("nan")), "finite"),
    (lambda value: value["training"]["validationCurve"][0].update(
        weightsSha256=digest("other")), "weights"),
    (lambda value: value["training"]["validationCurve"][-1].update(
        weightsSha256=digest("other")), "weights"),
    (lambda value: value["training"]["validationCurve"][1]["perPolicy"].pop(POLICIES[0]),
     "policy strata"),
])
def test_partial_or_unpaired_curves_cannot_be_pooled(mutate, reason):
    """A valid final model cannot excuse fabricated curve identity or missing observations."""
    spec = MatrixSpec((1,), (2,), 250)
    values = with_curves(records(spec))
    mutate(values[0][0]["variants"][0])
    with pytest.raises(ValueError, match=reason):
        combine_studies(values, spec)


def test_curve_declaration_requires_final_update_and_same_coverage_for_all_cells():
    """An enabled curve is a complete paired schedule rather than selected favorable updates."""
    spec = MatrixSpec((1,), (2, 3), 250)
    values = with_curves(records(spec))
    values[0][0]["config"]["validationUpdates"] = [100]
    with pytest.raises(ValueError, match="omits final"):
        combine_studies(values, spec)
    values = with_curves(records(spec))
    values[1][0]["config"]["validationUpdates"] = []
    with pytest.raises(ValueError, match="hyperparameters"):
        combine_studies(values, spec)


def basic_records(spec):
    """Adapt reader-only sufficient statistics to the actual Basic teacher labels."""
    values = with_curves(records(spec))
    for report, _digest in values:
        report["config"].update(sensorProfile="basic", sensorBins=8)
        report["variants"] = [report["variants"][0]]
        variant = report["variants"][0]
        for evaluation_record in (variant["before"], variant["after"],
                                  *variant["training"]["validationCurve"]):
            evaluation_record["perPolicy"] = {
                key.replace("motion-", "basic-"): value
                for key, value in evaluation_record["perPolicy"].items()}
            if "perScenarioPolicy" in evaluation_record:
                evaluation_record["perScenarioPolicy"] = {
                    key.replace("/motion-", "/basic-"): value
                    for key, value in evaluation_record["perScenarioPolicy"].items()}
    return values


def test_basic_full_matrix_preserves_its_actual_policy_strata_and_learning_curves():
    """Full-only Basic data reports balanced coverage without manufacturing extra ablations."""
    spec = MatrixSpec((1, 2, 3), (2, 4, 6), 250)
    result = combine_studies(basic_records(spec), spec)
    assert result["sensorProfile"] == "basic" and result["sensorBins"] == 8
    assert set(result["variants"]) == {"full"}
    assert result["replicatedSeedFoldCoverage"] and not result["maturityQualified"]
    assert not result["releaseQualified"] and result["profileChoice"] is None
    points = result["variants"]["full"]["heldOutLearningCurve"]
    assert [point["update"] for point in points] == [0, 100, 250]
    assert points[1]["measurements"]["basic-v2-stochastic/turnMae"]["count"] == 9
    assert points[1]["pairedDifferenceFromFull"]["turnMae"]["zero"] == 9


@pytest.mark.parametrize("bins", range(8, 33))
def test_explicit_motion_width_keeps_the_complete_seven_variant_comparison(bins):
    """Declaring a new width cannot weaken Motion pairing or silently select defaults."""
    spec = MatrixSpec((1,), (2,), 250)
    values = records(spec)
    values[0][0]["config"].update(sensorProfile="motion", sensorBins=bins)
    result = combine_studies(values, spec)
    assert result["sensorProfile"] == "motion" and result["sensorBins"] == bins
    assert set(result["variants"]) == set(VARIANTS)
    values[0][0]["variants"].pop()
    with pytest.raises(ValueError, match="seven distinct"):
        combine_studies(values, spec)


@pytest.mark.parametrize("identity", [
    {"sensorProfile": "basic"}, {"sensorBins": 8}, {"sensorProfile": "v3", "sensorBins": 8},
    {"sensorProfile": "basic", "sensorBins": True},
    {"sensorProfile": "motion", "sensorBins": 7},
    {"sensorProfile": "motion", "sensorBins": 33},
])
def test_incomplete_or_unsupported_sensor_identity_cannot_enter_matrix(identity):
    """An absent legacy declaration differs from a partial or invalid explicit identity."""
    spec = MatrixSpec((1,), (2,), 250)
    values = records(spec)
    values[0][0]["config"].update(identity)
    with pytest.raises(ValueError):
        combine_studies(values, spec)


def test_basic_missing_or_wrong_teacher_policies_and_motion_groups_reject():
    """Basic statistics cannot be relabeled as Motion or include undefined channel removals."""
    spec = MatrixSpec((1,), (2,), 250)
    values = basic_records(spec)
    variant = values[0][0]["variants"][0]
    variant["after"]["perPolicy"].pop("basic-v2-stochastic")
    variant["training"]["validationCurve"][-1]["perPolicy"].pop("basic-v2-stochastic")
    with pytest.raises(ValueError, match="policy strata"):
        combine_studies(values, spec)
    values = basic_records(spec)
    values[0][0]["variants"][0]["variant"] = "core-only"
    with pytest.raises(ValueError, match="exactly the Full"):
        combine_studies(values, spec)
    values = basic_records(spec)
    values[0][0]["config"]["sensorProfile"] = "motion"
    with pytest.raises(ValueError, match="seven distinct"):
        combine_studies(values, spec)


def test_cross_profile_or_width_cells_cannot_be_pooled_as_paired_data():
    """Even matching source/seed receipts cannot pair different sensor geometries."""
    spec = MatrixSpec((1,), (2, 3), 250)
    for identity in ({"sensorProfile": "motion", "sensorBins": 8},
                     {"sensorProfile": "basic", "sensorBins": 9}):
        values = basic_records(spec)
        values[1][0]["config"].update(identity)
        with pytest.raises(ValueError, match="hyperparameters"):
            combine_studies(values, spec)


def test_reader_rejects_hash_bound_native_input_width_that_disagrees_with_study(tmp_path):
    """A recomputed report receipt cannot reinterpret another native sensor dimension."""
    report = write_study(tmp_path)
    report["config"].update(sensorProfile="motion", sensorBins=32)
    variant = report["variants"][0]
    native_path = tmp_path / variant["variant"] / "native-before.json"
    native = json.loads(native_path.read_bytes())
    native["candidate"]["inputSize"] = 264
    native_path.write_text(json.dumps(native), encoding="utf-8")
    variant["nativeBefore"]["sha256"] = hashlib.sha256(native_path.read_bytes()).hexdigest()
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="native candidate sensor width"):
        read_study(tmp_path)
