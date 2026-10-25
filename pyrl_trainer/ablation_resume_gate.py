"""Exercise a real process restart between native-validated ablation variants."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

from .ablation_matrix import read_study
from .experiment_source import capture
from .tactical_ablation_gate import AblationStudy


def arguments(root, canonical, tactical, directory, resume=False):
    """Keep the real qualification small and CPU-only while preserving deployment checks."""
    return SimpleNamespace(
        updates=2, validation_update=(1, 2), cuda_graphs=False, corpus=root / "corpus",
        variant=["full", "no-heading"], validation_seed=2, batch_sequences=2,
        sequence_length=16, burn_in=0, seed=1, device="cpu", cpu_threads=1,
        device_memory_mib=512, canonical_runner=canonical, tactical_runner=tactical,
        directory=directory, native_steps=12, native_workers=1, deadline_seconds=300,
        resume=resume,
    )


def child(args):
    """Fail only after a complete real variant; the normal driver owns all checkpointing."""
    study = AblationStudy(arguments(args.directory, args.canonical_runner,
                                   args.tactical_runner, args.directory / args.phase,
                                   resume=False))
    study.report["sourceProvenance"]["command"] = [
        sys.executable, "-m", "pyrl_trainer.ablation_resume_gate", *sys.argv[1:]]
    if args.phase == "interrupted":
        actual = study.variant

        def interrupt(name):
            if name == "full":
                return actual(name)
            (study.args.directory / name).mkdir()
            (study.args.directory / name / "interrupted.txt").write_text("incomplete attempt")
            raise RuntimeError("qualification injected interruption")

        study.variant = interrupt
        try:
            study.run()
        except RuntimeError as error:
            if str(error) != "qualification injected interruption":
                raise
        else:
            raise ValueError("interruption was not exercised")
    else:
        study.run()


def qualify(args):
    """Compare resumed and uninterrupted model bytes plus independently admitted native reports."""
    args.directory.mkdir()
    subprocess.run([str(args.tactical_runner), "--seed-count", "2", "--steps", "12",
                    "--workers", "1", "--bins", "8", "--corpus-profile", "motion",
                    "--output", str(args.directory / "teachers.json"),
                    "--corpus-directory", str(args.directory / "corpus")],
                   check=True, timeout=120)
    command = [sys.executable, "-m", "pyrl_trainer.ablation_resume_gate",
               "--canonical-runner", str(args.canonical_runner),
               "--tactical-runner", str(args.tactical_runner),
               "--directory", str(args.directory)]
    subprocess.run([*command, "--phase", "interrupted"], check=True, timeout=300)
    interrupted = args.directory / "interrupted"
    before = json.loads((interrupted / "progress.json").read_bytes())
    if before["measurementComplete"] or len(before["variants"]) != 1:
        raise ValueError("interrupted study did not retain exactly one complete variant")
    # Continue the same directory under a genuinely separate process owner.
    resume = [sys.executable, "-m", "pyrl_trainer.tactical_ablation_gate",
              "--corpus", str(args.directory / "corpus"),
              "--canonical-runner", str(args.canonical_runner),
              "--tactical-runner", str(args.tactical_runner),
              "--directory", str(interrupted), "--resume", "--device", "cpu",
              "--variant", "full", "--variant", "no-heading", "--updates", "2",
              "--batch-sequences", "2", "--sequence-length", "16", "--burn-in", "0",
              "--cpu-threads", "1", "--device-memory-mib", "512", "--native-steps", "12",
              "--validation-seed", "2", "--validation-update", "1", "--validation-update", "2"]
    subprocess.run(resume, check=True, timeout=300)
    subprocess.run([*command, "--phase", "control"], check=True, timeout=300)
    resumed, _ = read_study(interrupted)
    control, _ = read_study(args.directory / "control")
    if resumed["variants"][0] != before["variants"][0]:
        raise ValueError("completed variant was not reused byte-for-byte")
    if not (interrupted / "interrupted-2-no-heading/interrupted.txt").is_file():
        raise ValueError("interrupted outputs were discarded")
    for first, second in zip(resumed["variants"], control["variants"]):
        for key in ("variant", "initialWeightsSha256", "trainedWeightsSha256"):
            if first[key] != second[key]:
                raise ValueError("resumed optimizer output differs: " + key)
        for key in ("scheduleSha256", "updates", "trainingRows", "burnInRows"):
            if first["training"][key] != second["training"][key]:
                raise ValueError("resumed paired schedule differs: " + key)
    provenance = capture([Path(__file__).name], args.directory)
    provenance["command"] = [
        sys.executable, "-m", "pyrl_trainer.ablation_resume_gate", *sys.argv[1:]]
    receipt = {"schemaVersion": 1, "kind": "native-ablation-resume-v1",
               "completedVariantReusedExactly": True, "interruptedOutputsPreserved": True,
               "resumedWeightsMatchUninterrupted": True, "variantCount": 2,
               "corpusManifestSha256": resumed["corpusManifestSha256"],
               "nativeSourceSha256": resumed["corpusNativeSourceSha256"],
               "runnerSha256": resumed["runnerSha256"],
               "summarySha256": hashlib.sha256((interrupted / "summary.json").read_bytes()
                                              ).hexdigest(),
               "sourceProvenance": provenance,
               "scope": "Completed-variant process restart; no mid-variant optimizer recovery"}
    (args.directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "sourceProvenance"}))


def main():
    """Use explicit ordinary native runners and one task-owned new qualification directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-runner", "tactical-runner", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("interrupted", "control"))
    args = parser.parse_args()
    args.canonical_runner = args.canonical_runner.resolve(strict=True)
    args.tactical_runner = args.tactical_runner.resolve(strict=True)
    args.directory = args.directory.resolve()
    if args.phase:
        child(args)
    else:
        qualify(args)


if __name__ == "__main__":
    main()
