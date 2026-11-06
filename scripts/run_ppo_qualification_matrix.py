"""Run the declared qualification scenarios in isolated Slither processes."""

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass, replace
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from tempfile import TemporaryDirectory

from pyrl_trainer.config import AUTHORITATIVE_REWARD_CONFIG, load_or_create_config, validate_config
from scripts.evaluate_ppo_episode import REFERENCES, evaluate_one
from scripts.summarize_ppo_qualification import summarize


@dataclass(frozen=True)
class ScenarioServer:
    """The pinned Slither source and settings shared by all evaluation cases."""

    root: Path
    config: Path
    config_hash: str
    math_backend: str


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _clean_revision(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError(f"qualification source tree has changes: {root}")
    return git("rev-parse", "HEAD")


def _completed_policies(manifest, runs):
    roles = {}
    for run in runs:
        spec = _read_json(run / "run.json")
        seed = spec["trainingSeed"]
        if seed not in manifest["trainingSeeds"] or f"seed{seed}-final" in roles:
            raise ValueError(f"unexpected or repeated training seed {seed}")
        keys = ("pyrlCommit", "slitherCommit", "configHash", "mathBackend", "worldSeed",
                "trainingSeeds", "evaluationWorldSeeds", "decisionBudget", "candidateRule")
        if any(spec[key] != manifest[key] for key in keys):
            raise ValueError(f"training configuration differs for seed {seed}")
        if (run / "exit-code.txt").read_text(encoding="utf-8-sig").strip() != "0":
            raise ValueError(f"training failed for seed {seed}")
        log = (run / "training.log").read_text(encoding="utf-8-sig")
        completed = re.findall(
            r"^\[qualification\] admitted_decisions=(\d+), decision_budget=(\d+), "
            r"alive_seconds=[\d.]+, updates=\d+, policy_v=(\d+)$", log, re.MULTILINE
        )
        if len(completed) != 1:
            raise ValueError(f"missing unique budget completion for seed {seed}")
        decisions, budget, version = map(int, completed[0])
        if budget != manifest["decisionBudget"] or decisions < budget or version <= 0:
            raise ValueError(f"training budget incomplete for seed {seed}")
        for suffix, relative, expected in (
            ("initial", "checkpoints/initial/latest.pt", 0),
            ("final", "checkpoints/latest.pt", version),
        ):
            checkpoint = run / relative
            if not checkpoint.is_file():
                raise ValueError(f"missing {suffix} checkpoint for seed {seed}")
            roles[f"seed{seed}-{suffix}"] = (checkpoint, expected)
    if len(roles) != 2 * len(manifest["trainingSeeds"]):
        raise ValueError("the declared training seeds have not all completed")
    return roles


def _validate_declared_seeds(manifest):
    for key in ("trainingSeeds", "evaluationWorldSeeds"):
        seeds = manifest[key]
        if (not seeds or any(isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed < 2**53 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError(f"invalid or repeated seeds in {key}")


def _effective_evaluation_settings(run):
    """Recover the effective training overrides from the pinned trainer's startup log."""
    log = (run / "training.log").read_text(encoding="utf-8-sig")
    startup = re.search(r"^\[main\] ws=.*$", log, re.MULTILINE)
    objective = re.search(r"^\[main\] reward_config_version=.*$", log, re.MULTILINE)
    if startup is None or objective is None:
        raise ValueError(f"missing effective trainer configuration in {run}")
    startup_fields = dict(re.findall(r"(?:^|, )([a-z_]+)=([^,]+)", startup[0][7:]))
    objective_fields = dict(re.findall(r", ([a-z_]+)=(\d+(?:\.\d+)?)(?=,|$)", objective[0]))
    coefficients = json.loads(objective[0].split("reward_coefficients=", 1)[1]
                              .split(", policy_parameterization=", 1)[0])
    if set(coefficients) != set(AUTHORITATIVE_REWARD_CONFIG):
        raise ValueError("training log does not identify the authoritative reward coefficients")
    width, depth = map(int, startup_fields["network"].split("x"))
    return {
        **coefficients,
        "net_hidden": width,
        "net_layers": depth,
        "train_device": startup_fields["train_device"],
        "infer_device": startup_fields["infer_device"],
        "max_actions_per_second": int(objective_fields["max_actions_per_second"]),
        "max_actions_per_tick": int(objective_fields["max_actions_per_tick"]),
        "evaluation_seed": int(objective_fields["evaluation_seed"]),
        "evaluation_max_seconds": float(objective_fields["evaluation_max_seconds"]),
    }


def _health(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/api/health")
        return json.load(connection.getresponse())
    finally:
        connection.close()


async def _ready(process, port):
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"Slither exited during startup with code {process.returncode}")
        try:
            health = await asyncio.to_thread(_health, port)
            if health.get("ok") and health.get("authority") == "rust":
                return health
            if health.get("interfaceFault"):
                raise RuntimeError(f"Slither reported a startup fault: {health['interfaceFault']}")
        except (OSError, http.client.HTTPException):
            pass
        await asyncio.sleep(0.2)
    raise TimeoutError("Slither did not become ready")


async def _stop(process):
    if process.returncode is not None:
        return
    with suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()


async def run_case(cfg, server, log_dir, seed, role):
    """Own one fresh loopback server for one policy and one scenario."""
    label, checkpoint, version = role
    log_dir.mkdir()
    log_dir = log_dir.resolve(strict=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    creation = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with TemporaryDirectory(prefix="world-", dir=log_dir) as scratch:
        scratch_path = Path(scratch).resolve(strict=True)
        if scratch_path.parent != log_dir:
            raise ValueError("scenario scratch directory is outside its case directory")
        database = scratch_path / "scenario.db"
        with (log_dir / "slither.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                "node", "--import", "tsx", "server/rustServer.ts",
                "--host", "127.0.0.1", "--port", str(port), "--config", str(server.config),
                "--fresh", "--seed", str(seed), "--db-path", str(database), cwd=server.root,
                stdout=log, stderr=asyncio.subprocess.STDOUT, **creation,
            )
            try:
                health = await _ready(process, port)
                if (health.get("configHash") != server.config_hash or
                        health.get("mathBackend") != server.math_backend):
                    raise ValueError("evaluation server differs from the declared configuration")
                result = await evaluate_one(
                    replace(cfg, ws_url=f"ws://127.0.0.1:{port}"), scenario_seed=seed,
                    checkpoint=checkpoint, reference=label if checkpoint is None else None,
                )
                if result["run_id"] != health["runId"] or result["policy_version"] != version:
                    raise ValueError("episode differs from the started run or selected policy")
                return {**result, "policy_label": label}
            finally:
                await _stop(process)


async def _run_matrix(cfg, server, output, manifest, roles):
    records = []
    for seed in manifest["evaluationWorldSeeds"]:
        for label, (checkpoint, version) in roles.items():
            result = await run_case(
                cfg, server, output / f"{seed}-{label}", seed, (label, checkpoint, version),
            )
            records.append(result)
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                f"[qualification-matrix] scenario={seed} policy={label} "
                f"integrity={result['integrity']} return={result['episode_return']:.6f}", flush=True
            )
    report = summarize(manifest, records)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report["complete"]


def main():
    """Use the completed run directories and the original trainer/server configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slither-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("training_runs", type=Path, nargs="+")
    args = parser.parse_args()
    slither = args.slither_root.resolve(strict=True)
    runs = [run.resolve(strict=True) for run in args.training_runs]
    manifest = _read_json(runs[0] / "run.json")
    _validate_declared_seeds(manifest)
    roles = _completed_policies(manifest, runs)
    roles.update(dict.fromkeys(REFERENCES, (None, 0)))
    if _clean_revision(slither) != manifest["slitherCommit"]:
        raise ValueError("Slither revision differs from the training runs")
    evaluation_revision = _clean_revision(Path(__file__).resolve().parents[1])
    trainer_config, server_config = (runs[0] / name for name in ("trainer.toml", "server.toml"))
    if not trainer_config.is_file() or not server_config.is_file():
        raise ValueError("missing training scenario configuration")
    effective = _effective_evaluation_settings(runs[0])
    if any(_effective_evaluation_settings(run) != effective for run in runs[1:]):
        raise ValueError("effective evaluator settings differ between training seeds")
    cfg = validate_config(replace(load_or_create_config(str(trainer_config)), **effective))
    if cfg.evaluation_max_seconds != 240:
        raise ValueError("the declared evaluation cap is 240 simulated seconds")
    server = ScenarioServer(slither, server_config, manifest["configHash"], manifest["mathBackend"])
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "matrix-run.json").write_text(json.dumps({
        **manifest, "evaluationCommit": evaluation_revision,
        "effectiveEvaluation": effective,
    }, indent=2), encoding="utf-8")
    complete = asyncio.run(_run_matrix(cfg, server, output, manifest, roles))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
