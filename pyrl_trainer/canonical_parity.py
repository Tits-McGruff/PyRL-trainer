"""Mandatory optimizer/export/Rust-inference/exact-byte-reimport parity gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import torch

from .canonical_graph import CanonicalGraph, CanonicalGraphManifest
from .sequence_distillation import CanonicalSequenceTrainer, SequenceBatch


def _require(condition, reason):
    """Keep qualification checks active even under Python's optimization flag."""
    if not condition:
        raise RuntimeError(reason)


def _floats(path, width, device):
    """Read immutable fixture Float32 bytes with explicit row interpretation."""
    values = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
    return torch.from_numpy(values.reshape(-1, width)).to(device)


def _sequence(model, observations, initial_state):
    """Retain every next state so an output-only check cannot hide recurrence errors."""
    state = initial_state
    outputs, states = [], []
    with torch.no_grad():
        for row in observations:
            output, state = model(row.unsqueeze(0), state)
            outputs.append(output.cpu())
            states.append(state.cpu())
    return torch.cat(outputs), torch.cat(states)


def _compare(actual, expected, device):
    """Apply the frozen short/long tolerances and report bounded error growth."""
    torch.testing.assert_close(
        actual[0], expected[0], atol=1e-4 if device == "cuda" else 1e-5, rtol=1e-4
    )
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    return [
        {
            "step": step,
            "maxAbs": (
                (actual[:step] - expected[:step]).abs().max().item() if actual.numel() else 0.0
            ),
        }
        for step in [1, 64, 128, 256]
    ]


def _optimize(model, observations, initial_state):
    """Perform actual Adam/autograd with method-v1 loss and a bounded 64-step BPTT window."""
    trainer = CanonicalSequenceTrainer(model)
    config = trainer.config
    rows = config.burn_in_steps + config.bptt_steps
    before = model.export_weights()
    values = observations[:rows].unsqueeze(1).repeat(1, config.batch_size, 1)
    labels = values[..., 0] * 0.2
    batch = SequenceBatch(
        values,
        labels,
        values[..., 1] > 0,
        torch.ones_like(labels),
        torch.ones(labels.shape, dtype=torch.bool, device=values.device),
        torch.zeros(labels.shape, dtype=torch.int64, device=values.device),
        initial_state.repeat(config.batch_size, 1),
    )
    update = trainer.update(batch)
    _require(update.optimizer_updated, "canonical sequence optimizer did not run")
    _require(model.export_weights() != before, "optimizer did not update canonical parameters")
    return {
        "loss": update.loss,
        "gradientNorm": update.gradient_norm,
        "batchSize": config.batch_size,
        "burnInSteps": config.burn_in_steps,
        "bpttSteps": config.bptt_steps,
        "trainingSamples": update.training_samples,
        "burnInSamples": update.burn_in_samples,
        "beforeSha256": hashlib.sha256(before).hexdigest(),
        "afterSha256": hashlib.sha256(model.export_weights()).hexdigest(),
    }


def _case(runner, root, operation, device):
    """Recompile and independently admit trained bytes with both Rust math backends."""
    label = "nonzero" if (root / "nonzero").exists() else "zero"
    reference = root / label
    manifest = CanonicalGraphManifest.read(reference / "manifest.json")
    model = CanonicalGraph(manifest, (reference / "weights.f32").read_bytes(), device)
    observations = _floats(root / "observations.f32", manifest.input_size, device)
    initial = (
        _floats(root / "initial-state.f32", manifest.total_state_size, device)
        if label == "nonzero"
        else model.initial_state(1)
    )
    optimization = _optimize(model, observations, initial)
    trained = model.export_weights()
    with tempfile.TemporaryDirectory(prefix="pyrl-canonical-roundtrip-") as directory:
        workspace = Path(directory)
        weight_path = workspace / "trained.f32"
        weight_path.write_bytes(trained)
        request = json.loads((root / f"request-{label}.json").read_text(encoding="utf-8"))
        request.pop("initializationSeed")
        request["weightsPath"] = str(weight_path)
        request["observationsPath"] = str((root / "observations.f32").resolve())
        if label == "nonzero":
            request["initialStatePath"] = str((root / "initial-state.f32").resolve())
        results = []
        for backend in ["rust-scalar-v1", "rust-sse2-v1"]:
            request["mathBackend"] = backend
            request_path = workspace / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output_directory = workspace / backend
            subprocess.run(
                [str(runner), str(request_path), str(output_directory)],
                check=True,
                capture_output=True,
                text=True,
            )
            admitted = CanonicalGraphManifest.read(output_directory / "manifest.json")
            _require(
                admitted.nodes == manifest.nodes and admitted.outputs == manifest.outputs,
                "Rust admission changed canonical node/output views",
            )
            _require(
                admitted.layout_sha256 == manifest.layout_sha256, "Rust layout identity mismatch"
            )
            _require(
                admitted.sensor_sha256 == manifest.sensor_sha256, "Rust sensor identity mismatch"
            )
            returned = (output_directory / "weights.f32").read_bytes()
            _require(returned == trained, "Rust admitted bytes differ from trained export")
            reimported = CanonicalGraph(admitted, returned, device)
            _require(
                reimported.export_weights() == returned,
                "PyTorch reimport changed exact weight bytes",
            )
            actual, states = _sequence(reimported, observations, initial)
            expected = _floats(output_directory / "outputs.f32", 2, "cpu")
            expected_states = (
                _floats(output_directory / "states.f32", manifest.total_state_size, "cpu")
                if manifest.total_state_size
                else torch.empty(256, 0)
            )
            receipt = json.loads((output_directory / "receipt.json").read_text(encoding="utf-8"))
            results.append(
                {
                    "case": operation,
                    "initialState": label,
                    "backend": backend,
                    "outputErrorGrowth": _compare(actual, expected, device),
                    "stateErrorGrowth": _compare(states, expected_states, device),
                    "nativeSourceSha256": receipt["sourceSha256"],
                    "nativeTarget": receipt["buildTarget"],
                    "weightsSha256": receipt["weightsSha256"],
                    "observationsSha256": receipt["observationsSha256"],
                    "initialStateSha256": receipt["initialStateSha256"],
                    "layoutSha256": admitted.layout_sha256,
                    "sensorSha256": admitted.sensor_sha256,
                    "optimization": optimization,
                }
            )
        return results


def run_gate(runner: Path, fixtures: Path, device: str):
    """Require the requested device; a missing GPU or Rust runner never silently skips."""
    runner = runner.resolve(strict=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA qualification requires an available CUDA device")
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    results = []
    for operation in ["dense", "mlp", "gru", "lstm", "rru", "mixed"]:
        results.extend(_case(runner, fixtures / operation, operation, device))
    source_hash = hashlib.sha256(b"pyrl.canonical-training-source.v1\0")
    for name in ["canonical_graph.py", "canonical_parity.py", "sequence_distillation.py"]:
        source_hash.update(name.encode("utf-8"))
        source_hash.update(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
    return {
        "trainerSourceSha256": source_hash.hexdigest(),
        "schemaVersion": 1,
        "gate": "canonical-optimizer-roundtrip-v1",
        "torchVersion": torch.__version__,
        "cudaVersion": torch.version.cuda,
        "device": device,
        "deviceName": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "pythonVersion": platform.python_version(),
        "seed": 42,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "dtype": "Float32",
        "trainingScope": "synthetic parity labels; no gameplay imitation-quality claim",
        "cases": results,
    }


def main():
    """Write a compact durable report only after every mandatory roundtrip succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate(args.runner, args.fixtures, args.device)
    with args.output_json.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    print(f"passed {len(report['cases'])} canonical optimizer roundtrips on {args.device}")


if __name__ == "__main__":
    main()
