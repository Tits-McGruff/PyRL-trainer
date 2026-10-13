"""Generate bounded canonical graph fixtures using only the Rust offline tool."""

import argparse
import json
from pathlib import Path
import struct
import subprocess


def _graph(operation, width):
    """Declare source DAGs; Rust decides all parameter and state packing."""
    nodes = [{"id": "input", "type": "Input", "outputSize": width}]
    edges = []

    def node(name, kind, **shape):
        nodes.append({"id": name, "type": kind, **shape})

    def edge(source, destination, source_port=0, ordinal=0):
        edges.append(
            {"from": source, "to": destination, "fromPort": source_port, "toPort": ordinal}
        )

    if operation == "routing":
        node("split", "Split", outputSizes=[49, 1, 1])
        edge("input", "split")
        outputs = [{"nodeId": "split", "port": 2}, {"nodeId": "split", "port": 1}]
        states = 0
    elif operation == "mixed":
        node("mlp", "MLP", inputSize=width, hiddenSizes=[7, 5], outputSize=4)
        node("gru", "GRU", inputSize=4, hiddenSize=3)
        node("lstm", "LSTM", inputSize=4, hiddenSize=2)
        node("concat", "Concat")
        node("rru", "RRU", inputSize=5, hiddenSize=3)
        node("head", "Dense", inputSize=3, outputSize=2)
        node("split", "Split", outputSizes=[1, 1])
        edge("input", "mlp")
        edge("mlp", "gru")
        edge("mlp", "lstm")
        edge("lstm", "concat", ordinal=0)
        edge("gru", "concat", ordinal=1)
        edge("concat", "rru")
        edge("rru", "head")
        edge("head", "split")
        outputs = [{"nodeId": "split", "port": 1}, {"nodeId": "split", "port": 0}]
        states = 10
    elif operation == "mlp":
        node("mlp", "MLP", inputSize=width, hiddenSizes=[9, 4, 6], outputSize=2)
        edge("input", "mlp")
        outputs, states = [{"nodeId": "mlp"}], 0
    elif operation == "dense":
        node("head", "Dense", inputSize=width, outputSize=2)
        edge("input", "head")
        outputs, states = [{"nodeId": "head"}], 0
    else:
        hidden = 3 if operation == "lstm" else 2
        node("recurrent", operation.upper(), inputSize=width, hiddenSize=hidden)
        edge("input", "recurrent")
        if hidden == 3:
            node("head", "Dense", inputSize=hidden, outputSize=2)
            edge("recurrent", "head")
        outputs = [{"nodeId": "head" if hidden == 3 else "recurrent"}]
        states = hidden * (2 if operation == "lstm" else 1)
    return {
        "type": "graph",
        "nodes": list(reversed(nodes)),
        "edges": list(reversed(edges)),
        "outputs": outputs,
        "outputSize": 2,
    }, states


def generate(runner, output):
    """Emit exact little-endian inputs and immutable Rust results with provenance."""
    output.mkdir(parents=True, exist_ok=False)
    records = []
    cases = [
        ("dense", "v3", 8),
        ("mlp", "basic", 8),
        ("gru", "motion", 8),
        ("lstm", "v3", 16),
        ("rru", "basic", 16),
        ("mixed", "motion", 16),
        ("routing", "v3", 8),
    ]
    for operation, profile, bins in cases:
        width = (
            (19 + 4 * bins) if profile == "v3" else 24 + (6 if profile == "basic" else 15) * bins
        )
        graph, state_count = _graph(operation, width)
        root = output / operation
        root.mkdir()
        values = [
            (((index * 1664525 + 1013904223) & 0xFFFF) / 32767.5 - 1)
            for index in range(256 * width)
        ]
        (root / "observations.f32").write_bytes(struct.pack(f"<{len(values)}f", *values))
        for nonzero in ([False, True] if state_count else [False]):
            label = "nonzero" if nonzero else "zero"
            request = {
                "schemaVersion": 1,
                "graphSpec": graph,
                "sensorProfile": profile,
                "bins": bins,
                "controlIntervalSeconds": 0.03,
                "lifeStateRule": "preserve",
                "mathBackend": "rust-scalar-v1",
                "initializationSeed": 42,
                "observationsPath": "observations.f32",
            }
            if nonzero:
                states = [
                    (index + 1) * 0.03125 * (-1 if index % 2 else 1) for index in range(state_count)
                ]
                (root / "initial-state.f32").write_bytes(struct.pack(f"<{state_count}f", *states))
                request["initialStatePath"] = "initial-state.f32"
            request_path = root / f"request-{label}.json"
            request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            subprocess.run(
                [str(runner), str(request_path), str(root / label)],
                check=True,
                capture_output=True,
                text=True,
            )
            receipt = json.loads((root / label / "receipt.json").read_text(encoding="utf-8"))
            records.append(
                {
                    "case": operation,
                    "initialState": label,
                    "sourceSha256": receipt["sourceSha256"],
                    "target": receipt["buildTarget"],
                    "steps": receipt["steps"],
                }
            )
    (output / "index.json").write_text(
        json.dumps({"schemaVersion": 1, "cases": records}, indent=2), encoding="utf-8"
    )


def main():
    """Require an explicit Rust runner and new destination; never overwrite evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    generate(args.runner.resolve(strict=True), args.output_directory.resolve())


if __name__ == "__main__":
    main()
