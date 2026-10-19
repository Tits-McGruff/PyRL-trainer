"""Paired graph-input ablations over immutable native tactical deliveries."""

from dataclasses import dataclass, replace
import time

import numpy as np
import torch

from .canonical_graph import CanonicalGraphError, distillation_loss
from .sequence_distillation import SequenceBatch, SequenceBoundary

# Split port zero is the scalar prefix; subsequent ports are canonical Motion channels.
VARIANTS = {
    "full": (), "no-bearing": (8,), "no-heading": (10, 11),
    "no-velocity": (12, 13), "no-urgency": (14,), "no-density": (15,),
    "core-only": tuple(range(7, 16)),
}


def ablation_graph(variant):
    """Select declared Split/Concat ports; Rust still owns packing and compilation."""
    if variant not in VARIANTS:
        raise CanonicalGraphError("unknown tactical ablation")
    nodes = [{"id": "input", "type": "Input", "outputSize": 264}]
    edges = []
    incoming, width = "input", 264
    if VARIANTS[variant]:
        selected = [port for port in range(16) if port not in VARIANTS[variant]]
        nodes.extend([{"id": "channels", "type": "Split", "outputSizes": [24] + [16] * 15},
                      {"id": "selected", "type": "Concat"}])
        edges.append({"from": "input", "to": "channels", "fromPort": 0, "toPort": 0})
        edges.extend({"from": "channels", "to": "selected", "fromPort": port,
                      "toPort": ordinal} for ordinal, port in enumerate(selected))
        incoming, width = "selected", 24 + (len(selected) - 1) * 16
    nodes.extend([
        {"id": "encode", "type": "MLP", "inputSize": width,
         "hiddenSizes": [64], "outputSize": 64},
        {"id": "memory", "type": "GRU", "inputSize": 64, "hiddenSize": 16},
        {"id": "motor", "type": "Dense", "inputSize": 16, "outputSize": 2},
    ])
    edges.extend({"from": source, "to": target, "fromPort": 0, "toPort": 0}
                 for source, target in [(incoming, "encode"), ("encode", "memory"),
                                        ("memory", "motor")])
    return {"type": "graph", "nodes": nodes, "edges": edges,
            "outputs": [{"nodeId": "motor"}], "outputSize": 2}


@dataclass(frozen=True)
class TacticalWindow:
    """One training start and its bounded true preceding deliveries within one native life."""

    sequence: int
    training_start: int
    burn_in: int


def paired_windows(corpus, sequences, config, updates, seed):
    """Cycle temporal blocks, including short fatal lives, with one shared seeded schedule."""
    if not isinstance(updates, int) or isinstance(updates, bool) or not 1 <= updates <= 10000:
        raise CanonicalGraphError("ablation update admission is 1..10000")
    if not sequences or any(sequence not in corpus.sequences for sequence in sequences):
        raise CanonicalGraphError("unknown or empty ablation training split")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xffffffff:
        raise CanonicalGraphError("ablation seed must be UInt32")
    indices = {sequence: index for index, sequence in enumerate(corpus.sequences)}
    blocks = max(sequence.rows for sequence in sequences) + config.bptt_steps - 1
    blocks //= config.bptt_steps
    rng = np.random.Generator(np.random.PCG64(seed))
    for update in range(updates):
        start = (update % blocks) * config.bptt_steps
        eligible = [indices[sequence] for sequence in sequences if sequence.rows > start]
        burn = min(config.burn_in_steps, start)
        yield tuple(TacticalWindow(int(index), start, burn)
                    for index in rng.choice(eligible, config.batch_size))


def _check_windows(corpus, manifest, windows, config):
    """Reject incompatible data or invalid spans before allocating a training slab."""
    if (manifest.input_size != 264 or manifest.sensor_sha256 != corpus.sensor.structural_sha256
            or manifest.control_interval_seconds != corpus.control_interval_seconds):
        raise CanonicalGraphError("ablation corpus deployment identity mismatch")
    if not windows or len(windows) > config.batch_size:
        raise CanonicalGraphError("invalid bounded ablation batch")
    burn = getattr(windows[0], "burn_in", -1)
    for window in windows:
        if not isinstance(window, TacticalWindow) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (window.sequence, window.training_start, window.burn_in)
        ):
            raise CanonicalGraphError("invalid ablation window identity")
        if not 0 <= window.sequence < len(corpus.sequences):
            raise CanonicalGraphError("unknown ablation sequence")
        sequence = corpus.sequences[window.sequence]
        if not (0 <= window.training_start < sequence.rows
                and window.burn_in == burn
                and 0 <= burn <= min(config.burn_in_steps, window.training_start)):
            raise CanonicalGraphError("invalid ablation delivery span")
    return replace(config, burn_in_steps=burn)


def window_batch(corpus, manifest, windows, config):
    """Copy only selected real deliveries; suffix padding never manufactures history."""
    actual = _check_windows(corpus, manifest, windows, config)
    shape = (actual.burn_in_steps + actual.bptt_steps, len(windows))
    observations = np.zeros((*shape, 264), dtype=np.float32)
    turns, boosts, weights = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    active = np.zeros(shape, dtype=np.bool_)
    boundaries = np.zeros(shape, dtype=np.int64)
    for column, window in enumerate(windows):
        sequence = corpus.sequences[window.sequence]
        begin = window.training_start - window.burn_in
        end = min(sequence.rows, window.training_start + actual.bptt_steps)
        source = slice(sequence.row_offset + begin, sequence.row_offset + end)
        count = end - begin
        observations[:count, column] = corpus.observations[source]
        turns[:count, column] = corpus.actions[source, 0]
        boosts[:count, column] = corpus.actions[source, 1]
        weights[:count, column] = 1
        active[:count, column] = True
        boundaries[0, column] = (SequenceBoundary.GENERATION if begin == 0
                                 else SequenceBoundary.GAP)
    initial_state = torch.zeros((len(windows), manifest.total_state_size), dtype=torch.float32)
    batch = SequenceBatch(*(torch.from_numpy(value) for value in
                            [observations, turns, boosts, weights, active, boundaries]),
                          initial_state)
    return batch, actual


def evaluate_agreement(model, corpus, sequences):
    """Measure full held-out native lives with carried student recurrence and no gradients."""
    if (model.manifest.input_size != 264
            or model.manifest.sensor_sha256 != corpus.sensor.structural_sha256
            or model.manifest.control_interval_seconds != corpus.control_interval_seconds
            or not sequences or any(sequence not in corpus.sequences for sequence in sequences)):
        raise CanonicalGraphError("agreement corpus deployment or sequence identity mismatch")
    totals = {"rows": 0, "turnSquaredError": 0.0, "turnAbsoluteError": 0.0,
              "boostDisagreements": 0, "lossSum": 0.0}
    per_scenario = {}
    device = model.flat_parameters.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for sequence in sequences:
            source = slice(sequence.row_offset, sequence.row_offset + sequence.rows)
            observations = torch.from_numpy(corpus.observations[source].copy()).to(device)
            labels = torch.from_numpy(corpus.actions[source].copy()).to(device)
            state = model.initial_state(1)
            outputs = []
            for row in observations:
                output, state = model(row.unsqueeze(0), state)
                outputs.append(output[0])
            prediction = torch.stack(outputs)
            error = prediction[:, 0] - labels[:, 0]
            loss = distillation_loss(prediction, labels[:, 0], labels[:, 1],
                                     torch.ones_like(labels[:, 0]))
            values = {"rows": sequence.rows, "turnSquaredError": error.square().sum().item(),
                      "turnAbsoluteError": error.abs().sum().item(),
                      "boostDisagreements": int(((prediction[:, 1] > 0.35)
                                                 != labels[:, 1].bool()).sum().item()),
                      "lossSum": loss.sum().item()}
            group = per_scenario.setdefault(sequence.scenario, dict.fromkeys(totals, 0))
            for key, value in values.items():
                totals[key] += value
                group[key] += value
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    rows = totals["rows"]
    if rows == 0:
        raise CanonicalGraphError("empty ablation agreement split")
    return {"rows": rows, "sequences": len(sequences), "turnMse": totals["turnSquaredError"] / rows,
            "turnMae": totals["turnAbsoluteError"] / rows,
            "boostDisagreement": totals["boostDisagreements"] / rows,
            "loss": totals["lossSum"] / rows, "perScenario": per_scenario,
            "torchEvaluationWallMs": (time.perf_counter() - started) * 1000}
