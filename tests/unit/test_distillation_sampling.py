"""Actual canonical updates from sampled portable teacher rows, including restart."""

from dataclasses import replace
import copy
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import CanonicalGraph, CanonicalGraphManifest
from pyrl_trainer.distillation_client import DistillationFetch
from pyrl_trainer.distillation_replay import TeacherReplay, stream_partition
from pyrl_trainer.distillation_sampling import ReplayWindowSampler
from pyrl_trainer.distillation_transport import (
    DatasetManifest,
    DistillationContractError,
    TeacherChunk,
)
from pyrl_trainer.sequence_distillation import (
    CanonicalSequenceTrainer,
    SequenceBoundary,
    SequenceTrainingConfig,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures"
REFERENCE = json.loads((ROOT / "teacher_chunk_v1/reference.json").read_text(encoding="utf-8"))
DATA = (ROOT / "teacher_chunk_v1/chunk.bin").read_bytes()
GRAPH = ROOT / "distillation_graph_v1"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def model(device="cpu"):
    """Load a 72-input GRU8/Dense2 independently compiled/initialized by Rust."""
    return CanonicalGraph(
        CanonicalGraphManifest.read(GRAPH / "manifest.json"),
        (GRAPH / "weights.f32").read_bytes(),
        device,
    )


def replay_fixture(generations=(1, 2, 3, 4, 5), chunks_per_stream=12):
    """Mutate only test wire metadata; rows remain normalized frozen synthetic vectors."""
    chunks = []
    sequence = 0
    for generation in generations:
        for fragment in range(chunks_per_stream):
            sequence += 1
            data = bytearray(DATA)
            tail = 25 + data[24] + 96
            # Every third fragment ends a life; the next begins its new physical ID.
            life = fragment // 3
            start = 4 if fragment == 0 else 1 if fragment % 3 == 0 else 2
            end = 2 if fragment % 3 == 2 else 1
            struct.pack_into(
                "<QQIQBB",
                data,
                tail + 15,
                sequence,
                generation,
                119,
                generation * 100 + life,
                start,
                end,
            )
            first, stride = tail + 47, 25 + 72 * 4
            for index in range(3):
                ordinal = fragment * 3 + index + 1
                struct.pack_into(
                    "<QQ", data, first + index * stride, generation * 1000 + ordinal * 4, ordinal
                )
                flag = data[first + index * stride + 20] & ~12
                if index == 0 and start in (1, 4):
                    flag |= 8
                if index == 2 and end == 2:
                    flag |= 4
                data[first + index * stride + 20] = flag
                struct.pack_into(
                    "<f", data, first + index * stride + 25 + 21 * 4, 1.0 if flag & 8 else -1.0
                )
            chunks.append(TeacherChunk.decode(bytes(data)))
    dataset = DatasetManifest.from_object(
        {
            "schemaVersion": 1,
            "runSeed": 42,
            "contract": REFERENCE["contract"],
            "sensorContractJson": REFERENCE["sensorContractJson"],
            "state": "stopped",
            "productionWatermark": f"{sequence:016x}",
            "storageWatermark": f"{sequence:016x}",
            "resolvedThrough": f"{sequence:016x}",
            "retentionFloor": "0000000000000001",
            "nativeDroppedChunks": "0000000000000000",
            "nativeDroppedSamples": "0000000000000000",
            "nativeDroppedBytes": "0000000000000000",
            "spoolDroppedChunks": "0000000000000000",
            "retainedChunks": len(chunks),
            "retainedBytes": f"{sum(len(c.raw_bytes) for c in chunks):016x}",
            "error": None,
        }
    )
    replay = TeacherReplay()
    replay.admit(DistillationFetch(dataset, 0, sequence, tuple(chunks), ()))
    return replay


def config():
    """Small real BPTT windows cross chunk and physical-life boundaries."""
    return SequenceTrainingConfig(batch_size=8, burn_in_steps=3, bptt_steps=12)


@pytest.mark.parametrize("device", DEVICES)
def test_sampler_runs_real_canonical_optimizer_with_exact_partition_and_boundaries(device):
    """Raw requests, rare weights and life state rules reach real CPU/CUDA training."""
    graph = model(device)
    sampler = ReplayWindowSampler(replay_fixture(), graph.manifest, config(), seed=912)
    sampled = sampler.sample()
    sampled.batch.validate(graph, config())
    assert sampled.partition == "training"
    assert all(
        stream_partition(42, item.baseline_slot, item.generation) == "training"
        for item in sampled.windows
    )
    assert sampler.batch_bytes <= sampler.tensor_bytes
    assert sampler.index_bytes <= 64 * 1024 * 1024
    assert (sampled.batch.boundaries[0] != SequenceBoundary.CONTINUE).all()
    assert (sampled.batch.boundaries == SequenceBoundary.LIFE).any()
    before = graph.export_weights()
    updated = CanonicalSequenceTrainer(graph, config()).update(sampled.batch)
    assert updated.optimizer_updated and graph.export_weights() != before
    assert updated.next_state.grad_fn is None
    assert updated.training_samples > 0 and updated.burn_in_samples == 24
    validation = sampler.sample("validation")
    assert all(
        stream_partition(42, item.baseline_slot, item.generation) == "validation"
        for item in validation.windows
    )
    assert not (
        {item.generation for item in sampled.windows}
        & {item.generation for item in validation.windows}
    )


def test_sampler_checkpoint_reproduces_next_windows_and_all_tensor_bits():
    """Restoring PCG64 and the replay index reproduces every next sample bit."""
    graph = model()
    replay = replay_fixture()
    original = ReplayWindowSampler(replay, graph.manifest, config(), seed=817)
    original.sample()
    saved = original.checkpoint_fields()
    expected = original.sample()
    restored = ReplayWindowSampler(replay, graph.manifest, config(), seed=1)
    restored.restore(saved)
    actual = restored.sample()
    assert actual.windows == expected.windows and actual.index_sha256 == expected.index_sha256
    for name in vars(actual.batch):
        assert torch.equal(getattr(actual.batch, name), getattr(expected.batch, name))
    assert restored.checkpoint_fields() == original.checkpoint_fields()


def test_sampler_missing_data_changed_config_or_rng_rejects_without_partial_restore():
    """Exact restart requires the whole immutable replay snapshot and supported generator state."""
    graph = model()
    sampler = ReplayWindowSampler(replay_fixture(), graph.manifest, config())
    saved = sampler.checkpoint_fields()
    for other in (
        ReplayWindowSampler(replay_fixture((1, 2)), graph.manifest, config()),
        ReplayWindowSampler(replay_fixture(), graph.manifest, replace(config(), burn_in_steps=2)),
    ):
        with pytest.raises(DistillationContractError, match="snapshot mismatch"):
            other.restore(saved)
    before = sampler.checkpoint_fields()
    bad = copy.deepcopy(saved)
    bad["rng"]["state"]["inc"] = 2
    with pytest.raises(DistillationContractError):
        sampler.restore(bad)
    assert sampler.checkpoint_fields() == before


def test_sampler_no_eligible_history_bad_contract_or_budget_never_advances_rng():
    """Admission failure remains isolated from already owned optimizer/replay state."""
    graph = model()
    replay = replay_fixture((1,), chunks_per_stream=1)
    sampler = ReplayWindowSampler(replay, graph.manifest, config())
    saved = sampler.checkpoint_fields()
    with pytest.raises(DistillationContractError, match="after burn-in"):
        sampler.sample()
    assert sampler.checkpoint_fields() == saved
    with pytest.raises(DistillationContractError, match="tensor RAM"):
        ReplayWindowSampler(replay, graph.manifest, config(), tensor_bytes=2 * 1024 * 1024)
    altered = replace(graph.manifest, control_interval_seconds=0.03)
    with pytest.raises(DistillationContractError, match="contract mismatch"):
        ReplayWindowSampler(replay, altered, config()).sample()


@pytest.mark.parametrize("device", DEVICES)
def test_native_reference_inference_matches_every_output_and_continuation(device):
    """The new GRU window graph preserves its independent Rust 256-delivery reference."""
    graph = model(device)
    observations = np.frombuffer((GRAPH / "observations.f32").read_bytes(), dtype="<f4").copy()
    observations = torch.from_numpy(observations.reshape(256, 72)).to(device)
    expected_output = np.frombuffer((GRAPH / "outputs.f32").read_bytes(), dtype="<f4").copy()
    expected_state = np.frombuffer((GRAPH / "states.f32").read_bytes(), dtype="<f4").copy()
    outputs, states = [], []
    state = graph.initial_state(1)
    with torch.no_grad():
        for row in observations:
            output, state = graph(row.unsqueeze(0), state)
            outputs.append(output.cpu())
            states.append(state.cpu())
    torch.testing.assert_close(
        torch.cat(outputs),
        torch.from_numpy(expected_output.reshape(256, 2)),
        atol=1e-4 if device == "cuda" else 1e-5,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        torch.cat(states), torch.from_numpy(expected_state.reshape(256, 8)), atol=1e-3, rtol=1e-3
    )
