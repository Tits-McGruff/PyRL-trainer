"""CPU/CUDA semantic parity against immutable whole-graph Rust sequences."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pyrl_trainer.canonical_graph import (
    CanonicalGraph,
    CanonicalGraphError,
    CanonicalGraphManifest,
    distillation_loss,
    _affine_dot,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "canonical_graph"
CASES = [
    (record["case"], record["initialState"])
    for record in json.loads((ROOT / "index.json").read_text(encoding="utf-8"))["cases"]
]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fixture(case, initial, device):
    """Verify recorded bytes before constructing any independent model views."""
    root = ROOT / case
    reference = root / initial
    manifest = CanonicalGraphManifest.read(reference / "manifest.json")
    receipt = json.loads((reference / "receipt.json").read_text(encoding="utf-8"))
    for filename, digest in [
        ("weights.f32", "weightsSha256"),
        ("outputs.f32", "outputsSha256"),
        ("states.f32", "statesSha256"),
    ]:
        assert hashlib.sha256((reference / filename).read_bytes()).hexdigest() == receipt[digest]
    weights = (reference / "weights.f32").read_bytes()
    model = CanonicalGraph(manifest, weights, device=device)
    observations = np.frombuffer((root / "observations.f32").read_bytes(), dtype="<f4").copy()
    assert hashlib.sha256(observations.tobytes()).hexdigest() == receipt["observationsSha256"]
    observations = torch.from_numpy(observations.reshape(256, manifest.input_size)).to(device)
    if initial == "nonzero":
        state = (
            torch.from_numpy(
                np.frombuffer((root / "initial-state.f32").read_bytes(), dtype="<f4").copy()
            )
            .to(device)
            .unsqueeze(0)
        )
    else:
        state = model.initial_state(1)
    expected = torch.from_numpy(
        np.frombuffer((reference / "outputs.f32").read_bytes(), dtype="<f4").copy()
    ).reshape(256, 2)
    expected_states = torch.from_numpy(
        np.frombuffer((reference / "states.f32").read_bytes(), dtype="<f4").copy()
    ).reshape(256, manifest.total_state_size)
    return model, observations, state, expected, expected_states


@pytest.mark.parametrize("case,initial", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_every_operation_and_256_step_state_parity(case, initial, device):
    """Single-step and long-horizon tolerances stay fixed for both outputs and state."""
    model, observations, state, expected, expected_states = _fixture(case, initial, device)
    before_weights = model.export_weights()
    collected, states = [], []
    with torch.no_grad():
        for row in observations:
            result, state = model(row.unsqueeze(0), state)
            collected.append(result.cpu())
            states.append(state.cpu())
    actual, actual_states = torch.cat(collected), torch.cat(states)
    single_atol = 1e-4 if device == "cuda" else 1e-5
    torch.testing.assert_close(actual[0], expected[0], atol=single_atol, rtol=1e-4)
    torch.testing.assert_close(actual_states[0], expected_states[0], atol=single_atol, rtol=1e-4)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_states, expected_states, atol=1e-3, rtol=1e-3)
    assert model.export_weights() == before_weights
    assert len(list(model.parameters())) == 1
    assert all(result.dtype == torch.float32 for result in (actual, actual_states))
    errors = []
    for step in [1, 64, 128, 256]:
        output_error = (actual[:step] - expected[:step]).abs().max().item()
        state_error = (
            (actual_states[:step] - expected_states[:step]).abs().max().item()
            if actual_states.numel()
            else 0.0
        )
        errors.append({"step": step, "outputMaxAbs": output_error, "stateMaxAbs": state_error})
    print(json.dumps({"case": case, "initial": initial, "device": device, "errorGrowth": errors}))


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_updates_flat_views_and_exact_method_v1_loss(device):
    """All canonical recurrent and affine nodes receive finite autograd gradients."""
    model, observations, state, _expected, _states = _fixture("mixed", "nonzero", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = model.export_weights()
    losses = []
    for row in observations[:64]:
        output, state = model(row.unsqueeze(0), state)
        turn = row[:1] * 0.2
        boost = (row[1:2] > 0).to(torch.float32)
        weight = torch.full_like(turn, 4.0)
        losses.append(distillation_loss(output, turn, boost, weight))
    loss = torch.cat(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = model.flat_parameters.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    for node in model.manifest.nodes:
        for view in node.parameters:
            assert view.tensor(gradient).abs().sum() > 0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    assert model.export_weights() != before
    cloned = CanonicalGraph(model.manifest, model.export_weights(), device=device)
    assert cloned.export_weights() == model.export_weights()


@pytest.mark.parametrize(
    "fault",
    ["equation", "sensor", "overlap", "state-order", "port", "outputs", "hidden", "cadence"],
)
def test_malformed_manifest_rejects_before_tensor_construction(fault):
    """Corrupt semantic versions, views and ordering fail at manifest ingress."""
    raw = json.loads((ROOT / "mixed/zero/manifest.json").read_text(encoding="utf-8"))
    if fault == "equation":
        raw["equations"]["revision"] = 2
    elif fault == "sensor":
        raw["sensorCanonicalJson"] += " "
    elif fault == "overlap":
        raw["nodes"][1]["parameterBlocks"][1]["offset"] = raw["nodes"][1]["parameterBlocks"][0][
            "offset"
        ]
    elif fault == "state-order":
        raw["nodes"][2]["stateBlocks"][0]["offset"] += 1
    elif fault == "port":
        raw["nodes"][2]["inputs"][0]["fromPort"] = 9
    elif fault == "outputs":
        raw["outputs"][0]["offset"] = 1
    elif fault == "hidden":
        raw["nodes"][2]["hiddenSize"] += 1
    else:
        raw["deployment"]["controlIntervalSeconds"] = 1.0
    with pytest.raises(CanonicalGraphError):
        CanonicalGraphManifest.from_bytes(json.dumps(raw).encode())


def test_bounds_weights_and_immutable_metadata():
    """No partial/nonfinite weights, oversized metadata or accidental aliasing."""
    root = ROOT / "dense/zero"
    encoded = (root / "manifest.json").read_bytes()
    manifest = CanonicalGraphManifest.from_bytes(encoded)
    weights = (root / "weights.f32").read_bytes()
    with pytest.raises(CanonicalGraphError):
        CanonicalGraph(manifest, weights[:-1])
    values = np.frombuffer(weights, dtype="<f4").copy()
    values[0] = np.nan
    with pytest.raises(CanonicalGraphError):
        CanonicalGraph(manifest, values.tobytes())
    with pytest.raises(CanonicalGraphError):
        CanonicalGraphManifest.from_bytes(b" " * (1024 * 1024 + 1))
    copied = copy.deepcopy(manifest)
    assert copied == manifest
    model = CanonicalGraph(manifest, weights)
    with pytest.raises(CanonicalGraphError):
        model(torch.zeros(1, manifest.input_size, dtype=torch.float64))
    with pytest.raises(CanonicalGraphError):
        model(torch.zeros(1, manifest.input_size), torch.zeros(1, 1))


def test_loss_uses_raw_boost_and_has_fixed_temperature_and_event_weights():
    """The exact objective has no deployed sigmoid/value head or realized-boost label."""
    outputs = torch.tensor([[0.2, 0.35], [-0.1, 0.45]], requires_grad=True)
    turn = torch.tensor([0.0, 0.1])
    boost = torch.tensor([0.0, 1.0])
    weights = torch.tensor([1.0, 4.0])
    result = distillation_loss(outputs, turn, boost, weights)
    expected = weights * (
        (outputs[:, 0] - turn).square()
        + torch.nn.functional.binary_cross_entropy_with_logits(
            (outputs[:, 1] - 0.35) / 0.10, boost, reduction="none"
        )
    )
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-7)
    result.mean().backward()
    assert outputs.grad[0, 1] > 0
    assert outputs.grad[1, 1] < 0


@pytest.mark.parametrize("field", ["schemaVersion", "layoutVersion", "outputSize"])
def test_noninteger_structural_revisions_reject(field):
    """Numeric-looking booleans/floats cannot impersonate versioned integer fields."""
    raw = json.loads((ROOT / "dense/zero/manifest.json").read_text(encoding="utf-8"))
    raw[field] = float(raw[field])
    with pytest.raises(CanonicalGraphError):
        CanonicalGraphManifest.from_bytes(json.dumps(raw).encode())


def test_distillation_labels_cannot_broadcast_or_promote_precision():
    """Malformed labels fail instead of silently changing weighting or FP32 semantics."""
    output, label = torch.zeros(2, 2), torch.zeros(2)
    with pytest.raises(CanonicalGraphError):
        distillation_loss(output, label[:1], label, label)
    with pytest.raises(CanonicalGraphError):
        distillation_loss(output, label.double(), label, label)


@pytest.mark.parametrize("case,initial", CASES)
def test_rust_json_value_member_order_preserves_canonical_equations_and_layout(case, initial):
    """Native portable envelopes sort JSON object members without changing deployed semantics."""
    path = ROOT / case / initial / "manifest.json"
    expected = CanonicalGraphManifest.read(path)
    raw = json.loads(path.read_bytes())
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    assert list(json.loads(encoded)["equations"]) == sorted(raw["equations"])
    actual = CanonicalGraphManifest.from_bytes(encoded)
    assert actual.nodes == expected.nodes
    assert actual.outputs == expected.outputs
    assert actual.total_parameters == expected.total_parameters
    assert actual.total_state_size == expected.total_state_size
    assert actual.layout_sha256 == expected.layout_sha256
    assert actual.sensor_sha256 == expected.sensor_sha256


@pytest.mark.parametrize("fault", ["missing", "extra", "formula", "digest"])
def test_reordered_equations_still_require_exact_fields_values_and_digest(fault):
    """Object-order normalization cannot admit an omitted field or a changed equation."""
    raw = json.loads((ROOT / "mixed/zero/manifest.json").read_bytes())
    if fault == "missing":
        del raw["equations"]["dtype"]
    elif fault == "extra":
        raw["equations"]["extra"] = "ignored"
    elif fault == "formula":
        raw["equations"]["gru"] += " changed"
    else:
        raw["equationsSha256"] = "0" * 64
    with pytest.raises(CanonicalGraphError, match="canonical"):
        CanonicalGraphManifest.from_bytes(json.dumps(raw, sort_keys=True).encode())


@pytest.mark.parametrize("width", [32, 51, 128, 147, 248, 256, 264, 504, 512])
@pytest.mark.parametrize("batch", [1, 8])
def test_compensated_cpu_affine_accuracy_and_gradients_with_strided_views(width, batch):
    """Compare real Float32 forward/backward with an independent Float64 oracle."""
    generator = torch.Generator().manual_seed(617)
    values = torch.randn(batch, width, generator=generator, requires_grad=True)
    storage = torch.randn(17, width + 1, generator=generator, requires_grad=True)
    weights = storage[:, :width]
    assert not weights.is_contiguous()
    cotangent = torch.randn(batch, 17, generator=generator)
    actual = _affine_dot(values, weights)
    expected_values = values.detach().double().requires_grad_(True)
    expected_storage = storage.detach().double().requires_grad_(True)
    expected = expected_values @ expected_storage[:, :width].T
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=5e-6, rtol=1e-5)
    (actual * cotangent).sum().backward()
    (expected * cotangent.double()).sum().backward()
    for actual_gradient, expected_gradient in [
        (values.grad, expected_values.grad), (storage.grad, expected_storage.grad)
    ]:
        assert actual_gradient.dtype == torch.float32
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient.double(), expected_gradient, atol=5e-6, rtol=1e-5
        )
    assert torch.count_nonzero(storage.grad[:, -1]) == 0


def test_compensated_cpu_affine_recovers_cancellation_and_preserves_nonfinite():
    """Recover cross-block small terms while retaining ordinary infinity/NaN results."""
    weights = torch.zeros(1, 512)
    weights[0, 0], weights[0, -1] = 1e8, -1e8
    weights[0, 16:32] = 1
    values = torch.ones(1, 512)
    assert _affine_dot(values, weights).item() == 16
    overflowing = torch.full((1, 512), torch.finfo(torch.float32).max)
    assert torch.isposinf(_affine_dot(values, overflowing)).all()
    overflowing[0, -1] = -torch.inf
    assert torch.isnan(_affine_dot(values, overflowing)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_short_cpu_and_cuda_affine_keep_direct_operation(device):
    """CUDA and short CPU rows preserve the existing matrix result exactly."""
    generator = torch.Generator().manual_seed(616)
    for width in ([31] if device == "cpu" else [31, 32, 504]):
        values = torch.randn(8, width, generator=generator).to(device)
        weights = torch.randn(17, width, generator=generator).to(device)
        assert torch.equal(_affine_dot(values, weights), values @ weights.T)
