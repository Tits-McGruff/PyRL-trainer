"""Differentiable Float32 views of Rust's canonical deployed graph manifest.

Rust is the compiler, initializer and deployment validator. This module never
packs weights or substitutes stock recurrent modules. All learned values belong
to one flat Parameter; manifest views address that Parameter directly.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Optional, Tuple

from slither_contracts import CONSTANTS, models as wire

import numpy as np
import torch
from torch import nn

MAX_MANIFEST_BYTES = CONSTANTS["maxManifestBytes"]
EQUATION_FIELDS_V1 = tuple(CONSTANTS["equationFieldOrder"])
EQUATIONS_SHA256_V1 = hashlib.sha256(
    CONSTANTS["equationHashDomain"].encode() + json.dumps(
        {key: CONSTANTS["equations"][key] for key in EQUATION_FIELDS_V1},
        ensure_ascii=False, separators=(",", ":"),
    ).encode()
).hexdigest()


class CanonicalGraphError(ValueError):
    """Malformed, unsupported or incompatible canonical training contract."""


def _integer(value, maximum, minimum=0):
    """Reject booleans, nonintegers and dimensions outside native ceilings."""
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise CanonicalGraphError("invalid bounded canonical integer")
    return value


def _keys(value, names):
    """Require an exact versioned object shape rather than silently ignoring fields."""
    expected = names.__annotations__ if isinstance(names, type) else names.split()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CanonicalGraphError("unsupported canonical object fields")


def _identifier(value):
    """Apply the native raw-UTF8 identifier limit without Unicode normalization."""
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= 64:
        raise CanonicalGraphError("invalid canonical node identifier")
    return value


def _digest(value):
    """Require the native lowercase SHA-256 representation."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CanonicalGraphError("invalid canonical digest")
    return value


@dataclass(frozen=True)
class ParameterView:
    """One logical view with Rust-provided element offsets and strides."""

    name: str
    offset: int
    length: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]

    def tensor(self, flat: torch.Tensor) -> torch.Tensor:
        """Create a differentiable view without allocating another parameter."""
        return flat.as_strided(self.shape, self.strides, self.offset)


@dataclass(frozen=True)
class CanonicalNode:
    """Validated immutable node descriptors in Rust's topological order."""

    identifier: str
    operation: str
    input_size: int
    output_sizes: Tuple[int, ...]
    hidden_sizes: Tuple[int, ...]
    hidden_size: Optional[int]
    inputs: Tuple[Tuple[str, int], ...]
    parameters: Tuple[ParameterView, ...]
    states: Tuple[Tuple[str, int, int], ...]


def _parameter_views(raw, node, owners):
    """Check addressed elements, including disjoint interleaved affine biases."""
    if not isinstance(raw, list) or len(raw) > 18:
        raise CanonicalGraphError("invalid canonical parameter block count")
    result = []
    names = set()
    for block in raw:
        _keys(block, wire.TrainingParameterBlock)
        name = block["name"]
        if not isinstance(name, str) or not 1 <= len(name) <= 32 or name in names:
            raise CanonicalGraphError("invalid or duplicate parameter name")
        names.add(name)
        shape = tuple(_integer(size, 512, 1) for size in block["shape"])
        strides = tuple(_integer(size, 513, 1) for size in block["strides"])
        if len(shape) not in (1, 2) or len(strides) != len(shape):
            raise CanonicalGraphError("invalid parameter view rank")
        count = _integer(block["length"], 1_000_000, 1)
        offset = _integer(block["offset"], len(owners))
        if count != math.prod(shape):
            raise CanonicalGraphError("parameter shape/count mismatch")
        start, stop = node["parameterOffset"], node["parameterOffset"] + node["parameterLength"]
        for logical in range(count):
            index = (
                offset + logical * strides[0]
                if len(shape) == 1
                else (
                    offset + (logical // shape[1]) * strides[0] + (logical % shape[1]) * strides[1]
                )
            )
            if not start <= index < stop or index >= len(owners) or owners[index]:
                raise CanonicalGraphError("overlapping or out-of-range parameter view")
            owners[index] = 1
        result.append(ParameterView(name, offset, count, shape, strides))
    return tuple(result)


def _expected_views(operation, input_size, output_sizes, hidden_sizes, hidden_size):
    """Validate logical operator shapes; never calculate packing offsets."""
    expected = {}
    if operation in ("Dense", "MLP"):
        source = input_size
        for layer, destination in enumerate(hidden_sizes + (sum(output_sizes),)):
            expected[f"layer{layer}.W"] = (destination, source)
            expected[f"layer{layer}.b"] = (destination,)
            source = destination
    elif operation in ("GRU", "LSTM", "RRU"):
        gates = {"GRU": "zrh", "LSTM": "ifog", "RRU": "cr"}[operation]
        for gate in gates:
            expected[f"W{gate}"] = (hidden_size, input_size)
            expected[f"U{gate}"] = (hidden_size, hidden_size)
            expected[f"b{gate}"] = (hidden_size,)
    return expected


def _state_views(raw, operation, hidden, state_range, owners):
    """Check declared hidden/cell semantics and exact recurrent ownership."""
    offset, length = state_range
    expected = (
        ("hidden", "cell")
        if operation == "LSTM"
        else (("hidden",) if operation in ("GRU", "RRU") else ())
    )
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise CanonicalGraphError("invalid recurrent state blocks")
    if not expected:
        if offset is not None or length != 0:
            raise CanonicalGraphError("nonrecurrent node owns state")
        return ()
    if length != hidden * len(expected):
        raise CanonicalGraphError("recurrent state length mismatch")
    offset = _integer(offset, len(owners))
    result = []
    for name, block in zip(expected, raw):
        _keys(block, wire.TrainingStateBlock)
        _integer(block["offset"], len(owners))
        _integer(block["length"], 512, 1)
        if block["name"] != name or block["offset"] != offset or block["length"] != hidden:
            raise CanonicalGraphError("hidden/cell layout mismatch")
        if offset is None or offset + hidden > len(owners):
            raise CanonicalGraphError("state view exceeds total state")
        for index in range(offset, offset + hidden):
            if owners[index]:
                raise CanonicalGraphError("overlapping recurrent states")
            owners[index] = 1
        result.append((name, offset, hidden))
        offset += hidden
    return tuple(result)


def _node(raw, prior, parameter_owners, state_owners):
    """Validate one DAG node against prior source ports and admitted flat views."""
    # pylint: disable=too-many-branches,too-many-statements
    _keys(
        raw,
        (
            wire.TrainingNode
        ),
    )
    identifier = _identifier(raw["id"])
    operation = raw["operation"]
    if identifier in prior or operation not in (
        "Input",
        "Dense",
        "MLP",
        "GRU",
        "LSTM",
        "RRU",
        "Concat",
        "Split",
    ):
        raise CanonicalGraphError("duplicate node or unsupported operation")
    input_size = _integer(raw["inputSize"], 512, 1)
    output_sizes = tuple(_integer(size, 512, 1) for size in raw["outputSizes"])
    hidden_sizes = tuple(_integer(size, 512, 1) for size in raw["hiddenSizes"])
    if not 1 <= len(output_sizes) <= 16 or sum(output_sizes) > 512 or len(hidden_sizes) > 8:
        raise CanonicalGraphError("invalid node widths")
    if operation != "Split" and len(output_sizes) != 1:
        raise CanonicalGraphError("multiple outputs on nonsplit node")
    if operation != "MLP" and hidden_sizes:
        raise CanonicalGraphError("hidden layers on non-MLP node")
    hidden = raw["hiddenSize"]
    if operation in ("GRU", "LSTM", "RRU"):
        hidden = _integer(hidden, 512, 1)
        if output_sizes != (hidden,):
            raise CanonicalGraphError("recurrent hidden/output mismatch")
    elif hidden is not None:
        raise CanonicalGraphError("hidden state on nonrecurrent node")
    inputs = []
    incoming_width = 0
    if not isinstance(raw["inputs"], list) or len(raw["inputs"]) > 128:
        raise CanonicalGraphError("invalid incoming edges")
    for ordinal, edge in enumerate(raw["inputs"]):
        _keys(edge, wire.TrainingInput)
        source = _identifier(edge["fromId"])
        _integer(edge["inputOrdinal"], 127)
        port = _integer(edge["fromPort"], 15)
        if (
            edge["inputOrdinal"] != ordinal
            or source not in prior
            or port >= len(prior[source].output_sizes)
        ):
            raise CanonicalGraphError("invalid topological edge/port/order")
        inputs.append((source, port))
        incoming_width += prior[source].output_sizes[port]
    if operation == "Input":
        if inputs or output_sizes != (input_size,):
            raise CanonicalGraphError("invalid graph input")
    elif not inputs or incoming_width != input_size:
        raise CanonicalGraphError("resolved input width mismatch")
    if operation in ("Concat", "Split") and sum(output_sizes) != input_size:
        raise CanonicalGraphError("routing node width mismatch")
    _integer(raw["parameterOffset"], len(parameter_owners))
    _integer(raw["parameterLength"], len(parameter_owners))
    _integer(raw["stateLength"], len(state_owners))
    views = _parameter_views(raw["parameterBlocks"], raw, parameter_owners)
    if {view.name: view.shape for view in views} != _expected_views(
        operation, input_size, output_sizes, hidden_sizes, hidden
    ):
        raise CanonicalGraphError("parameter names/shapes differ from canonical equations")
    states = _state_views(
        raw["stateBlocks"],
        operation,
        hidden,
        (raw["stateOffset"], raw["stateLength"]),
        state_owners,
    )
    return CanonicalNode(
        identifier,
        operation,
        input_size,
        output_sizes,
        hidden_sizes,
        hidden,
        tuple(inputs),
        views,
        states,
    )


@dataclass(frozen=True)
class CanonicalGraphManifest:
    """Bounded immutable runtime facts; Rust remains the layout digest verifier."""

    canonical_json: bytes
    input_size: int
    total_parameters: int
    total_state_size: int
    nodes: Tuple[CanonicalNode, ...]
    outputs: Tuple[Tuple[str, int], ...]
    life_state_rule: str
    control_interval_seconds: float
    layout_sha256: str
    sensor_sha256: str

    @classmethod
    def from_bytes(cls, encoded: bytes):
        """Parse a Rust manifest and reject unsupported or unsafe tensor descriptions."""
        if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_MANIFEST_BYTES:
            raise CanonicalGraphError("manifest exceeds its byte limit")
        try:
            raw = json.loads(encoded)
            return cls._from_object(raw, encoded)
        except (
            KeyError,
            TypeError,
            IndexError,
            UnicodeError,
            RecursionError,
            json.JSONDecodeError,
        ) as error:
            raise CanonicalGraphError("malformed canonical manifest") from error

    @classmethod
    def _from_object(cls, raw, encoded):
        """Resolve nodes in supplied compiler order and prove full numeric ownership."""
        _keys(
            raw,
            (
                wire.CanonicalTrainingGraphManifestV1
            ),
        )
        if (
            _integer(raw["schemaVersion"], 1, 1) != 1
            or _integer(raw["layoutVersion"], 1, 1) != 1
            or _integer(raw["outputSize"], 2, 2) != 2
        ):
            raise CanonicalGraphError("unsupported canonical manifest revision")
        _keys(raw["equations"], " ".join(EQUATION_FIELDS_V1))
        equations = json.dumps(
            {name: raw["equations"][name] for name in EQUATION_FIELDS_V1},
            ensure_ascii=False, separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(CONSTANTS["equationHashDomain"].encode() + equations).hexdigest()
        if digest != EQUATIONS_SHA256_V1 or raw["equationsSha256"] != digest:
            raise CanonicalGraphError("unsupported canonical equations")
        sensor_bytes = raw["sensorCanonicalJson"].encode("utf-8")
        sensor_hash = hashlib.sha256(
            CONSTANTS["sensorHashDomain"].encode() + sensor_bytes
        ).hexdigest()
        if len(sensor_bytes) > 65536 or raw["sensorSha256"] != sensor_hash:
            raise CanonicalGraphError("sensor canonical digest mismatch")
        input_size = _integer(raw["inputSize"], 504, 1)
        sensor = json.loads(sensor_bytes)
        if sensor["inputSize"] != input_size or len(sensor["order"]) != input_size:
            raise CanonicalGraphError("sensor width/order mismatch")
        parameters = _integer(raw["totalParameters"], 1_000_000)
        state_size = _integer(raw["totalStateSize"], 16384)
        parameter_owners, state_owners = bytearray(parameters), bytearray(state_size)
        if not isinstance(raw["nodes"], list) or not 1 <= len(raw["nodes"]) <= 64:
            raise CanonicalGraphError("invalid node count")
        nodes = {}
        parameter_cursor, state_cursor = 0, 0
        for node_raw in raw["nodes"]:
            node = _node(node_raw, nodes, parameter_owners, state_owners)
            if node_raw["parameterOffset"] != parameter_cursor:
                raise CanonicalGraphError("noncontiguous canonical node parameters")
            if node.states and node_raw["stateOffset"] != state_cursor:
                raise CanonicalGraphError("noncontiguous canonical node states")
            parameter_cursor += node_raw["parameterLength"]
            state_cursor += node_raw["stateLength"]
            nodes[node.identifier] = node
        if (
            raw["topologicalOrder"] != list(nodes)
            or parameter_cursor != parameters
            or state_cursor != state_size
            or parameter_owners.count(1) != parameters
            or state_owners.count(1) != state_size
        ):
            raise CanonicalGraphError("incomplete canonical ownership/order")
        inputs = [node for node in nodes.values() if node.operation == "Input"]
        if len(inputs) != 1 or inputs[0].input_size != input_size:
            raise CanonicalGraphError("input contract mismatch")
        outputs = _outputs(raw["outputs"], nodes)
        cadence, life_rule = _deployment(raw["deployment"])
        return cls(
            encoded,
            input_size,
            parameters,
            state_size,
            tuple(nodes.values()),
            outputs,
            life_rule,
            cadence,
            _digest(raw["layoutSha256"]),
            _digest(raw["sensorSha256"]),
        )

    @classmethod
    def read(cls, path: Path):
        """Bound actual reads even if a file grows after stat."""
        with Path(path).open("rb") as stream:
            return cls.from_bytes(stream.read(MAX_MANIFEST_BYTES + 1))


def _outputs(raw, nodes):
    """Validate exact flattened turn/boost output port ordering."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
        raise CanonicalGraphError("invalid graph outputs")
    result, offset = [], 0
    for output in raw:
        _keys(output, wire.TrainingOutput)
        identifier = _identifier(output["nodeId"])
        _integer(output["offset"], 2)
        _integer(output["length"], 2, 1)
        port = _integer(output["port"], 15)
        if identifier not in nodes or port >= len(nodes[identifier].output_sizes):
            raise CanonicalGraphError("invalid graph output port")
        width = nodes[identifier].output_sizes[port]
        if output["offset"] != offset or output["length"] != width:
            raise CanonicalGraphError("graph output range mismatch")
        result.append((identifier, port))
        offset += width
    if offset != 2:
        raise CanonicalGraphError("graph must output turn and boost")
    return tuple(result)


def _deployment(raw):
    """Reject unsupported motor interpretation without conflating sensor identity."""
    _keys(
        raw,
        (
            wire.TrainingDeploymentManifest
        ),
    )
    interval = raw["controlIntervalSeconds"]
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(interval)
        or not 0.008 <= interval <= 0.06
    ):
        raise CanonicalGraphError("unsupported deployment cadence")
    if (
        raw["fixedStepHz"],
        raw["turnOutputIndex"],
        raw["boostOutputIndex"],
        raw["boostThreshold"],
    ) != (60, 0, 1, 0.35):
        raise CanonicalGraphError("unsupported output mapping")
    if raw["lifeStateRule"] not in ("reset", "preserve"):
        raise CanonicalGraphError("unsupported recurrent life rule")
    return interval, raw["lifeStateRule"]


def _compensated_cpu_dot(values, weights):
    """Reduce bounded 16-term CPU partials in a Float32 error-corrected tree.

    Callers select widths 128..512. Padding to the next power of two permits at
    most 32 partial matrices and five tree levels. A nonfinite partial cannot
    become finite through later additions, so one final guard preserves ordinary
    infinity/NaN propagation without per-level finite masks. All products,
    sums and corrections remain Float32.
    """
    width = values.shape[-1]
    padded = 1 << (width - 1).bit_length()
    if padded != width:
        values = torch.nn.functional.pad(values, (0, padded - width))
        weights = torch.nn.functional.pad(weights, (0, padded - width))
    blocks = padded // 16
    left = values.reshape(values.shape[0], blocks, 16).transpose(0, 1)
    right = weights.reshape(weights.shape[0], blocks, 16).permute(1, 2, 0)
    partial = torch.bmm(left, right)
    correction = None
    while partial.shape[0] > 1:
        left, right = partial[0::2], partial[1::2]
        updated = left + right
        error = torch.where(
            left.abs() >= right.abs(),
            (left - updated) + right,
            (right - updated) + left,
        )
        correction = (
            error if correction is None
            else (correction[0::2] + correction[1::2]) + error
        )
        partial = updated
    return torch.where(partial[0].isfinite(), partial[0] + correction[0], partial[0])


class _CpuAffineDot(torch.autograd.Function):  # pylint: disable=abstract-method
    """Keep only original inputs for the analytic derivative of a linear dot.

    The correction improves Float32 forward rounding; the mathematical operation
    remains values @ weights.T. Saving the original two views avoids retaining
    the internal reduction tree for every TBPTT delivery.
    """

    @staticmethod
    def forward(ctx, values, weights):  # pylint: disable=arguments-differ
        """Evaluate the compensated CPU sum without recording its internal graph."""
        ctx.save_for_backward(values, weights)
        return _compensated_cpu_dot(values, weights)

    @staticmethod
    def backward(ctx, gradient):  # pylint: disable=arguments-differ
        """Return ordinary Float32 matrix adjoints for the requested input views."""
        values, weights = ctx.saved_tensors
        # Backward may run after forward's autocast guard has left its scope.
        with torch.autocast(device_type="cpu", enabled=False):
            values_gradient = gradient @ weights if ctx.needs_input_grad[0] else None
            weights_gradient = gradient.T @ values if ctx.needs_input_grad[1] else None
        return values_gradient, weights_gradient


def _affine_dot(values, weights):
    """Use accurate bounded CPU long rows; retain direct short/CUDA operations.

    CPU widths below 128 have short BLAS reductions. Longer rows use 16-term
    partials, compensated Float32 reduction and the analytic matrix derivative.
    Canonical callers validate dimensions/dtype and disable autocast; this helper
    introduces no parameters, recurrent state, compiler cache or RNG.
    """
    if values.device.type != "cpu" or values.shape[-1] < 128:
        return values @ weights.T
    return _CpuAffineDot.apply(values, weights)


class CanonicalGraph(nn.Module):
    """Exact differentiable canonical DAG with one flat Float32 Parameter."""

    def __init__(self, manifest: CanonicalGraphManifest, weights: bytes, device="cpu"):
        super().__init__()
        self.manifest = manifest
        if not isinstance(weights, bytes) or len(weights) != manifest.total_parameters * 4:
            raise CanonicalGraphError("canonical weight byte count mismatch")
        values = np.frombuffer(weights, dtype="<f4").copy()
        if not np.isfinite(values).all():
            raise CanonicalGraphError("nonfinite canonical weights")
        self.flat_parameters = nn.Parameter(
            torch.from_numpy(values).to(device=device, dtype=torch.float32)
        )
        if self.flat_parameters.device.type == "cuda":
            configure_distillation_precision()
        self._views = {
            node.identifier: {view.name: view for view in node.parameters}
            for node in manifest.nodes
        }

    def export_weights(self) -> bytes:
        """Export exact finite little-endian Float32 bytes, without pickle or model heads."""
        values = self.flat_parameters.detach().cpu().numpy()  # pylint: disable=not-callable
        values = values.astype("<f4", copy=False)
        if not np.isfinite(values).all():
            raise CanonicalGraphError("nonfinite trained canonical weights")
        return values.tobytes()

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """Allocate zero recurrence with the same device/dtype as canonical weights."""
        return self.flat_parameters.new_zeros(
            (_integer(batch_size, 4096, 1), self.manifest.total_state_size)
        )

    def _weights(self, node):
        """Create logical views lazily so device moves retain current storage."""
        return {
            name: view.tensor(self.flat_parameters)
            for name, view in self._views[node.identifier].items()
        }

    @staticmethod
    def _recurrent(node, values, state, weights):
        """Custom recurrent equations preserve Rust's gate convention and state order."""
        previous = state.narrow(-1, node.states[0][1], node.hidden_size)

        def gate(name, recurrent=previous):
            return (
                _affine_dot(values, weights[f"W{name}"])
                + _affine_dot(recurrent, weights[f"U{name}"])
            ) + weights[f"b{name}"]

        if node.operation == "GRU":
            update, reset = torch.sigmoid(gate("z")), torch.sigmoid(gate("r"))
            candidate = torch.tanh(gate("h", reset * previous))
            hidden = (1 - update) * previous + update * candidate
            return hidden, (hidden,)
        if node.operation == "RRU":
            candidate, reset = torch.tanh(gate("c")), torch.sigmoid(gate("r"))
            hidden = (1 - reset) * previous + reset * candidate
            return hidden, (hidden,)
        cell = state.narrow(-1, node.states[1][1], node.hidden_size)
        updated_cell = torch.sigmoid(gate("f")) * cell + torch.sigmoid(gate("i")) * torch.tanh(
            gate("g")
        )
        hidden = torch.sigmoid(gate("o")) * torch.tanh(updated_cell)
        return hidden, (hidden, updated_cell)

    def forward(self, observations: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Advance exactly one delivered observation for every batch row."""
        if observations.ndim != 2 or observations.shape[1] != self.manifest.input_size:
            raise CanonicalGraphError("invalid canonical observation shape")
        if (
            observations.dtype != torch.float32
            or observations.device != self.flat_parameters.device
        ):
            raise CanonicalGraphError("canonical observations must use model device and Float32")
        if state is None:
            state = self.initial_state(observations.shape[0])
        if (
            state.shape != (observations.shape[0], self.manifest.total_state_size)
            or state.dtype != torch.float32
            or state.device != observations.device
        ):
            raise CanonicalGraphError("invalid canonical recurrent state")
        if self.flat_parameters.dtype != torch.float32:
            raise CanonicalGraphError("canonical model must remain Float32")
        with torch.autocast(device_type=observations.device.type, enabled=False):
            return self._evaluate(observations, state)

    def _evaluate(self, observations, state):
        """Execute declared ports with Float32 autocast disabled for the entire DAG."""
        if observations.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise CanonicalGraphError("distillation parity requires TF32 disabled")
        ports, next_states = {}, []
        for node in self.manifest.nodes:
            if node.operation == "Input":
                ports[node.identifier] = (observations,)
                continue
            incoming = [ports[source][port] for source, port in node.inputs]
            values = incoming[0] if len(incoming) == 1 else torch.cat(incoming, dim=-1)
            weights = self._weights(node)
            if node.operation in ("Dense", "MLP"):
                for layer in range(len(node.hidden_sizes) + 1):
                    values = torch.tanh(
                        _affine_dot(values, weights[f"layer{layer}.W"]) + weights[f"layer{layer}.b"]
                    )
            elif node.operation in ("GRU", "LSTM", "RRU"):
                values, updated = self._recurrent(node, values, state, weights)
                next_states.extend(updated)
            ports[node.identifier] = (
                torch.split(values, node.output_sizes, dim=-1)
                if node.operation == "Split"
                else (values,)
            )
        output = torch.cat([ports[source][port] for source, port in self.manifest.outputs], dim=-1)
        recurrence = torch.cat(next_states, dim=-1) if next_states else state
        return output, recurrence


def distillation_loss(
    outputs: torch.Tensor,
    teacher_turn: torch.Tensor,
    teacher_boost_request: torch.Tensor,
    event_weight: torch.Tensor,
):
    """Method-v1 weighted turn MSE plus temperature-.10 boost BCE, without deployment heads."""
    if outputs.ndim < 1 or outputs.shape[-1] != 2:
        raise CanonicalGraphError("distillation requires canonical two-output rows")
    expected_shape = outputs.shape[:-1]
    for label in (teacher_turn, teacher_boost_request, event_weight):
        if label.shape != expected_shape or label.device != outputs.device:
            raise CanonicalGraphError("distillation label shape/device mismatch")
    if (
        outputs.dtype != torch.float32
        or teacher_turn.dtype != torch.float32
        or event_weight.dtype != torch.float32
        or teacher_boost_request.dtype not in (torch.float32, torch.bool)
    ):
        raise CanonicalGraphError("distillation requires Float32 numeric labels and weights")
    turn = (outputs[..., 0] - teacher_turn).square()
    boost = (outputs[..., 1] - 0.35) / 0.10
    boost_loss = torch.nn.functional.softplus(boost)  # pylint: disable=not-callable
    boost_loss = boost_loss - teacher_boost_request * boost
    return event_weight * (turn + boost_loss)


def configure_distillation_precision():
    """Select the initial FP32 parity envelope in an explicit distillation process."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
