"""Bounded delivered-observation sequence training for canonical distillation."""

from dataclasses import dataclass
from enum import IntEnum
import math
from types import FunctionType

import torch

from .canonical_graph import CanonicalGraph, CanonicalGraphError, distillation_loss


class SequenceBoundary(IntEnum):
    """Reasons supplied by the committed stream decoder, never inferred from chunk ends."""

    CONTINUE = 0
    GENERATION = 1
    LIFE = 2
    GAP = 3
    CAPTURE_START = 4


@dataclass(frozen=True)
class SequenceTrainingConfig:
    """Method-v1 defaults and explicit finite sequence allocation ceilings."""

    batch_size: int = 32
    bptt_steps: int = 64
    burn_in_steps: int = 32
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0

    def __post_init__(self):
        for value, low, high in [
            (self.batch_size, 1, 256),
            (self.bptt_steps, 1, 256),
            (self.burn_in_steps, 0, 255),
        ]:
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise CanonicalGraphError("invalid bounded sequence training configuration")
        if self.bptt_steps + self.burn_in_steps > 256:
            raise CanonicalGraphError("burn-in plus BPTT exceeds 256 deliveries")
        for value in [self.learning_rate, self.gradient_clip]:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise CanonicalGraphError("invalid Adam learning rate or gradient clipping")


@dataclass(frozen=True)
class SequenceBatch:
    """Time-major windows; inactive suffixes are padding, not simulated deliveries."""

    observations: torch.Tensor
    teacher_turn: torch.Tensor
    teacher_boost_request: torch.Tensor
    event_weight: torch.Tensor
    active: torch.Tensor
    boundaries: torch.Tensor
    initial_state: torch.Tensor

    def validate(self, model: CanonicalGraph, config: SequenceTrainingConfig):
        # pylint: disable=too-many-branches
        """Validate the whole window before any optimizer or recurrent state changes."""
        if self.observations.ndim != 3:
            raise CanonicalGraphError("sequence observations must be time/batch/input")
        for value in [
            self.teacher_turn,
            self.teacher_boost_request,
            self.event_weight,
            self.active,
            self.boundaries,
            self.initial_state,
        ]:
            if value.device != self.observations.device:
                raise CanonicalGraphError("sequence tensors must share one source device")
        time, batch, width = self.observations.shape
        if not config.burn_in_steps < time <= config.burn_in_steps + config.bptt_steps:
            raise CanonicalGraphError("sequence has no training rows or exceeds the BPTT bound")
        if not 1 <= batch <= config.batch_size or width != model.manifest.input_size:
            raise CanonicalGraphError("sequence batch/input width mismatch")
        for value in [self.observations, self.teacher_turn, self.event_weight, self.initial_state]:
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise CanonicalGraphError("sequence numeric data must be finite Float32")
        for value in [
            self.teacher_turn,
            self.teacher_boost_request,
            self.event_weight,
            self.active,
            self.boundaries,
        ]:
            if value.shape != (time, batch):
                raise CanonicalGraphError("sequence label/mask shape mismatch")
        if self.initial_state.shape != (batch, model.manifest.total_state_size):
            raise CanonicalGraphError("sequence initial recurrent width mismatch")
        if self.active.dtype != torch.bool or self.boundaries.dtype != torch.int64:
            raise CanonicalGraphError("sequence masks require bool/int64")
        if self.teacher_boost_request.dtype not in (torch.bool, torch.float32):
            raise CanonicalGraphError("boost labels must be committed request booleans")
        if not ((self.teacher_boost_request == 0) | (self.teacher_boost_request == 1)).all():
            raise CanonicalGraphError("boost labels must be zero or one")
        if (self.teacher_turn.abs() > 1).any() or (self.observations.abs() > 1).any():
            raise CanonicalGraphError("normalized observations and applied turns exceed bounds")
        if (self.boundaries < 0).any() or (self.boundaries > 4).any():
            raise CanonicalGraphError("unsupported sequence boundary")
        if (self.event_weight < 0).any() or (self.event_weight > 4).any():
            raise CanonicalGraphError("event weights exceed method-v1 admission")
        if (self.event_weight[self.active] <= 0).any():
            raise CanonicalGraphError("active sequence rows need positive event weights")
        if ((~self.active[:-1]) & self.active[1:]).any():
            raise CanonicalGraphError("padding cannot hide a missing interior delivery")
        if not self.active[config.burn_in_steps :].any():
            raise CanonicalGraphError("batch has no active training deliveries")

    def on_device(self, device):
        """Transfer one admitted window once; detach caller-owned data/initial history."""
        return SequenceBatch(
            *(
                value.detach().to(device)
                for value in [
                    self.observations,
                    self.teacher_turn,
                    self.teacher_boost_request,
                    self.event_weight,
                    self.active,
                    self.boundaries,
                    self.initial_state,
                ]
            )
        )


@dataclass(frozen=True)
class SequenceUpdate:
    """Metrics and pre-update-weight continuation, detached for the next truncation.

    Burn-in reconstructs only the supplied bounded prefix. Callers must provide
    state before its first row, not after an overlapping previous prefix.
    """

    loss: float
    turn_mse: float
    turn_mae: float
    boost_disagreement: float
    training_samples: int
    burn_in_samples: int
    unknown_state_starts: int
    gradient_norm: float
    optimizer_updated: bool
    next_state: torch.Tensor


def _before_delivery(state, boundary, active, life_rule):
    """Apply declared resets; transport/capture gaps always terminate unknown history."""
    # Dynamo must compare tensors with primitive integers: IntEnum operands can
    # otherwise fold to a Python False and silently remove a delivered reset.
    reset = (
        (boundary == int(SequenceBoundary.GENERATION))
        | (boundary == int(SequenceBoundary.GAP))
        | (boundary == int(SequenceBoundary.CAPTURE_START))
    )
    if life_rule == "reset":
        reset = reset | (boundary == int(SequenceBoundary.LIFE))
    return torch.where((reset & active).unsqueeze(-1), torch.zeros_like(state), state)


def _delivery(model, batch, index, state):
    """Advance only active rows; a sealed chunk boundary has no implicit reset."""
    active = batch.active[index]
    before = _before_delivery(
        state, batch.boundaries[index], active, model.manifest.life_state_rule
    )
    output, proposed = model(batch.observations[index], before)
    return output, torch.where(active.unsqueeze(-1), proposed, state)


def _sequence_forward(model, data, burn_in):
    """Unroll one entire window, preserving canonical equations and TBPTT cuts."""
    state = data.initial_state
    with torch.no_grad():
        for index in range(burn_in):
            _output, state = _delivery(model, data, index, state)
    state = state.detach()
    predictions = []
    for index in range(burn_in, data.observations.shape[0]):
        output, state = _delivery(model, data, index, state)
        predictions.append(output)
    return torch.stack(predictions), state


@dataclass(frozen=True)
class _ForwardWindow:
    """Inference-only storage; inactive prefix/suffix padding consumes no deliveries."""

    observations: torch.Tensor
    active: torch.Tensor
    boundaries: torch.Tensor
    initial_state: torch.Tensor


def _pad_forward_window(data, burn_in, capacity):
    """Align real burn-in to one fixed detach boundary without advancing padded state."""
    prefix = capacity.burn_in_steps - burn_in
    suffix = capacity.bptt_steps - (data.observations.shape[0] - burn_in)
    def padded(value):
        return torch.cat((value.new_zeros((prefix, *value.shape[1:])), value,
                          value.new_zeros((suffix, *value.shape[1:]))))
    return _ForwardWindow(padded(data.observations), padded(data.active),
                          padded(data.boundaries), data.initial_state)


class _WindowExecution:
    """Optional qualification accelerator with one bounded compiled signature per owner.

    Shorter burn-in/training windows use inactive padding. Other signatures keep
    using eager CUDA. A separate code object isolates
    Dynamo's per-code cache across independent graph/ablation owners; no global
    compiler reset affects other trainers. Loss, finite checks and Adam stay eager.
    These counters describe dispatch, not proof that a backend captured every call.
    """

    def __init__(self, enabled, capacity=SequenceTrainingConfig()):
        if not isinstance(enabled, bool):
            raise CanonicalGraphError("cuda_graphs must be an explicit boolean")
        self.enabled = enabled
        self.capacity = capacity
        self.signature = None
        self.compiled = None
        self.compiled_calls = 0
        self.eager_calls = 0

    @staticmethod
    def _admitted(model, data):
        """Bound FX unrolling and private graph-pool work before invoking the compiler."""
        time, batch, _width = data.observations.shape
        nodes = model.manifest.nodes
        work = sum(8 + 8 * (len(node.parameters) + len(node.hidden_sizes)) for node in nodes)
        return (time <= 96 and batch <= 32 and len(nodes) <= 8
                and model.manifest.total_parameters <= 65536
                and model.manifest.total_state_size <= 256 and time * work <= 32768)

    def __call__(self, model, data, burn_in):
        training_steps = data.observations.shape[0] - burn_in
        if (not self.enabled or burn_in > self.capacity.burn_in_steps
                or training_steps > self.capacity.bptt_steps
                or self.capacity.burn_in_steps + self.capacity.bptt_steps > 96
                or not self._admitted(model, data)):
            self.eager_calls += 1
            return _sequence_forward(model, data, burn_in)
        window = _pad_forward_window(data, burn_in, self.capacity)
        if not self._admitted(model, window):
            self.eager_calls += 1
            return _sequence_forward(model, data, burn_in)
        tensors = (window.observations, window.active, window.boundaries, window.initial_state)
        signature = tuple((tuple(value.shape), value.stride(), value.dtype,
                           value.device) for value in tensors)
        if self.signature is not None and self.signature != signature:
            self.eager_calls += 1
            return _sequence_forward(model, data, burn_in)
        if self.compiled is None:
            # torch.compile caches on code objects. Own one cache for this model,
            # rather than exhausting a shared frame's recompile limit across jobs.
            isolated = FunctionType(_sequence_forward.__code__.replace(), globals())
            self.compiled = torch.compile(isolated, backend="cudagraphs",
                                          fullgraph=True, dynamic=False)
            self.signature = signature
        torch.compiler.cudagraph_mark_step_begin()
        outputs, state = self.compiled(model, window, self.capacity.burn_in_steps)
        self.compiled_calls += 1
        return outputs[:training_steps], state

    def statistics(self):
        """Return bounded diagnostics without serializing derived compiler state."""
        return {"requested": "cudagraphs" if self.enabled else "eager",
                "compiledWindowCalls": self.compiled_calls, "eagerWindowCalls": self.eager_calls,
                "compiledSignatures": int(self.compiled is not None)}


class CanonicalSequenceTrainer:
    # pylint: disable=too-few-public-methods
    """Adam/TBPTT owner for independent canonical windows, beside existing PPO."""

    def __init__(
        self, model: CanonicalGraph, config=SequenceTrainingConfig(), *, cuda_graphs=False
    ):
        self.window_execution = _WindowExecution(cuda_graphs, config)
        if cuda_graphs and model.flat_parameters.device.type != "cuda":
            raise CanonicalGraphError("CUDA window execution requires a CUDA model")
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.completed_updates = 0

    def update(self, batch: SequenceBatch) -> SequenceUpdate:
        """Burn in without gradients, train one bounded segment, and detach its continuation."""
        batch.validate(self.model, self.config)
        data = batch.on_device(self.model.flat_parameters.device)
        outputs, state = self.window_execution(self.model, data, self.config.burn_in_steps)
        start = self.config.burn_in_steps
        active = data.active[start:]
        count = int(active.sum().item())
        losses = distillation_loss(
            outputs,
            data.teacher_turn[start:],
            data.teacher_boost_request[start:],
            data.event_weight[start:],
        )
        objective = torch.where(active, losses, torch.zeros_like(losses)).sum() / count
        if not torch.isfinite(objective) or not torch.isfinite(state).all():
            raise CanonicalGraphError("nonfinite sequence inference or objective")
        self.optimizer.zero_grad(set_to_none=True)
        gradient_norm = 0.0
        updated = objective.requires_grad
        if updated:
            objective.backward()
            gradient = self.model.flat_parameters.grad
            if gradient is None or not torch.isfinite(gradient).all():
                self.optimizer.zero_grad(set_to_none=True)
                raise CanonicalGraphError("nonfinite or missing sequence gradients")
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            if not torch.isfinite(norm):
                raise CanonicalGraphError("nonfinite clipped sequence gradient norm")
            gradient_norm = norm.item()
            self.optimizer.step()
            self.completed_updates += 1
        with torch.no_grad():
            turn_error = outputs[..., 0] - data.teacher_turn[start:]
            unknown = (
                (data.boundaries == SequenceBoundary.GAP)
                | (data.boundaries == SequenceBoundary.CAPTURE_START)
            ) & data.active
            disagreement = (outputs[..., 1] > 0.35) != data.teacher_boost_request[start:].bool()
            return SequenceUpdate(
                objective.item(),
                turn_error[active].square().mean().item(),
                turn_error[active].abs().mean().item(),
                disagreement[active].float().mean().item(),
                count,
                int(data.active[:start].sum().item()),
                int(unknown.sum().item()),
                gradient_norm,
                updated,
                state.detach().clone(),
            )

    def evaluate(self, batch: SequenceBatch):
        """Evaluate unweighted validation without optimizer, gradient or parameter mutation."""
        batch.validate(self.model, self.config)
        data = batch.on_device(self.model.flat_parameters.device)
        state = data.initial_state
        predictions = []
        with torch.no_grad():
            for index in range(data.observations.shape[0]):
                output, state = _delivery(self.model, data, index, state)
                if index >= self.config.burn_in_steps:
                    predictions.append(output)
            outputs = torch.stack(predictions)
            if not torch.isfinite(outputs).all() or not torch.isfinite(state).all():
                raise CanonicalGraphError("nonfinite validation inference")
            start = self.config.burn_in_steps
            active = data.active[start:]
            error = outputs[..., 0] - data.teacher_turn[start:]
            disagreement = (outputs[..., 1] > 0.35) != data.teacher_boost_request[start:].bool()
            return {
                "turn_mse": error[active].square().mean().item(),
                "turn_mae": error[active].abs().mean().item(),
                "boost_disagreement": disagreement[active].float().mean().item(),
                "samples": int(active.sum().item()),
            }
