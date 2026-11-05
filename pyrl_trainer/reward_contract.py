"""Validate native reward-v1 intervals and apply trainer-owned objective weights.

Wire shapes come from slither-contracts. This module owns trainer admission and
weighting, never simulation events or sensor-label reconstruction.
"""

from dataclasses import dataclass, fields
from copy import deepcopy
import math
import re
from typing import Any, Mapping, cast, get_args

from slither_contracts import CONSTANTS
from slither_contracts.models import (
    ExternalRewardDeltasV1,
    ExternalRewardEndReason,
    ExternalRewardEnvelopeV1,
    ExternalRewardStateV1,
    ExternalRewardCommandV1,
    ExternalRewardAppliedCommandV1,
    ExternalRewardControlV1,
    ExternalRewardControlSourceV1,
    ExternalRewardTransitionKind,
)


class RewardContractError(RuntimeError):
    """Fatal missing, unsupported, or malformed reward contract."""


class RewardDiscontinuityError(RewardContractError):
    """An interval cannot be attached to the current pending decision.

    Recovery must explicitly discard unresolved experience and rebase. Retrying
    the same interval or inventing a terminal transition is never appropriate.
    """


_HEX_ID = re.compile(r"[0-9a-f]{16}\Z")
_MAX_EXACT = 2**53 - 1
_SIGNED_INTEGERS = frozenset(("lengthDelta", "retainedGrowthDelta"))
_COUNTS = frozenset(("foodCountDelta", "killsDelta"))
_SIGNED_FLOATS = frozenset(("pointsDelta", "otherPointsDelta"))
REWARD_COMPONENT_KEYS = ("survival", "food", "growth", "points", "kills", "death")


def _identity(value: Any, *, zero_allowed: bool = False) -> str:
    if not isinstance(value, str) or not _HEX_ID.fullmatch(value):
        raise RewardContractError("reward identity must be a 16-digit lowercase hex string")
    if not zero_allowed and int(value, 16) == 0:
        raise RewardContractError("reward identity must be positive")
    return value


def _number(value: Any, name: str, *, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardContractError(f"reward {name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise RewardContractError(f"reward {name} is not finite") from exc
    if not math.isfinite(result) or (not signed and result < 0):
        raise RewardContractError(f"reward {name} is nonfinite or has an invalid sign")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise RewardContractError(f"reward {name} must be an exact integer in range")
    return value


def _shape(raw: Any, model: type, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != set(model.__annotations__):
        raise RewardContractError(f"reward {name} has missing or unknown fields")
    return raw


def require_reward_version(welcome: Mapping[str, Any]) -> None:
    """Check independently of Protocol 2, sensor layout, and packet versions."""
    version = welcome.get("rewardVersion")
    if CONSTANTS.get("externalRewardVersion") != 1:
        raise RewardContractError("installed slither-contracts does not support reward version 1")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise RewardContractError("server must advertise authoritative rewardVersion 1")


def _validate_lifecycle(raw: Mapping[str, Any]) -> None:
    kind = raw["transitionKind"]
    reason = raw["reason"]
    if kind not in get_args(ExternalRewardTransitionKind):
        raise RewardContractError("unsupported reward transition kind")
    if reason is not None and reason not in get_args(ExternalRewardEndReason):
        raise RewardContractError("unsupported reward closure reason")
    predecessor = raw["previousObservationSequence"]
    if predecessor is not None:
        _identity(predecessor)
        if int(raw["observationSequence"], 16) <= int(predecessor, 16):
            raise RewardContractError("reward observation sequence does not advance")
    if kind == "initial":
        if predecessor is not None or reason is not None or raw["fromStep"] != raw["toStep"]:
            raise RewardContractError("invalid initial reward baseline")
    elif predecessor is None:
        raise RewardContractError("noninitial reward requires a predecessor")
    if kind in ("initial", "live") and reason is not None:
        raise RewardContractError("reward transition kind and reason disagree")
    if kind == "terminal" and reason != "death":
        raise RewardContractError("reward transition kind and reason disagree")
    if kind == "truncated" and reason in (None, "death"):
        raise RewardContractError("reward transition kind and reason disagree")


def _command(raw: Any, cutoff: int) -> dict[str, Any]:
    value = dict(_shape(raw, ExternalRewardCommandV1, "command"))
    _identity(value["sequence"])
    if value["clientActionId"] is not None:
        _identity(value["clientActionId"])
    _identity(value["clientTick"], zero_allowed=True)
    _identity(value["acceptedAtStep"], zero_allowed=True)
    if int(value["acceptedAtStep"], 16) > cutoff:
        raise RewardContractError("command receipt is after its observation boundary")
    return value


def _validate_control_selection(value: Mapping[str, Any], cutoff: int) -> None:
    source, selected = value["source"], value["selectedAtStep"]
    if source is not None and source not in get_args(ExternalRewardControlSourceV1):
        raise RewardContractError("unsupported physical control source")
    if (source is None) != (selected is None):
        raise RewardContractError("control source omits its physical boundary")
    if selected is not None:
        _identity(selected, zero_allowed=True)
        if int(selected, 16) >= cutoff:
            raise RewardContractError("control selection has not completed physics")


def _control(raw: Any, cutoff: int) -> dict[str, Any]:
    value = dict(_shape(raw, ExternalRewardControlV1, "control"))
    _validate_control_selection(value, cutoff)
    source, selected = value["source"], value["selectedAtStep"]
    for name in ("acceptedCount", "appliedCount", "supersededCount"):
        _integer(value[name], name, 0, _MAX_EXACT)
    accepted = value["acceptedCommand"]
    if accepted is not None:
        accepted = _command(accepted, cutoff)
    applied = value["appliedCommand"]
    if applied is not None:
        applied = dict(_shape(applied, ExternalRewardAppliedCommandV1, "applied command"))
        applied["command"] = _command(applied["command"], cutoff)
        first = int(_identity(applied["firstAppliedStep"], zero_allowed=True), 16)
        if (selected is None or first > int(selected, 16)
                or first < int(applied["command"]["acceptedAtStep"], 16)):
            raise RewardContractError("invalid physical command application boundary")
    if ((accepted is None) != (value["acceptedCount"] == 0)
            or (applied is None) != (value["appliedCount"] == 0)
            or (source == "held" and applied is None)):
        raise RewardContractError("control counts omit their command identity")
    if applied is not None:
        command = applied["command"]
        if accepted is None or int(command["sequence"], 16) > int(accepted["sequence"], 16):
            raise RewardContractError("applied command was never accepted")
        if command["sequence"] == accepted["sequence"] and command != accepted:
            raise RewardContractError("accepted and applied command identity disagree")
    pending = int(accepted is not None and
                  (applied is None or applied["command"]["sequence"] != accepted["sequence"]))
    if value["acceptedCount"] != value["appliedCount"] + value["supersededCount"] + pending:
        raise RewardContractError("command counts do not conserve accepted inputs")
    return dict(value, acceptedCommand=accepted, appliedCommand=applied)


def _control_progress(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for counter, marker in (("acceptedCount", "acceptedCommand"),
                            ("appliedCount", "appliedCommand")):
        old, new = previous[marker], current[marker]
        if current[counter] < previous[counter]:
            raise RewardDiscontinuityError("command counters regressed")
        if current[counter] == previous[counter]:
            if new != old:
                raise RewardDiscontinuityError("command identity changed without its counter")
            continue
        if old is not None:
            if marker == "appliedCommand":
                old, new = old["command"], new["command"]
            if new["sequence"] <= old["sequence"]:
                raise RewardDiscontinuityError("command counter advanced without a new sequence")
    if current["supersededCount"] < previous["supersededCount"]:
        raise RewardDiscontinuityError("command supersession counter regressed")
    old, new = previous["selectedAtStep"], current["selectedAtStep"]
    if old is not None and (new is None or int(new, 16) < int(old, 16)):
        raise RewardDiscontinuityError("physical control selection regressed")
    if new == old and current["source"] != previous["source"]:
        raise RewardDiscontinuityError("control source changed without a new selection")
    if (old is not None and current["appliedCount"] > previous["appliedCount"]
            and current["appliedCommand"]["firstAppliedStep"] <= old):
        raise RewardDiscontinuityError("new application is before the previous selection")


def parse_reward(raw: Any) -> ExternalRewardEnvelopeV1:
    """Validate one complete interval without changing stream state."""
    raw = _shape(raw, ExternalRewardEnvelopeV1, "envelope")
    require_reward_version(raw)
    if not isinstance(raw["runId"], str) or not raw["runId"] or len(raw["runId"]) > 256:
        raise RewardContractError("reward runId must be a bounded nonempty string")
    for name in ("worldEpoch", "leaseId", "physicalSnakeId", "observationSequence"):
        _identity(raw[name])
    _integer(raw["snakeId"], "snakeId", 1, 16_777_216)
    for name in ("fromStep", "toStep"):
        _identity(raw[name], zero_allowed=True)
    if int(raw["toStep"], 16) < int(raw["fromStep"], 16):
        raise RewardContractError("reward interval steps run backwards")
    _validate_lifecycle(raw)
    state = dict(_shape(raw["state"], ExternalRewardStateV1, "state"))
    for name, value in state.items():
        _integer(value, name, 0 if name == "retainedGrowth" else 1, _MAX_EXACT)
    if state["retainedGrowth"] != max(state["bodyLength"] - state["startingLength"], 0):
        raise RewardContractError("reward boundary state has inconsistent retained growth")
    deltas = dict(_shape(raw["deltas"], ExternalRewardDeltasV1, "deltas"))
    for name, value in deltas.items():
        if name in _SIGNED_INTEGERS:
            _integer(value, name, -_MAX_EXACT, _MAX_EXACT)
        elif name in _COUNTS:
            _integer(value, name, 0, _MAX_EXACT)
        else:
            deltas[name] = _number(value, name, signed=name in _SIGNED_FLOATS)
    if raw["transitionKind"] == "initial" and any(value != 0 for value in deltas.values()):
        raise RewardContractError("initial reward must contain zero deltas")
    # Prefix credits and the physical observation clock have distinct boundaries.
    # A valid interval can contain live-time credit before physics advances.
    accounted = (deltas["survivalPointsDelta"] + deltas["foodPointsDelta"]
                 + deltas["killPointsDelta"] + deltas["otherPointsDelta"]
                 - deltas["boostPointsSpentDelta"])
    if not math.isfinite(accounted) or not math.isclose(
        deltas["pointsDelta"], accounted, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise RewardContractError("reward point-accounting identity failed")
    # Own each shallow map: a transport decoder cannot mutate admitted data.
    control = _control(raw["control"], int(raw["toStep"], 16))
    return cast(ExternalRewardEnvelopeV1, dict(raw, state=state, control=control, deltas=deltas))


def _life(reward: ExternalRewardEnvelopeV1) -> tuple[str, str, str, str, int]:
    return (reward["runId"], reward["worldEpoch"], reward["leaseId"],
            reward["physicalSnakeId"], reward["snakeId"])


class RewardStream:
    """One explicitly assigned life, with constant-sized sequence admission state."""

    def __init__(self) -> None:
        self._life: tuple[str, str, str, str, int] | None = None
        self._sequence: str | None = None
        self._step: str | None = None
        self._closed = False
        self._state: ExternalRewardStateV1 | None = None
        self._control: dict[str, Any] | None = None

    def accept(self, raw: Any, *, snake_id: int) -> ExternalRewardEnvelopeV1:
        """Admit once, only after all identity and predecessor checks pass."""
        reward = parse_reward(raw)
        if reward["snakeId"] != snake_id:
            raise RewardDiscontinuityError("reward does not belong to the assigned snake")
        if self._closed:
            raise RewardDiscontinuityError("reward received after life closure")
        if self._life is None:
            if reward["transitionKind"] != "initial":
                raise RewardDiscontinuityError("reward stream requires an initial baseline")
        elif (_life(reward) != self._life
              or reward["transitionKind"] == "initial"
              or reward["previousObservationSequence"] != self._sequence
              or reward["fromStep"] != self._step):
            raise RewardDiscontinuityError("reward identity, predecessor, or interval has a gap")
        if self._state is not None:
            previous, current = self._state, reward["state"]
            if (current["startingLength"] != previous["startingLength"]
                    or current["bodyLength"] - previous["bodyLength"]
                    != reward["deltas"]["lengthDelta"]
                    or current["retainedGrowth"] - previous["retainedGrowth"]
                    != reward["deltas"]["retainedGrowthDelta"]):
                raise RewardDiscontinuityError("reward boundary state and length deltas disagree")
        self._commit(reward)
        return reward

    def rebase(self, raw: Any, *, snake_id: int) -> None:
        """Discard this interval after explicit reconnect/assignment tail cleanup.

        This is never called automatically by accept and returns no trainable
        reward. A terminal or truncated interval cannot establish a live baseline.
        """
        reward = parse_reward(raw)
        if reward["snakeId"] != snake_id or reward["transitionKind"] not in ("initial", "live"):
            raise RewardDiscontinuityError("cannot rebase from this reward interval")
        self._commit(reward)

    def discard_closure(self, raw: Any, *, snake_id: int) -> None:
        """Retire a life after an explicit transport gap, without trainable reward.

        Reclaim may be followed immediately by death/cutoff instead of a live
        baseline. Validate the known life, but never reward its unresolved tail.
        """
        reward = parse_reward(raw)
        if (self._closed or reward["snakeId"] != snake_id
                or reward["transitionKind"] not in ("terminal", "truncated")
                or (self._life is not None and _life(reward) != self._life)):
            raise RewardDiscontinuityError("cannot discard closure for this reward life")
        self._commit(reward)

    def _commit(self, reward: ExternalRewardEnvelopeV1) -> None:
        if (self._state is not None and self._life == _life(reward)
                and self._state["startingLength"] != reward["state"]["startingLength"]):
            raise RewardDiscontinuityError("starting length changed within the same reward life")
        if self._control is not None and self._life == _life(reward):
            _control_progress(self._control, reward["control"])
        self._life = _life(reward)
        self._sequence = reward["observationSequence"]
        self._step = reward["toStep"]
        self._closed = reward["transitionKind"] in ("terminal", "truncated")
        self._state = dict(reward["state"])
        self._control = deepcopy(reward["control"])


@dataclass(frozen=True)
class RewardCoefficients:
    """Initial experimental objective, separate from the server wire version."""

    survival: float = 0.001
    food: float = 0.10
    growth: float = 0.05
    points: float = 0.01
    kills: float = 1.0
    death_penalty: float = 0.50

    def __post_init__(self) -> None:
        for field in fields(self):
            _number(getattr(self, field.name), field.name)


def reward_components(
    reward: ExternalRewardEnvelopeV1,
    coefficients: RewardCoefficients = RewardCoefficients(),
) -> dict[str, float]:
    """Weight a validated interval; point survival is paid only by the time term."""
    delta = reward["deltas"]
    parts = {
        "survival": coefficients.survival * delta["aliveSecondsDelta"],
        "food": coefficients.food * delta["foodValueDelta"],
        "growth": coefficients.growth * delta["retainedGrowthDelta"],
        "points": coefficients.points * (delta["pointsDelta"] - delta["survivalPointsDelta"]),
        "kills": coefficients.kills * delta["killsDelta"],
        "death": -coefficients.death_penalty if reward["transitionKind"] == "terminal" else 0.0,
    }
    if (not all(math.isfinite(value) for value in parts.values())
            or not math.isfinite(sum(parts.values()))):
        raise RewardContractError("weighted reward overflow")
    return parts
