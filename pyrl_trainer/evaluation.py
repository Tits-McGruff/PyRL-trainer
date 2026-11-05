"""Bounded frozen-evaluator accounting over accepted authoritative intervals."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import torch
from slither_contracts.models import ExternalRewardDeltasV1, ExternalRewardEnvelopeV1

from .reward_contract import REWARD_COMPONENT_KEYS, RewardContractError

EVALUATION_CONFIG_VERSION = 1


class EvaluationEpisodeComplete(Exception):
    """A valid duration-capped episode ended; request a fresh assignment next."""


def evaluation_config_record(cfg) -> dict[str, Any]:
    """Identify comparable live evaluator scores separately from PPO coefficients."""
    return {"max_seconds": float(cfg.evaluation_max_seconds),
            "base_seed": int(cfg.evaluation_seed), "sampling": "stochastic",
            "score": "undiscounted_authoritative_reward_v1",
            "cutoff": "first_accepted_observation_at_or_after_limit"}


def policy_snapshot_sha256(snapshot: dict[str, Any]) -> str:
    """Hash exact CPU parameter bytes and layouts without serializing source or models."""
    digest = hashlib.sha256(b"pyrl-evaluation-policy-v1\0")
    for name, tensor in sorted(snapshot["model"].items()):
        value = tensor.detach().to("cpu").contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(header.encode("utf-8") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class EvaluationEpisode:
    """One immutable-policy episode; raw and weighted totals use constant memory."""

    policy_version: int
    policy_sha256: str
    seed: int
    max_seconds: float
    first: ExternalRewardEnvelopeV1 | None = None
    last: ExternalRewardEnvelopeV1 | None = None
    raw: dict[str, float] = field(default_factory=lambda: {
        name: 0 for name in ExternalRewardDeltasV1.__annotations__
    })
    components: dict[str, float] = field(default_factory=lambda: {
        name: 0.0 for name in REWARD_COMPONENT_KEYS
    })
    finished: bool = False

    def receive(self, reward: ExternalRewardEnvelopeV1, components: dict[str, float]) -> None:
        """Accept only a stream-validated interval; never infer state from sensors."""
        if self.finished:
            raise RewardContractError("evaluation already closed")
        if self.first is None:
            self.first = reward
        raw = {name: value + reward["deltas"][name] for name, value in self.raw.items()}
        parts = {name: value + components[name] for name, value in self.components.items()}
        if not all(math.isfinite(value) for value in
                   (*raw.values(), *parts.values(), sum(parts.values()))):
            raise RewardContractError("evaluation accounting overflow")
        self.raw, self.components, self.last = raw, parts, reward

    @property
    def limit_reached(self) -> bool:
        """Cut only at a complete received interval, reporting actual exposure."""
        return self.raw["elapsedSeconds"] >= self.max_seconds

    def finish(self, kind: str, reason: str, *, integrity: bool) -> dict[str, Any] | None:
        """Return one complete or explicitly excluded prefix result, at most once."""
        if self.finished:
            return None
        self.finished = True
        if self.first is None or self.last is None:
            return None
        ticks = int(self.last["toStep"], 16) - int(self.first["toStep"], 16)
        return {
            "schema_version": 1, "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256, "sampling_seed": self.seed,
            "run_id": self.first["runId"], "world_epoch": self.first["worldEpoch"],
            "lease_id": self.first["leaseId"], "physical_snake_id": self.first["physicalSnakeId"],
            "snake_id": self.first["snakeId"], "initial_step": self.first["toStep"],
            "cutoff_step": self.last["toStep"],
            "cutoff_observation_sequence": self.last["observationSequence"],
            "lifetime_ticks": ticks, "elapsed_seconds": self.raw["elapsedSeconds"],
            "starting_body_length": self.first["state"]["startingLength"],
            "initial_observed_body_length": self.first["state"]["bodyLength"],
            "final_body_length": self.last["state"]["bodyLength"],
            "initial_retained_growth": self.first["state"]["retainedGrowth"],
            "retained_growth": self.last["state"]["retainedGrowth"],
            "alive_seconds": self.raw["aliveSecondsDelta"],
            "duration_limit_seconds": self.max_seconds,
            "duration_overshoot_seconds": max(0.0, self.raw["elapsedSeconds"] - self.max_seconds),
            "raw_deltas": dict(self.raw), "reward_components": dict(self.components),
            "episode_return": sum(self.components.values()),
            "transition_kind": kind, "reason": reason, "integrity": integrity,
        }
