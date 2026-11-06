"""Actor client for the Slither training server."""

import asyncio
import json
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import websockets

from .binary_sensors import decode_binary_sensors
from .config import Config, MAX_WS_MESSAGE_BYTES, PROTOCOL_VERSION
from .evaluation import EvaluationEpisode, EvaluationEpisodeComplete, policy_snapshot_sha256
from .sensor_contract import SensorContract, SensorContractError
from .reward_contract import (
    RewardCoefficients, RewardContractError, RewardDiscontinuityError, RewardStream,
    parse_reward, require_reward_version, reward_components,
)
from .utils import (
    build_index,
    clamp,
    compute_stride,
)


@dataclass
class Transition:  # pylint: disable=too-few-public-methods
    """Single environment transition for PPO."""

    obs: np.ndarray
    action_turn: float
    turn_latent: float
    action_boost: float
    logp: float
    value: float
    reward: float
    done: float  # 1.0 if episode ended at this step else 0.0
    policy_version: int = 0
    reward_components: Dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    alive_seconds: float = 0.0
    action_id: Optional[str] = None


class ActorClient:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """WebSocket client that controls one training or frozen-evaluation snake."""

    def __init__(
        self,
        actor_id: int,
        cfg: Config,
        shared_state: Any,  # Avoid circular type hint for SharedState.
        experience_q: asyncio.Queue,
        *,
        evaluation: bool = False,
    ):
        self.actor_id = actor_id
        self.cfg = cfg
        self.shared_state = shared_state
        self.experience_q = experience_q
        self.evaluation = bool(evaluation)

        self.snake_id: Optional[int] = None
        self.resume_token: Optional[str] = None
        self.tick_rate: int = 60
        self.stride: int = 1

        self.sensor_order: List[str] = []
        self.sensor_idx: Dict[str, int] = {}

        self.last_sent_tick: Optional[int] = None
        self.last_sent_time: float = 0.0

        self.last_obs: Optional[np.ndarray] = None
        self.rollout: List[Transition] = []

        self.episodes: int = 0
        self.steps: int = 0

        self.last_sensor_tick: Optional[int] = None
        self.last_assign_tick: Optional[int] = None
        self.last_gen: Optional[int] = None
        self.assign_count: int = 0

        self.pending_transition: Optional[Transition] = None
        self.episode_return: float = 0.0
        self.episode_actions: int = 0
        self.episode_policy_snapshot: Optional[Dict[str, Any]] = None
        self.episode_policy_model: Optional[Any] = None
        self.reward_coefficients = RewardCoefficients(
            survival=cfg.reward_survival_per_second, food=cfg.reward_food_value_scale,
            growth=cfg.reward_retained_growth_scale, points=cfg.reward_task_points_scale,
            kills=cfg.reward_kills_scale, death_penalty=cfg.reward_death_penalty_magnitude,
        )
        self._reward_stream = RewardStream()
        self._reward_rebase = False
        self._reward_has_observation = False
        self._episode_closed = False
        self._episode_integrity = True
        self.discarded_reward_intervals = 0
        self.evaluation_episode: Optional[EvaluationEpisode] = None
        self.evaluation_generator: Optional[torch.Generator] = None
        self._evaluation_name = self._fresh_evaluation_name() if self.evaluation else ""
        self._action_id_counter = 0
        self._last_issued_action_id = 0
        self.last_reward_envelope = None

    def _fresh_evaluation_name(self) -> str:
        """Avoid legacy name-based reclaim after an intentional capped cutoff."""
        return f"{self.cfg.bot_name[:2]}-e-{secrets.token_hex(8)}"

    @property
    def _log_label(self) -> str:
        return "eval" if self.evaluation else f"actor {self.actor_id}"

    def _reset_per_snake_state(self) -> None:
        self.last_obs = None
        self.pending_transition = None
        self.rollout.clear()
        self.last_sent_tick = None
        self.episode_return = 0.0
        self.episode_actions = 0
        self.episode_policy_snapshot = None
        self.episode_policy_model = None
        self._reward_stream = RewardStream()
        self._reward_rebase = False
        self._reward_has_observation = False
        self._episode_closed = False
        self._episode_integrity = True
        self.evaluation_episode = None
        self.evaluation_generator = None
        self.last_reward_envelope = None

    def _reset_connection_state(self) -> None:
        """Reset socket-local fields while retaining reclaimable snake state."""
        self.sensor_order = []
        self.sensor_idx = {}
        self.last_sent_time = 0.0

    def _start_evaluation_policy(self) -> None:
        if not self.evaluation:
            return
        callback = getattr(self.shared_state, "begin_evaluation_policy", None)
        if not callable(callback):
            raise RuntimeError("evaluation actor requires frozen-policy support")
        snapshot, model = callback()
        self.episode_policy_snapshot = snapshot
        self.episode_policy_model = model
        seed = (int(self.cfg.evaluation_seed) + self.episodes) % (2**63)
        self.evaluation_generator = torch.Generator(device=self.cfg.infer_device).manual_seed(seed)
        self.evaluation_episode = EvaluationEpisode(
            int(snapshot["update_steps"]), policy_snapshot_sha256(snapshot), seed,
            float(self.cfg.evaluation_max_seconds),
        )

    def _finish_evaluation(self, kind: str, reason: str, *, integrity: bool) -> None:
        if not self.evaluation or self.evaluation_episode is None:
            return
        result = self.evaluation_episode.finish(kind, reason, integrity=integrity)
        if result is None:
            return
        callback = getattr(self.shared_state, "record_evaluation_result", None)
        if not callable(callback):
            raise RuntimeError("evaluation actor requires complete-result recording")
        callback(result, self.episode_policy_snapshot)
        print(f"[eval] result={json.dumps(result, sort_keys=True, allow_nan=False)}")

    def _check_evaluation_limit(self) -> None:
        """End at a validated observation boundary, without inventing death or reward."""
        if not self.evaluation:
            return
        if not self._episode_integrity:
            raise EvaluationEpisodeComplete()
        if self.evaluation_episode is not None and self.evaluation_episode.limit_reached:
            self._finish_evaluation("truncated", "evaluationTimeLimit", integrity=True)
            self._episode_closed = True
            raise EvaluationEpisodeComplete()

    async def _truncate_disconnected_rollout(self) -> None:
        """Flush completed PPO work without inventing a terminal transition."""
        self._finish_evaluation("incomplete", "transportGap", integrity=False)
        bootstrap = (
            float(self.pending_transition.value)
            if self.pending_transition is not None
            else 0.0
        )
        self.pending_transition = None

        if self.rollout and not self.evaluation:
            await self.experience_q.put(
                (self.actor_id, list(self.rollout), bootstrap)
            )
            self._record_admitted_rollout()
            self.rollout.clear()

        self.last_obs = None
        self.last_sent_tick = None
        self._reward_rebase = True
        self._episode_integrity = False

    def _record_admitted_rollout(self) -> None:
        """Count only completed transitions after their queue admission succeeds."""
        if self.evaluation:
            return
        callback = getattr(self.shared_state, "record_admitted_rollout", None)
        if callable(callback):
            callback(self.rollout)

    def _record_episode(self, lifetime_ticks: int, *, kind: str, reason: str) -> None:
        if self.evaluation:
            self._finish_evaluation(kind, reason, integrity=self._episode_integrity)
            return
        callback = getattr(self.shared_state, "record_episode", None)
        if not callable(callback):
            return
        callback(self.episode_return, lifetime_ticks)

    async def run(self) -> None:
        """Connect to the server and keep the control loop alive."""
        url = self.cfg.ws_url
        while True:
            name = (self._evaluation_name if self.evaluation
                    else f"{self.cfg.bot_name}-{self.actor_id:03d}")
            try:
                self._reset_connection_state()
                async with websockets.connect(
                    url, max_size=MAX_WS_MESSAGE_BYTES
                ) as ws:
                    await self._handshake(ws, name)
                    await self._loop(ws, name)
            except asyncio.CancelledError:
                raise
            except EvaluationEpisodeComplete:
                self.snake_id = None
                self.resume_token = None
                self.last_assign_tick = None
                self.last_sensor_tick = None
                self._reset_per_snake_state()
                self._evaluation_name = self._fresh_evaluation_name()
            except (SensorContractError, RewardContractError):
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                await self._truncate_disconnected_rollout()
                print(
                    f"[{self._log_label}] disconnected, "
                    f"reason={type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(0.5)

    async def _handshake(self, ws, name: str) -> None:
        hello = {
            "type": "hello",
            "clientType": "bot",
            "version": PROTOCOL_VERSION,
            "binarySensors": True,
            "binarySensorsVersion": 2,
            "rewardVersion": 1,
        }
        await ws.send(json.dumps(hello))

        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "welcome":
                self._apply_welcome(msg)
                join = {"type": "join", "mode": "player", "name": name[:24]}
                if self.resume_token:
                    join["resumeToken"] = self.resume_token
                await ws.send(json.dumps(join))
                break
            if msg.get("type") == "error":
                raise RewardContractError(
                    f"server error during handshake: {msg.get('message')}"
                )

        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "stateReplaced":
                await self._handle_state_replaced(msg, ws, name)
                continue
            if msg.get("type") == "reclaimResult" and not msg.get("reclaimed"):
                await self._truncate_disconnected_rollout()
                self.resume_token = None
                self.snake_id = None
                self.last_sensor_tick = None
                self.last_assign_tick = None
                self.last_gen = None
                await ws.send(
                    json.dumps(
                        {"type": "join", "mode": "player", "name": name[:24]}
                    )
                )
                continue
            if msg.get("type") == "assign":
                await self._on_assign(
                    msg.get("snakeId"),
                    msg.get("resumeToken"),
                    reclaimed=bool(msg.get("reclaimed", False)),
                    replacement_reason=msg.get("replacementReason"),
                )
                break
            if msg.get("type") == "error":
                raise RuntimeError(
                    f"server error during assign wait: {msg.get('message')}"
                )

    def _apply_welcome(self, msg: Dict[str, Any]) -> None:
        """Apply Protocol 2 fields and enforce the model's sensor contract."""
        if msg.get("protocolVersion") != PROTOCOL_VERSION:
            raise RuntimeError("server welcome did not confirm Protocol 2")
        self.tick_rate = int(msg.get("tickRate", 60))
        self.stride = compute_stride(
            self.tick_rate, self.cfg.max_actions_per_second
        )

        spec = msg.get("sensorSpec") or {}
        expected_contract = getattr(self.shared_state, "sensor_contract", None)
        if expected_contract is not None:
            actual_contract = SensorContract.from_spec(spec)
            if actual_contract != expected_contract:
                raise SensorContractError(
                    "server sensor contract changed; refusing to run the current model"
                )
            self.sensor_order = list(actual_contract.order)
        else:
            self.sensor_order = list(spec.get("order") or [])
            if not self.sensor_order:
                raise RuntimeError("server welcome omitted sensor order")
        self.sensor_idx = build_index(self.sensor_order)
        require_reward_version(msg)
        if msg.get("binarySensorsVersion") != 2:
            raise RewardContractError("server must support reward-bearing binarySensorsVersion 2")

    async def _handle_state_replaced(
        self, msg: Dict[str, Any], ws, name: str
    ) -> None:
        """End the old episode and join imported authority without a stale token."""
        welcome = msg.get("welcome")
        if not isinstance(welcome, dict):
            raise RuntimeError("stateReplaced omitted the replacement welcome")
        self._finish_evaluation("incomplete", "stateReplaced", integrity=False)
        await self._truncate_disconnected_rollout()
        self.snake_id = None
        self.resume_token = None
        self.last_sensor_tick = None
        self.last_assign_tick = None
        self.last_gen = None
        self.last_sent_time = 0.0
        self._reset_per_snake_state()
        self._apply_welcome(welcome)
        await ws.send(
            json.dumps({"type": "join", "mode": "player", "name": name[:24]})
        )

    async def _on_assign(
        self,
        snake_id: int,
        resume_token: str,
        reclaimed: bool = False,
        replacement_reason: Optional[str] = None,
    ) -> None:
        if replacement_reason not in (None, "generation"):
            raise RewardContractError("unsupported assignment replacement reason")
        if not isinstance(snake_id, int) or snake_id <= 0:
            raise RuntimeError("server assignment omitted a valid snakeId")
        if not isinstance(resume_token, str) or not resume_token:
            raise RuntimeError("Protocol 2 assignment omitted resumeToken")

        previous_snake = self.snake_id
        now_tick = self.last_sensor_tick
        lived = None
        if self.last_assign_tick is not None and now_tick is not None:
            lived = int(now_tick - self.last_assign_tick)

        self.assign_count += 1

        if reclaimed:
            if previous_snake is None:
                raise RuntimeError(
                    "server reported a reclaimed assignment without a prior snake"
                )
            if int(snake_id) != int(previous_snake):
                raise RuntimeError(
                    "server reclaim changed the assigned snake identity"
                )
            self.resume_token = resume_token
            self.last_sent_tick = None
            self._reward_rebase = True
            self._episode_integrity = False
            print(
                f"[{self._log_label}] reclaimed snake {snake_id}, "
                f"assigns={self.assign_count}"
            )
            return

        size_str = ""
        if self.last_obs is not None and "size_norm" in self.sensor_idx:
            size_value = self.last_obs[self.sensor_idx["size_norm"]]
            size_str = f", size_norm={size_value:.3f}"

        if lived is None:
            print(
                f"[{self._log_label}] assign {previous_snake} -> {snake_id}, "
                f"assigns={self.assign_count}"
            )
        else:
            print(
                f"[{self._log_label}] assign {previous_snake} -> {snake_id}, "
                f"lived_ticks={lived}{size_str}, assigns={self.assign_count}"
            )

        if previous_snake is not None and not self._episode_closed and self._reward_has_observation:
            if self._episode_integrity:
                raise RewardContractError("replacement assignment omitted the old-life closure")
            await self._truncate_disconnected_rollout()

        self.snake_id = int(snake_id)
        self.resume_token = resume_token
        self._reset_per_snake_state()
        self._start_evaluation_policy()
        self.episodes += 1
        self.last_assign_tick = now_tick

    def _should_send_action(self, tick: int) -> bool:
        if self.last_sent_tick is not None:
            if self.last_sent_tick == tick:
                return False
            if (tick - self.last_sent_tick) < self.stride:
                return False
        if self.cfg.max_actions_per_second > 0 and self.last_sent_time > 0.0:
            min_interval = 1.0 / float(self.cfg.max_actions_per_second)
            if (time.monotonic() - self.last_sent_time) < min_interval:
                return False
        return True



    def _sample_action(
        self, obs: np.ndarray
    ) -> Tuple[float, float, float, float, float, int]:
        """Sample from the live training policy or the episode's frozen evaluator."""
        policy_version = int(
            getattr(self.shared_state, "published_policy_version", 0)
        )
        policy_model = None
        if self.evaluation:
            if self.episode_policy_model is None:
                raise RuntimeError("evaluation episode has no frozen policy")
            policy_model = self.episode_policy_model
            policy_version = int(
                (self.episode_policy_snapshot or {}).get(
                    "update_steps", policy_version
                )
            )

        kwargs = {"policy_model": policy_model}
        if self.evaluation:
            kwargs["generator"] = self.evaluation_generator
        with torch.no_grad():
            turn, boost, logp, value, turn_latent = self.shared_state.act(
                obs, **kwargs,
            )
        return turn, boost, logp, value, turn_latent, policy_version

    def _training_collection_open(self) -> bool:
        """Whether a valid pending decision may enter this actor's rollout."""
        return not self.evaluation

    def _accept_reward(self, raw: Any):
        if self.snake_id is None:
            raise RewardContractError("reward has no assigned snake")
        if self._episode_closed:
            raise RewardDiscontinuityError("reward received after episode closure")
        if self._reward_rebase:
            self._reward_stream.rebase(raw, snake_id=self.snake_id)
            self._reward_rebase = False
            self.discarded_reward_intervals += 1
            self._reward_has_observation = True
            return None
        reward = self._reward_stream.accept(raw, snake_id=self.snake_id)
        self._reward_has_observation = True
        return reward

    def _account_reward(self, reward) -> None:
        if reward is None:
            return
        components = reward_components(reward, self.reward_coefficients)
        total = float(sum(components.values()))
        episode_return = self.episode_return + total
        if not math.isfinite(episode_return):
            raise RewardContractError("accumulated episode reward overflow")
        if self.pending_transition is not None and self._training_collection_open():
            pending = self.pending_transition
            parts = {key: pending.reward_components.get(key, 0.0) + value
                     for key, value in components.items()}
            pending_reward = pending.reward + total
            elapsed = pending.elapsed_seconds + reward["deltas"]["elapsedSeconds"]
            alive = pending.alive_seconds + reward["deltas"]["aliveSecondsDelta"]
            if not all(math.isfinite(value) for value in
                       (*parts.values(), pending_reward, elapsed, alive)):
                raise RewardContractError("accumulated decision reward overflow")
            pending.reward = pending_reward
            pending.reward_components = parts
            pending.elapsed_seconds = elapsed
            pending.alive_seconds = alive
        self.episode_return = episode_return
        if self.evaluation and self.evaluation_episode is not None:
            self.evaluation_episode.receive(reward, components)
        if not self.evaluation:
            callback = getattr(self.shared_state, "record_received_reward", None)
            if callable(callback):
                callback(reward)

    def _observe_command_metadata(self, reward) -> None:
        """Observe native counts without treating command acceptance as physical execution."""
        control = reward["control"]
        previous = (None if self.last_reward_envelope is None
                    else self.last_reward_envelope["control"])
        applied = control["appliedCommand"]
        accepted = control["acceptedCommand"]
        old_accepted = None if previous is None else previous["acceptedCommand"]
        if (previous is not None and self._last_issued_action_id and accepted is not None
                and accepted != old_accepted and accepted["clientActionId"] is None):
            raise RewardContractError(
                "Rust accepted a PPO command without its client action identity"
            )
        self._validate_echoed_command(accepted, old_accepted)
        if applied is not None:
            old_applied = None if previous is None else previous["appliedCommand"]
            self._validate_echoed_command(applied["command"],
                                          None if old_applied is None else old_applied["command"])
        if previous is not None and not self.evaluation:
            delta = tuple(control[key] - previous[key]
                          for key in ("acceptedCount", "appliedCount", "supersededCount"))
            callback = getattr(self.shared_state, "record_command_counts", None)
            if callable(callback):
                callback(*delta)
            if (applied is not None and delta[1] > 0
                    and applied["command"]["clientActionId"] is not None):
                delay = (int(applied["firstAppliedStep"], 16)
                         - int(applied["command"]["clientTick"], 16))
                callback = getattr(self.shared_state, "record_command_delay", None)
                if delay >= 0 and callable(callback):
                    callback(delay)
        self.last_reward_envelope = reward

    def _validate_echoed_command(self, command, previous) -> None:
        if command is None or command == previous:
            return
        identity = command["clientActionId"]
        if identity is None:
            if previous is not None and previous["clientActionId"] is not None:
                raise RewardContractError("tagged command stream lost its client action identity")
            return
        value = int(identity, 16)
        if self._last_issued_action_id and value > self._action_id_counter:
            raise RewardContractError("server echoed an unknown client action identity")
        if previous is not None and previous["clientActionId"] is not None:
            if identity <= previous["clientActionId"]:
                raise RewardContractError("server replayed a client action identity")
        self._action_id_counter = max(self._action_id_counter, value)

    async def _send_action(self, ws, tick: int, turn: float, boost: float) -> str:
        """Issue one uniquely tagged command; identity survives transport retries and lifetimes."""
        if self._action_id_counter >= 2**64 - 1:
            raise RewardContractError("client action identity exhausted")
        self._action_id_counter += 1
        identity = f"{self._action_id_counter:016x}"
        await ws.send(json.dumps({"type": "action", "tick": tick, "snakeId": self.snake_id,
                                  "turn": clamp(turn, -1.0, 1.0),
                                  "boost": clamp(boost, 0.0, 1.0), "actionId": identity}))
        self._last_issued_action_id = self._action_id_counter
        if self.pending_transition is not None:
            self.pending_transition.action_id = identity
        self.last_sent_tick = tick
        self.last_sent_time = time.monotonic()
        return identity

    def _observation_and_reward(self, msg: Dict[str, Any]) -> np.ndarray:
        if msg.get("snakeId") != self.snake_id:
            raise RewardDiscontinuityError("observation belongs to another assigned snake")
        obs = np.asarray(msg.get("sensors"), dtype=np.float32)
        if obs.shape != (len(self.sensor_order),) or not np.isfinite(obs).all():
            raise SensorContractError(
                "observation does not match the admitted finite sensor vector"
            )
        raw_reward = parse_reward(msg.get("reward"))
        if raw_reward["transitionKind"] not in ("initial", "live"):
            raise RewardContractError("ordinary observation cannot close a life")
        tick = msg.get("tick")
        if (isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
                or int(raw_reward["toStep"], 16) != tick):
            raise RewardContractError("reward cutoff does not match the observation tick")
        reward = self._accept_reward(raw_reward)
        self._observe_command_metadata(raw_reward)
        self._account_reward(reward)
        self.last_sensor_tick = tick
        if self.last_assign_tick is None:
            self.last_assign_tick = tick
        return obs

    def _cutoff_value(self, obs: np.ndarray) -> float:
        model = self.episode_policy_model if self.evaluation else self.shared_state.infer_model
        with torch.no_grad():
            tensor = torch.as_tensor(obs, device=self.cfg.infer_device).unsqueeze(0)
            _mean, _boost, value = model(tensor)
        return float(value.item())

    async def _submit_environment_closure(self, bootstrap: float) -> None:
        if self.rollout and not self.evaluation:
            await self.experience_q.put((self.actor_id, list(self.rollout), bootstrap))
            self._record_admitted_rollout()
        self.rollout.clear()

    async def _handle_transition(self, msg: Dict[str, Any]) -> None:
        """Close the pending old-life action before accepting its successor."""
        raw = parse_reward(msg.get("reward"))
        terminal = raw["transitionKind"] == "terminal"
        if not terminal and raw["transitionKind"] != "truncated":
            raise RewardContractError("lifecycle message must close an old life")
        final = msg.get("finalSensors")
        final_obs = None
        if terminal:
            if final is not None:
                raise RewardContractError("terminal closure cannot carry bootstrap sensors")
        else:
            final_obs = np.asarray(final, dtype=np.float32)
            if final_obs.shape != (len(self.sensor_order),) or not np.isfinite(final_obs).all():
                raise RewardContractError("truncation requires the final old-life observation")
        if self._reward_rebase:
            self._reward_stream.discard_closure(raw, snake_id=self.snake_id)
            self._reward_rebase = False
            self.discarded_reward_intervals += 1
            reward = None
        else:
            reward = self._accept_reward(raw)
        self._observe_command_metadata(raw)
        self._account_reward(reward)
        self.last_sensor_tick = int(raw["toStep"], 16)
        bootstrap = 0.0
        if not terminal and (self.rollout or self.pending_transition is not None):
            bootstrap = self._cutoff_value(final_obs)
        if self.pending_transition is not None:
            if self._training_collection_open():
                self.pending_transition.done = float(terminal)
                self.rollout.append(self.pending_transition)
            self.pending_transition = None
        await self._submit_environment_closure(bootstrap)
        self._episode_closed = True
        lifetime = (0 if self.last_assign_tick is None
                    else max(0, self.last_sensor_tick - self.last_assign_tick))
        self._record_episode(lifetime, kind=raw["transitionKind"], reason=raw["reason"])

    async def _store_training_action(
        self,
        obs: np.ndarray,
        sampled_action: Tuple[float, float, float, float, float, int],
    ) -> None:
        """Finish the prior transition and stage the newly sampled training action."""
        turn, boost, logp, value, turn_latent, policy_version = sampled_action
        if self.pending_transition is not None:
            self.rollout.append(self.pending_transition)
            self.pending_transition = None

        if len(self.rollout) >= self.cfg.horizon:
            await self.experience_q.put(
                (self.actor_id, list(self.rollout), float(value))
            )
            self._record_admitted_rollout()
            self.rollout.clear()

        self.pending_transition = Transition(
            obs=obs,
            action_turn=turn,
            turn_latent=turn_latent,
            action_boost=boost,
            logp=logp,
            value=value,
            reward=0.0,
            done=0.0,
            policy_version=policy_version,
        )

    async def _handle_sensors(self, msg: Dict[str, Any], ws) -> None:
        if self.snake_id is None:
            return
        obs = self._observation_and_reward(msg)
        self._check_evaluation_limit()
        tick = int(msg["tick"])

        self.last_obs = obs

        if not self._should_send_action(tick):
            return

        sampled_action = self._sample_action(obs)
        turn, boost, _logp, _value, _turn_latent, _policy_version = sampled_action
        if not self.evaluation:
            await self._store_training_action(obs, sampled_action)

        self.steps += 1
        self.episode_actions += 1

        await self._send_action(ws, tick, turn, boost)

    async def _loop(self, ws, name: str) -> None:
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                await self._handle_sensors(decode_binary_sensors(raw), ws)
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "transition":
                await self._handle_transition(msg)
                continue

            if msg_type == "assign":
                await self._on_assign(
                    msg.get("snakeId"),
                    msg.get("resumeToken"),
                    reclaimed=bool(msg.get("reclaimed", False)),
                    replacement_reason=msg.get("replacementReason"),
                )
                continue

            if msg_type == "stateReplaced":
                await self._handle_state_replaced(msg, ws, name)
                continue

            if msg_type == "error":
                raise RuntimeError(
                    f"server protocol error: {msg.get('message')}"
                )

            if msg_type == "stats":
                generation = msg.get("gen")
                if generation is not None and generation != self.last_gen:
                    self.last_gen = generation
                    print(
                        f"[{self._log_label}] gen={generation}, "
                        f"tick={msg.get('tick')}, "
                        f"alive={msg.get('alive')}/{msg.get('aliveTotal')}"
                    )
                continue

            if msg_type != "sensors":
                continue

            await self._handle_sensors(msg, ws)
