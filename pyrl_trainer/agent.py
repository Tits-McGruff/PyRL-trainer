"""Actor client for the Slither training server."""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import websockets

from .config import Config, PROTOCOL_VERSION
from .sensor_contract import SensorContract, SensorContractError
from .utils import (
    build_index,
    clamp,
    compute_stride,
    default_reward_components,
)


MAX_WS_MESSAGE_BYTES = int(
    os.environ.get("SLITHER_WS_MAX_MESSAGE", str(8 * 1024 * 1024))
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


class ActorClient:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """WebSocket client that controls a single snake."""

    def __init__(
        self,
        actor_id: int,
        cfg: Config,
        shared_state: Any,  # Avoid circular type hint for SharedState.
        experience_q: asyncio.Queue,
    ):
        self.actor_id = actor_id
        self.cfg = cfg
        self.shared_state = shared_state
        self.experience_q = experience_q

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

    def _reset_per_snake_state(self) -> None:
        self.last_obs = None
        self.pending_transition = None
        self.rollout.clear()
        self.last_sent_tick = None
        self.episode_return = 0.0
        self.episode_actions = 0

    def _reset_connection_state(self) -> None:
        """Reset socket-local fields while retaining reclaimable snake state."""
        self.sensor_order = []
        self.sensor_idx = {}
        self.last_sent_time = 0.0

    async def _truncate_disconnected_rollout(self) -> None:
        """Flush completed PPO work without inventing a terminal transition."""
        bootstrap = (
            float(self.pending_transition.value)
            if self.pending_transition is not None
            else 0.0
        )
        self.pending_transition = None

        if self.rollout:
            await self.experience_q.put(
                (self.actor_id, list(self.rollout), bootstrap)
            )
            self.rollout.clear()

        self.last_obs = None
        self.last_sent_tick = None

    def _record_reward_components(self, components: Dict[str, float]) -> None:
        callback = getattr(self.shared_state, "record_reward_components", None)
        if callable(callback):
            callback(components)

    def _record_episode(self, lifetime_ticks: int) -> None:
        callback = getattr(self.shared_state, "record_episode", None)
        if callable(callback):
            callback(self.episode_return, lifetime_ticks)

    async def run(self) -> None:
        """Connect to the server and keep the control loop alive."""
        url = self.cfg.ws_url
        name = f"{self.cfg.bot_name}-{self.actor_id:03d}"
        while True:
            try:
                self._reset_connection_state()
                async with websockets.connect(
                    url, max_size=MAX_WS_MESSAGE_BYTES
                ) as ws:
                    await self._handshake(ws, name)
                    await self._loop(ws, name)
            except asyncio.CancelledError:
                raise
            except SensorContractError:
                raise
            except Exception as exc:  # pylint: disable=broad-exception-caught
                await self._truncate_disconnected_rollout()
                print(
                    f"[actor {self.actor_id}] disconnected, "
                    f"reason={type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(0.5)

    async def _handshake(self, ws, name: str) -> None:
        hello = {
            "type": "hello",
            "clientType": "bot",
            "version": PROTOCOL_VERSION,
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
                raise RuntimeError(
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

    async def _handle_state_replaced(
        self, msg: Dict[str, Any], ws, name: str
    ) -> None:
        """End the old episode and join imported authority without a stale token."""
        welcome = msg.get("welcome")
        if not isinstance(welcome, dict):
            raise RuntimeError("stateReplaced omitted the replacement welcome")
        await self._finalize_terminal_episode(death_penalty=0.0)
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
    ) -> None:
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
            print(
                f"[actor {self.actor_id}] reclaimed snake {snake_id}, "
                f"assigns={self.assign_count}"
            )
            return

        size_str = ""
        if self.last_obs is not None and "size_norm" in self.sensor_idx:
            size_value = self.last_obs[self.sensor_idx["size_norm"]]
            size_str = f", size_norm={size_value:.3f}"

        if lived is None:
            print(
                f"[actor {self.actor_id}] assign {previous_snake} -> {snake_id}, "
                f"assigns={self.assign_count}"
            )
        else:
            print(
                f"[actor {self.actor_id}] assign {previous_snake} -> {snake_id}, "
                f"lived_ticks={lived}{size_str}, assigns={self.assign_count}"
            )

        if previous_snake is not None:
            await self._finalize_terminal_episode()

        self.snake_id = int(snake_id)
        self.resume_token = resume_token
        self._reset_per_snake_state()
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

    async def _finalize_terminal_episode(
        self, death_penalty: float = -0.5
    ) -> None:
        had_episode = (
            self.snake_id is not None
            or self.episode_actions > 0
            or self.pending_transition is not None
            or bool(self.rollout)
        )
        if self.pending_transition is not None:
            penalty = float(death_penalty)
            self.pending_transition.reward += penalty
            self.pending_transition.done = 1.0
            self.rollout.append(self.pending_transition)
            self.pending_transition = None
            self.episode_return += penalty
            if penalty:
                self._record_reward_components({"death": penalty})

        if self.rollout:
            await self.experience_q.put(
                (self.actor_id, list(self.rollout), 0.0)
            )
            self.rollout.clear()

        lifetime_ticks = 0
        if self.last_assign_tick is not None and self.last_sensor_tick is not None:
            lifetime_ticks = max(
                0, int(self.last_sensor_tick - self.last_assign_tick)
            )
        if had_episode:
            self._record_episode(lifetime_ticks)

    async def _handle_sensors(self, msg: Dict[str, Any], ws) -> None:
        if self.snake_id is None:
            return
        if int(msg.get("snakeId", -1)) != self.snake_id:
            return

        tick = int(msg.get("tick", 0))
        self.last_sensor_tick = tick
        if self.last_assign_tick is None:
            self.last_assign_tick = tick

        sensors = msg.get("sensors") or []
        obs = np.asarray(sensors, dtype=np.float32)
        if obs.shape[0] != len(self.sensor_order):
            return

        if self.last_obs is not None and self.pending_transition is not None:
            components = default_reward_components(
                self.last_obs, obs, self.sensor_idx
            )
            reward = float(sum(components.values()))
            self.pending_transition.reward += reward
            self.episode_return += reward
            self._record_reward_components(components)

        self.last_obs = obs

        if not self._should_send_action(tick):
            return

        with torch.no_grad():
            (
                turn,
                boost,
                logp,
                value,
                turn_latent,
            ) = self.shared_state.act(obs, turn_std=self.cfg.turn_std)

        if self.pending_transition is not None:
            self.rollout.append(self.pending_transition)
            self.pending_transition = None

        if len(self.rollout) >= self.cfg.horizon:
            await self.experience_q.put(
                (self.actor_id, list(self.rollout), float(value))
            )
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
        )
        self.steps += 1
        self.episode_actions += 1

        action_msg = {
            "type": "action",
            "tick": tick,
            "snakeId": self.snake_id,
            "turn": clamp(turn, -1.0, 1.0),
            "boost": clamp(boost, 0.0, 1.0),
        }
        await ws.send(json.dumps(action_msg))
        self.last_sent_tick = tick
        self.last_sent_time = time.monotonic()

    async def _loop(self, ws, name: str) -> None:
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "assign":
                await self._on_assign(
                    msg.get("snakeId"),
                    msg.get("resumeToken"),
                    reclaimed=bool(msg.get("reclaimed", False)),
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
                        f"[actor {self.actor_id}] gen={generation}, "
                        f"tick={msg.get('tick')}, "
                        f"alive={msg.get('alive')}/{msg.get('aliveTotal')}"
                    )
                continue

            if msg_type != "sensors":
                continue

            await self._handle_sensors(msg, ws)
