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
from .utils import build_index, compute_stride, default_reward, clamp


MAX_WS_MESSAGE_BYTES = int(os.environ.get("SLITHER_WS_MAX_MESSAGE", str(8 * 1024 * 1024)))


@dataclass
class Transition:  # pylint: disable=too-few-public-methods
    """Single environment transition for PPO."""
    obs: np.ndarray
    action_turn: float
    action_boost: float
    logp: float
    value: float
    reward: float
    done: float  # 1.0 if episode ended at this step else 0.0


class ActorClient:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """WebSocket client that controls a single snake."""
    def __init__(self,
                 actor_id: int,
                 cfg: Config,
                 shared_state: Any,  # Avoid circular type hint for SharedState.
                 experience_q: asyncio.Queue):
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

    def _reset_per_snake_state(self) -> None:
        self.last_obs = None
        self.pending_transition = None
        self.rollout.clear()
        self.last_sent_tick = None

    def _reset_connection_state(self) -> None:
        self.snake_id = None
        self.sensor_order = []
        self.sensor_idx = {}
        self.last_sensor_tick = None
        self.last_assign_tick = None
        self.last_gen = None
        self.last_sent_time = 0.0
        self._reset_per_snake_state()

    async def run(self) -> None:
        """Connect to the server and keep the control loop alive."""
        url = self.cfg.ws_url
        name = f"{self.cfg.bot_name}-{self.actor_id:03d}"
        while True:
            try:
                self._reset_connection_state()
                async with websockets.connect(url, max_size=MAX_WS_MESSAGE_BYTES) as ws:
                    await self._handshake(ws, name)
                    await self._loop(ws)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(
                    f"[actor {self.actor_id}] disconnected, reason={type(e).__name__}: {e}"
                )
                await asyncio.sleep(0.5)

    async def _handshake(self, ws, name: str) -> None:
        hello = {"type": "hello", "clientType": "bot", "version": PROTOCOL_VERSION}
        await ws.send(json.dumps(hello))

        # Wait for welcome
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "welcome":
                if msg.get("protocolVersion") != PROTOCOL_VERSION:
                    raise RuntimeError("server welcome did not confirm Protocol 2")
                self.tick_rate = int(msg.get("tickRate", 60))
                self.stride = compute_stride(self.tick_rate, self.cfg.max_actions_per_second)
                spec = msg.get("sensorSpec") or {}
                self.sensor_order = list(spec.get("order") or [])
                self.sensor_idx = build_index(self.sensor_order)
                join = {"type": "join", "mode": "player", "name": name[:24]}
                if self.resume_token:
                    join["resumeToken"] = self.resume_token
                await ws.send(json.dumps(join))
                break
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during handshake: {msg.get('message')}")

        # Wait for assign
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "reclaimResult" and not msg.get("reclaimed"):
                self.resume_token = None
                await ws.send(json.dumps({"type": "join", "mode": "player", "name": name[:24]}))
                continue
            if msg.get("type") == "assign":
                await self._on_assign(msg.get("snakeId"), msg.get("resumeToken"))
                break
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during assign wait: {msg.get('message')}")

    async def _on_assign(self, snake_id: int, resume_token: str) -> None:
        if not isinstance(snake_id, int) or snake_id <= 0:
            raise RuntimeError("server assignment omitted a valid snakeId")
        if not isinstance(resume_token, str) or not resume_token:
            raise RuntimeError("Protocol 2 assignment omitted resumeToken")
        prev = self.snake_id
        now_tick = getattr(self, "last_sensor_tick", None)
        lived = None
        if getattr(self, "last_assign_tick", None) is not None and now_tick is not None:
            lat = getattr(self, "last_assign_tick", None)
            lived = int(now_tick - lat)

        self.assign_count = int(getattr(self, "assign_count", 0)) + 1

        # Extract size from previous observation if available
        size_str = ""
        if self.last_obs is not None and "size_norm" in self.sensor_idx:
            size_val = self.last_obs[self.sensor_idx["size_norm"]]
            size_str = f", size_norm={size_val:.3f}"

        if lived is None:
            print(
                f"[actor {self.actor_id}] assign {prev} -> {snake_id}, "
                f"assigns={self.assign_count}"
            )
        else:
            print(
                f"[actor {self.actor_id}] assign {prev} -> {snake_id}, "
                f"lived_ticks={lived}{size_str}, assigns={self.assign_count}"
            )
        if prev is not None:
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
            if (time.time() - self.last_sent_time) < min_interval:
                return False
        return True

    async def _finalize_terminal_episode(self, death_penalty: float = -0.5) -> None:
        if self.pending_transition is not None:
            self.pending_transition.reward += float(death_penalty)
            self.pending_transition.done = 1.0
            self.rollout.append(self.pending_transition)
            self.pending_transition = None

        if self.rollout:
            await self.experience_q.put((self.actor_id, list(self.rollout), 0.0))
            self.rollout.clear()

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
            r = default_reward(self.last_obs, obs, self.sensor_idx)
            self.pending_transition.reward += r

        self.last_obs = obs

        if not self._should_send_action(tick):
            return

        with torch.no_grad():
            turn, boost, logp, value = self.shared_state.act(obs, turn_std=self.cfg.turn_std)

        if self.pending_transition is not None:
            self.rollout.append(self.pending_transition)
            self.pending_transition = None

        if len(self.rollout) >= self.cfg.horizon:
            await self.experience_q.put((self.actor_id, list(self.rollout), float(value)))
            self.rollout.clear()

        self.pending_transition = Transition(
            obs=obs,
            action_turn=turn,
            action_boost=boost,
            logp=logp,
            value=value,
            reward=0.0,
            done=0.0
        )
        self.steps += 1

        action_msg = {
            "type": "action",
            "tick": tick,
            "snakeId": self.snake_id,
            "turn": clamp(turn, -1.0, 1.0),
            "boost": clamp(boost, 0.0, 1.0),
        }
        await ws.send(json.dumps(action_msg))
        self.last_sent_tick = tick
        self.last_sent_time = time.time()

    async def _loop(self, ws) -> None:
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue

            msg = json.loads(raw)
            t = msg.get("type")

            if t == "assign":
                await self._on_assign(msg.get("snakeId"), msg.get("resumeToken"))
                continue

            if t == "error":
                m = msg.get("message")
                raise RuntimeError(f"server protocol error: {m}")

            if t == "stats":
                gen = msg.get("gen")
                if gen is not None and gen != getattr(self, "last_gen", None):
                    self.last_gen = gen
                    print(
                        f"[actor {self.actor_id}] gen={gen}, tick={msg.get('tick')}, "
                        f"alive={msg.get('alive')}/{msg.get('aliveTotal')}"
                    )
                continue

            if t != "sensors":
                continue

            await self._handle_sensors(msg, ws)
