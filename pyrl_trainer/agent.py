import asyncio
import json
import time
import os
import websockets
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .config import Config, PROTOCOL_VERSION
from .utils import build_index, compute_stride, default_reward, clamp


# We need PROTOCOL_VERSION from somewhere, assuming it's in config or we define it here.
# Since it was global in trainer.py, let's put it in config or here.
# For now, I'll rely on it being imported from config if I put it there, or just define it.
PROTOCOL_VERSION = 1


MAX_WS_MESSAGE_BYTES = int(os.environ.get("SLITHER_WS_MAX_MESSAGE", str(8 * 1024 * 1024)))


@dataclass
class Transition:
    obs: np.ndarray
    action_turn: float
    action_boost: float
    logp: float
    value: float
    reward: float
    done: float  # 1.0 if episode ended at this step else 0.0


class ActorClient:  # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 actor_id: int,
                 cfg: Config,
                 shared_state: Any, # Avoid circular type hint for SharedState
                 experience_q: asyncio.Queue):
        self.actor_id = actor_id
        self.cfg = cfg
        self.shared_state = shared_state
        self.experience_q = experience_q

        self.snake_id: Optional[int] = None
        self.tick_rate: int = 60
        self.stride: int = 1

        self.sensor_order: List[str] = []
        self.sensor_idx: Dict[str, int] = {}

        self.last_sent_tick: Optional[int] = None
        self.last_sent_time: float = 0.0

        self.prev_obs: Optional[np.ndarray] = None
        self.rollout: List[Transition] = []

        self.episodes: int = 0
        self.steps: int = 0

        self.last_sensor_tick: Optional[int] = None
        self.last_assign_tick: Optional[int] = None
        self.last_gen: Optional[int] = None
        self.assign_count: int = 0

        self.pending_transition: Optional[Transition] = None

    async def run(self) -> None:
        url = self.cfg.ws_url
        name = f"{self.cfg.bot_name}-{self.actor_id:03d}"
        while True:
            try:
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
                self.tick_rate = int(msg.get("tickRate", 60))
                self.stride = compute_stride(self.tick_rate, self.cfg.max_actions_per_second)
                spec = msg.get("sensorSpec") or {}
                self.sensor_order = list(spec.get("order") or [])
                self.sensor_idx = build_index(self.sensor_order)
                join = {"type": "join", "mode": "player", "name": name[:24]}
                await ws.send(json.dumps(join))
                await ws.send(json.dumps({"type": "viz", "enabled": False}))
                break
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during handshake: {msg.get('message')}")

        # Wait for assign
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "assign":
                await self._on_assign(msg.get("snakeId"))
                break
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during assign wait: {msg.get('message')}")

    async def _on_assign(self, snake_id: int) -> None:
        prev = self.snake_id
        now_tick = getattr(self, "last_sensor_tick", None)
        lived = None
        if getattr(self, "last_assign_tick", None) is not None and now_tick is not None:
            lat = getattr(self, "last_assign_tick", None)
            lived = int(now_tick - lat)

        self.assign_count = int(getattr(self, "assign_count", 0)) + 1
        
        # Extract size from previous observation if available
        size_str = ""
        if self.prev_obs is not None and "size_norm" in self.sensor_idx:
            size_val = self.prev_obs[self.sensor_idx["size_norm"]]
            size_str = f", size_norm={size_val:.3f}"

        if lived is None:
            print(f"[actor {self.actor_id}] assign {prev} -> {snake_id}, assigns={self.assign_count}")
        else:
            print(f"[actor {self.actor_id}] assign {prev} -> {snake_id}, lived_ticks={lived}{size_str}, assigns={self.assign_count}")

            # *** CRITICAL FIX: Handle Termination ***
            # The previous snake died. If we have a pending transition, finalize it as terminal.
            if self.pending_transition is not None:
                # We don't have a "next obs" because we died, but essentially we transitioned to a terminal state.
                # Use a small death penalty or just the regular reward up to this point.
                # Let's say reward is -0.5 for dying.
                self.pending_transition.reward = -0.5
                self.pending_transition.done = 1.0
                self.rollout.append(self.pending_transition)
                
            # Flush the rollout immediately so the learner sees the death event.
            if self.rollout:
                # bootstrap 0.0 because it's terminal
                await self.experience_q.put((self.actor_id, list(self.rollout), 0.0))
                self.rollout.clear()

        self.snake_id = int(snake_id)
        self.prev_obs = None
        self.pending_transition = None # Clear pending since we are a new snake
        self.last_sent_tick = None
        self.episodes += 1
        self.last_assign_tick = now_tick

    def _should_send_action(self, tick: int) -> bool:
        if self.last_sent_tick == tick:
            return False
        if self.stride <= 1:
            return True
        return (tick % self.stride) == 0

    async def _loop(self, ws) -> None:
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue

            msg = json.loads(raw)
            t = msg.get("type")

            if t == "assign":
                await self._on_assign(msg.get("snakeId"))
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

            if self.snake_id is None:
                continue
            if int(msg.get("snakeId", -1)) != self.snake_id:
                continue

            tick = int(msg.get("tick", 0))
            self.last_sensor_tick = tick
            sensors = msg.get("sensors") or []
            obs = np.asarray(sensors, dtype=np.float32)
            if obs.shape[0] != len(self.sensor_order):
                continue

            # *** CRITICAL FIX: Transition Alignment ***
            # 1. Finalize the pending transition from the PREVIOUS tick using CURRENT obs.
            if self.pending_transition is not None:
                # Reward is based on the change from prev_obs (stored in pending) to obs (current)
                r = default_reward(self.pending_transition.obs, obs, self.sensor_idx)
                self.pending_transition.reward = r
                self.rollout.append(self.pending_transition)
                self.pending_transition = None

            # 3. Choose NEW action for the CURRENT state (and get its value)
            self.prev_obs = obs # pending_transition source obs
            
            with torch.no_grad():
                turn, boost, logp, value = self.shared_state.act(obs, turn_std=self.cfg.turn_std)

            # 2. Check if rollout is full and send it
            # We do this AFTER acting so we have 'value' to use as bootstrap
            if len(self.rollout) >= self.cfg.horizon:
                # bootstrap with current value.
                # If we die later, this chunk is valid (non-terminal end).
                await self.experience_q.put((self.actor_id, list(self.rollout), float(value)))
                self.rollout.clear()

            # 4. Store as PENDING.
            # done is 0.0 for now. It becomes 1.0 only if we die.
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

            if self._should_send_action(tick):
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
