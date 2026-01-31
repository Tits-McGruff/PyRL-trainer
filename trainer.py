from pathlib import Path
"""Slither bot trainer with PPO-style learning and TOML-based configuration."""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# pip install websockets torch
import websockets
import torch
from torch import nn
from torch import optim
PROTOCOL_VERSION = 1


MAX_WS_MESSAGE_BYTES = int(os.environ.get("SLITHER_WS_MAX_MESSAGE", str(8 * 1024 * 1024)))
# --- Config loading from config.toml (auto-generated if missing), env vars override ---
import sys

try:
    if sys.version_info >= (3, 11):
        import tomllib as _toml_reader  # type: ignore
    else:
        import tomli as _toml_reader  # type: ignore
except ImportError:
    _toml_reader = None  # type: ignore


def _toml_quote(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(float(v))
    if isinstance(v, str):
        return _toml_quote(v)
    raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _toml_dump_sections(sections: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for section_name, kv in sections.items():
        lines.append(f"[{section_name}]")
        for k, v in kv.items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _read_config_toml(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if _toml_reader is not None:
        return _toml_reader.loads(raw.decode("utf-8"))
    raise RuntimeError("Reading TOML requires Python 3.11+ (tomllib) or `pip install tomli`.")


def _write_config_toml(path: Path, sections: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Slither trainer configuration\n"
        "# Auto-generated because it was missing.\n"
        "# Environment variables prefixed with SLITHER_ override values in this file.\n\n"
    )
    body = _toml_dump_sections(sections)
    path.write_text(header + body, encoding="utf-8")


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None else default


def _default_train_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_net_hidden() -> int:
    return 512 if torch.cuda.is_available() else 256


def _default_net_layers() -> int:
    return 4 if torch.cuda.is_available() else 2


@dataclass
class Config:
    ws_url: str = "ws://localhost:5174"
    bot_name: str = "NNTrainer"
    actors: int = 4

    max_actions_per_tick: int = 1
    max_actions_per_second: int = 120

    horizon: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    lr: float = 3e-4
    minibatch: int = 256
    epochs: int = 4

    train_device: str = ""
    infer_device: str = "cpu"

    net_hidden: int = 0
    net_layers: int = 0

    turn_std: float = 0.15

    log_every_seconds: float = 5.0

    ckpt_dir: str = "./checkpoints"
    save_every_updates: int = 100
    keep_last: int = 5


def _defaults_config() -> Config:
    cfg = Config()
    cfg.train_device = _default_train_device()
    cfg.net_hidden = _default_net_hidden()
    cfg.net_layers = _default_net_layers()
    return cfg


def _config_sections(cfg: Config) -> Dict[str, Dict[str, Any]]:
    return {
        "connection": {
            "ws_url": cfg.ws_url,
            "bot_name": cfg.bot_name,
            "actors": cfg.actors,
        },
        "server": {
            "max_actions_per_tick": cfg.max_actions_per_tick,
            "max_actions_per_second": cfg.max_actions_per_second,
        },
        "training": {
            "horizon": cfg.horizon,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "ppo_clip": cfg.ppo_clip,
            "ent_coef": cfg.ent_coef,
            "vf_coef": cfg.vf_coef,
            "lr": cfg.lr,
            "minibatch": cfg.minibatch,
            "epochs": cfg.epochs,
            "turn_std": cfg.turn_std,
        },
        "devices": {
            "train_device": cfg.train_device,
            "infer_device": cfg.infer_device,
        },
        "model": {
            "net_hidden": cfg.net_hidden,
            "net_layers": cfg.net_layers,
        },
        "logging": {
            "log_every_seconds": cfg.log_every_seconds,
        },
        "checkpointing": {
            "ckpt_dir": cfg.ckpt_dir,
            "save_every_updates": cfg.save_every_updates,
            "keep_last": cfg.keep_last,
        },
    }


def _flatten_sections(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[kk] = vv
        else:
            out[k] = v
    return out


def _merge_cfg(base: Config, flat: Dict[str, Any]) -> Config:
    cfg = Config(**asdict(base))
    for k, v in flat.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _apply_env_overrides(cfg: Config) -> Config:
    cfg.ws_url = _env_str("SLITHER_WS_URL", cfg.ws_url)
    cfg.bot_name = _env_str("SLITHER_BOT_NAME", cfg.bot_name)
    cfg.actors = _env_int("SLITHER_ACTORS", cfg.actors)

    cfg.max_actions_per_tick = _env_int("SLITHER_MAX_APT", cfg.max_actions_per_tick)
    cfg.max_actions_per_second = _env_int("SLITHER_MAX_APS", cfg.max_actions_per_second)

    cfg.horizon = _env_int("SLITHER_HORIZON", cfg.horizon)
    cfg.gamma = _env_float("SLITHER_GAMMA", cfg.gamma)
    cfg.gae_lambda = _env_float("SLITHER_GAE_LAMBDA", cfg.gae_lambda)
    cfg.ppo_clip = _env_float("SLITHER_PPO_CLIP", cfg.ppo_clip)
    cfg.ent_coef = _env_float("SLITHER_ENT_COEF", cfg.ent_coef)
    cfg.vf_coef = _env_float("SLITHER_VF_COEF", cfg.vf_coef)
    cfg.lr = _env_float("SLITHER_LR", cfg.lr)
    cfg.minibatch = _env_int("SLITHER_MINIBATCH", cfg.minibatch)
    cfg.epochs = _env_int("SLITHER_EPOCHS", cfg.epochs)

    cfg.train_device = _env_str("SLITHER_TRAIN_DEVICE", cfg.train_device or _default_train_device())
    cfg.infer_device = _env_str("SLITHER_INFER_DEVICE", cfg.infer_device)

    cfg.net_hidden = _env_int("SLITHER_NET_HIDDEN", cfg.net_hidden or _default_net_hidden())
    cfg.net_layers = _env_int("SLITHER_NET_LAYERS", cfg.net_layers or _default_net_layers())

    cfg.turn_std = _env_float("SLITHER_TURN_STD", cfg.turn_std)
    cfg.log_every_seconds = _env_float("SLITHER_LOG_EVERY", cfg.log_every_seconds)

    cfg.ckpt_dir = _env_str("SLITHER_CKPT_DIR", cfg.ckpt_dir)
    cfg.save_every_updates = _env_int("SLITHER_SAVE_EVERY_UPDATES", cfg.save_every_updates)
    cfg.keep_last = _env_int("SLITHER_KEEP_LAST", cfg.keep_last)

    return cfg


def load_or_create_config(config_path: str = "config.toml") -> Config:
    path = Path(config_path)
    defaults = _defaults_config()

    if not path.exists():
        _write_config_toml(path, _config_sections(defaults))

    try:
        data = _read_config_toml(path)
        flat = _flatten_sections(data)
        cfg = _merge_cfg(defaults, flat)
    except Exception:
        cfg = defaults

    cfg = _apply_env_overrides(cfg)

    if not cfg.train_device:
        cfg.train_device = _default_train_device()
    if cfg.net_hidden <= 0:
        cfg.net_hidden = _default_net_hidden()
    if cfg.net_layers <= 0:
        cfg.net_layers = _default_net_layers()

    return cfg



class PolicyValueNet(nn.Module):
    """
    Simple MLP policy/value head.
    Outputs:
      - turn_mean in [-1, 1] via tanh
      - boost_logit, converted to probability with sigmoid
      - value scalar
    """
    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        layers = int(max(1, layers))

        blocks: List[nn.Module] = []
        # First layer maps obs -> hidden
        blocks.append(nn.Linear(obs_dim, hidden))
        blocks.append(nn.ReLU())

        # Additional hidden layers (hidden -> hidden)
        for _ in range(layers - 1):
            blocks.append(nn.Linear(hidden, hidden))
            blocks.append(nn.ReLU())

        self.net = nn.Sequential(*blocks)
        self.turn_head = nn.Linear(hidden, 1)
        self.boost_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        turn_mean = torch.tanh(self.turn_head(x)).squeeze(-1)
        boost_logit = self.boost_head(x).squeeze(-1)
        value = self.value_head(x).squeeze(-1)
        return turn_mean, boost_logit, value


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def build_index(order: List[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(order)}


def compute_stride(tick_rate: int, max_actions_per_second: int) -> int:
    """
    Ensures we never exceed maxActionsPerSecond.
    If tick_rate <= max APS, stride is 1, send each tick.
    If tick_rate > max APS, stride > 1, send every stride ticks, last action held by server.
    """
    if max_actions_per_second <= 0:
        return 1
    return max(1, math.ceil(tick_rate / max_actions_per_second))


def default_reward(prev_obs: Optional[np.ndarray],
                   obs: np.ndarray,
                   idx: Dict[str, int]) -> float:
    """
    Placeholder reward, purely to make the training loop runnable.
    Replace with your real reward shaping.
    Uses deltas in size_norm and points_pct, plus a tiny alive bonus.
    All inputs are in [-1, 1] per manual.
    """
    alive_bonus = 0.001
    if prev_obs is None:
        return alive_bonus

    r = alive_bonus

    if "size_norm" in idx:
        r += 0.05 * float(obs[idx["size_norm"]] - prev_obs[idx["size_norm"]])

    if "points_pct" in idx:
        r += 0.02 * float(obs[idx["points_pct"]] - prev_obs[idx["points_pct"]])

    # Mild safety shaping, keep it small; negative when close to hazards or wall.
    # hazard_i and wall_i are clearance, -1 very close, +1 clear, so penalize low values.
    hazard_labels = [k for k in idx.keys() if k.startswith("hazard_")]
    wall_labels = [k for k in idx.keys() if k.startswith("wall_")]
    if hazard_labels:
        h = np.mean([obs[idx[k]] for k in hazard_labels])
        r += 0.01 * float(h)  # more clearance, slightly positive
    if wall_labels:
        w = np.mean([obs[idx[k]] for k in wall_labels])
        r += 0.01 * float(w)

    return float(r)


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
                 shared_state: "SharedState",
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

        # Debug counters: assignment churn, tick tracking
        self.last_sensor_tick: Optional[int] = None
        self.last_assign_tick: Optional[int] = None
        self.last_gen: Optional[int] = None
        self.assign_count: int = 0

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
                self._on_assign(msg.get("snakeId"))
                break
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during assign wait: {msg.get('message')}")

    def _on_assign(self, snake_id: int) -> None:
        prev = self.snake_id
        now_tick = getattr(self, "last_sensor_tick", None)
        lived = None
        if getattr(self, "last_assign_tick", None) is not None and now_tick is not None:
            lat = getattr(self, "last_assign_tick", None)
            lived = int(now_tick - lat)

        self.assign_count = int(getattr(self, "assign_count", 0)) + 1
        if lived is None:
            print(f"[actor {self.actor_id}] assign {prev} -> {snake_id}, assigns={self.assign_count}")
        else:
            print(f"[actor {self.actor_id}] assign {prev} -> {snake_id}, lived_ticks={lived}, assigns={self.assign_count}")

        self.snake_id = int(snake_id)
        self.prev_obs = None
        self.rollout.clear()
        self.last_sent_tick = None
        self.episodes += 1
        self.last_assign_tick = now_tick

    def _should_send_action(self, tick: int) -> bool:
        """
        Enforces:
          - maxActionsPerTick = 1: do not send twice for same tick
          - maxActionsPerSecond: stride-based gating, based on welcome.tickRate
        """
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
                self._on_assign(msg.get("snakeId"))
                continue

            if t == "error":
                # Per manual, message can be omitted.
                m = msg.get("message")
                raise RuntimeError(f"server protocol error: {m}")

            if t == "stats":
                # Optional; could log if you want.
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
                # Stale sensors, ignore.
                continue

            tick = int(msg.get("tick", 0))
            self.last_sensor_tick = tick
            sensors = msg.get("sensors") or []
            obs = np.asarray(sensors, dtype=np.float32)
            if obs.shape[0] != len(self.sensor_order):
                # Forward compatibility; ignore malformed frames.
                continue

            # Reward computed from obs and previous obs, replace with your shaping.
            reward = default_reward(self.prev_obs, obs, self.sensor_idx)
            self.prev_obs = obs

            # Policy inference
            with torch.no_grad():
                turn, boost, logp, value = self.shared_state.act(obs, turn_std=self.cfg.turn_std)

            # Episode handling: server indicates death by sending a new assign.
            # So done is 0 here; we will flush and mark done when assign arrives.
            done = 0.0

            # Store transition
            self.rollout.append(
                Transition(
                    obs=obs,
                    action_turn=turn,
                    action_boost=boost,
                    logp=logp,
                    value=value,
                    reward=reward,
                    done=done,
                )
            )
            self.steps += 1

            # Send action, rate limited
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

            # Flush rollout when horizon reached
            if len(self.rollout) >= self.cfg.horizon:
                await self.experience_q.put((self.actor_id, self.rollout))
                self.rollout = []


class SharedState:  # pylint: disable=too-many-instance-attributes
    """
    Shared policy/value network weights.
    Learner updates the train_model; actors use an inference copy to avoid training contention.
    """
    def __init__(self, obs_dim: int, cfg: Config):
        self.cfg = cfg
        self.obs_dim = obs_dim

        self.train_model = PolicyValueNet(obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers).to(cfg.train_device)
        self.infer_model = PolicyValueNet(obs_dim, hidden=cfg.net_hidden, layers=cfg.net_layers).to(cfg.infer_device)

        self._sync_infer_from_train()

        self.optimizer = optim.Adam(self.train_model.parameters(), lr=cfg.lr)
        self.update_steps = 0

    def _sync_infer_from_train(self) -> None:
        self.infer_model.load_state_dict(self.train_model.state_dict())

    def sync_infer_from_train(self) -> None:
        """Sync the inference model weights from the training model."""
        self._sync_infer_from_train()

    def act(self, obs: np.ndarray, turn_std: float) -> Tuple[float, float, float, float]:
        """
        Returns turn, boost, logp, value.
        """
        o = torch.tensor(obs, dtype=torch.float32, device=self.cfg.infer_device).unsqueeze(0)
        turn_mean, boost_logit, value = self.infer_model(o)
        turn_mean = turn_mean.item()
        boost_prob = torch.sigmoid(boost_logit).item()
        value = value.item()

        # Simple stochasticity: Normal around mean for turn, Bernoulli for boost.
        # Note: logp is approximate because we clamp to [-1,1] after sampling.
        turn_sample = np.random.normal(loc=turn_mean, scale=turn_std)
        turn = float(np.clip(turn_sample, -1.0, 1.0))
        boost = 1.0 if (np.random.rand() < boost_prob) else 0.0

        # Approx logp
        # Normal log prob for unclipped sample, and Bernoulli log prob for boost.
        normal_logp = -0.5 * (((turn_sample - turn_mean) / turn_std) ** 2) - math.log(turn_std) - 0.5 * math.log(2 * math.pi)
        bern_logp = math.log(boost_prob + 1e-8) if boost > 0.5 else math.log(1.0 - boost_prob + 1e-8)
        logp = float(normal_logp + bern_logp)

        return turn, boost, logp, value

    def ppo_update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        One PPO update over the provided batch.
        Expects tensors on train_device.
        """
        obs = batch["obs"]
        act_turn = batch["act_turn"]
        act_boost = batch["act_boost"]
        old_logp = batch["old_logp"]
        returns = batch["returns"]
        adv = batch["adv"]

        turn_mean, boost_logit, value = self.train_model(obs)
        boost_prob = torch.sigmoid(boost_logit)

        # Recompute approximate logp under current policy.
        # Same approximation as actor; good enough as a starting point.
        turn_std = torch.tensor(self.cfg.turn_std, device=obs.device)
        normal = torch.distributions.Normal(turn_mean, turn_std)
        logp_turn = normal.log_prob(act_turn)

        bern = torch.distributions.Bernoulli(probs=boost_prob)
        logp_boost = bern.log_prob(act_boost)

        logp = logp_turn + logp_boost
        ratio = torch.exp(logp - old_logp)

        clipped = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
        policy_loss = -(torch.min(ratio * adv, clipped * adv)).mean()

        value_loss = ((returns - value) ** 2).mean()

        entropy = (normal.entropy() + bern.entropy()).mean()

        loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % 1 == 0:
            self._sync_infer_from_train()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }



def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _list_ckpts(ckpt_dir: str) -> list[Path]:
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    return sorted(d.glob("ckpt_*.pt"))


def _rotate_ckpts(ckpt_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = _list_ckpts(ckpt_dir)
    if len(ckpts) <= keep_last:
        return
    for p in ckpts[: max(0, len(ckpts) - keep_last)]:
        try:
            p.unlink()
        except OSError:
            pass


def save_checkpoint(cfg: Config, shared_state: SharedState) -> None:
    _ensure_dir(cfg.ckpt_dir)
    step = int(shared_state.update_steps)

    payload = {
        "update_steps": step,
        "obs_dim": int(shared_state.obs_dim),
        "net_hidden": int(cfg.net_hidden),
        "net_layers": int(cfg.net_layers),
        "model": shared_state.train_model.state_dict(),
        "optimizer": shared_state.optimizer.state_dict(),
    }

    ckpt_dir = Path(cfg.ckpt_dir)
    latest_path = ckpt_dir / "latest.pt"
    latest_arch_path = ckpt_dir / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"

    torch.save(payload, latest_path)
    torch.save(payload, latest_arch_path)

    numbered = ckpt_dir / f"ckpt_h{cfg.net_hidden}_l{cfg.net_layers}_{step:08d}.pt"
    torch.save(payload, numbered)

    _rotate_ckpts(cfg.ckpt_dir, cfg.keep_last)


def load_checkpoint_if_present(cfg: Config, shared_state: SharedState) -> Optional[Path]:
    # pylint: disable=too-many-return-statements,too-many-branches
    """
    Loads a checkpoint if present and compatible with the current model shape.
    Returns the Path that was loaded, or None if no compatible checkpoint exists.
    """
    ckpt_dir = Path(cfg.ckpt_dir)
    latest_arch = ckpt_dir / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"
    latest = ckpt_dir / "latest.pt"

    cand = None
    if latest_arch.exists():
        cand = latest_arch
    elif latest.exists():
        cand = latest
    else:
        return None

    try:
        payload = torch.load(cand, map_location=cfg.train_device)
    except Exception:
        return None

    # Quick compatibility checks to avoid noisy state_dict tracebacks.
    if int(payload.get("obs_dim", -1)) != int(shared_state.obs_dim):
        return None

    ckpt_hidden = payload.get("net_hidden", None)
    ckpt_layers = payload.get("net_layers", None)
    if ckpt_hidden is not None and int(ckpt_hidden) != int(cfg.net_hidden):
        return None
    if ckpt_layers is not None and int(ckpt_layers) != int(cfg.net_layers):
        return None

    model_sd = payload.get("model", None)
    opt_sd = payload.get("optimizer", None)
    if not isinstance(model_sd, dict) or not isinstance(opt_sd, dict):
        return None

    current_sd = shared_state.train_model.state_dict()

    # If keys differ, it's almost certainly an architecture mismatch.
    if set(model_sd.keys()) != set(current_sd.keys()):
        return None

    # If any tensor shape differs, skip.
    for k, v in model_sd.items():
        if k not in current_sd:
            return None
        try:
            if tuple(v.shape) != tuple(current_sd[k].shape):
                return None
        except Exception:
            return None

    shared_state.train_model.load_state_dict(model_sd)
    shared_state.optimizer.load_state_dict(opt_sd)
    shared_state.update_steps = int(payload.get("update_steps", 0))
    shared_state.sync_infer_from_train()
    return cand


def gae(rollout: List[Transition], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes advantages and returns for a single rollout.
    done is assumed 0 inside rollout; terminal handling can be added by setting done when assign arrives.
    """
    t_len = len(rollout)
    adv = np.zeros(t_len, dtype=np.float32)
    ret = np.zeros(t_len, dtype=np.float32)

    last_gae = 0.0
    last_value = 0.0  # bootstrap off 0 by default; can be replaced with value(next_obs) if you keep next obs

    for t in reversed(range(t_len)):
        r = rollout[t].reward
        v = rollout[t].value
        d = rollout[t].done
        next_v = last_value if t == t_len - 1 else rollout[t + 1].value
        delta = r + gamma * (1.0 - d) * next_v - v
        last_gae = delta + gamma * lam * (1.0 - d) * last_gae
        adv[t] = last_gae

    ret = adv + np.asarray([tr.value for tr in rollout], dtype=np.float32)
    return adv, ret


def collate_rollouts(rollouts: List[List[Transition]], cfg: Config, device: str) -> Dict[str, torch.Tensor]:
    # pylint: disable=too-many-locals
    obs = np.concatenate([np.stack([tr.obs for tr in ro], axis=0) for ro in rollouts], axis=0)
    act_turn = np.concatenate([np.asarray([tr.action_turn for tr in ro], dtype=np.float32) for ro in rollouts], axis=0)
    act_boost = np.concatenate([np.asarray([tr.action_boost for tr in ro], dtype=np.float32) for ro in rollouts], axis=0)
    old_logp = np.concatenate([np.asarray([tr.logp for tr in ro], dtype=np.float32) for ro in rollouts], axis=0)

    advs = []
    rets = []
    for ro in rollouts:
        a, r = gae(ro, cfg.gamma, cfg.gae_lambda)
        advs.append(a)
        rets.append(r)

    adv = np.concatenate(advs, axis=0)
    ret = np.concatenate(rets, axis=0)

    # Normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    batch = {
        "obs": torch.tensor(obs, dtype=torch.float32, device=device),
        "act_turn": torch.tensor(act_turn, dtype=torch.float32, device=device),
        "act_boost": torch.tensor(act_boost, dtype=torch.float32, device=device),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32, device=device),
        "adv": torch.tensor(adv, dtype=torch.float32, device=device),
        "returns": torch.tensor(ret, dtype=torch.float32, device=device),
    }
    return batch


async def learner_loop(cfg: Config, shared_state: SharedState, experience_q: asyncio.Queue) -> None:
    # pylint: disable=too-many-locals
    last_log = time.time()
    updates = 0
    total_steps = 0

    pending_rollouts: List[List[Transition]] = []

    while True:
        _actor_id, rollout = await experience_q.get()
        pending_rollouts.append(rollout)
        total_steps += len(rollout)

        # Train when we have enough samples.
        if sum(len(r) for r in pending_rollouts) < cfg.minibatch:
            continue

        batch = collate_rollouts(pending_rollouts, cfg, device=cfg.train_device)
        pending_rollouts = []

        # Multiple epochs over the same batch; still simple, but effective as a baseline.
        metrics_accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(cfg.epochs):
            metrics = shared_state.ppo_update(batch)
            for k in metrics_accum:
                metrics_accum[k] += metrics[k]

        updates += 1


        # Checkpointing cadence

        if cfg.save_every_updates > 0 and (updates % cfg.save_every_updates) == 0:

            try:

                save_checkpoint(cfg, shared_state)

            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[learner] checkpoint save failed: {type(e).__name__}: {e}")
        now = time.time()
        if now - last_log >= cfg.log_every_seconds:
            denom = float(cfg.epochs)
            msg = (
                f"[learner] updates={updates}, steps={total_steps}, "
                f"loss={metrics_accum['loss']/denom:.4f}, "
                f"policy={metrics_accum['policy_loss']/denom:.4f}, "
                f"value={metrics_accum['value_loss']/denom:.4f}, "
                f"entropy={metrics_accum['entropy']/denom:.4f}"
            )
            print(msg)
            last_log = now


async def discover_obs_dim(ws_url: str) -> int:
    """
    Connect once, read welcome, return sensorCount; does not join as player.
    This avoids hardcoding obs dim and honors sensorSpec order.
    """
    async with websockets.connect(ws_url, max_size=MAX_WS_MESSAGE_BYTES) as ws:
        hello = {"type": "hello", "clientType": "bot", "version": PROTOCOL_VERSION}
        await ws.send(json.dumps(hello))
        while True:
            raw = await ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "welcome":
                spec = msg.get("sensorSpec") or {}
                sensor_count = int(spec.get("sensorCount", 0))
                if sensor_count <= 0:
                    order = spec.get("order") or []
                    sensor_count = len(order)
                return sensor_count
            if msg.get("type") == "error":
                raise RuntimeError(f"server error during welcome: {msg.get('message')}")


async def main() -> None:
    cfg = load_or_create_config(os.environ.get("SLITHER_CONFIG", "config.toml"))

    # GPU fast-path: TF32 matmuls are typically a free speed win on Ada GPUs for MLP training.
    if str(cfg.train_device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    obs_dim = await discover_obs_dim(cfg.ws_url)
    print(
        f"[main] ws={cfg.ws_url}, obs_dim={obs_dim}, actors={cfg.actors}, "
        f"train_device={cfg.train_device}, infer_device={cfg.infer_device}"
    )

    shared_state = SharedState(obs_dim=obs_dim, cfg=cfg)


    # Auto-resume from latest checkpoint if present

    try:

        loaded = load_checkpoint_if_present(cfg, shared_state)

        if loaded is not None:

            print(f"[main] resumed from {loaded} at update_steps={shared_state.update_steps}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[main] checkpoint load failed: {type(e).__name__}: {e}")
    experience_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.actors * 4)

    learner = asyncio.create_task(learner_loop(cfg, shared_state, experience_q))

    actors = []
    for i in range(cfg.actors):
        actors.append(asyncio.create_task(ActorClient(i, cfg, shared_state, experience_q).run()))

    await asyncio.gather(learner, *actors)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
