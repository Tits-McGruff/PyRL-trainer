"""Configuration loading and defaults for the trainer."""

import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

PROTOCOL_VERSION = 2
MAX_WS_MESSAGE_BYTES = int(os.environ.get("SLITHER_WS_MAX_MESSAGE", str(8 * 1024 * 1024)))

# --- Config loading from config.toml ---

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
    except (TypeError, ValueError):
        return int(default)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None else default


def _default_train_device() -> str:
    import torch  # pylint: disable=import-outside-toplevel
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_net_hidden() -> int:
    import torch  # pylint: disable=import-outside-toplevel
    return 512 if torch.cuda.is_available() else 256


def _default_net_layers() -> int:
    import torch  # pylint: disable=import-outside-toplevel
    return 4 if torch.cuda.is_available() else 2


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Trainer configuration values."""
    ws_url: str = "ws://localhost:3000"
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
            except (TypeError, ValueError):
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
    """Load config from disk or create defaults if missing."""
    path = Path(config_path)
    defaults = _defaults_config()

    if not path.exists():
        _write_config_toml(path, _config_sections(defaults))

    try:
        data = _read_config_toml(path)
        flat = _flatten_sections(data)
        cfg = _merge_cfg(defaults, flat)
    except (OSError, RuntimeError, ValueError, TypeError):
        cfg = defaults

    cfg = _apply_env_overrides(cfg)

    # Re-validate defaults if 0
    if cfg.net_hidden <= 0:
        cfg.net_hidden = _default_net_hidden()
    if cfg.net_layers <= 0:
        cfg.net_layers = _default_net_layers()
    # Ensure train_device is set if it was empty
    if not cfg.train_device:
        cfg.train_device = _default_train_device()

    return cfg
