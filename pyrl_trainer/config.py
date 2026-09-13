"""Configuration loading and defaults for the trainer."""

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import tomli_w

PROTOCOL_VERSION = 2
MAX_WS_MESSAGE_BYTES = int(
    os.environ.get("SLITHER_WS_MAX_MESSAGE", str(8 * 1024 * 1024))
)

try:
    if sys.version_info >= (3, 11):
        import tomllib as _toml_reader  # type: ignore
    else:
        import tomli as _toml_reader  # type: ignore
except ImportError:
    _toml_reader = None  # type: ignore


def _read_config_toml(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    if _toml_reader is not None:
        return _toml_reader.loads(raw.decode("utf-8"))
    raise RuntimeError(
        "Reading TOML requires Python 3.11+ (tomllib) or `pip install tomli`."
    )


def _write_config_toml(path: Path, sections: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Slither trainer configuration\n"
        "# Auto-generated because it was missing.\n"
        "# Environment variables prefixed with SLITHER_ override values in this file.\n\n"
    )
    body = tomli_w.dumps(sections)
    path.write_text(header + body, encoding="utf-8")


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    return parsed


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value is not None else default


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
    batch_size: int = 1024
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
            "batch_size": cfg.batch_size,
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


def _flatten_sections(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                out[nested_key] = nested_value
        else:
            out[key] = value
    return out


def _merge_cfg(base: Config, flat: Dict[str, Any]) -> Config:
    cfg = Config(**asdict(base))
    for key, value in flat.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _apply_env_overrides(cfg: Config) -> Config:
    cfg.ws_url = _env_str("SLITHER_WS_URL", cfg.ws_url)
    cfg.bot_name = _env_str("SLITHER_BOT_NAME", cfg.bot_name)
    cfg.actors = _env_int("SLITHER_ACTORS", cfg.actors)

    cfg.max_actions_per_tick = _env_int(
        "SLITHER_MAX_APT", cfg.max_actions_per_tick
    )
    cfg.max_actions_per_second = _env_int(
        "SLITHER_MAX_APS", cfg.max_actions_per_second
    )

    cfg.horizon = _env_int("SLITHER_HORIZON", cfg.horizon)
    cfg.batch_size = _env_int("SLITHER_BATCH_SIZE", cfg.batch_size)
    cfg.gamma = _env_float("SLITHER_GAMMA", cfg.gamma)
    cfg.gae_lambda = _env_float("SLITHER_GAE_LAMBDA", cfg.gae_lambda)
    cfg.ppo_clip = _env_float("SLITHER_PPO_CLIP", cfg.ppo_clip)
    cfg.ent_coef = _env_float("SLITHER_ENT_COEF", cfg.ent_coef)
    cfg.vf_coef = _env_float("SLITHER_VF_COEF", cfg.vf_coef)
    cfg.lr = _env_float("SLITHER_LR", cfg.lr)
    cfg.minibatch = _env_int("SLITHER_MINIBATCH", cfg.minibatch)
    cfg.epochs = _env_int("SLITHER_EPOCHS", cfg.epochs)

    cfg.train_device = _env_str(
        "SLITHER_TRAIN_DEVICE", cfg.train_device or _default_train_device()
    )
    cfg.infer_device = _env_str("SLITHER_INFER_DEVICE", cfg.infer_device)

    cfg.net_hidden = _env_int(
        "SLITHER_NET_HIDDEN", cfg.net_hidden or _default_net_hidden()
    )
    cfg.net_layers = _env_int(
        "SLITHER_NET_LAYERS", cfg.net_layers or _default_net_layers()
    )

    cfg.turn_std = _env_float("SLITHER_TURN_STD", cfg.turn_std)
    cfg.log_every_seconds = _env_float(
        "SLITHER_LOG_EVERY", cfg.log_every_seconds
    )

    cfg.ckpt_dir = _env_str("SLITHER_CKPT_DIR", cfg.ckpt_dir)
    cfg.save_every_updates = _env_int(
        "SLITHER_SAVE_EVERY_UPDATES", cfg.save_every_updates
    )
    cfg.keep_last = _env_int("SLITHER_KEEP_LAST", cfg.keep_last)

    return cfg


def _validated_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _validated_float(
    name: str,
    value: Any,
    minimum: float,
    *,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if minimum_inclusive:
        if parsed < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    elif parsed <= minimum:
        raise ValueError(f"{name} must be > {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def validate_config(cfg: Config) -> Config:
    """Normalize supported scalar types and reject unsafe trainer settings."""
    cfg.actors = _validated_int("actors", cfg.actors, 1)
    cfg.max_actions_per_tick = _validated_int(
        "max_actions_per_tick", cfg.max_actions_per_tick, 1
    )
    cfg.max_actions_per_second = _validated_int(
        "max_actions_per_second", cfg.max_actions_per_second, 0
    )
    cfg.horizon = _validated_int("horizon", cfg.horizon, 1)
    cfg.batch_size = _validated_int("batch_size", cfg.batch_size, 1)
    cfg.minibatch = _validated_int("minibatch", cfg.minibatch, 1)
    cfg.epochs = _validated_int("epochs", cfg.epochs, 1)
    cfg.net_hidden = _validated_int("net_hidden", cfg.net_hidden, 1)
    cfg.net_layers = _validated_int("net_layers", cfg.net_layers, 1)
    cfg.save_every_updates = _validated_int(
        "save_every_updates", cfg.save_every_updates, 0
    )
    cfg.keep_last = _validated_int("keep_last", cfg.keep_last, 0)

    cfg.gamma = _validated_float(
        "gamma", cfg.gamma, 0.0, maximum=1.0, minimum_inclusive=False
    )
    cfg.gae_lambda = _validated_float(
        "gae_lambda", cfg.gae_lambda, 0.0, maximum=1.0
    )
    cfg.ppo_clip = _validated_float(
        "ppo_clip", cfg.ppo_clip, 0.0, minimum_inclusive=False
    )
    cfg.ent_coef = _validated_float("ent_coef", cfg.ent_coef, 0.0)
    cfg.vf_coef = _validated_float("vf_coef", cfg.vf_coef, 0.0)
    cfg.lr = _validated_float("lr", cfg.lr, 0.0, minimum_inclusive=False)
    cfg.turn_std = _validated_float(
        "turn_std", cfg.turn_std, 0.0, minimum_inclusive=False
    )
    cfg.log_every_seconds = _validated_float(
        "log_every_seconds",
        cfg.log_every_seconds,
        0.0,
        minimum_inclusive=False,
    )

    for name in ("ws_url", "bot_name", "train_device", "infer_device", "ckpt_dir"):
        value = getattr(cfg, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        setattr(cfg, name, value.strip())

    return cfg


def load_or_create_config(config_path: str = "config.toml") -> Config:
    """Load, override, and validate trainer configuration."""
    path = Path(config_path)
    defaults = _defaults_config()

    if not path.exists():
        _write_config_toml(path, _config_sections(defaults))

    try:
        data = _read_config_toml(path)
        flat = _flatten_sections(data)
        cfg = _merge_cfg(defaults, flat)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"failed to load config {path}: {exc}") from exc

    cfg = _apply_env_overrides(cfg)
    return validate_config(cfg)
