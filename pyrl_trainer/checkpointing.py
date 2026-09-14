"""Durable trainer checkpoint save and recovery helpers."""

import copy
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import (
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_REWARD_CONFIG,
    REWARD_CONFIG_VERSION,
    Config,
    reward_config_record,
)
from .sensor_contract import SensorContract, SensorContractError
from .utils import ensure_dir


_REJECTION_CORRUPT = "corrupt"
_REJECTION_UNSUPPORTED_SCHEMA = "unsupported_schema"
_REJECTION_ARCHITECTURE = "architecture_mismatch"
_REJECTION_SENSOR = "sensor_mismatch"
_REJECTION_REWARD = "reward_mismatch"
_REJECTION_APPLY = "state_apply_failure"


def _list_numbered(cfg: Config) -> List[Path]:
    directory = Path(cfg.ckpt_dir)
    if not directory.exists():
        return []
    pattern = f"ckpt_h{cfg.net_hidden}_l{cfg.net_layers}_*.pt"
    return sorted(directory.glob(pattern))


def _rotate_numbered(cfg: Config) -> None:
    if cfg.keep_last <= 0:
        return
    checkpoints = _list_numbered(cfg)
    for path in checkpoints[: max(0, len(checkpoints) - cfg.keep_last)]:
        try:
            path.unlink()
        except OSError:
            pass


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    """Write a torch payload to a sibling temp file and atomically replace."""
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _expected_sensor_contract(shared_state: Any) -> Optional[SensorContract]:
    contract = shared_state.sensor_contract
    if contract is not None and not isinstance(contract, SensorContract):
        raise TypeError("shared_state.sensor_contract must be a SensorContract")
    return contract


def save_checkpoint(cfg: Config, shared_state: Any) -> None:
    """Persist a schema-v2 checkpoint atomically and rotate old recovery files."""
    ensure_dir(cfg.ckpt_dir)
    step = int(shared_state.update_steps)
    if step < 0:
        raise ValueError("shared_state.update_steps must be >= 0")

    payload: Dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "update_steps": step,
        "obs_dim": int(shared_state.obs_dim),
        "net_hidden": int(cfg.net_hidden),
        "net_layers": int(cfg.net_layers),
        "reward_config_version": REWARD_CONFIG_VERSION,
        "reward_config": reward_config_record(cfg),
        "model": shared_state.train_model.state_dict(),
        "optimizer": shared_state.optimizer.state_dict(),
    }
    contract = _expected_sensor_contract(shared_state)
    if contract is not None:
        payload.update(contract.checkpoint_fields())

    directory = Path(cfg.ckpt_dir)
    numbered = directory / (
        f"ckpt_h{cfg.net_hidden}_l{cfg.net_layers}_{step:08d}.pt"
    )
    latest_arch = directory / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"
    latest = directory / "latest.pt"

    _atomic_torch_save(payload, numbered)
    _atomic_torch_save(payload, latest_arch)
    _atomic_torch_save(payload, latest)
    _rotate_numbered(cfg)


def _checkpoint_candidates(cfg: Config) -> List[Tuple[int, Path]]:
    """Return unique candidate paths with deterministic equal-step priority."""
    directory = Path(cfg.ckpt_dir)
    if not directory.exists():
        return []

    latest_arch = directory / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"
    latest = directory / "latest.pt"

    candidates: List[Tuple[int, Path]] = []
    seen = set()
    for priority, path in (
        (0, latest_arch),
        (1, latest),
        *((2, path) for path in _list_numbered(cfg)),
    ):
        if path.exists() and path not in seen:
            candidates.append((priority, path))
            seen.add(path)
    return candidates


def _integer_field(
    payload: Dict[str, Any], key: str, minimum: int
) -> Optional[int]:
    """Parse one checkpoint integer field without accepting fractional values."""
    raw = payload.get(key)
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(raw, float) and not raw.is_integer():
        return None
    if value < minimum:
        return None
    return value


def _schema_version(payload: Dict[str, Any]) -> Optional[int]:
    raw = payload.get("checkpoint_schema_version")
    if raw is None:
        return 1
    try:
        version = int(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(raw, float) and not raw.is_integer():
        return None
    return version


def _valid_reward_record(record: Any) -> Optional[Dict[str, float]]:
    if not isinstance(record, dict) or set(record) != set(LEGACY_REWARD_CONFIG):
        return None

    normalized: Dict[str, float] = {}
    for key in LEGACY_REWARD_CONFIG:
        try:
            value = float(record[key])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        normalized[key] = value

    if normalized["reward_growth_scale"] < 0.0:
        return None
    if normalized["reward_food_approach_scale"] < 0.0:
        return None
    if normalized["reward_food_delta_clip"] <= 0.0:
        return None
    if normalized["reward_survival_bonus"] < 0.0:
        return None
    if not -1.0 <= normalized["reward_wall_threshold"] <= 1.0:
        return None
    if normalized["reward_wall_penalty_scale"] < 0.0:
        return None
    if not -1.0 <= normalized["reward_hazard_threshold"] <= 1.0:
        return None
    if normalized["reward_hazard_penalty_magnitude"] < 0.0:
        return None
    if normalized["reward_death_penalty_magnitude"] < 0.0:
        return None
    return normalized


def _reward_compatible(
    payload: Dict[str, Any], cfg: Config, schema_version: int
) -> Tuple[bool, bool]:
    """Return (compatible, legacy-default-assumption-used)."""
    current = reward_config_record(cfg)
    if schema_version == 1:
        return current == LEGACY_REWARD_CONFIG, current == LEGACY_REWARD_CONFIG

    try:
        reward_version = int(payload.get("reward_config_version", -1))
    except (TypeError, ValueError):
        return False, False
    if reward_version != REWARD_CONFIG_VERSION:
        return False, False

    stored = _valid_reward_record(payload.get("reward_config"))
    if stored is None:
        return False, False
    return stored == current, False


def _architecture_and_sensor_compatible(
    payload: Dict[str, Any],
    cfg: Config,
    shared_state: Any,
) -> Optional[str]:
    obs_dim = _integer_field(payload, "obs_dim", 1)
    net_hidden = _integer_field(payload, "net_hidden", 1)
    net_layers = _integer_field(payload, "net_layers", 1)
    update_steps = _integer_field(payload, "update_steps", 0)
    if (
        obs_dim != int(shared_state.obs_dim)
        or net_hidden != int(cfg.net_hidden)
        or net_layers != int(cfg.net_layers)
        or update_steps is None
    ):
        return _REJECTION_ARCHITECTURE

    expected_contract = _expected_sensor_contract(shared_state)
    if expected_contract is not None:
        try:
            checkpoint_contract = SensorContract.from_checkpoint(payload)
        except (SensorContractError, TypeError):
            return _REJECTION_SENSOR
        if checkpoint_contract != expected_contract:
            return _REJECTION_SENSOR

    model_state = payload.get("model")
    optimizer_state = payload.get("optimizer")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        return _REJECTION_ARCHITECTURE

    current_state = shared_state.train_model.state_dict()
    if set(model_state) != set(current_state):
        return _REJECTION_ARCHITECTURE

    for key, value in model_state.items():
        try:
            if tuple(value.shape) != tuple(current_state[key].shape):
                return _REJECTION_ARCHITECTURE
        except (AttributeError, TypeError):
            return _REJECTION_ARCHITECTURE
    return None


def _inspect_payload(
    payload: Any,
    cfg: Config,
    shared_state: Any,
) -> Tuple[Optional[int], bool, Optional[str]]:
    """Validate compatibility without mutating live state."""
    if not isinstance(payload, dict):
        return None, False, _REJECTION_CORRUPT

    schema_version = _schema_version(payload)
    if schema_version not in (1, CHECKPOINT_SCHEMA_VERSION):
        return None, False, _REJECTION_UNSUPPORTED_SCHEMA

    mismatch = _architecture_and_sensor_compatible(payload, cfg, shared_state)
    if mismatch is not None:
        return None, False, mismatch

    compatible, legacy_assumption = _reward_compatible(
        payload, cfg, schema_version
    )
    if not compatible:
        return None, False, _REJECTION_REWARD

    return schema_version, legacy_assumption, None


def _load_candidate(
    path: Path, cfg: Config, shared_state: Any
) -> Tuple[Optional[Dict[str, Any]], Optional[int], bool, Optional[str]]:
    """Deserialize and classify one candidate without mutating live state."""
    try:
        payload = torch.load(path, map_location=cfg.train_device)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ):
        return None, None, False, _REJECTION_CORRUPT

    schema_version, legacy_assumption, rejection = _inspect_payload(
        payload, cfg, shared_state
    )
    if rejection is not None or schema_version is None:
        return None, schema_version, legacy_assumption, rejection
    return payload, schema_version, legacy_assumption, None


def load_checkpoint_if_present(cfg: Config, shared_state: Any) -> Optional[Path]:
    """Load the highest-update compatible recoverable checkpoint transactionally."""
    candidates = _checkpoint_candidates(cfg)
    if not candidates:
        return None

    original_model = copy.deepcopy(shared_state.train_model.state_dict())
    original_optimizer = copy.deepcopy(shared_state.optimizer.state_dict())
    original_steps = int(shared_state.update_steps)

    valid: List[Tuple[int, int, str, Path, Dict[str, Any], int, bool]] = []
    for priority, candidate in candidates:
        payload, schema_version, legacy_assumption, rejection = _load_candidate(
            candidate, cfg, shared_state
        )
        if payload is None or rejection is not None or schema_version is None:
            continue

        valid.append(
            (
                int(payload["update_steps"]),
                priority,
                str(candidate),
                candidate,
                payload,
                schema_version,
                legacy_assumption,
            )
        )

    valid.sort(key=lambda item: (-item[0], item[1], item[2]))

    for (
        _step,
        _priority,
        _path_key,
        candidate,
        payload,
        schema_version,
        legacy_assumption,
    ) in valid:
        try:
            shared_state.train_model.load_state_dict(payload["model"])
            shared_state.optimizer.load_state_dict(payload["optimizer"])
            shared_state.update_steps = int(payload["update_steps"])
            shared_state.sync_infer_from_train()
        except (KeyError, RuntimeError, TypeError, ValueError):
            shared_state.train_model.load_state_dict(original_model)
            shared_state.optimizer.load_state_dict(original_optimizer)
            shared_state.update_steps = original_steps
            shared_state.sync_infer_from_train()
            continue

        contract = _expected_sensor_contract(shared_state)
        layout = contract.layout_version if contract is not None else "none"
        reward_version = (
            "legacy-defaults" if legacy_assumption else str(REWARD_CONFIG_VERSION)
        )
        print(
            f"[checkpoint] restored path={candidate}, schema={schema_version}, "
            f"update_steps={shared_state.update_steps}, sensor_layout={layout}, "
            f"reward_config_version={reward_version}, "
            f"legacy_default_assumption={legacy_assumption}"
        )
        return candidate

    return None
