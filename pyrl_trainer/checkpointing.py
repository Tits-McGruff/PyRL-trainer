"""Durable trainer checkpoint save and recovery helpers."""

import copy
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .config import Config
from .sensor_contract import SensorContract, SensorContractError
from .utils import ensure_dir


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
    contract = getattr(shared_state, "sensor_contract", None)
    if contract is not None and not isinstance(contract, SensorContract):
        raise TypeError("shared_state.sensor_contract must be a SensorContract")
    return contract


def save_checkpoint(cfg: Config, shared_state: Any) -> None:
    """Persist a checkpoint atomically and rotate old files for this model shape."""
    ensure_dir(cfg.ckpt_dir)
    step = int(shared_state.update_steps)

    payload: Dict[str, Any] = {
        "update_steps": step,
        "obs_dim": int(shared_state.obs_dim),
        "net_hidden": int(cfg.net_hidden),
        "net_layers": int(cfg.net_layers),
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

    # Publish a durable numbered recovery point first. Pointer-like latest files
    # are replaced only after that recovery point is complete.
    _atomic_torch_save(payload, numbered)
    _atomic_torch_save(payload, latest_arch)
    _atomic_torch_save(payload, latest)
    _rotate_numbered(cfg)


def _checkpoint_candidates(cfg: Config) -> List[Path]:
    directory = Path(cfg.ckpt_dir)
    if not directory.exists():
        return []

    latest_arch = directory / f"latest_h{cfg.net_hidden}_l{cfg.net_layers}.pt"
    latest = directory / "latest.pt"
    numbered = list(reversed(_list_numbered(cfg)))

    candidates: List[Path] = []
    for path in (latest_arch, latest, *numbered):
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates


def _payload_compatible(  # pylint: disable=too-many-return-statements
    payload: Any,
    cfg: Config,
    shared_state: Any,
) -> bool:
    if not isinstance(payload, dict):
        return False

    try:
        if int(payload.get("obs_dim", -1)) != int(shared_state.obs_dim):
            return False
        if int(payload.get("net_hidden", -1)) != int(cfg.net_hidden):
            return False
        if int(payload.get("net_layers", -1)) != int(cfg.net_layers):
            return False
        int(payload.get("update_steps", 0))
    except (TypeError, ValueError):
        return False

    expected_contract = _expected_sensor_contract(shared_state)
    if expected_contract is not None:
        try:
            checkpoint_contract = SensorContract.from_checkpoint(payload)
        except (SensorContractError, TypeError):
            return False
        if checkpoint_contract != expected_contract:
            return False

    model_state = payload.get("model")
    optimizer_state = payload.get("optimizer")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        return False

    current_state = shared_state.train_model.state_dict()
    if set(model_state) != set(current_state):
        return False

    for key, value in model_state.items():
        try:
            if tuple(value.shape) != tuple(current_state[key].shape):
                return False
        except (AttributeError, TypeError):
            return False
    return True


def load_checkpoint_if_present(cfg: Config, shared_state: Any) -> Optional[Path]:
    """Load the newest valid model- and sensor-compatible checkpoint."""
    candidates = _checkpoint_candidates(cfg)
    if not candidates:
        return None

    original_model = copy.deepcopy(shared_state.train_model.state_dict())
    original_optimizer = copy.deepcopy(shared_state.optimizer.state_dict())
    original_steps = int(shared_state.update_steps)

    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location=cfg.train_device)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
        ):
            continue

        if not _payload_compatible(payload, cfg, shared_state):
            continue

        try:
            shared_state.train_model.load_state_dict(payload["model"])
            shared_state.optimizer.load_state_dict(payload["optimizer"])
            shared_state.update_steps = int(payload.get("update_steps", 0))
            shared_state.sync_infer_from_train()
        except (KeyError, RuntimeError, TypeError, ValueError):
            shared_state.train_model.load_state_dict(original_model)
            shared_state.optimizer.load_state_dict(original_optimizer)
            shared_state.update_steps = original_steps
            shared_state.sync_infer_from_train()
            continue
        return candidate

    return None
