"""Native v4 structural metadata and legacy PPO sensor identity compatibility."""

from dataclasses import replace
import copy
import json
from pathlib import Path

import pytest
import torch

from pyrl_trainer.checkpointing import load_checkpoint_if_present, save_checkpoint
from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState
from pyrl_trainer.sensor_contract import SensorContract, SensorContractError

pytestmark = pytest.mark.unit
REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/teacher_chunk_v1/reference.json").read_text(
        encoding="utf-8"
    )
)
RAW = json.loads(REFERENCE["sensorContractJson"])


def spec():
    """Use exact native bytes/digest retained by the Rust teacher reference."""
    return {
        "layoutVersion": "v4",
        "profile": "basic",
        "sensorCount": 72,
        "order": RAW["order"],
        "contract": {
            "canonicalJson": REFERENCE["sensorContractJson"],
            "sha256": REFERENCE["contract"]["sensorSha256"],
        },
    }


def test_v4_spec_checkpoint_roundtrip_retains_native_bytes_and_dimensions():
    """Sensor count alone cannot stand in for profile, revisions or native structural identity."""
    contract = SensorContract.from_spec(spec())
    assert (
        contract.profile,
        contract.normalization_revision,
        contract.bins,
        contract.scalar_count,
        contract.channel_count,
        contract.sensor_count,
    ) == ("basic", 0, 8, 24, 6, 72)
    assert contract.canonical_json == REFERENCE["sensorContractJson"]
    assert SensorContract.from_checkpoint(contract.checkpoint_fields()) == contract
    assert SensorContract.from_checkpoint(
        {"sensor_layout_version": "v3", "sensor_order": ["a", "b"]}
    ) == SensorContract("v3", ("a", "b"))


@pytest.mark.parametrize(
    "field",
    [
        "sensor_profile",
        "sensor_normalization_revision",
        "sensor_structural_sha256",
        "sensor_canonical_json",
        "sensor_input_size",
        "sensor_bins",
        "sensor_scalar_count",
        "sensor_channel_count",
    ],
)
def test_v4_missing_checkpoint_structural_fields_cannot_be_guessed(field):
    """Legacy two-field metadata remains v3-only; incomplete v4 never pads or infers a digest."""
    record = SensorContract.from_spec(spec()).checkpoint_fields()
    record.pop(field)
    with pytest.raises(SensorContractError):
        SensorContract.from_checkpoint(record)


def test_v4_malformed_welcome_same_width_reordered_or_changed_profile_rejects():
    """Native SHA and channel order remain exact before any model/controller initialization."""
    for change in ("missing", "digest", "bytes", "profile", "order", "bool-width"):
        value = copy.deepcopy(spec())
        if change == "missing":
            value.pop("contract")
        elif change == "digest":
            value["contract"]["sha256"] = "00" * 32
        elif change == "bytes":
            value["contract"]["canonicalJson"] += " "
        elif change == "profile":
            value["profile"] = "motion"
        elif change == "order":
            value["order"][0], value["order"][1] = value["order"][1], value["order"][0]
        else:
            value["sensorCount"] = True
        with pytest.raises(SensorContractError):
            SensorContract.from_spec(value)
    contract = SensorContract.from_spec(spec())
    with pytest.raises(SensorContractError):
        replace(contract, normalization_revision=False)


def test_existing_ppo_checkpoint_owner_preserves_v4_metadata_and_rejects_incomplete_resume(
    tmp_path,
):
    """The existing PPO schema owner enforces the richer v4 compatibility boundary."""
    contract = SensorContract.from_spec(spec())
    cfg = Config(
        ckpt_dir=str(tmp_path),
        keep_last=3,
        net_hidden=8,
        net_layers=1,
        train_device="cpu",
        infer_device="cpu",
        lr=3e-4,
    )
    state = SharedState(72, cfg, sensor_contract=contract)
    state.update_steps = 4
    save_checkpoint(cfg, state)
    restored = SharedState(72, cfg, sensor_contract=contract)
    assert load_checkpoint_if_present(cfg, restored) is not None
    assert restored.update_steps == 4
    for path in tmp_path.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("sensor_structural_sha256")
        torch.save(payload, path)
    rejected = SharedState(72, cfg, sensor_contract=contract)
    assert load_checkpoint_if_present(cfg, rejected) is None
    assert rejected.update_steps == 0
