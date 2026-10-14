"""Strict bounded shadow metadata; all inference and teacher ownership remain in Rust."""

import math
from .distillation_transport import DistillationContractError, exact_keys, hex_identity, integer


def _finite(value, maximum=None):
    """Reject bool/coercion and nonfinite diagnostic numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DistillationContractError("invalid shadow numeric metric")
    if maximum is not None and not 0 <= value <= maximum:
        raise DistillationContractError("shadow metric range")
    return value


def _totals(value):
    """Keep native sample/disagreement counters as exact unsigned words."""
    exact_keys(
        value,
        "samples turnAbsoluteError turnSquaredError boostDisagreements "
        "turnMae turnMse boostDisagreementRate",
    )
    samples = int(hex_identity(value["samples"], 16), 16)
    disagreements = int(hex_identity(value["boostDisagreements"], 16), 16)
    if disagreements > samples:
        raise DistillationContractError("shadow disagreements exceed samples")
    for name in ("turnAbsoluteError", "turnSquaredError"):
        if _finite(value[name]) < 0:
            raise DistillationContractError("negative shadow metric")
    _finite(value["turnMae"], 2)
    _finite(value["turnMse"], 4)
    _finite(value["boostDisagreementRate"], 1)
    return value


def shadow_snapshot(value, slot):
    """Validate one fixed native segment without interpreting observations or weights."""
    exact_keys(
        value,
        "shadowSlot artifactSha256 captureEpoch baselineSlot activationSequence activationStep "
        "completedStep lastSampleStep generation physicalSnakeId enabled error rawOutputs "
        "totals strategies lifeStages strata cruising preDeath deaths",
    )
    if integer(value["shadowSlot"], 0, 3) != slot:
        raise DistillationContractError("shadow slot ordering mismatch")
    hex_identity(value["artifactSha256"], 64)
    hex_identity(value["captureEpoch"], 32, True)
    integer(value["baselineSlot"], 0, 119)
    for name in (
        "activationSequence",
        "activationStep",
        "completedStep",
        "lastSampleStep",
        "generation",
        "physicalSnakeId",
        "deaths",
    ):
        hex_identity(value[name], 16, name == "activationSequence")
    if max(value["activationStep"], value["lastSampleStep"]) > value["completedStep"]:
        raise DistillationContractError("shadow sample chronology")
    if not isinstance(value["enabled"], bool) or (
        value["error"] is not None
        and (not isinstance(value["error"], str) or len(value["error"].encode("utf-8")) > 1024)
    ):
        raise DistillationContractError("invalid shadow state")
    outputs = value["rawOutputs"]
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise DistillationContractError("invalid shadow output count")
    for output in outputs:
        _finite(output)
    for name in ("totals", "cruising", "preDeath"):
        _totals(value[name])
    for name, count in (("strategies", 4), ("lifeStages", 3), ("strata", 6)):
        if not isinstance(value[name], list) or len(value[name]) != count:
            raise DistillationContractError("shadow stratum count")
        for stratum in value[name]:
            _totals(stratum)
    return value


def shadow_metrics(value):
    """Read the current diagnostic process and four fixed native slots."""
    exact_keys(value, "schemaVersion processEpoch slots")
    integer(value["schemaVersion"], 1, 1)
    hex_identity(value["processEpoch"], 32, True)
    slots = value["slots"]
    if not isinstance(slots, list) or len(slots) != 4:
        raise DistillationContractError("shadow metrics slot count")
    for slot, snapshot in enumerate(slots):
        if snapshot is not None:
            shadow_snapshot(snapshot, slot)
    return value


def shadow_receipt(value):
    """Validate a durable explicit activation/stop result, retaining exact source scope."""
    exact_keys(
        value,
        "schemaVersion processEpoch requestId commandSequence activationSequence "
        "previousActivationSequence runId configRevision captureEpoch artifactSha256 "
        "shadowSlot baselineSlot effectiveStep enabled acknowledgedAt",
    )
    integer(value["schemaVersion"], 1, 1)
    for name in ("processEpoch", "requestId", "captureEpoch"):
        hex_identity(value[name], 32, True)
    hex_identity(value["artifactSha256"], 64)
    for name in (
        "commandSequence",
        "activationSequence",
        "configRevision",
        "effectiveStep",
        "previousActivationSequence",
    ):
        hex_identity(value[name], 16, name not in ("effectiveStep", "previousActivationSequence"))
    integer(value["shadowSlot"], 0, 3)
    integer(value["baselineSlot"], 0, 119)
    integer(value["acknowledgedAt"], 1, (1 << 53) - 1)
    if (
        not isinstance(value["runId"], str)
        or not value["runId"]
        or "\0" in value["runId"]
        or len(value["runId"].encode("utf-8")) > 128
        or not isinstance(value["enabled"], bool)
    ):
        raise DistillationContractError("invalid shadow receipt scope")
    if value["activationSequence"] > value["commandSequence"]:
        raise DistillationContractError("shadow receipt chronology")
    if value["enabled"]:
        if (
            value["activationSequence"] != value["commandSequence"]
            or value["previousActivationSequence"] >= value["activationSequence"]
        ):
            raise DistillationContractError("shadow receipt chronology")
    elif value["previousActivationSequence"] != value["activationSequence"]:
        raise DistillationContractError("shadow receipt chronology")
    return value


def online_state(value):
    """Admit the optional independent checkpoint journal; empty state means disabled."""
    if value == {}:
        return value
    exact_keys(value, "schemaVersion configuration pending receipt snapshot")
    integer(value["schemaVersion"], 1, 1)
    configuration = value["configuration"]
    exact_keys(configuration, "shadowSlot baselineSlot configRevision")
    integer(configuration["shadowSlot"], 0, 3)
    integer(configuration["baselineSlot"], 0, 119)
    hex_identity(configuration["configRevision"], 16, True)
    if value["pending"] is not None:
        exact_keys(
            value["pending"], "artifactSha256 requestId processEpoch expectedActivationSequence"
        )
        hex_identity(value["pending"]["artifactSha256"], 64)
        hex_identity(value["pending"]["requestId"], 32, True)
        hex_identity(value["pending"]["processEpoch"], 32, True)
        hex_identity(value["pending"]["expectedActivationSequence"], 16)
    receipt = value["receipt"]
    if receipt is not None:
        shadow_receipt(receipt)
        if not receipt["enabled"] or any(
            receipt[name] != configuration[name]
            for name in ("shadowSlot", "baselineSlot", "configRevision")
        ):
            raise DistillationContractError("job shadow deployment mismatch")
    snapshot = value["snapshot"]
    if snapshot is not None:
        shadow_snapshot(snapshot, configuration["shadowSlot"])
        if receipt is None or any(
            snapshot[name] != receipt[name]
            for name in ("artifactSha256", "captureEpoch", "activationSequence", "baselineSlot")
        ):
            raise DistillationContractError("job shadow metric segment mismatch")
    return value
