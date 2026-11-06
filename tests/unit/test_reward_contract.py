"""Authoritative reward parsing, objective weights, and life chronology."""

from copy import deepcopy
from importlib.resources import files
import json

import pytest

from pyrl_trainer.reward_contract import (
    RewardCoefficients,
    RewardContractError,
    RewardDiscontinuityError,
    RewardStream,
    parse_reward,
    require_reward_version,
    reward_components,
)

pytestmark = pytest.mark.unit


def native_fixture():
    """This exact fixture is checked against real Rust food/collision physics."""
    return json.loads(files("slither_contracts").joinpath(
        "external-reward-v1-fixture.json"
    ).read_text(encoding="utf-8"))


def initial():
    """Establish the same physical life's zero-reward delivery baseline."""
    payload = deepcopy(native_fixture()[0])
    payload.update(observationSequence="0000000000000001",
                   previousObservationSequence=None, toStep="0000000000000000",
                   transitionKind="initial", reason=None)
    payload["deltas"] = {name: 0 for name in payload["deltas"]}
    payload["state"]["bodyLength"] = payload["state"]["startingLength"]
    payload["state"]["retainedGrowth"] = 0
    return payload


def live():
    """Reuse the native food event without its death closure."""
    payload = deepcopy(native_fixture()[0])
    payload.update(transitionKind="live", reason=None)
    return payload


def test_native_terminal_and_truncation_fixture_has_exact_objective():
    """Simultaneous native events retain their complete weighted reward."""
    victim, owner = map(parse_reward, native_fixture())
    victim_parts = reward_components(victim)
    owner_parts = reward_components(owner)
    assert victim_parts == pytest.approx({
        "survival": 0, "food": 0.1, "growth": 0.05,
        "points": 0.2, "kills": 0, "death": -0.5,
    })
    assert owner_parts == pytest.approx({
        "survival": 0, "food": 0.1, "growth": 0.05,
        "points": 4.2, "kills": 1, "death": 0,
    })
    assert sum(victim_parts.values()) == pytest.approx(-0.15)
    assert sum(owner_parts.values()) == pytest.approx(5.35)


def test_live_time_scales_and_survival_points_are_not_paid_twice():
    """Time is the only survival payment regardless of observation duration."""
    payload = live()
    payload["deltas"] = dict(initial()["deltas"], elapsedSeconds=10,
                             aliveSecondsDelta=10, pointsDelta=6,
                             survivalPointsDelta=6)
    parts = reward_components(parse_reward(payload))
    assert parts["survival"] == pytest.approx(0.01)
    assert parts["points"] == 0
    payload["deltas"].update(elapsedSeconds=20, aliveSecondsDelta=20,
                             pointsDelta=12, survivalPointsDelta=12)
    assert reward_components(parse_reward(payload))["survival"] == pytest.approx(0.02)


def test_signed_boost_cost_and_growth_are_preserved():
    """Boost economics and loss of retained growth reduce return."""
    payload = live()
    payload["deltas"] = dict(initial()["deltas"], elapsedSeconds=1,
                             pointsDelta=-1, boostPointsSpentDelta=1,
                             lengthDelta=-1, retainedGrowthDelta=-1)
    parts = reward_components(parse_reward(payload))
    assert parts["points"] == pytest.approx(-0.01)
    assert parts["growth"] == pytest.approx(-0.05)
    assert parts["death"] == 0


def test_prefix_credit_can_precede_physical_time_without_changing_the_clock():
    """Prefix live-time credit need not fit inside the physical sensor interval."""
    payload = live()
    payload["deltas"] = dict(initial()["deltas"], elapsedSeconds=0,
                             aliveSecondsDelta=1 / 60, pointsDelta=0.01,
                             survivalPointsDelta=0.01)
    payload["toStep"] = payload["fromStep"]
    admitted = parse_reward(payload)
    assert admitted["deltas"]["elapsedSeconds"] == 0
    parts = reward_components(admitted)
    assert parts["survival"] == pytest.approx(0.001 / 60)
    assert parts["points"] == 0


def test_configured_weights_use_raw_quantities_without_sensor_labels():
    """Trainer coefficients operate independently of the observation layout."""
    reward = parse_reward(live())
    coefficients = RewardCoefficients(food=2, growth=3, points=4, kills=5,
                                      death_penalty=6, survival=7)
    parts = reward_components(reward, coefficients)
    assert parts["food"] == 2
    assert parts["growth"] == 3
    assert parts["points"] == 80


def test_admission_closes_life_once_and_keeps_final_tail():
    """A completed life cannot supply its last event a second time."""
    stream = RewardStream()
    assert sum(reward_components(stream.accept(initial(), snake_id=7)).values()) == 0
    closure = stream.accept(native_fixture()[0], snake_id=7)
    assert closure["deltas"]["foodCountDelta"] == 1
    assert closure["transitionKind"] == "terminal"
    with pytest.raises(RewardDiscontinuityError, match="after life closure"):
        stream.accept(native_fixture()[0], snake_id=7)


@pytest.mark.parametrize("kind,reason", [
    ("terminal", "death"), ("truncated", "manualGeneration"),
])
def test_cutoff_elapsed_time_does_not_manufacture_live_credit(kind, reason):
    """Physical tail time and prefix live-time credits are distinct facts."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    payload = deepcopy(native_fixture()[0])
    payload.update(transitionKind=kind, reason=reason, toStep="0000000000000002")
    payload["deltas"].update(elapsedSeconds=2 / 60, aliveSecondsDelta=0)
    closure = stream.accept(payload, snake_id=7)
    parts = reward_components(closure)
    assert closure["deltas"]["elapsedSeconds"] == pytest.approx(2 / 60)
    assert parts["survival"] == 0
    assert parts["food"] == pytest.approx(0.1)
    assert parts["death"] == (-0.5 if kind == "terminal" else 0)


def test_successor_baseline_cannot_consume_predecessor_reward():
    """Assignment handling explicitly resets admission after closing the old life."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    stream.accept(native_fixture()[0], snake_id=7)
    owner = deepcopy(native_fixture()[1])
    baseline = deepcopy(owner)
    baseline.update(transitionKind="initial", reason=None,
                    observationSequence=owner["previousObservationSequence"],
                    previousObservationSequence=None, toStep=owner["fromStep"])
    baseline["deltas"] = {name: 0 for name in baseline["deltas"]}
    baseline["state"]["bodyLength"] = baseline["state"]["startingLength"]
    baseline["state"]["retainedGrowth"] = 0
    with pytest.raises(RewardDiscontinuityError, match="after life closure"):
        stream.accept(baseline, snake_id=20)
    assert stream.rebase(baseline, snake_id=20) is None
    with pytest.raises(RewardDiscontinuityError, match="assigned snake"):
        stream.accept(native_fixture()[0], snake_id=20)
    closure = stream.accept(owner, snake_id=20)
    assert closure["deltas"]["killsDelta"] == 1
    assert reward_components(closure)["death"] == 0


@pytest.mark.parametrize("change", [
    {"leaseId": "0000000000000099"},
    {"worldEpoch": "0000000000000099"},
    {"physicalSnakeId": "0000000000000099"},
    {"runId": "other-run"},
    {"snakeId": 20},
    {"previousObservationSequence": "0000000000000002"},
    {"fromStep": "0000000000000001"},
])
def test_mismatch_does_not_advance_pending_stream(change):
    """Failed admission leaves the correct pending predecessor available."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    bad = live()
    bad.update(change)
    with pytest.raises(RewardDiscontinuityError):
        stream.accept(bad, snake_id=7)
    assert stream.accept(live(), snake_id=7)["deltas"]["foodValueDelta"] == 1


def test_multiple_intervals_are_admitted_once_before_closure():
    """Several observation intervals can contribute to one decision reward."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    first = stream.accept(live(), snake_id=7)
    with pytest.raises(RewardDiscontinuityError):
        stream.accept(live(), snake_id=7)
    last = deepcopy(native_fixture()[0])
    last.update(observationSequence="0000000000000004",
                previousObservationSequence="0000000000000003",
                fromStep="0000000000000001", toStep="0000000000000002")
    last["state"]["bodyLength"] += 1
    last["state"]["retainedGrowth"] += 1
    second = stream.accept(last, snake_id=7)
    assert first["deltas"]["foodCountDelta"] + second["deltas"]["foodCountDelta"] == 2
    assert sum(reward_components(first).values()) + sum(
        reward_components(second).values()
    ) == pytest.approx(0.2)


def test_reclaim_requires_explicit_discard_and_rebase():
    """A missed delivery cannot silently cross a reconnect boundary."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    missed = live()
    missed.update(previousObservationSequence="0000000000000002")
    with pytest.raises(RewardDiscontinuityError):
        stream.accept(missed, snake_id=7)
    assert stream.rebase(missed, snake_id=7) is None
    tail = deepcopy(native_fixture()[0])
    tail.update(previousObservationSequence=missed["observationSequence"],
                observationSequence="0000000000000004",
                fromStep=missed["toStep"], toStep="0000000000000002")
    tail["state"]["bodyLength"] += 1
    tail["state"]["retainedGrowth"] += 1
    assert stream.accept(tail, snake_id=7)["transitionKind"] == "terminal"


@pytest.mark.parametrize("payload", [None, {}, {"rewardVersion": 2},
                                      {"rewardVersion": True}])
def test_missing_or_unsupported_contract_fails(payload):
    """Missing fields and unsupported versions never select a fallback."""
    with pytest.raises(RewardContractError):
        parse_reward(payload)


@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_welcome_requires_explicit_version(version):
    """Sensor labels and truthy values cannot advertise reward support."""
    with pytest.raises(RewardContractError, match="rewardVersion 1"):
        require_reward_version({"rewardVersion": version})


@pytest.mark.parametrize("field,value", [
    ("foodValueDelta", -1), ("foodValueDelta", float("nan")),
    ("foodPointsDelta", float("inf")), ("foodCountDelta", 1.5),
    ("killsDelta", True), ("killsDelta", 2**53),
    ("retainedGrowthDelta", 0.5), ("pointsDelta", float("-inf")),
    ("pointsDelta", 0), ("aliveSecondsDelta", -1),
    ("foodValueDelta", 10**1000),
])
def test_invalid_raw_values_fail_without_fallback(field, value):
    """Malformed authoritative quantities cannot enter return calculations."""
    payload = live()
    payload["deltas"][field] = value
    with pytest.raises(RewardContractError):
        parse_reward(payload)


@pytest.mark.parametrize("change", [
    {"transitionKind": "unknown"}, {"transitionKind": "terminal"},
    {"transitionKind": "truncated", "reason": "death"},
    {"reason": "disconnect"}, {"reason": "generation"},
    {"previousObservationSequence": None},
    {"observationSequence": "0000000000000001"},
    {"worldEpoch": "0000000000000000"}, {"worldEpoch": "BAD"},
    {"snakeId": 2**24 + 1}, {"runId": ""},
    {"toStep": "0000000000000000", "fromStep": "0000000000000001"},
])
def test_invalid_lifecycle_and_identity_fail(change):
    """Closures have explicit reasons and ordered exact identities."""
    payload = live()
    payload.update(change)
    with pytest.raises(RewardContractError):
        parse_reward(payload)


def test_initial_must_be_zero_and_schema_is_complete():
    """A baseline is not a reward event and cannot hide omitted fields."""
    payload = initial()
    payload["deltas"]["foodValueDelta"] = 1
    with pytest.raises(RewardContractError, match="zero deltas"):
        parse_reward(payload)
    payload = live()
    del payload["deltas"]["foodValueDelta"]
    with pytest.raises(RewardContractError, match="missing or unknown"):
        parse_reward(payload)
    payload = live()
    payload["delivered_wallet_delta_norm"] = 1
    with pytest.raises(RewardContractError, match="missing or unknown"):
        parse_reward(payload)


def test_invalid_coefficients_and_weighted_overflow_fail():
    """Configuration and arithmetic errors must be reported explicitly."""
    with pytest.raises(RewardContractError):
        RewardCoefficients(food=-1)
    with pytest.raises(RewardContractError):
        RewardCoefficients(points=float("nan"))
    with pytest.raises(RewardContractError, match="overflow"):
        reward_components(parse_reward(live()), RewardCoefficients(points=1e308))
    with pytest.raises(RewardContractError, match="overflow"):
        reward_components(parse_reward(live()), RewardCoefficients(food=1e308, points=5e306))


@pytest.mark.parametrize("change", [
    {"startingLength": 0}, {"bodyLength": 0}, {"bodyLength": True},
    {"bodyLength": 1.5}, {"startingLength": 2**53}, {"retainedGrowth": -1},
    {"retainedGrowth": 0}, {"retainedGrowth": float("nan")},
])
def test_invalid_boundary_state_fails_without_sensor_reconstruction(change):
    """Absolute body measurements are mandatory exact authoritative facts."""
    payload = live()
    payload["state"].update(change)
    with pytest.raises(RewardContractError):
        parse_reward(payload)


def test_missing_or_unknown_boundary_state_fields_fail():
    """A missing snapshot cannot degrade into normalized size-sensor inference."""
    payload = live()
    payload.pop("state")
    with pytest.raises(RewardContractError):
        parse_reward(payload)
    payload = live()
    payload["state"]["size_norm"] = 1
    with pytest.raises(RewardContractError, match="state"):
        parse_reward(payload)


@pytest.mark.parametrize("change", ["body", "starting", "lengthDelta", "retainedGrowthDelta"])
def test_boundary_state_disagreement_does_not_advance_stream(change):
    """A coherent sequence cannot conceal a length/delta mismatch."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    wrong = live()
    if change == "body":
        wrong["state"].update(bodyLength=14, retainedGrowth=2)
    elif change == "starting":
        wrong["state"].update(startingLength=13, bodyLength=14)
    else:
        wrong["deltas"][change] = 0
    with pytest.raises(RewardDiscontinuityError, match="state and length"):
        stream.accept(wrong, snake_id=7)
    assert stream.accept(live(), snake_id=7)["state"]["bodyLength"] == 13


def test_prebaseline_growth_is_reported_but_never_paid_as_spawn_reward():
    """The first snapshot can already contain growth; its deltas still start at zero."""
    stream = RewardStream()
    baseline = initial()
    baseline["state"].update(bodyLength=13, retainedGrowth=1)
    assert sum(reward_components(stream.accept(baseline, snake_id=7)).values()) == 0
    following = live()
    following["state"].update(bodyLength=14, retainedGrowth=2)
    result = stream.accept(following, snake_id=7)
    assert result["state"]["retainedGrowth"] == 2
    assert reward_components(result)["growth"] == pytest.approx(0.05)


def test_rebase_cannot_change_starting_length_of_a_known_life():
    """A transport gap discards unknown rewards but does not rewrite life identity facts."""
    stream = RewardStream()
    stream.accept(initial(), snake_id=7)
    bad = live()
    bad["state"].update(startingLength=13, bodyLength=14)
    with pytest.raises(RewardDiscontinuityError, match="starting length"):
        stream.rebase(bad, snake_id=7)
    assert stream.accept(live(), snake_id=7)["state"]["startingLength"] == 12
