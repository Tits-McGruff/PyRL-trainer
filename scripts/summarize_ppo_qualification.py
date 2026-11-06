"""Summarize authoritative evaluation records on their common scenario set."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


PREFIX = "[qualification-episode] "
COMPARABLE = (
    "config_hash", "sim_speed", "tick_rate", "max_actions_per_second",
    "duration_limit_seconds", "sampling_seed", "reward_config",
)


def _measurements(record):
    raw = record["raw_deltas"]
    rewards = record["reward_components"]
    values = {
        "return": record["episode_return"],
        "survival_reward": rewards["survival"],
        "task_reward": sum(rewards[key] for key in ("food", "growth", "points", "kills")),
        "death_reward": rewards["death"],
        "food_value": raw["foodValueDelta"],
        "food_per_alive_second": record["food_value_per_alive_second"],
        "retained_growth": record["retained_growth"],
        "task_net_points": record["task_net_points"],
        "boost_spent": raw["boostPointsSpentDelta"],
        "kills": raw["killsDelta"],
        "alive_seconds": record["alive_seconds"],
        "elapsed_seconds": record["elapsed_seconds"],
        "sampled_turn_saturation": record["sampled_turn_saturation"],
        "decisions_per_alive_second": record["decisions_per_alive_second"],
    }
    if record["learned_turn_std"] is not None:
        values["learned_turn_std"] = record["learned_turn_std"]
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("evaluation record has nonfinite measurements")
    return values


def _statistics(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def _summarize_policy(records, scenarios):
    measured = [_measurements(records[scenario]) for scenario in scenarios]
    return {
        "policy_sha256": records[scenarios[0]]["policy_sha256"],
        "policy_version": records[scenarios[0]]["policy_version"],
        "end_reasons": dict(Counter(records[seed]["reason"] for seed in scenarios)),
        "end_kinds": dict(Counter(records[seed]["transition_kind"] for seed in scenarios)),
        "metrics": {key: _statistics([item[key] for item in measured]) for key in measured[0]},
    }


def _paired(candidate, baseline, scenarios):
    pairs = []
    for seed in scenarios:
        left, right = _measurements(candidate[seed]), _measurements(baseline[seed])
        pairs.append({"world_seed": seed, **{
            key: left[key] - right[key] for key in
            ("return", "survival_reward", "task_reward", "food_value",
             "food_per_alive_second", "retained_growth", "task_net_points")
        }})
    return {
        "scenarios": pairs,
        "differences": {
            key: _statistics([pair[key] for pair in pairs])
            for key in pairs[0] if key != "world_seed"
        },
    }


def summarize(manifest, records):
    """Require matching policy identities and settings, then pair only intact episodes."""
    seeds = manifest["trainingSeeds"]
    scenarios = manifest["evaluationWorldSeeds"]
    references = {label: tuple(manifest[key][axis] for axis in ("turn", "boost"))
                  for label, key in (("full-lock", "turnReference"),
                                     ("straight", "straightReference"))}
    policies = {label: {} for seed in seeds for label in
                (f"seed{seed}-initial", f"seed{seed}-final")}
    policies.update({reference: {} for reference in references})
    identities = {}
    settings = {}
    for record in records:
        label, scenario = record["policy_label"], record["world_seed"]
        if label not in policies or scenario not in scenarios:
            raise ValueError(f"undeclared policy or scenario: {label}, {scenario}")
        if scenario in policies[label]:
            raise ValueError(f"duplicate evaluation record: {label}, {scenario}")
        reference = label if label in references else None
        controls = record["fixed_controls"]
        if (record["reference"] != reference or (reference is not None
                and tuple(controls) != references[reference])):
            raise ValueError(f"evaluation controls do not match the label: {label}")
        identity = (record["policy_sha256"], record["policy_version"])
        if identities.setdefault(label, identity) != identity:
            raise ValueError(f"evaluation changed policy snapshot: {label}")
        comparison = tuple(record[key] for key in COMPARABLE)
        if settings.setdefault(scenario, comparison) != comparison:
            raise ValueError(f"evaluation settings differ for world seed {scenario}")
        policies[label][scenario] = record

    excluded = {}
    for scenario in scenarios:
        issues = {}
        for label, episodes in policies.items():
            episode = episodes.get(scenario)
            if episode is None:
                issues[label] = "missing"
            elif episode["integrity"] is not True or episode["transition_kind"] == "incomplete":
                issues[label] = episode["reason"]
        if issues:
            excluded[scenario] = issues
    matched = [scenario for scenario in scenarios if scenario not in excluded]
    return {
        "complete": bool(scenarios) and not excluded,
        "matched_world_seeds": matched,
        "excluded_world_seeds": excluded,
        "policies": {label: _summarize_policy(episodes, matched)
                     for label, episodes in policies.items()} if matched else {},
        "paired": {
            seed: {
                name: _paired(policies[f"seed{seed}-final"], policies[label], matched)
                for name, label in (("initial", f"seed{seed}-initial"),
                                    ("full-lock", "full-lock"), ("straight", "straight"))
            }
            for seed in seeds
        } if matched else {},
    }


def main():
    """Read the declared manifest and evaluation log lines or plain JSONL records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    records = []
    for path in args.records:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith(PREFIX) or line.startswith("{"):
                records.append(json.loads(line.removeprefix(PREFIX)))
    print(json.dumps(summarize(manifest, records), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
