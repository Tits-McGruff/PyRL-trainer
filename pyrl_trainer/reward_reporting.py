"""Reward exposure presentation without actor or rollout state ownership."""

from typing import Dict

from .reward_contract import REWARD_COMPONENT_KEYS


def format_reward_exposure(stats: Dict[str, float]) -> str:
    """Describe received events separately from the rollout-admission denominator."""
    rates = ",".join(f"{key}={stats[f'reward_{key}_per_alive_second']:.4f}"
                     for key in REWARD_COMPONENT_KEYS)
    cumulative = ",".join(f"{key}={stats[f'reward_total_{key}']:.3f}"
                          for key in REWARD_COMPONENT_KEYS)
    return (
        f"received[intervals={int(stats['received_intervals'])},"
        f"total_intervals={int(stats['received_interval_count'])},"
        f"elapsed_s={stats['received_elapsedSeconds']:.3f},"
        f"alive_s={stats['received_aliveSecondsDelta']:.3f},"
        f"food_value={stats['received_foodValueDelta']:.3f},"
        f"food_count={int(stats['received_foodCountDelta'])},"
        f"growth={stats['received_retainedGrowthDelta']:.3f},"
        f"task_points={stats['received_pointsDelta'] - stats['received_survivalPointsDelta']:.3f},"
        f"kills={int(stats['received_killsDelta'])},"
        f"boost_spent={stats['received_boostPointsSpentDelta']:.3f},"
        f"total_food={stats['received_total_foodValueDelta']:.3f},"
        f"total_alive_s={stats['received_total_aliveSecondsDelta']:.3f}] "
        f"admitted[transitions={int(stats['admitted_transitions'])},"
        f"elapsed_s={stats['admitted_elapsed_seconds']:.3f},"
        f"alive_s={stats['admitted_alive_seconds']:.3f},"
        f"decision_s={stats['decision_seconds_mean']:.5f}] "
        f"reward_per_admitted_alive_second[{rates}] reward_total[{cumulative}] "
        f"commands[accepted={int(stats.get('commands_accepted', 0))},"
        f"applied={int(stats.get('commands_applied', 0))},"
        f"superseded={int(stats.get('commands_superseded', 0))},"
        f"delay_samples={int(stats.get('command_delay_samples', 0))},"
        f"delay_steps_mean={stats.get('command_delay_steps_mean', 0):.2f},"
        f"delay_steps_max={int(stats.get('command_delay_steps_max', 0))},"
        f"fence_intervals={int(stats.get('epoch_fence_intervals', 0))},"
        f"fence_seconds={stats.get('epoch_fence_seconds', 0):.3f}]"
    )
