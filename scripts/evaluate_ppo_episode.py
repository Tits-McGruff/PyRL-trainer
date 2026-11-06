"""Score one frozen policy or fixed reference on a fresh Slither scenario."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from pyrl_trainer.__main__ import discover_sensor_contract
from pyrl_trainer.agent import ActorClient
from pyrl_trainer.checkpointing import load_frozen_evaluation_checkpoint
from pyrl_trainer.config import load_or_create_config, reward_config_record
from pyrl_trainer.evaluation import EvaluationEpisode
from pyrl_trainer.learner import SharedState
from pyrl_trainer.reward_contract import RewardContractError


REFERENCES = {"full-lock": (1.0, 0.0), "straight": (0.0, 0.0)}


class EpisodeState(SharedState):
    """Capture the first episode, including an incomplete result for exclusion."""

    def __init__(self, cfg, sensor_contract):
        super().__init__(sensor_contract.sensor_count, cfg, sensor_contract=sensor_contract)
        self.result = asyncio.get_running_loop().create_future()

    def record_evaluation_result(self, result, snapshot):
        if not self.result.done():
            self.result.set_result(result)


class QualificationActor(ActorClient):  # pylint: disable=too-few-public-methods
    """Use the regular reward and lifecycle path for learned and fixed controls."""

    def __init__(self, state, scenario_seed, reference):
        super().__init__(0, state.cfg, state, asyncio.Queue(), evaluation=True)
        self.scenario_seed = scenario_seed
        self.reference = reference
        self.scenario = {}
        self.episode_evidence = {}
        self.turn_samples = self.turn_saturated = 0
        self.turn_abs_sum = 0.0

    def _apply_welcome(self, msg):
        super()._apply_welcome(msg)
        if msg.get("worldSeed") != self.scenario_seed:
            raise RewardContractError("evaluation server does not have the declared world seed")
        core = msg.get("settings", {}).get("core", {})
        if not msg.get("configHash") or "simSpeed" not in core:
            raise RewardContractError("evaluation server omitted scenario configuration")
        self.scenario = {
            "world_seed": self.scenario_seed,
            "config_hash": msg["configHash"],
            "sim_speed": core["simSpeed"],
            "tick_rate": self.tick_rate,
            "max_actions_per_second": self.cfg.max_actions_per_second,
        }

    def _start_evaluation_policy(self):
        if self.reference is None:
            super()._start_evaluation_policy()
            return
        turn, boost = REFERENCES[self.reference]
        identity = json.dumps({"turn": turn, "boost": boost}, sort_keys=True).encode()
        digest = hashlib.sha256(b"pyrl-fixed-reference-v1\0" + identity).hexdigest()
        self.evaluation_episode = EvaluationEpisode(
            0, digest, self.cfg.evaluation_seed, self.cfg.evaluation_max_seconds
        )

    def _sample_action(self, obs):
        if self.reference is None:
            action = super()._sample_action(obs)
        else:
            turn, boost = REFERENCES[self.reference]
            action = (turn, boost, 0.0, 0.0, 0.0, 0)
        magnitude = abs(action[0])
        self.turn_samples += 1
        self.turn_saturated += int(magnitude > 0.95)
        self.turn_abs_sum += magnitude
        return action

    def _finish_evaluation(self, kind, reason, *, integrity):
        if not self.shared_state.result.done() and self.evaluation_episode is not None:
            samples = max(1, self.turn_samples)
            self.episode_evidence = {
                **self.scenario,
                "issued_decisions": self.turn_samples,
                "mean_abs_sampled_turn": self.turn_abs_sum / samples,
                "sampled_turn_saturation": self.turn_saturated / samples,
                "learned_turn_std": (None if self.episode_policy_model is None
                                     else self.episode_policy_model.steering_std().item()),
            }
        super()._finish_evaluation(kind, reason, integrity=integrity)


async def evaluate_one(cfg, *, scenario_seed, checkpoint=None, reference=None, timeout=360.0):
    """Run to the first complete or excluded result and release the owned connection."""
    if (checkpoint is None) == (reference is None):
        raise ValueError("select one checkpoint or fixed reference")
    if reference is not None and reference not in REFERENCES:
        raise ValueError("unknown fixed reference")
    contract = await asyncio.wait_for(discover_sensor_contract(cfg.ws_url), timeout=15)
    if (contract.layout_version, contract.profile) != ("v4", "motion"):
        raise RewardContractError("qualification requires the v4 Motion sensor contract")
    state = EpisodeState(cfg, contract)
    if checkpoint is not None:
        load_frozen_evaluation_checkpoint(checkpoint, cfg, state)
    actor = QualificationActor(state, scenario_seed, reference)
    task = asyncio.create_task(actor.run())
    try:
        done, _ = await asyncio.wait(
            (task, state.result), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
        if state.result not in done:
            raise TimeoutError("evaluation did not produce an episode within the wall-time limit")
        result = state.result.result()
        raw = result["raw_deltas"]
        alive = result["alive_seconds"]
        return {
            **result, **actor.episode_evidence,
            "reward_config": reward_config_record(cfg),
            "reference": reference,
            "fixed_controls": REFERENCES[reference] if reference is not None else None,
            "task_net_points": raw["pointsDelta"] - raw["survivalPointsDelta"],
            "food_value_per_alive_second": raw["foodValueDelta"] / alive if alive else 0.0,
            "decisions_per_alive_second": actor.episode_evidence["issued_decisions"] / alive
            if alive else 0.0,
        }
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def main():
    """Connect to a scenario started independently with the declared world seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--scenario-seed", required=True, type=int)
    parser.add_argument("--label", help="Policy label for the paired qualification report.")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--reference", choices=REFERENCES)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error("--config must name the declared trainer configuration")
    cfg = load_or_create_config(str(args.config))
    cfg.ws_url = args.ws_url
    result = asyncio.run(evaluate_one(
        cfg, scenario_seed=args.scenario_seed, checkpoint=args.checkpoint, reference=args.reference
    ))
    result["policy_label"] = args.label or args.reference
    print(f"[qualification-episode] {json.dumps(result, sort_keys=True, allow_nan=False)}")


if __name__ == "__main__":
    main()
