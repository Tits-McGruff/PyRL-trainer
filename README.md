# PyRL Trainer

Async multi-actor PPO trainer for a Slither-style WebSocket game server.

The trainer launches multiple bot actors that play in parallel, streams rollouts into a learner, and continuously updates a shared policy/value model with PPO.

## What It Does

- Connects to a game server over WebSocket (`hello` -> `welcome` -> `join` -> `assign`).
- Spawns multiple actors (`Config.actors`) to read sensors, sample actions, and stream rollouts.
- Applies action pacing using tick stride and wall-clock gating.
- Runs a learner loop that batches rollouts, computes GAE, and applies PPO updates.
- Syncs updated training weights to a dedicated inference model used by actors.
- Auto-saves checkpoints and auto-resumes the newest compatible recoverable checkpoint on startup.
- Auto-creates `config.toml` with defaults if missing.

## Repository Layout

- `pyrl_trainer/__main__.py`: process entrypoint, config load, server sensor discovery, checkpoint restore, and actor/learner task supervision.
- `pyrl_trainer/agent.py`: actor client, handshake, action loop, reconnect handling, reward application, and rollout construction.
- `pyrl_trainer/learner.py`: shared model state, PPO update, diagnostics, and learner queue loop.
- `pyrl_trainer/checkpointing.py`: the sole checkpoint persistence/recovery implementation, including atomic publication, candidate selection, compatibility checks, rollback, and restoration.
- `pyrl_trainer/network.py`: `PolicyValueNet` MLP policy/value network.
- `pyrl_trainer/config.py`: config schema, TOML read/write, validation, environment overrides, and reward/checkpoint schema constants.
- `pyrl_trainer/utils.py`: reward shaping, stride/rate helpers, and misc utilities.
- `tests/`: unit, integration, system, and e2e coverage.
- `docs/API-Instructions`: protocol and API notes for the server interface.

## Requirements

- Python 3.10+.
- A running compatible WebSocket game server.
- PyTorch, NumPy, websockets, and TOML support as listed in `requirements.txt`.
- `constraints.txt` records the top-level dependency versions established by the passing Python 3.10, 3.11, and 3.14 CI matrix. `torchvision` and `torchaudio` are not required by this trainer.

## Setup

Linux/macOS:

```bash
./install.sh
```

Windows:

```bat
install.bat
```

The install scripts create `.venv`, upgrade packaging tools, and install the constrained dependency set. The normal requirements file uses the CUDA 12.6 PyTorch package index; CI explicitly installs CPU PyTorch while applying the same version constraints.

## Run

Linux/macOS:

```bash
./run.sh
```

Windows:

```bat
run.bat
```

Direct Python entrypoint:

```bash
python -m pyrl_trainer
```

`run.sh` and `run.bat` are one-shot launchers. They do not restart the trainer after a fatal failure. Use an external supervisor if unattended restart policy is required.

On startup, the trainer:

- loads config from `SLITHER_CONFIG` or `config.toml`,
- validates and normalizes configuration before runtime tasks start,
- connects once to discover the exact Protocol 2 sensor contract,
- builds model state from that contract,
- attempts a transactional checkpoint resume from `ckpt_dir`,
- starts the learner and actor tasks only after checkpoint restoration is complete.

A fatal learner or actor task failure cancels and awaits its sibling trainer tasks before the original failure escapes. Ordinary actor disconnects remain actor-local reconnect events with the existing fixed 0.5 second retry delay.

## Configuration

Default config file: `config.toml`, auto-generated if missing.

Main sections:

- `[connection]`: `ws_url`, `bot_name`, `actors`
- `[server]`: `max_actions_per_tick`, `max_actions_per_second`
- `[training]`: `horizon`, `batch_size`, `gamma`, `gae_lambda`, `ppo_clip`, `ent_coef`, `vf_coef`, `lr`, `minibatch`, `epochs`, `turn_std`
- `[rewards]`: complete reward-shaping configuration described below
- `[devices]`: `train_device`, `infer_device`
- `[model]`: `net_hidden`, `net_layers`
- `[logging]`: `log_every_seconds`
- `[checkpointing]`: `ckpt_dir`, `save_every_updates`, `keep_last`

Environment variables (`SLITHER_*`) override TOML values. General overrides are:

- `SLITHER_CONFIG`, `SLITHER_WS_URL`, `SLITHER_BOT_NAME`, `SLITHER_ACTORS`
- `SLITHER_MAX_APT`, `SLITHER_MAX_APS`
- `SLITHER_HORIZON`, `SLITHER_BATCH_SIZE`, `SLITHER_GAMMA`, `SLITHER_GAE_LAMBDA`, `SLITHER_PPO_CLIP`
- `SLITHER_ENT_COEF`, `SLITHER_VF_COEF`, `SLITHER_LR`, `SLITHER_MINIBATCH`, `SLITHER_EPOCHS`
- `SLITHER_TRAIN_DEVICE`, `SLITHER_INFER_DEVICE`
- `SLITHER_NET_HIDDEN`, `SLITHER_NET_LAYERS`
- `SLITHER_TURN_STD`, `SLITHER_LOG_EVERY`
- `SLITHER_CKPT_DIR`, `SLITHER_SAVE_EVERY_UPDATES`, `SLITHER_KEEP_LAST`
- `SLITHER_WS_MAX_MESSAGE`

Example:

```bash
SLITHER_WS_URL=ws://127.0.0.1:3000 SLITHER_ACTORS=8 python -m pyrl_trainer
```

## Reward Configuration

The default generated config contains:

```toml
[rewards]
growth_scale = 10.0
food_approach_scale = 0.5
food_delta_clip = 0.1
survival_bonus = 0.0001
wall_threshold = -0.5
wall_penalty_scale = 0.05
hazard_threshold = -0.5
hazard_penalty_magnitude = 0.1
death_penalty_magnitude = 0.5
```

The corresponding environment overrides are:

- `SLITHER_REWARD_GROWTH_SCALE`
- `SLITHER_REWARD_FOOD_APPROACH_SCALE`
- `SLITHER_REWARD_FOOD_DELTA_CLIP`
- `SLITHER_REWARD_SURVIVAL_BONUS`
- `SLITHER_REWARD_WALL_THRESHOLD`
- `SLITHER_REWARD_WALL_PENALTY_SCALE`
- `SLITHER_REWARD_HAZARD_THRESHOLD`
- `SLITHER_REWARD_HAZARD_PENALTY`
- `SLITHER_REWARD_DEATH_PENALTY`

For a transition with a previous observation, the default reward is the sum of five components. `growth` is `growth_scale * points_delta_norm`. `food_approach` is `food_approach_scale * clamp(current_closeness - previous_closeness, -food_delta_clip, +food_delta_clip)`. `survival` is the configured per-transition survival bonus. If `wall_dist_norm < wall_threshold`, the wall contribution is `-wall_penalty_scale * (-wall_dist_norm)^2`. If the existing front-hazard-bin average is below `hazard_threshold`, the hazard contribution is `-hazard_penalty_magnitude`. Missing optional sensor labels contribute zero to the relevant component. The first observation has no previous observation and therefore produces zero reward components.

Protocol 2 calls the food field `nearest_food_dist_norm`, but its numeric meaning is closeness on the trainer side: values increase as food gets closer, with roughly `-1` representing the far/no-food limit and `+1` representing zero distance. The wire field name remains unchanged for Protocol 2 compatibility.

A genuine terminal death subtracts `death_penalty_magnitude` exactly once from the pending transition. A `stateReplaced` event terminates the old transition with zero death contribution.

All reward values are validated as finite floats. Scale/bonus/penalty magnitudes must be non-negative, `food_delta_clip` must be positive, and the wall/hazard thresholds must be within `[-1, 1]`.

## Policy Diagnostics

Turn actions keep the existing fixed-standard-deviation Gaussian policy followed by `tanh`. `turn_std` controls continuous-turn exploration and remains fixed rather than learned. The learner reports `entropy_turn` and `entropy_boost` separately and also retains `entropy` as their sum. Because `turn_std` is fixed, turn entropy is constant with respect to the trainable turn mean; `ent_coef` therefore has a learnable entropy effect through the Bernoulli boost policy, while the PPO loss formula itself remains unchanged.

The learner also reports `turn_mean_abs_mean`, `turn_mean_abs_max`, and `turn_saturated_fraction`. Saturation is the fraction of evaluated samples for which `abs(tanh(turn_mean)) > 0.95`. These are diagnostics only; no hard clamp is applied to `turn_mean`.

## Checkpoints

Checkpoint persistence and recovery are owned exclusively by `pyrl_trainer/checkpointing.py`. Files are saved under `ckpt_dir`, default `./checkpoints`:

- `latest.pt`
- `latest_h{hidden}_l{layers}.pt`
- `ckpt_h{hidden}_l{layers}_{step}.pt`

Each destination is published atomically through a temporary sibling file followed by `os.replace`. A save commits the numbered recovery checkpoint first, then the architecture-specific latest file, then the generic latest file, then rotates old numbered checkpoints. A crash between those publications can therefore leave pointer files at an older step without invalidating the newer numbered checkpoint.

Startup deserializes recoverable candidates, validates them before mutating live state, ranks compatible candidates by stored `update_steps`, and restores the highest step. Equal-step ties prefer the architecture-specific latest file, then generic `latest.pt`, then a numbered checkpoint. Corrupt or incompatible candidates are skipped. If model/optimizer application fails, the pre-load model, optimizer, update count, and inference policy are restored before the loader tries the next candidate.

New checkpoints use schema version 2 and store the complete effective reward configuration as reward-config version 1. Automatic resume requires compatible model architecture, sensor contract, and reward configuration. Schema-v2 checkpoints with different rewards are rejected. Legacy checkpoints without reward metadata are accepted only when the current reward configuration exactly matches the canonical pre-v2 defaults documented above; using changed rewards causes those legacy checkpoints to be skipped. Unknown future checkpoint schemas or reward-config versions are skipped safely. There is no force-load override in this remediation.

## Testing

Run the complete suite:

```bash
python -m pytest -q
```

Run by marker:

```bash
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m system
python -m pytest -m e2e
```

Run lint with the same command used by CI:

```bash
pylint --disable=duplicate-code,too-many-instance-attributes,too-many-locals,try-except-raise $(git ls-files '*.py')
```

CI runs lint, the full suite, and all four marker groups on Python 3.10, 3.11, and 3.14 with CPU PyTorch.

## Notes

- The actor enforces action pacing via `max_actions_per_second`, using stride plus a wall-clock gate.
- `max_actions_per_tick` is retained in config for server alignment but is not directly enforced by actor code.
- The WebSocket message-size limit has one source, `pyrl_trainer.config.MAX_WS_MESSAGE_BYTES`, controlled by `SLITHER_WS_MAX_MESSAGE`.
- If connection fails at startup, verify the server is running and reachable at `ws_url`.
