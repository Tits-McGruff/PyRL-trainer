# PyRL Trainer

Async multi-actor PPO trainer for a Slither-style WebSocket game server.

The trainer launches multiple bot actors that play in parallel, streams rollouts into a learner, and continuously updates a shared policy/value model with PPO.

## What It Does

- Connects to a game server over WebSocket (`hello` -> `welcome` -> `join` -> `assign`).
- Spawns multiple actors (`Config.actors`) to read sensors, sample actions, and stream rollouts.
- Applies action pacing using tick stride and wall-clock gating.
- Runs a learner loop that batches rollouts, computes GAE, and applies PPO updates.
- Syncs updated training weights to a dedicated inference model used by actors.
- Auto-saves checkpoints and auto-resumes compatible checkpoints on startup.
- Auto-creates `config.toml` with defaults if missing.

## Repository Layout

- `pyrl_trainer/__main__.py`: process entrypoint, config load, server sensor discovery, actor/learner task startup.
- `pyrl_trainer/agent.py`: actor client, handshake, action loop, rollout construction.
- `pyrl_trainer/learner.py`: shared state, PPO update, learner queue loop, checkpoint save/load.
- `pyrl_trainer/network.py`: `PolicyValueNet` MLP policy/value network.
- `pyrl_trainer/config.py`: config schema, TOML read/write, environment overrides.
- `pyrl_trainer/utils.py`: reward shaping, stride/rate helpers, misc utilities.
- `tests/`: unit, integration, system, and e2e coverage.
- `docs/API-Instructions`: protocol and API notes for the server interface.

## Requirements

- Python 3.10+ (3.11+ recommended).
- A running compatible WebSocket game server.
- PyTorch + NumPy + websockets (installed by `requirements.txt`).

## Setup

Linux/macOS:

```bash
./install.sh
```

Windows:

```bat
install.bat
```

The install scripts create `.venv`, upgrade packaging tools, and install dependencies.

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

On startup, the trainer:

- loads config from `SLITHER_CONFIG` or `config.toml`,
- connects once to discover `obs_dim` from server `sensorSpec`,
- attempts checkpoint resume from `ckpt_dir`,
- starts learner + actor tasks.

## Configuration

Default config file: `config.toml` (auto-generated if missing).

Main sections:

- `[connection]`: `ws_url`, `bot_name`, `actors`
- `[server]`: `max_actions_per_tick`, `max_actions_per_second`
- `[training]`: `horizon`, `gamma`, `gae_lambda`, `ppo_clip`, `ent_coef`, `vf_coef`, `lr`, `minibatch`, `epochs`, `turn_std`
- `[devices]`: `train_device`, `infer_device`
- `[model]`: `net_hidden`, `net_layers`
- `[logging]`: `log_every_seconds`
- `[checkpointing]`: `ckpt_dir`, `save_every_updates`, `keep_last`

Environment variables (`SLITHER_*`) override TOML values:

- `SLITHER_CONFIG`, `SLITHER_WS_URL`, `SLITHER_BOT_NAME`, `SLITHER_ACTORS`
- `SLITHER_MAX_APT`, `SLITHER_MAX_APS`
- `SLITHER_HORIZON`, `SLITHER_GAMMA`, `SLITHER_GAE_LAMBDA`, `SLITHER_PPO_CLIP`
- `SLITHER_ENT_COEF`, `SLITHER_VF_COEF`, `SLITHER_LR`, `SLITHER_MINIBATCH`, `SLITHER_EPOCHS`
- `SLITHER_TRAIN_DEVICE`, `SLITHER_INFER_DEVICE`
- `SLITHER_NET_HIDDEN`, `SLITHER_NET_LAYERS`
- `SLITHER_TURN_STD`, `SLITHER_LOG_EVERY`
- `SLITHER_CKPT_DIR`, `SLITHER_SAVE_EVERY_UPDATES`, `SLITHER_KEEP_LAST`
- `SLITHER_WS_MAX_MESSAGE`

Example:

```bash
SLITHER_WS_URL=ws://127.0.0.1:5174 SLITHER_ACTORS=8 python -m pyrl_trainer
```

## Checkpoints

Saved to `ckpt_dir` (default `./checkpoints`):

- `latest.pt`
- `latest_h{hidden}_l{layers}.pt`
- `ckpt_h{hidden}_l{layers}_{step}.pt`

Rotation keeps the most recent `keep_last` numbered checkpoints.
Resume only occurs if checkpoint architecture and observation shape are compatible.

## Testing

Run all tests:

```bash
pytest
```

Run by marker:

```bash
pytest -m unit
pytest -m integration
pytest -m system
pytest -m e2e
```

## Notes

- The actor enforces action pacing via `max_actions_per_second` (stride + wall-clock gate).
- `max_actions_per_tick` is in config for server alignment but is not directly enforced by actor code.
- If connection fails at startup, verify the server is running and reachable at `ws_url`.
