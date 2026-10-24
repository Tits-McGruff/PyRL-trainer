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
SLITHER_WS_URL=ws://127.0.0.1:5174 SLITHER_ACTORS=8 python -m pyrl_trainer
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


## Canonical graph distillation tools

Select the canonical job owner with `python -m pyrl_trainer --mode distillation`.
Its remaining flags go directly to the dedicated distillation runner;
`python -m pyrl_trainer --mode distillation --help` lists them. The standalone
`python -m pyrl_trainer.distillation_run` invocation remains available.
`python -m pyrl_trainer` and `--mode ppo` retain PPO startup from
`config.toml`/`SLITHER_CONFIG`. Distillation mode does not load that PPO
configuration or import its actor/learner owners, construct actors, or start a learner queue.

New canonical jobs expose the following immutable settings. All ranges are checked
before source/checkpoint access; unavailable or physically oversized CUDA requests
fail explicitly. The initial device admission ceiling remains 8 GiB. The selected
ceiling need not be entirely free: the admitted graph/window estimate must fit free
VRAM, and actual Torch allocation peaks are checked during updates.

| CLI control | New-job default | Public range |
|---|---:|---|
| `--seed` | 1 | UInt32 |
| `--cpu-threads` | min(2, available process CPUs) | 1..available, respecting affinity |
| `--device-memory-mib` | 8192 | 512..8192, within physical CUDA capacity |
| `--replay-ram-mib` | 256 | 64..2048 |
| `--spool-gib` | Observe producer (4 GiB by default) | Require matching 1..64 GiB producer disk quota |
| `--learning-rate` | 0.0003 | finite 0.000001..0.01 |
| `--batch-size` | 32 | 1..256 sequence windows |
| `--bptt` | 64 | 16..256 delivered observations |
| `--burn-in` | 32 | 0..128 delivered observations |

Burn-in plus BPTT must be at most 256; incompatible combinations are rejected.
Replay RAM bounds raw/decoded chunks and bookkeeping. It is independent of HTTP
transfers, sampler tensors, model/optimizer memory and disk checkpoint work. Larger
quotas do not eagerly allocate their capacity. Full loaded resource/soak qualification
remains open. `--device cpu` explicitly selects CPU training.

Exact resume requires matching trainer source/numerical environment and preserves
every saved choice. New-job defaults never replace checkpoint settings; library
defaults and smaller library-only windows remain supported. Omit settings flags to
use the checkpoint, or repeat the exact same values. Changing thread/device/replay
budgets, seed, learning rate or
sequence settings requires a new job. The runner reports effective choices under
`settings` in its final JSON result. PPO settings and dependencies remain separate.

The HTTP consumer reads the producer's admitted global storage policy before rows.
Its byte/count/lease ceilings are immutable source provenance in version-2 client
checkpoints, saved atomically with replay, cursors, sampler, optimizer and RNG.
Select the producer quota with Slither's `--distillation-spool-gib N`, then use
`--spool-gib N` to require that value. A policy change stops the fetch before new
rows or cursor advancement and requires a new job. Offline exact resume retains
the saved policy without network access. Legacy providers that return 404 for
`GET /api/distillation/v1/storage` are recorded as unreported, with no guessed
quota; explicit budget requests require reported admission. Other errors remain
failures. The ceiling does not reserve that amount of disk or RAM, and loaded
storage-capacity qualification remains open.

`pyrl_trainer.canonical_graph` provides a separate differentiable representation
of Rust's deployed graph. It uses one flat Float32 parameter tensor and the
Rust manifest's exact parameter views, custom recurrent equations, sensor
identity and output order. It does not use PPO's `PolicyValueNet` or `ActorClient`.
The graph/loss/parity and bounded sequence-training primitives are implemented.
The HTTP teacher consumer and strict Rust sample decoder are also implemented;
Bounded replay and stable whole-stream partitioning are implemented in
`distillation_replay`; bounded window sampling and an initial standalone durable job owner are now
implemented. Portable candidates, native upload admission and explicit shadow
activation/metrics are implemented. Saved-library deployment and integrated
resource/gameplay qualification remain in progress.
The sequence API is documented in
[`docs/canonical-sequence-training.md`](docs/canonical-sequence-training.md).

Run CPU or CUDA qualification with an explicitly built Rust runner:

```powershell
.venv/Scripts/python.exe -m pyrl_trainer.canonical_parity --runner C:/path/to/canonical-training-graph.exe --fixtures tests/fixtures/canonical_graph --device cuda --output-json new-roundtrip.json
```

This gate performs real optimizer updates, exports canonical Float32 bytes,
independently admits them in Rust, evaluates both native math backends, and
reimports exact bytes into PyTorch. An unavailable requested GPU fails the gate.
TF32 and autocast are disabled for the initial parity envelope.
CPU affine rows of width 128 or more use batched 16-term Float32 matrix partials
and an error-corrected reduction tree to reduce cancellation across CPU BLAS
implementations and batch shapes. The analytic matrix derivative saves only the
original input/weight views, avoiding a retained correction graph per delivery.
Short CPU rows and CUDA retain their direct matrix operation. Parameters,
activations and gradients stay Float32; gradients and strided views are checked
against an independent Float64 oracle. Arithmetic remains bound to
the recorded trainer/native sources and environment; old jobs are not exact
continuations under a changed trainer source. Unit tests also
compare every output and next recurrent state over frozen 256-delivery sequences.
Compact CPU/RTX4080 reports and reproduction notes are under
[`docs/performance/canonical_graph`](docs/performance/canonical_graph/README.md).
These synthetic gates measure semantic/optimizer parity, not gameplay quality.
Use `--include-large` in both `scripts/generate_canonical_graph_fixtures.py`
and `pyrl_trainer.canonical_parity` to require Motion/B32 GRU/LSTM/RRU graphs
with five 256-wide feature layers and 96 recurrent units. CI generates these
with the pinned native runner and requires all 18 CPU optimizer roundtrips;
missing large fixtures fail rather than being skipped.


An existing CPU Torch installation can satisfy the version constraint even
when the CUDA index is configured. To replace it explicitly inside `.venv`:

```powershell
.venv/Scripts/python.exe -m pip install --force-reinstall --no-deps -c constraints.txt --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```


`pyrl_trainer.distillation_client.DistillationClient` reads the dedicated
`/api/distillation/v1/` namespace without joining an actor or sending actions.
Configure a server origin and capture epoch, call `fetch()` to obtain bounded
chunks and explicit unavailable ranges, then `commit()` only when the job owns
that proposal. Save `checkpoint_fields()` atomically with replay, sampler,
optimizer and RNG state. A fetch never advances the local cursor or acknowledges
a game transaction. Downloads use finite storage pins that are released after
validation, including on failure. Restarts bind the source, complete native
contract, canonical sensor bytes and recorded run seed; legacy unknown seeds
are rejected. The frozen teacher fixture is produced by Rust's
`teacher_chunk_reference` tool and tests codec interoperability, not gameplay.

Optional native shadow evaluation is explicit. Add all three flags to an online
new-job invocation of `pyrl_trainer.distillation_run`:

```powershell
--shadow-slot 0 --shadow-baseline-slot 1 --shadow-config-revision 0000000000000001
```

The chosen slot is 0..3 and follows one selected baseline teacher. The run must
already admit baseline v2 and matching v4 sensors/cadence; ordinary v3 production
defaults remain unchanged pending qualification. Candidate upload alone does not
activate a student. An opted-in job freezes the complete request and immutable
candidate in a self-contained checkpoint before issuing the ordered native
activation. Lost replies retry the same process/request/observed-slot identity;
a changed process or replaced segment fails explicitly. Exact job resume cannot
change shadow configuration. Metrics are Rust-computed unweighted summaries,
including exact counters and separate delivery/publication steps. After activation,
job polling returns no snapshot while the separate native mailbox still holds
the exact prior slot; after the new segment is observed, interruptions fail. They never
consume a dataset cursor, send gameplay ACKs or rewrite controlling weights.
`DistillationClient.shadow_metrics`, `activate_shadow` and `stop_shadow` also
expose these bounded diagnostic routes for explicit library use.


The live qualification gate accepts `--replay-min-mib` below the selected
`--replay-mib` capacity to require actual retained ownership before training.
For a near-capacity 2-GiB case, use `--replay-mib 2048 --replay-min-mib 2040`.
The floor uses the ordinary replay accounting; its receipt separates raw/decoded
data bytes from total charged ownership. Only validated samples from the active
capture epoch can satisfy it. Explicit loading gets a separate absolute
`--replay-fill-seconds` setup allowance (1..10,800, default 1,800), followed by
the existing optimizer-work deadline. Resume preserves the requested floor;
sustain fails if live eviction ever lowers ownership beneath it. This is an
optional qualification load, outside ordinary job controls and defaults.

The `live-distillation` CI job runs the cross-repository qualification host against
real Rust v4/v2 capture over HTTP. It trains with the default 32-window batch,
32-step burn-in and 64-step BPTT, saves at update 125, and launches a second Python
process to reproduce update 126's complete numeric witness. Update 250 freezes an
immutable candidate for Rust validation and four actual shadow slots. The host
uses a disposable database/spool and admits no actor or controller traffic. Its
compact reports include process RSS, Torch allocator peaks, native step timing and
full source identities, plus matching producer storage policy across the fresh
trainer process. CPU CI exercises 4/64 GiB ceilings with one/three Rust workers;
these are admission/provenance checks, not full-disk allocation benchmarks.
Local CUDA runs use the same gate with explicit `cuda:0`;
CPU CI never silently substitutes for CUDA qualification.

The pinned host also trains a second student matching the active Dense world
architecture. It saves both trained architectures, verifies graph-mismatched
spawn rejection, exports the compatible portable original exactly, and deploys
through Rust. After catalog deletion and cold-file reclamation, the live retained
brain must keep its distilled origin and produce actual neural inference.

The host also accepts `--workload P0|P1 --student-artifact PATH --contention-seconds 60`
for a sustained contention window. Supply an independently Rust-validated Motion/B16
portable student with the matching 1/60-second teacher cadence. The disposable world
uses the modest native graph with 55/300 random evolved individuals and ten stochastic
v2 teachers. After exact trainer restart and four-shadow activation, a third Python
process restores the saved job and runs real Adam updates while fetching new live
chunks for 60..300 seconds. Both optimizer progress and fresh source-cursor progress
are required. Reports preserve failed real-time gates, replay/RSS/VRAM peaks, capture
settlement/drops and each shadow's sample progress. The window includes Python startup
and restore; active optimizer-window duration is reported separately. This does not
qualify a mature population, crowded geometry, a long soak or production defaults.
The live CI job includes a 60-second P0 CPU contention smoke and retains its
measured timing result even when the real-time performance gate fails.

### Canonical TBPTT capacity probe

Use an explicitly Rust-compiled manifest and initialized flat weights to measure
actual canonical model, Adam and full-window TBPTT allocation:

```powershell
python -m pyrl_trainer.distillation_capacity_gate --manifest PATH/manifest.json --weights PATH/weights.f32 --device cuda:0 --output capacity.json
```

The probe shares the running job's conservative memory estimate and rejects an
oversized request before model/device allocation. Select `--device cpu` explicitly
for CPU measurement. `--batch-sequences`, `--sequence-length`, `--burn-in`,
`--cpu-threads` and `--device-memory-mib` use the public job bounds; the combined
window remains at most 256 deliveries. Reports are create-new and include exact
source/input identities, completed optimizer work, process RSS and CUDA allocated
and reserved peaks. An admitted run must complete all requested updates within
its budget. A rejected case reports `qualificationComplete: false`.

Inputs are synthetic Float32 windows with fully active rows and rare-event weight
four. This measures training allocation independently of teacher encoding, replay
RAM, HTTP, trainer checkpoints, live game contention, shadows and the integrated
soak. The device budget is an admission ceiling; successful probes do not imply
that the ceiling was allocated or that every graph within a parameter count has
the same memory cost.

Dataset metadata admits the protocol's 64-GiB producer ceiling while every intake
and restored manifest must fit the producer's observed byte and object policy.
This avoids an obsolete 4-GiB parser limit blocking a larger explicitly selected
spool. Reading a large inventory does not allocate its advertised bytes: chunk
fetching, decoding and replay retain their independent bounds. Legacy endpoints
without a reported policy remain explicitly unreported; the client does not guess
their selected quota.

### Loaded replay RAM probe

Generate a synthetic fixture with Slither's standalone Rust
`scripts/qualification/teacher-replay-fixture` harness, which calls the production
teacher encoder and checks each native roundtrip. Then run:

```powershell
python -m pyrl_trainer.distillation_replay_capacity_gate --fixture PATH --ram-mib 2048 --output replay-capacity.json
```

Select 64..2048 MiB explicitly. The fixture must load through rotation at the
selected ceiling; a partial load cannot qualify. The host checks the exact bundle
hash, genuine decoded chunk identities, per-batch RAM accounting, whole-chunk
retention, exact-retry idempotence and independent-decode replay restore. It records
actual process RSS both after filling and while the original and restored replay
owners coexist. Their combined restore peak is distinct from one buffer's retained
RAM budget. Reports are create-new; fixture files remain read-only for reuse.

The fixture is length-framed Rust TeacherSampleV1 data, with no Python sample
encoder. This probe covers decoder/replay/restore memory independently of HTTP,
optimizer work, durable trainer checkpoint peaks, live game contention and the
integrated soak. Large synthetic inputs are disposable and should be removed after
required cross-platform reports are retained.

Capacity probes require Python assertions enabled and reject `python -O` before
fixture/output or device admission. Replay reports include interpreter, platform,
NumPy/Torch versions, pointer width and CPU-thread facts alongside measured RSS.


### Loaded durable trainer checkpoint probe

Use the same Rust replay fixture and a compatible Rust-produced canonical manifest
and initialized weights. Keep the new workspace separate from the read-only inputs:

```powershell
python -m pyrl_trainer.distillation_checkpoint_capacity_gate --fixture PATH --manifest PATH/manifest.json --weights PATH/weights.f32 --ram-mib 2048 --device cuda:0 --workspace NEW_DIRECTORY --output checkpoint-capacity.json
```

The probe fills the selected 64..2048 MiB replay through retention, completes a real
default-window Adam update, and saves through the production fsynced job serializer.
It restores an independent job and requires the next windows, loss, weights, Adam
moments, recurrent state and private RNG to match exactly. Reports include checkpoint
bytes and save/restore process peaks while both replay/model owners coexist. CUDA
uses the existing explicit device admission; `--device cpu` runs the CPU alternative.

The output must be new and outside the new workspace. A complete progress report is
fsynced before success cleanup; failures retain the checkpoint workspace. The
fixture is never edited. This offline test does not qualify HTTP, live game
contention, pending candidate backlog or the integrated
soak. It rejects disabled Python assertions before input access.


Use `--checkpoint-saves 4` to exercise production retention of the latest three
complete files. The host verifies serial progression, exact retained inventory,
oldest-file removal and retained physical bytes after every publication, then
requires exact next-update restore from the latest checkpoint. The default remains
one publication; reports state the selected publication count explicitly.


### Tactical profile and width qualification

The native tactical qualifier can export either v4 Basic or Motion with an
explicit B8..32 width. `load_tactical_corpus` validates the exact native canonical
metadata, digest, row bytes and matching teacher policy names before exposing
read-only arrays. Existing Motion/B16 corpora retain their interpretation.

`tactical_ablation_gate` derives its complete graph input and deployment request
from that validated contract. Motion keeps its seven declared channel ablations
at the chosen width; Basic trains Full and rejects unavailable Motion-only
groups. Window padding, agreement evaluation and native before/after student
physics use the same declared width. Separate profile corpora contain their
actual teacher trajectories and do not establish paired cross-profile data.
These tools enable qualification; they do not select sensor defaults or prove
policy maturity or real-time performance.

### Replicated tactical ablation evidence

Run `pyrl_trainer.tactical_ablation_gate` separately for each requested initializer
seed and held-out geometry seed, using one immutable native corpus, runner pair,
device/environment and optimizer budget. The corpus must contain every held-out
seed plus training seeds. Each study still trains all seven variants on paired
windows and performs actual before/after native student trials. Reports now retain
separate deterministic/stochastic teacher-policy and scenario/policy statistics;
these strata cannot cancel each other inside one aggregate agreement score.

Combine completed studies with the reader-only `pyrl_trainer.ablation_matrix`:

```text
python -m pyrl_trainer.ablation_matrix --study SEED1_FOLD2 --study SEED1_FOLD3 --initialization-seed 1 --validation-seed 2 --validation-seed 3 --updates 5000 --output matrix.json
```

Supply every cell of the requested seed/fold Cartesian grid. A study's `--seed`
selects initialization/sampling and `--validation-seed` excludes that complete
native geometry seed from training. Repeated-initialization/fold coverage is
reported only with at least three distinct identities on each axis; the example
above intentionally does not meet that coverage. More optimizer updates alone do
not prove a mature model.

The reader verifies all seven Motion ablations or the sole Basic Full variant,
matched windows within each cell, matching
initial parameters across folds, unchanged sources/environment/hyperparameters,
actual native outcome reports and exact portable bytes. It recomputes the trainer
source identity from the retained source inventory. Results retain per-cell paired
differences from Full, cell-wise macro averages and both policy strata. The tool
never chooses a winner, claims independent folds/confidence intervals, promotes a
sensor profile or qualifies representative populations, sensing cost or real-time
performance. Reports preserve the declared profile and B8..32 width. Older
studies without either field retain their original Motion/B16 interpretation;
partial declarations reject. Different profiles or widths cannot enter the same
paired matrix, and Basic statistics retain their own teacher-policy names.
CI verifies real two-update Basic/B8 and Motion/B32 studies as separate one-cell
matrices.

Exact job source identity also includes `sensor_contract.py`; a changed sensor
parser cannot retain the identity of a cold exact trainer restart. Checkpoint files
remain untouched and exact restoration still requires matching trainer source.


For held-out learning curves, add repeated `--validation-update 250 --validation-update 500 --validation-update 1000` flags to each child study. The
final requested optimizer update is included automatically; checkpoint work is
bounded at sixteen positive points plus the initial model. Checkpoints evaluate
the same held-out native lives without changing packed weights, CPU/CUDA RNG or
the paired Adam trajectory. The matrix reader validates complete matching curves
and reports per-update cell distributions and differences from Full, including
both teacher-policy strata. Final-only studies remain supported. Agreement curves
provide maturity evidence alongside the existing native gameplay and performance
gates; they do not automatically choose production sensor defaults.
