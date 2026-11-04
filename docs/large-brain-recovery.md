# Large-brain artifacts and admission-only checkpoint migration

Production graphs use the pinned Slither contract: at most 2,000,000 Float32
parameters (8,000,000 raw weight bytes), tensor width 4096, and 16,384 recurrent
state floats. The canonical parser, portable artifact reader/writer and upload
receipt validator use those same ceilings. A shape stride can be width + 1 for
interleaved affine biases. Rust still independently recompiles every upload.

Checkpoint weights use the raw-byte ceiling. Adam has two Float32 moment arrays
(up to 16,000,000 bytes together) plus scalar/serialization overhead inside its
existing 32 MiB member allowance. RNG's 256 KiB allowance is independent of graph
size. Detached recurrent state permits 256 windows times 16,384 floats times four
bytes (16 MiB). Replay, sequence/device admission, aggregate checkpoint bytes and pending
candidate count remain bounded separately. Checkpoint ZIP members remain stored,
with exact inventory and immutable member digests; the format version is unchanged.

## Use a completed model without restarting training

An artifact export does not execute the trainer or require equivalent source or
numerical environments. It verifies the complete checkpoint member inventory,
all immutable role/replay/candidate digests, the canonical graph and finite weights.
It preserves the original trainer source and update count as portable provenance.
It does not deserialize optimizer state, claim a deterministic restart, publish
anything to Slither, or modify the checkpoint.

```powershell
python -m pyrl_trainer.distillation_export `
  path\to\job-000000000000008d.distill-job `
  path\to\completed-5000.slither-brain
```

The output must not already exist. Confirm the printed weight digest against the
checkpoint's `job.json` digest for `weights.f32`. In Slither use **Brain Library**,
choose the original `.slither-brain` file, enter a name and click **Import**. This
is an explicit library save. Alternatively use the existing uploaded candidate's
digest and **Save candidate**. Select **Details**, verify the graph/sensor/cadence
compatibility indicators, then explicitly click **Spawn** in a matching world.
The model can be exported individually or included in a selected-brain checkpoint
archive. Uploads alone remain transient and never spawn or save themselves.

## Continue a job across the reviewed admission-only patch

Ordinary `distillation_run --resume` still requires exact trainer source and the
recorded numerical environment. There is no ignore-source or arbitrary-hash flag.
Do not edit checkpoint metadata by hand. Model computation, losses, optimizer,
sampling, RNG and serialized payload semantics have not changed in this patch.

`python -m pyrl_trainer.distillation_migrate` supports only the built-in
`large-brain-admission-v1` approval. Its compact approval contains per-file hashes
of the reviewed upstream source, enumerated manual ceiling edits, and the exact
successor source/contracts. Any other source change rejects and requires a new
code review and a deliberately versioned migration; supplying matching arbitrary
source files or hashes cannot approve it. No historical source tree is bundled.

Keep the old checkout and its runtime intact, and create a separate Git worktree
for the patched source. Use the original job's Python/Torch/NumPy/CUDA environment.
Install the patch's pinned `requirements-contracts.txt` in that environment only
after stopping its trainer process. Preserve the old source checkout until the
migration has completed. Example (paths are operator-selected):

```powershell
git fetch origin codex/large-brain-lifecycle
git worktree add ..\PyRL-large-brain origin/codex/large-brain-lifecycle
Set-Location ..\PyRL-large-brain
# Run with the original numerical environment's Python:
python -m pip install -r requirements-contracts.txt
python -m pyrl_trainer.distillation_migrate `
  path\to\original.distill-job path\to\new-admission-v1.distill-job `
  --source-root path\to\old-checkout\pyrl_trainer
```

The tool proves the old source files and contract identity explain the checkpoint's
actual source hash, verifies the original checkpoint, writes a new temporary ZIP,
and copies every member except `job.json` byte for byte. Only its source identity
changes. It then runs the ordinary complete restore validation: graph identity,
full Adam settings/moments, environment, replay/index/sampler, Python/Torch/CUDA RNG,
recurrent state and all pending candidate job/deployment/update identities. It
verifies immutable bytes again and publishes exclusively to a new path, with a
sidecar receipt recording old/new file/source digests and the unchanged update
count. The original is never overwritten. Source files changed after process
startup, a different numerical environment, corruption or an unapproved edit must
reject before successor publication. Migration performs no training or HTTP calls.

Resume the new path through the normal runner only when more training is intended:

```powershell
python -m pyrl_trainer.distillation_run --resume path\to\new-admission-v1.distill-job `
  --job-directory path\to\existing-job-directory --updates 100 --offline
```

`--updates` requests additional work; do not resume a completed job merely to save
its model. Export its saved weights instead. If the original source files are no
longer available or differ from the reviewed revisions, retain the checkpoint and
recover the exact source from Git/the preserved deployment. Migration never guesses
source equivalence. Export remains available without trainer restart.

## Deployment order and data preservation

Deploy Slither first, rebuilding its source-identified native addon and browser.
Keep its database, managed checkpoint directory, diagnostic spool/candidates and
server TOML. A source change uses the supported labelled latest-checkpoint recovery;
it does not promise bit-identical continuation across builds. Then update PyRL and
its exact Slither contracts pin. Smaller portable v1 artifacts and valid checkpoint
payload formats remain readable. Source identity changes still require the explicit
migration for restart; this is separate from portable artifact compatibility.

The portable envelope remains 64 MiB. Slither's 512 MiB population checkpoint,
2 GiB/100-record library, 64 MiB/8-record transient candidate store, 64 MiB shadow
workspace and 4 GiB save archive remain aggregate resource policies. They need not
admit every possible collection of individually valid graphs simultaneously.
