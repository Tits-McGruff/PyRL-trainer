# Canonical sequence training API

`pyrl_trainer.sequence_distillation.CanonicalSequenceTrainer` owns Adam and one
bounded BPTT update over independent delivered-observation windows. It is
separate from PPO and does not open a socket or instantiate an ActorClient.
Teacher transport and strict Rust stream decoding are implemented separately in
`distillation_client` and `distillation_transport`. Whole-chunk replay and stable stream partitioning are implemented in
`distillation_replay`. Bounded window sampling and an initial standalone DistillationJobV1 owner now
connect replay to actual training and atomic restart. Portable candidate snapshots and upload/native admission are now implemented.
Native shadows, saved-library deployment and integrated resource qualification
remain open.

Method-v1 defaults are batch 32, burn-in 32, BPTT 64, Adam learning rate 3e-4,
gradient norm clipping 1.0 and Float32. Explicit configuration allows up to 256
windows and a combined burn-in/training length of at most 256 deliveries.
Learning rate and clip are finite, positive and at most 1. Short full-sequence
BPTT is supported by zero burn-in within the same bounds. Unbounded whole-life
BPTT is rejected.

`SequenceBatch` holds:

- `observations`: `[time,batch,input]` Float32, finite normalized values in [-1,1].
- `teacher_turn`: `[time,batch]` Float32 committed applied turn in [-1,1].
- `teacher_boost_request`: `[time,batch]` bool or Float32 zero/one request labels.
- `event_weight`: `[time,batch]` Float32, positive for active rows and at most 4.
- `active`: `[time,batch]` bool. Inactive rows are suffix padding and never advance
  recurrence or contribute loss. Interior missing rows cannot be hidden as padding.
- `boundaries`: `[time,batch]` int64 `SequenceBoundary` values.
- `initial_state`: `[batch,totalStateSize]` finite Float32 state before the first
  supplied row. All source tensors share one device; they are detached and
  transferred once to the model device.

Time advances once per delivered row, including burn-in. The caller supplies
real deployment deliveries; the trainer never interpolates missing samples or
adds a cadence tick. Ordinary chunk boundaries use `CONTINUE` and do not reset.

`GENERATION` resets state. `LIFE` applies the manifest's Reset/Preserve rule.
`GAP` and `CAPTURE_START` terminate unknown history and start from zero, even
under Preserve. Capture-start has its own reason and is never called a physical
life start. Their counts are reported as unknown-state starts. A gap reset also
cuts backward dependency on the preceding segment.

Burn-in runs without autograd and has no loss contribution. Its final state is
detached before the bounded training segment. The objective is the mean of
`eventWeight*((rawTurn-teacherTurn)^2 + softplus(z)-request*z)`, where
`z=(rawBoost-.35)/.10`, over active training samples. Weighting does not change
unweighted turn MSE/MAE or raw-threshold boost-disagreement metrics.

Each update returns bounded metrics and a detached next state computed with the
pre-update weights. This is normal truncated-training continuation, not a claim
of exact reconstruction after changing weights. Burn-in reconstructs only the
bounded prefix supplied. For overlapping windows, pass state before that prefix;
do not replay an overlap from the previous window's final state. Independent
mid-life windows need explicit capture/gap semantics and bounded reconstruction.

Invalid windows reject before model/optimizer changes. Nonfinite forward/loss
or gradient results reject before the optimizer step. Constant routing graphs
or graphs whose outputs do not depend on their parameters return metrics with
`optimizer_updated=false`. They never fabricate a learned parameter or head.
The higher-level job runner must account for such non-updating graphs in its
bounded termination policy.


`TeacherReplay` owns whole decoded chunks with a default 256 MiB RAM ceiling
and 65,536-object ceiling. Admission charges raw bytes, decoded arrays, per-object
metadata headroom and a fixed 256 KiB bookkeeping reserve. The configurable byte
library ceiling is 2..2048 MiB; the public runner admits 64..2048 MiB. This bounds replay ownership; HTTP transfers, tensors,
optimizer/device memory and durable checkpoint work need separate job admission.
The job consumes segment views before the next replay mutation.

`admit(fetch)` validates the entire settled chunk/gap proposal before replacing
replay or advancing its local cursor. Exact retries do not resurrect evicted data.
Ordinary chunk seals continue recurrence. Missing/evicted prefixes use `GAP`;
only a proved previous-life end followed by the actual next physical-life start
uses `LIFE`. Explicit capture/generation reasons remain distinct. A missing global
sequence for another multiplexed owner does not break a locally proved contiguous
sample-ordinal stream. Numeric step gaps alone are not interpolated deliveries.

Partition revision 1 hashes `slither.distillation-partition.v1` plus a NUL byte
and little-endian `(u32 runSeed, u32 baselineSlot, u64 generation)` with SHA256.
The first eight digest bytes interpreted as a little-endian unsigned integer
select validation when modulo five is zero. Every physical life/fragment in that
stream has the same split. Final held-out run seeds remain a job-level gate.
Rare strata receive one 4x loss multiplier regardless of overlap; cruising receives
1x. `raw_counts()` preserves unweighted row/stratum/request/realized/death counts.
This buffer and partition API are primitives; they do not claim durable optimizer,
sampler/RNG restart, qualified resource admission or a complete training job.


`ReplayWindowSampler` samples independent windows uniformly by delivered training
row within one partition. Its own PCG64 state is checkpointed with a digest of
the exact retained chunks, graph/deployment contract and sequence configuration.
Each window has up to 32 burn-in plus 64 training deliveries by default. It uses
zero initial state and an explicit unknown-prefix boundary when sampling within
a life. Interior proved life boundaries honor Reset/Preserve; gaps reset. Short
training suffixes use padding that neither advances recurrence nor contributes
loss. The sampler does not create observations between delivered samples.
The stream index has a separate 64 MiB admission including replacement staging;
CPU batch tensors have a separate 128 MiB default ceiling. These are admission
bounds, with final integrated workload qualification still required.

`DistillationJob` owns the canonical model/Adam, replay, sampler, private Python
and Torch CPU/CUDA generators, metrics and the last detached window continuation.
It normalizes the default two Torch CPU threads against available CPUs. CUDA0
is preferred explicitly; unavailable CUDA fails rather than selecting CPU.
The initial device budget is 8 GiB, with cold graph/window admission and observed
CUDA allocation peaks. Device/optimizer failures pause the job and prohibit
publishing possibly partial state as an exact checkpoint. Normal short-history
waiting leaves the job and sampler ready for more captured rows. No loader workers
or actor/controller objects are created.

Standalone usage with an independently generated matching native graph manifest:

```sh
python -m pyrl_trainer.distillation_run --manifest graph/manifest.json --weights graph/weights.f32 --origin http://localhost:5174 --capture-epoch CAPTURE_EPOCH_HEX --job-directory distillation-jobs --device cuda:0 --updates 1000
python -m pyrl_trainer.distillation_run --resume CHECKPOINT_PATH --offline --job-directory distillation-jobs --updates 1000
```

New-job inputs must match the captured structural sensor bytes/digest and cadence.
Use `--device cpu` explicitly for CPU training. The runner has finite update and
wall limits, validates every 50 successful updates, checkpoints every 60 wall
seconds or 1,000 updates, and saves start/final recovery boundaries. Resume rejects
explicit changed source, device, seed, CPU threads, device/replay budgets, learning
rate or sequence settings. New CLI jobs use training seed 1 and expose the bounded
controls in the README; library defaults and resumed settings keep their meanings. Offline resume never
fetches data or joins a controller.

A `.distill-job` checkpoint is a bounded, uncompressed, self-contained ZIP,
separate from PPO schema2 and from any deployed brain artifact. It contains the
canonical graph/Float32 weights, local Adam/private RNG payloads, detached
recurrence, window identities, source/environment/method records, HTTP admission
settings and cursor, replay/sampler decisions, and every required immutable raw
teacher chunk with its binary digest index. ZIP central-directory allocation is
admitted before metadata parsing. Payload digests, exact schemas, complete replay
bytes, canonical dimensions and finite optimizer/state values are checked before
a fresh owner is returned. Torch optimizer/RNG reads use `weights_only=True`;
those local trainer payloads are never a deployable candidate.

Publication flushes the complete file, creates an exclusive final hard link,
flushes publication where supported and retains the newest three valid containers.
The disk reserve remains 1 GiB. A finite on-disk ceiling is derived from the recorded
replay quota, bounded fixed roles, at most four portable candidates and ZIP overhead;
it is not a memory allocation. Raw chunk bytes stream individually, and fixed-role,
8 MiB central-directory and 65,536-chunk limits remain independently enforced.
Readers validate ZIP64 locators for both count and size cases, with consistent
classic/wide counts and offsets; classic ZIP permits exactly 65,535 entries without
a locator when its sizes fit. Replay raw-byte admission precedes chunk reads, then
full decoded ownership is charged during restore. Missing required
replay data, trainer-source mismatch or numerical-environment mismatch rejects
exact restart; no observations are refetched or silently reconstructed. CPU/CUDA
regressions compare the very next weight bytes, Adam tensors, private RNG draws,
sampled spans and detached recurrence after restart. Random windows intentionally
reconstruct only bounded prefixes, not unlimited recurrent history.

Every 250th successful optimizer update now freezes an immutable `.slither-brain`
candidate. The bounded queue retains four pending exports, records discarded
pending candidates, and checkpoints their complete bytes. Later updates cannot
mutate them. Candidates include canonical graph/Float32 bytes, teacher/run/config
identity, method/seed, processed sample counts, trained generation extent,
validation and numerical provenance. The export contains no Python model/pickle.

The online runner checkpoints the pending handoff before posting it to the same
configured server's dedicated candidate namespace. Rust independently recompiles
packing, equations, sensors and deployment before returning a content-ID receipt.
Only matching complete receipts release pending ownership; lost-response retries
retain/send identical bytes. Offline jobs retain pending exports in their job
checkpoint. Candidate upload neither activates a controlling policy nor saves it
into the brain library. Those actions and native shadows remain explicit later
integration. Shared PPO v4 sensor contracts now preserve native structural bytes,
profile, normalization revision, digest and dimensions; incomplete v4 checkpoints
reject while legacy v3 metadata remains readable.
