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


## Native tactical corpus input

`pyrl_trainer.tactical_corpus.load_tactical_corpus(directory)` loads the native
qualifier's `native-tactical-corpus-v1` completion manifest and three exact
little-endian files: observations (N by 264 Float32), actions (N by two Float32
turn/boost-request values), and delivered step IDs (N UInt32). It verifies
canonical sensor bytes/digest, cadence, bounded sizes/hashes, finite normalized
values and complete non-overlapping scenario/seed/mode sequence ownership.
The binary plus metadata producer ceiling is 256 MiB; validation scratch is
bounded in blocks. Returned arrays are read-only and contain delivered rows,
without interpolation of held decisions.

`corpus.split_by_seed({2})` selects entire held-out native trajectories; it
rejects empty, unknown or all-seed validation partitions. This prevents window
leakage when comparing feature ablations on identical data. These are synthetic
physical scenarios with baseline-v2 Motion teachers, not a representative
trained-population distribution or a student-quality pass. Each sequence starts
at a fresh native policy state and ends at death or the bounded horizon.

The retained small fixture is generated by the actual Rust CLI. CI also builds
the pinned producer, exports a fresh two-seed corpus, verifies it through this
reader and retains its exact bytes beside canonical parity evidence. No Python
world, teacher policy or sensor implementation is introduced.


## Paired tactical feature ablation

Run `python -m pyrl_trainer.tactical_ablation_gate` with explicit `--corpus`,
`--canonical-runner`, `--tactical-runner` and a new `--directory`. The default
study uses CUDA0, 250 updates, 32 sequences, 64-step BPTT and up to 32 real
preceding burn-in deliveries. `--device cpu` is explicit; there is no silent
CUDA fallback. `--validation-seed` excludes complete native trajectories from
training and selects the same held-out physical seed for native student trials.

Seven declared graph variants cover Full, removal of bearing residual, heading
pair, velocity pair, urgency, density, and a scalar/six-channel core. Rust owns
all manifests, parameter packing and initialization. Split/Concat selects inputs
before the first trainable layer while retaining the full Motion/B16 input
contract. This measures information/parameter effects and does not imply a
released sensor subset or reduced sensing work.

The schedule cycles temporal blocks with one private seeded generator. Short
lives remain eligible. Initial blocks use exact zero physical-life state and
available history; later windows use bounded burn-in from a labelled unknown
prefix. Only suffixes are padded. All variants must reconcile identical schedule
hashes and training/burn-in row counts. Validation carries recurrent state over
entire held-out lives, independently of training windows.

Before/after artifacts undergo normal native admission and drive actual native
control/physics against constant opponents. Whole-graph parity uses existing
short/long canonical tolerances. Reports retain source snapshots, immutable
native binary fingerprints, initialized/trained artifacts, per-scenario agreement
and actual gameplay outcomes. Negative quality remains measured data; a completed
study never sets `releaseQualified` true. Representative populations, native
performance, sensor freeze and integrated soak are separate acceptance gates.


### Held-out agreement checkpoints

Pass repeated `--validation-update` flags to `tactical_ablation_gate` to measure
learning rather than only initialized/final quality. For example, an explicit
2,000-update study can request updates 250, 500, 1,000 and 1,500; the final 2,000
point is included automatically. At most sixteen distinct positive checkpoints
are admitted, plus the initial update-zero point. Omit the flags for the existing
final-only study. Every variant uses the same checkpoints and paired training
window schedule.

Each checkpoint evaluates the same complete held-out native lives with separate
student recurrent state. Its compact receipt records packed-weight identity,
row/sequence exposure, aggregate error/loss, both teacher-policy sufficient
statistics and descriptive evaluation time. Full scenario/policy strata remain in
the initialized/final records. Evaluation must preserve packed parameters and
CPU/CUDA RNG state. Regression checks use a frozen Rust-compiled recurrent graph
and real Adam/TBPTT to require identical final packed weights, optimizer tensors,
window-schedule digest and update trace with checkpoints enabled or disabled.

`ablation_matrix` rejects partial or reordered curves, changed exposure or endpoint
weights, and unreconciled policy statistics. Enabled matrices include a
`heldOutLearningCurve` per variant: cellwise distributions and paired differences
from Full at each measured update. Older final-only studies stay readable.
Intermediate agreement curves do not include native gameplay evaluations and do
not automatically declare convergence, model maturity, a winning profile or a
release pass. Those decisions require the retained curves and remaining gameplay,
representative-population, capacity and real-time evidence.


`CanonicalSequenceTrainer(..., cuda_graphs=True)` is an opt-in qualification
path for whole-window `torch.compile(backend="cudagraphs")` execution. The tactical
ablation CLI exposes it as `--cuda-graphs --device cuda:0`. Durable CUDA jobs select this bounded path automatically, while CPU jobs stay
eager. Their numerical restart environment records `sequenceExecution` as
`bounded-cuda-window-v1` or `eager-v1`, and their allocator policy includes private
graph pools within the existing device budget. This option preserves custom canonical gate
equations, explicit state, delivered resets, padding and the burn-in detach;
loss, finite checks, clipping and Adam remain outside the compiled window.
Returned continuations own their storage across subsequent graph replays.

Each trainer compiles at most one tensor-shape/stride/device signature. Shorter
burn-in prefixes are right-aligned with inactive padding before the fixed detach
boundary; shorter training suffixes use inactive padding after their last real
row. Padding never advances recurrence or contributes loss. The initial sequence
configuration bounds this storage. Other signatures execute eagerly on the same
CUDA device. Compilation admits at
most 96 deliveries, 32 lanes, eight nodes, 65,536 parameters, 256 recurrent state
values and 32,768 conservative unrolled operator units. Larger admitted training
windows stay eager. Each owner has an isolated compiler code cache; the trainer
never resets process-wide compiler caches. Unsupported compilation fails rather
than concealing an error. CPU models reject an explicit CUDA-graph request.

Qualification reports add `config.cudaGraphs` and training `windowExecution`
dispatch counts when requested. Compiled dispatch counts alone do not prove that
PyTorch captured every backend call. Compilation and warmup are included in study
wall time; a warmed microbenchmark is not a throughput or maturity gate. The
qualification process caps the CUDA caching allocator at its declared device
budget, including private graph pools. Compiler caches are derived state and are
not serialized. Compiler state remains derived: durable restart restores canonical parameters,
Adam, sampler and private RNG, then compiles the next eligible CUDA window afresh.
The exact next-update job regression covers this path when CUDA is available;
full GPU process-restart and live-contention measurements are still required
for integration qualification.

The retained [RTX 4080 qualification](performance/canonical_graph/cuda-window-windows-v1/evidence.json)
compares 32 paired native-corpus windows at batch 32 / burn-in 32 / BPTT 64.
Loss, gradients, parameters, recurrent continuation and both Adam moment tensors
matched eager CUDA exactly in this run. The final 16 updates averaged 26.86 ms
compiled versus 278.42 ms eager; total measured update time, including compilation,
was 33.26 seconds compiled versus 9.24 seconds eager. A separate 100-update run
used the compiled path for every window and passed 256-delivery native export
parity (output/state maximum errors 2.84e-7 / 8.05e-7). The capsule retains exact
measured sources, experiment, receipts and links to the already-retained input
corpus/artifact. These measurements qualify this bounded path; they do not
establish maturity, native real-time performance or a six-hour integrated soak.


The live qualification helper also has an explicit six-hour sustain mode:
`--phase sustain --soak-hours 6 --seconds 21600 --max-wall-seconds 21900` with the
existing matching capture/checkpoint arguments. Other phases and ordinary windows
keep their previous bounds. Every minute must advance both optimizer and live
intake counters, with at most 120 seconds between progress receipts. The helper
records bounded progress and saves ordinary durable snapshots every ten minutes,
using the existing three-file rotation. Progress files are atomic observations;
the `.distill-job` files remain the exact restart authority. The Slither host
supplies the persistent workspace, native clock and generation-barrier evidence.
This mode provides the harness for a long measurement; it does not itself claim
that a six-hour or full-capacity qualification has passed.


CLI and live qualification reports include `windowExecution` dispatch counts for
the current process. Those counters restart at zero after loading a job; they are
not serialized compiler state or lifetime optimizer counts. The separate-process
restart regression saves a warm owner, starts a fresh interpreter and compares
its next update's parameters, Adam state, recurrence, sampled windows and private
RNG witnesses. CUDA cases run only on a real available device.
