# Authoritative PPO reward contract v1

The repair plan is maintained in Slither at
`docs/todo/ppo-reward-action-policy-repair-plan.md`.

The first implementation stage adds the shared structural contract, native
accounting and lifecycle delivery, binary codecs, and `pyrl_trainer.reward_contract`.
The current PPO entry point and both actor implementations require this contract,
consume its deltas, and close old-life transitions explicitly. The local server
routes negotiated JSON/binary observations and lifecycle closures.

The contracts dependency and interoperability checkout both pin the Slither
implementation commit. That commit must be published before a fresh remote
install or CI checkout can resolve the pin. Local tests consume the matching
contracts checkout, including the native-verified packaged fixture.

## Ownership and versions

Rust produces committed event quantities. PyRL validates ordered intervals and
applies trainer coefficients. Sensor labels never reconstruct food, growth,
points, kills, live time, or death.

Reward version, sensor version, Protocol 2, packet-envelope version, reward weight
configuration, policy parameterization, and checkpoint schema are separate
identities. `require_reward_version()` requires an explicit `rewardVersion: 1`.

The explicit reward-capable binary envelope is SNS2 revision 2. It preserves the
44-byte revision-1 header and little-endian Float32 sensor body, then appends a
little-endian Uint32 reward byte length and at most 4096 bytes of UTF-8 JSON.
The reward's public snake ID and ending step must match the packet header.
Packet revision and size bounds are generated shared constants. Revision-1
packets retain their existing layout. The hello explicitly requests
`rewardVersion: 1`, `binarySensors: true`, and `binarySensorsVersion: 2`.
The hub deliberately retains the original payload for peers that do not opt in.

The Rust source `native/src/contracts.rs` generates the Python TypedDicts. Runtime
validation checks their field inventories and applies trainer-specific semantic
checks. Missing fields, unsupported versions, nonfinite values, nonintegral
counts and inconsistent point accounting raise `RewardContractError`.

## Intervals and lifecycle

An envelope contains explicit run, world, assignment lease, physical snake and
visible snake identities. Observation sequences and step boundaries are exact
16-digit lowercase hexadecimal strings. The predecessor and interval start must
match the last accepted envelope for the same assigned life.

- `initial` establishes a zero-reward baseline with no predecessor.
- `live` carries accumulated ordinary deltas since the admitted predecessor.
- `terminal` closes a real death with the complete final interval.
- `truncated` closes a non-death cutoff, including generation replacement.

`RewardStream.accept()` never automatically repairs a gap. A mismatch leaves its
admission state unchanged. Reconnect handling must discard unresolved experience
before explicitly using `rebase()` to establish a new live baseline. That method
returns no trainable reward. It cannot rebase from a closed-life envelope.

Consumers must complete old-life reward before accepting the next assignment.
Real death uses zero bootstrap. A generation cutoff requires the final old-life
observation; the replacement snake's observation and the pending transition's
starting value are not valid substitutes.

Every envelope also has required `state.startingLength`, `state.bodyLength`, and
`state.retainedGrowth` measurements in exact integer body segments. Starting
length is a positive constant for the physical life; body length is positive and
frozen before terminal cleanup; retained growth is
`max(bodyLength - startingLength, 0)`. Initial deltas remain zero even if the
initial snapshot already contains growth. Stream admission verifies that the
two signed length deltas match successive snapshots and that starting length
does not change when rebasing a known life after a gap. These measurements do
not add reward components.

Native generation assignments now retain the old-life closure JSON and a final
sensor vector for live cutoffs through the existing receipt-driven handoff.
Terminal closures have no bootstrap vector. God Mode death before the cutoff and
collision on the cutoff step both remain terminal. These fields reach the coarse
native bridge and Node routes them before the successor assignment. Actors
validate and account for the closure before resetting pending life state.

Observation time follows physical geometry: the ordinary pre-movement boundary
is `sourceCompletedStep * fixedDt`, while a post-step closure is
`(sourceCompletedStep + 1) * fixedDt`. Prefix live-time and survival-point credits
are separately reported authoritative accounting facts. They must not shift the
physical observation clock forward by one tick. A final interval can therefore
contain physical elapsed time with no additional live-time credit.
Conversely, a prefix interval may credit live time before physics advances.
Neither producer nor consumer requires `aliveSecondsDelta <= elapsedSeconds`
within each individual packet.

## Initial objective

Default weights in `RewardCoefficients` are:

```text
survival = 0.001 * aliveSecondsDelta
food     = 0.10  * foodValueDelta
growth   = 0.05  * retainedGrowthDelta
points   = 0.01  * (pointsDelta - survivalPointsDelta)
kills    = 1.00  * killsDelta
death    = -0.50 for terminal death; otherwise zero
```

Food value is distinct from pellet count. Retained growth is
`max(finalizedBodyLength - startingLength, 0)` and its interval delta is signed.
Boost can reduce both retained growth and net points. Terminal body deletion does
not erase the final growth measurement. Survival points are removed from the
weighted points term so time is the only survival payment.

The raw economic identity is:

```text
pointsDelta = survivalPointsDelta + foodPointsDelta + killPointsDelta
            + otherPointsDelta - boostPointsSpentDelta
```

Food, growth, economic progress and kills intentionally overlap. One food-value
unit producing 20 points and one growth segment contributes `0.35` task reward.
Changing these experimental weights must be recorded in trainer configuration
and checkpoint compatibility metadata, separately from raw reward version 1.

Signed retained growth telescopes over an undiscounted closed loss/regrowth
cycle. It does not prove that discounted or complete food-economic cycles are
unprofitable. Weighted components, scalar reward, pending reward accumulations,
and episode returns must all remain finite.

## Received and admitted diagnostics

Actors retain weighted components and simulated duration with each pending
decision. Components enter `reward[...]` only after a completed rollout is
successfully queued. Dropping a pending decision on disconnect does not count
its reward as admitted to PPO.

Raw `received[...]` counters include accepted noninitial intervals while an
actor waits at the collection barrier. They report interval and cumulative food,
growth, economic and live-time exposure separately from admitted transitions.
Component rates use admitted credited live seconds as their denominator, and
decision duration uses admitted physical seconds. Evaluator events are excluded
from training diagnostics.

The diagnostic warning for absent task events requires both 10,000 intervals
and 60 aggregate credited live seconds since the last event or warning reset.
It uses raw task quantities, independently of configured weights, and survives
routine log drains. It never changes training settings.

## Frozen evaluator

Evaluation freezes one published model for an entire episode. A dedicated
`torch.Generator` samples turn and boost, seeded by `evaluation_seed` plus the
assigned episode index. Constructing the frozen model and creating an empty
optimizer state for its checkpoint also preserve the training RNG state.

`evaluation_max_seconds` defaults to 240 simulated seconds. Scoring ends at the
first accepted ordinary observation at or after that physical duration. The
complete interval is included; no fractional food event or reward is invented.
Results report the actual duration and any overshoot. This local cutoff is
`truncated` with result reason `evaluationTimeLimit`, without a death penalty.
A real terminal closure received first remains terminal.

After a duration cutoff, the evaluator closes its connection and joins with a
fresh name and no resume token. This avoids explicit-token and legacy name-based
reclaim of the capped life. Ordinary server disconnect-grace behavior still
applies to the old lease; the trainer does not manufacture a server-side death.
Record this environment behavior when comparing policies, especially at high
simulation speeds.

Each `[eval] result=...` JSON record includes exact policy-parameter SHA-256,
policy version, sampling seed, run/world/life identity, initial and cutoff steps,
cutoff observation sequence, physical and credited time, all raw reward deltas,
all weighted components, repaired return, ending reason, and integrity status.
Raw length and retained-growth deltas are exact changes since the admitted
baseline. Results separately report the life starting length, initially observed
length/growth, absolute final body length, and final retained growth directly
from the authoritative state snapshots. No normalized sensor values are used.

Transport gaps and state replacement record an incomplete prefix and exclude it
from promotion. Valid zero-step deaths and negative scores can establish the
first incumbent. Best-policy artifacts and `best_evaluator.json` retain the
winning result. Evaluation configuration has its own version: changing the
duration, base seed, sampling, score, or cutoff definition resets the comparable
leaderboard while leaving model resume compatibility separate.

## Verification

`tests/unit/test_reward_contract.py` checks parser rejection, stream chronology,
rebase behavior and weighted reward. It consumes
`slither_contracts/external-reward-v1-fixture.json`, the same compact fixture
checked by Slither's real native food/collision/kill physics test.

For the real Node-to-Python codec test, set `PYRL_SLITHER_ROOT` to the matching
Slither checkout and run:

```sh
python -m pytest tests/integration/test_reward_binary_interop.py -q
```

That test invokes the actual TypeScript encoder with the native-verified fixture,
decodes its stdout bytes in Python, and checks identity, compact sensors and
terminal objective value. A configured checkout or encoder failure fails the
test; ordinary test discovery skips it only when no checkout was requested.

With that Slither checkout's Node dependencies installed and its production
addon built, the same environment enables the real frozen-evaluator socket test:

```sh
python -m pytest tests/e2e/test_native_evaluation.py -q
```

It starts a task-owned seeded native server and database, consumes actual binary
observations in the Python actor, reaches two duration cutoffs, and verifies
distinct native life/lease identities after reconnect. A configured checkout
with missing dependencies or an unavailable/stale addon fails startup explicitly.
The interoperability CI job builds the production server and runs both checks.

The same module also runs a four-actor PPO smoke scenario. Existing UI controls
place a snake on real food, kill that physical life through ordinary game death
semantics, and finish two generations. Rust supplies every reward event and
cutoff observation. Two complete actor barriers feed the actual PPO optimizer;
the checks cover nonzero survival/food/growth, boost expenditure, one terminal
penalty, old-life truncation values, coherent policy versions, near-zero initial
KL, finite updates, and exact evaluator/recovery checkpoint roundtrips. A
separate frozen evaluator supplies the saved best candidate. Placements arrange
events only in this test; no scripted reward or food steering enters training.
The smoke keeps horizon 256 and closes partial contributions at explicit lifecycle
boundaries rather than waiting for untrained gameplay to discover useful events.

Slither's native real-socket regression also exercises food consumption, retained
growth, terminal death and manual-generation truncation through the production
addon and negotiated binary transport. The Motion actor tests exercise the real
PPO policy, final-state bootstrap and four-actor reward admission. Steering now
uses a trainable, checkpointed `turn_log_std`; trainer reward-config version 2 and
policy parameterization version 2 reject historical fixed-variance objectives.

These checks establish implementation behavior. They do not establish useful
learning or gameplay quality; the plan's separate learning qualification remains
required. Publishing the paired dependency revision and qualifying command/epoch
application boundaries remain part of the rollout work.
