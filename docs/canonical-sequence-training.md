# Canonical sequence training API

`pyrl_trainer.sequence_distillation.CanonicalSequenceTrainer` owns Adam and one
bounded BPTT update over independent delivered-observation windows. It is
separate from PPO and does not open a socket or instantiate an ActorClient.
Teacher transport, stream decoding, replay/sampling, job persistence and normal
candidate deployment remain later integration work.

Method-v1 defaults are batch 32, burn-in 32, BPTT 64, Adam learning rate 3e-4,
gradient norm clipping 1.0 and Float32. Explicit configuration allows up to 128
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
