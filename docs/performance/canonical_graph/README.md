# Canonical optimizer roundtrip evidence

The two JSON reports retain the mandatory Stage F gate on CPU and the local
RTX 4080 with Torch `2.14.0+cu126`. Both use Float32, fixed seed 42, TF32 disabled,
Adam at 3e-4 and gradient clipping at 1.0. Each graph receives an actual optimizer
update with the method-v1 loss. Both native scalar and SSE2 backends independently
compile its source graph and validate the trained canonical weight count/finite
values before inference. Returned Float32 weight bytes are compared exactly,
reimported into PyTorch, and used for output/state comparison over 256 deliveries.

The reports include native source/target identity, graph and sensor digests,
trained weight digests, numerical error growth and Torch/CUDA/device versions.
Single-step tolerances are CPU 1e-5/1e-4 and CUDA 1e-4/1e-4 (atol/rtol).
The 256-step tolerance is 1e-3/1e-3. No tolerance was widened.
Synthetic labels exercise optimizer/export semantics; these reports do not
measure gameplay imitation quality, corpus coverage or sensor workload cost.

PyRL CI runs the live CPU roundtrip against the pinned native contract and
retains its report as an artifact. CUDA requires an available GPU and fails
explicitly when unavailable. Reproduce with a new report path:

```powershell
.venv/Scripts/python.exe -m pyrl_trainer.canonical_parity --runner C:/path/to/canonical-training-graph.exe --fixtures tests/fixtures/canonical_graph --device cuda --output-json new-cuda-report.json
```


The `*-sequence-roundtrip-v1.json` reports exercise the actual
`CanonicalSequenceTrainer`: 32 windows, 32 burn-in deliveries and 64 BPTT
deliveries, with 1,024 burn-in and 2,048 training samples per update.
The earlier reports retain the initial graph/optimizer primitive evidence.
CI's live gate now uses the sequence trainer too.
