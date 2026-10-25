# Performance results

Keep compact summaries and deliberate regression fixtures in Git. Routine run
outputs use CI artifacts; significant published results may use ordinary release
assets with a recorded SHA-256. Experiments identify clean Git commits or explicit
WIP commits. Local dirty diagnostics require opt-in and visibly record dirty state;
they do not copy source files into outputs.

The [original evidence release](https://github.com/Tits-McGruff/PyRL-trainer/releases/tag/qualification-evidence-v1-20261024)
remains unchanged. Its archive is 227,246 bytes with SHA-256
`7ab54e5e5e863ea52b8b73086d7211b5b1c905dace408d8b27e20f39aa513f82`.
The completed migration's catalog and exceptional source patch remain inspectable
at tag `qualification-resume-20261025`. They have no ongoing reconstruction or CI
requirements. More recent result folders retain compact asset receipts/summaries.

Correctness uses `tests/fixtures/cuda_window_replay_v1` (trained brain, observations,
outputs, states and receipt), `compensated_dot_counterexample_v1`, and the bounded
semantic/protocol fixtures. Ordinary tests require no historical asset downloads.
Restart checkpoints separately retain exact runtime trainer/module hashes because
optimizer/RNG/replay restoration depends on the loaded implementation.
