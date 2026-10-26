# Native tactical profile fixtures

Exact synthetic Rust-produced Basic/B8 and Motion/B32 corpora, including
committed observations/actions/steps and complete canonical graph initializations.
Native commit 147cefc41ef56c116fd4a214615901c2255a5e42 generated these files.
The native source identity is recorded per case in receipt.json; compressed and
decoded hashes bind every file. No owner game, database, process inventory or
audio data is included.

The native qualifier used two seeds, 24 physical steps, one worker and 0.06-second
teacher cadence. Each graph was initialized natively with seed 1 using the
complete declared profile. Tests verify read-only row ownership, real optimizer
updates, held-out teacher groups, port labels and structural rejection.
Motion/B16 compatibility remains covered by the separate legacy fixture.

The two-update native deployment smoke is reproducible in the canonical-roundtrip
CI job. These small fixtures do not qualify model maturity, profile choice or
runtime performance.
