# Native learning-curve test graph

Frozen `canonical-training-graph` output from Slither commit `e8673ff`, using the
first recorded Motion/B16 delivery and 0.06-second cadence from the existing
`tactical_corpus_v1` fixture. The graph is Input264, GRU4, Dense2 with native
initialization seed 1. `evidence.json` binds original/compressed bytes and native
source identity. Unit checks read the Rust manifest and packed weights directly
to verify validation-checkpoint noninterference with real canonical Adam/TBPTT.
