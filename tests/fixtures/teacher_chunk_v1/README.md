# Frozen Rust teacher codec reference

`chunk.bin` and `reference.json` were emitted together by Slither's actual
`native/src/bin/teacher_chunk_reference.rs` tool. The reference records the
compiled native source digest, exact canonical sensor bytes and complete
contract/descriptor. This is a synthetic portable-codec fixture, with no claim
that its rows came from a committed world step.

The fixture covers UTF-8 run IDs, full u64 identities, signed-zero Float32 bits,
request versus realized boost, post-step death, all six stratum flags and exact
boundary metadata. Unit tests retain its bytes. The canonical native CI job
also builds the current Rust encoder and compares its decoded sample semantics;
a changed source-specific teacher digest does not rewrite the frozen fixture.

Generate a new current reference into an absent directory:

```sh
cargo run --manifest-path /path/to/slither_neuroevo/native/Cargo.toml --release --bin teacher_chunk_reference -- /path/to/new-reference
PYRL_TEACHER_CHUNK_REFERENCE=/path/to/new-reference python -m pytest -q tests/unit/test_distillation_transport.py
```
