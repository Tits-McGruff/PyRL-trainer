# Qualification evidence storage

Git retains compact run references, exact provenance, checksums, selected readable results and deliberate correctness fixtures. Full corpora, model collections, profiles, resource/timing sequences and complete reports belong in artifact storage. Normally tracked source is identified by repository, exact commit, path inventory and content hash, never copied into compressed evidence objects.

The v1 migration preserves every original performance file, including original source snapshots, in a deduplicated release asset. Those old source copies are historical preservation only. New runs do not produce them. The artifact-only release tag points at the pre-feature base, so the release tag does not keep the old performance blobs reachable in Git. No existing branch history was rewritten.

- PyRL release: https://github.com/Tits-McGruff/PyRL-trainer/releases/tag/qualification-evidence-v1-20261024
- Slither release: https://github.com/Tits-McGruff/slither_neuroevo/releases/tag/qualification-evidence-v1-20261024

Each repository's `catalog.json` records the original repository/commit, every original logical path, exact byte length and SHA-256, and release URL/archive hash/size. Archive members are deduplicated `objects/<sha256>` regular files. Fetch validates the archive size/hash, the complete bounded inventory, member confinement/type/size, all object hashes and every materialized file. The migration downloaded the published assets again and restored/verified all files before deleting any payload at the branch tips. Required audits fail with a fetch command when the cache is absent; they do not skip or infer success from a missing artifact.

From PyRL, prepare its historical cache with:

```sh
python -m pyrl_trainer.evidence_store fetch --catalog docs/performance/catalog.json --cache .cache/qualification-evidence
python -m pytest -q tests/evidence
```

From Slither, use the maintained PyRL checkout (not an archived module):

```sh
python ../PyRL-trainer/pyrl_trainer/evidence_store.py fetch --catalog docs/performance/catalog.json --cache .cache/qualification-evidence
node node_modules/tsx/dist/cli.mjs scripts/run-tests.ts evidence-required --reporter=dot
python scripts/qualification/run-python-tests.py --kind evidence
```

Set `PYRL_SOURCE_REPOSITORY` to that checkout for replicated matrix verification, and `SLITHER_SOURCE_REPOSITORY` when it is not the sibling checkout. Historical CI checks out complete Git history at pinned revisions for source reconstruction. `QUALIFICATION_EVIDENCE_ROOT` can select an already verified materialized root. `fetch --archive FILE` supports an explicitly downloaded release asset; its checksum is still mandatory. Use `evidence_store.py verify --catalog ... --root ...` to rehash a cache independently. Cache files are immutable inputs; never edit them.

`run.json` files identify their old logical prefix/manifest and preserve recorded identity metadata. The catalog locates the full original configuration, commands, seeds, dependency/toolchain/environment information and outputs. Original run metadata is retained as recorded; the migration does not invent missing historical measurements. Existing README performance results remain readable. Paths in older reports now refer to the restored cache, not checked-in payloads.

`source-provenance.json` maps historical source versions to ordinary Git paths and exact commits. Canonical LF hashes establish Git content; raw SHA-256 plus LF/CRLF/mixed recipes restore measured bytes. Unique experimental versions are uncompressed textual patches in `benchmarks/patches`, against the recorded ordinary base commit. Every source version was reconstructed and raw-hash checked during migration. `benchmarks/historical-patches.json` additionally preserves original experimental patches and their recorded base metadata/preimage blob identities. Patches are historical reconstruction inputs, not alternate production implementations. No archived Python reader is executed by replicated verification: the maintained reader reconstructs source identities from this provenance and reproduces all three matrices, including prefix equality.

New PyRL studies record the same runtime trainer hash as before, plus source provenance, dependency hashes, Python identity and reproduction command; their existing numerical environment/configuration, native source/runner hashes, settings, split seeds, schedules and artifact hashes remain mandatory. A dirty experiment records only an explicit patch against its exact base. Compatibility/restart hashes still bind raw loaded source bytes. Retaining a hash does not require retaining another copy of the module.

Ordinary correctness tests use small named fixtures. PyRL's `tests/fixtures/cuda_window_replay_v1` holds the actual trained brain, all 256 observations, Rust outputs/states and receipt. PyRL's `tests/fixtures/compensated_dot_counterexample_v1` additionally extracts one real 504-term operand row from the old failing LSTM export and checks the current CPU dot and its gradients against independent expectations. Slither's `native/fixtures/learned-cohorts-v1` retains only Full/Core-only brains for single-policy and three-seed family contracts; the native cohort unit test uses two of those same files. `scripts/qualification/fixtures` holds selected actual public resource-counter rows and four minimized first-step numerical failures. Full recurrent numerical exports and all historical performance assertions remain required external qualification. Existing semantic/protocol/interoperability fixtures, the five native source-identity inputs and pre-feature `docs/todo/evidence` remain intact.

`evidence-fixtures.json` explicitly approves fixture paths, sizes, hashes and purposes. The maintained `pyrl_trainer.evidence_policy` CI tool rejects compressed source, generated performance payloads, unknown/changed fixtures and excessive file/total sizes. It also inspects unignored candidate files locally. Updating approval records is a deliberate reviewed change, not a promotion side effect. Runtime output folders and the external cache must remain ignored. Do not expand the legacy `docs/todo/evidence` exemption for new experiments.

For a new durable promotion, create a completed run directory inside the repository with `provenance.json`. It must contain `repositories` (repository URLs and exact commit SHAs), `sources` (path/hash/EOL inventories), `build`, `toolchain`, `dependencies`, `commands`, `settings`, `seeds` and `environment`. Then export to a file outside that run directory:

```sh
python PATH_TO_PYRL/pyrl_trainer/evidence_store.py export --root . --run IGNORED_RUN_DIR --archive OUTSIDE_RUN.tar.gz --catalog NEW_CATALOG.json --url https://github.com/OWNER/REPO/releases/download/UNIQUE_TAG/OUTSIDE_RUN.tar.gz
```

The exporter refuses Python/Rust/TypeScript module copies, including compressed copies. Experimental differences belong in explicit base-commit patches. `--legacy-migration` exists only to preserve the old tracked bundles and is never the new-run promotion path.

Routine CI runs continue using `actions/upload-artifact` with failure on missing required outputs. Durable promotion is explicit: export an immutable run to a release asset (or a checksum-addressed object store), record its manifest and checksum, publish, download it afresh, verify complete restoration, and only then add the compact reference to Git. Use a unique release/version/object key; never overwrite an existing asset to reuse a manifest. GitHub Actions retention is finite, so long-lived claims need durable promotion. Release assets remain deletable by maintainers; keep a separately verified backup under the same content hash when stronger retention is required. New exports should select a completed run's output directory and provenance, without copies of tracked source or unrelated host inventories. Publish only resource counters passed through Slither's guarded exporter.

No deletion commit reclaims old Git objects. A later feature-branch rewrite requires separate owner approval, an inventory of refs/clones and source-commit reachability, and verification that all historical claims can still reconstruct their exact sources. The preserved historical release assets provide an independent recovery copy, but merely recording a SHA does not preserve a commit that every Git host has discarded. Do not prune or force-push as part of this migration.
