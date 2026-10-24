# Motion/B8 CUDA cell, 8,000 updates

Initialization seed 1 and held-out geometry fold 2 completed all seven paired variants in 3,180.53 seconds. Each variant executed 8,000 compiled CUDA windows with zero eager windows, then passed portable export/parity and native held-out gameplay validation. The corpus has 119,814 actual native observations; the held-out fold has 14,640 rows in 28 sequences.

Full held-out turn MAE falls from 0.04240647 at 6,000 updates to 0.03844980 at 8,000. This is one synthetic tactical cell, with no replicated coverage, maturity, profile/default selection, representative ecology or integrated release acceptance claim. Lower agreement error from an ablation is not sufficient to choose sensor channels.

`summary.json` retains per-variant curves and gameplay receipts. `catalog.json` locates the complete checksummed outputs in the immutable [release asset](https://github.com/Tits-McGruff/PyRL-trainer/releases/tag/qualification-resume-20261025). The asset was downloaded afresh, completely restored/rehash-verified and independently admitted by the maintained study reader. Source uses Git/path/hash/EOL provenance; no source snapshots are included. The corpus metadata records all generated input hashes and the reproduction command. Resource publication contains guarded aggregate counters only; the shared host was not asserted idle.

```sh
python -m pyrl_trainer.evidence_store fetch --catalog docs/performance/canonical_graph/motion-b8-cuda8000-v1/catalog.json --cache .cache/qualification-evidence
```
