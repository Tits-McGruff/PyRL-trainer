# Motion/B8 Full, 32,000-update screen

The Full-only seed-1/fold-2 CUDA run completes 32,000 compiled windows with zero eager windows in 1,279.73 seconds. All recorded weight hashes through 8,000 updates exactly match the earlier seven-variant cell. The bounded loss trace has 401 samples at an 80-update interval; all eleven requested held-out checkpoints and the initial model are retained.

| Measured outcome in this cell | 8,000 updates | 32,000 updates |
|---|---:|---:|
| Held-out turn MAE | 0.03844980 | 0.02108827 |
| Native surviving trials | 13 | 12 |
| Native total survival | 39.08333 s | 37.56667 s |
| Native total food earned | 20.89149 | 14.18816 |
| Native kills | 4 | 4 |

The native evaluation uses fourteen short held-out tactical scenarios. These results do not establish a population ecology or a general gameplay regression. MAE also falls from 0.02502261 at 24,000 to 0.02108827 at 32,000, so there is no demonstrated plateau. A larger replicated grid is not justified by agreement alone. Maturity, channel/profile/default selection and integrated acceptance remain open.

`summary.json` records the checkpoint curve and the native receipts. `catalog.json` identifies complete outputs, corpus metadata, aggregate-only counters and Git/path/hash/EOL provenance in the immutable [release asset](https://github.com/Tits-McGruff/PyRL-trainer/releases/tag/qualification-long-curve-20261025). The asset was downloaded afresh, completely restored/rehash-verified and admitted by the maintained study reader. It contains no source snapshots or host process inventories. Compiler and other shared-host activity are covered by aggregate counters; this is not an idle-host timing claim.

Download the linked release asset directly and verify its recorded SHA-256 when inspecting historical output.
