# Rollout helper modularization

This implements the bounded PyRL packet in the owner's evidence-verified modularization plan dated 7 November 2026. The starting checkout was clean at `b1c9d8b67f0905571b8f4372ed0a042d744a6575`.

| Historical public import | Implementation source | Change |
|---|---|---|
| `pyrl_trainer.learner.gae` | `pyrl_trainer/rollout_processing.py` | Re-exported function; body and signature unchanged |
| `pyrl_trainer.learner.split_rollout_by_policy_version` | `pyrl_trainer/rollout_processing.py` | Re-exported function; body and signature unchanged |
| `pyrl_trainer.learner.collate_rollouts` | `pyrl_trainer/rollout_processing.py` | Re-exported function; body and signature unchanged |
| `pyrl_trainer.learner.format_reward_exposure` | `pyrl_trainer/reward_reporting.py` | Re-exported function; body and signature unchanged |

`learner.py` decreases from 914 to 747 lines. The new modules contain the existing pure algorithms and formatting, not additional state owners. Function ASTs match before/after. Historical imports resolve to the same function objects as the new implementation imports. `Transition`, `ActorClient`, `SharedState`, the optimizer, learner loop, reward attribution and synchronized policy fences retain their owners. Float32 operations, advantage normalization, bootstrap overrides, contiguous policy segmentation and return layouts are unchanged. Function implementation `__module__` values now identify their new modules; public import paths remain available.

The new rollout module imports the existing actor data type and configuration. Neither module imports the learner or starts work at import time. A rollout-math change now opens the 148-line algorithm module plus the relevant data contract when necessary, rather than the whole learner owner. The reward formatter can be read independently in its 40-line module. Module depth stays at one package level; no dependencies or framework were added.

`distillation_job.SOURCE_FILES` and all of its files are untouched. The learner and the two new modules are outside that independent distillation implementation's source list and import closure. This packet changes no distillation restart identity, checkpoint format, artifact schema, model/optimizer serialization or actor/learner wire contract.

Validation uses the repository's Python 3.12.10 virtual environment:

- Four function AST comparisons and historical import identity checks passed.
- Compilation of `pyrl_trainer` passed.
- Pylint with the repository CI options passed for all three changed modules (10.00/10, exit 0). The unused `learner.gae` compatibility re-export has a local documented suppression.
- All 29 additional native artifact/teacher tests passed, with the deliberately excluded `real_250th` qualification case deselected (exit 0). The current Slither debug tools generated the teacher reference; the tracked `large_brain_migration_v1.json` fixture was present.
- All 9 live reward-codec/native-evaluator tests passed against the current Slither checkout (exit 0).
- Full default pytest passed: 804 passed, 13 skipped (exit 0). Nine skips are the live cross-repository cases, all separately enabled and passed above. Together these cover the baseline 813 passing cases with four remaining optional skips. Before/after discovery is the same 817 test IDs.

The first pytest invocation used the Slither working directory and failed while collecting unrelated files. It is a harness error, not a source regression; the corrected command runs from this repository and follows `pytest.ini`. An initial lint trial also identified unnecessary import aliases; those were corrected without changing the function bodies.
