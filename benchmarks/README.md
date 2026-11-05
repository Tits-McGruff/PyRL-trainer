# Experimental drivers

Maintained benchmark and ablation programs live in `pyrl_trainer` and `scripts`.
Use a clean commit or a temporary `experiment/...` branch for a new experiment.
The unique historical `experiment.py` patch and its base/hash metadata remain
inspectable at the unchanged `qualification-resume-20261025` tag. They are not
runtime dependencies or an ongoing source reconstruction system.

## PPO reward and steering qualification

Use the bounded PPO command in the trainer README to produce each seed's
`initial/latest.pt` and final `latest.pt`. Evaluate those exact files and the
fixed controls on the same declared world seeds. For each policy and scenario,
start a fresh Slither server with its own scratch database, matching settings,
and the declared `--seed`. Run one episode per server.

```powershell
python -m scripts.evaluate_ppo_episode --config $trainerConfig --ws-url $wsUrl --scenario-seed 1001 --checkpoint $checkpoint
python -m scripts.evaluate_ppo_episode --config $trainerConfig --ws-url $wsUrl --scenario-seed 1001 --reference full-lock
```

The fixed controls are `turn=1, boost=0` for `full-lock` and `turn=0, boost=0`
for `straight`. Keep the evaluator cap, sampling seed, action pacing, and
server settings common across comparisons. The `[qualification-episode]`
line contains one JSON record with the world seed, configuration hash,
policy identity, actual starting and cutoff steps, reward and steering
measurements, and integrity status. World seeds and settings define the
matched scenario. Asynchronous join timing can differ between processes.
Retain and report incomplete episodes separately from paired results.
