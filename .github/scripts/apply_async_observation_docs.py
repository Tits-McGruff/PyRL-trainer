from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


# Make the existing stride test explicitly exercise a burst where observations
# outnumber policy actions and every intermediate observation contributes reward.
path = "tests/integration/test_actor_transitions.py"
needle = '''@pytest.mark.asyncio\nasync def test_import_replacement_rejoins_without_stale_token():\n'''
test = '''@pytest.mark.asyncio\nasync def test_burst_observation_stream_processes_every_sensor_between_actions(monkeypatch):\n    """Queued sensor bursts remain ordered even when action pacing skips ticks."""\n    cfg = Config(max_actions_per_second=0, horizon=20)\n    actor = ActorClient(0, cfg, DummySharedState(), asyncio.Queue())\n    actor.snake_id = 1\n    actor.stride = 3\n    actor.sensor_order = ["points_delta_norm"]\n    actor.sensor_idx = {"points_delta_norm": 0}\n\n    monkeypatch.setattr(\n        "pyrl_trainer.agent.default_reward_components",\n        lambda prev, obs, idx, reward_cfg: {\n            "growth": float(obs[0] - prev[0]),\n            "food_approach": 0.0,\n            "survival": 0.0,\n            "wall": 0.0,\n            "hazard": 0.0,\n        },\n    )\n\n    ws = DummyWS()\n    for tick in [10, 11, 12, 13, 14]:\n        await actor._handle_sensors(\n            {\n                "type": "sensors",\n                "tick": tick,\n                "snakeId": 1,\n                "sensors": [float(tick)],\n            },\n            ws,\n        )\n\n    assert len(ws.sent) == 2\n    assert [json.loads(value)["tick"] for value in ws.sent] == [10, 13]\n    assert actor.last_sensor_tick == 14\n    assert actor.last_obs is not None\n    assert actor.last_obs[0] == pytest.approx(14.0)\n    assert len(actor.rollout) == 1\n    assert actor.rollout[0].reward == pytest.approx(3.0)\n    assert actor.pending_transition is not None\n    assert actor.pending_transition.reward == pytest.approx(1.0)\n\n\n'''
text = read(path)
if needle not in text:
    raise RuntimeError("actor transition insertion point missing")
write(path, text.replace(needle, test + needle, 1))

# README clarification for the actual actor behavior.
path = "README.md"
replace_once(path,
'''- Spawns multiple actors (`Config.actors`) to read sensors, sample actions, and stream rollouts.\n- Applies action pacing using tick stride and wall-clock gating.\n''',
'''- Spawns multiple actors (`Config.actors`) to read the server's ordered asynchronous sensor stream, sample actions, and stream rollouts.\n- Processes every sensor observation in order even when action pacing skips that tick; action pacing uses tick stride and wall-clock gating.\n''')
replace_once(path,
'''- The actor enforces action pacing via `max_actions_per_second`, using stride plus a wall-clock gate.\n''',
'''- The actor enforces action pacing via `max_actions_per_second`, using stride plus a wall-clock gate. Sensor packets are an ordered asynchronous stream; receiving one does not imply the server is waiting for an action or transport acknowledgement, and multiple observations may be processed between sent actions.\n''')

# Replace stale server-interface claims in the trainer-local API guide.
path = "docs/API-Instructions"
text = read(path)
text = text.replace(
'''10. `stats` broadcasts about once per second. `fitnessData` is included by\n    default in the current build but should be treated as optional.\n    `fitnessHistory` is omitted unless a new generation entry has been\n    appended since the last broadcast. `viz` is included only when viz\n    streaming is enabled and MT inference is off (`mtEnabled` in\n    `server/config.toml`). `hofEntry` is included only when a new HoF entry\n    is produced.\n''',
'''10. `stats` is replaceable display/diagnostic output and should be treated as\n    optional by training clients. PPO actors should drive their control and\n    reward chronology from `sensors`, `assign`, and lifecycle messages rather\n    than assuming a particular stats cadence or legacy inference-pool mode.\n''')
text = text.replace(
'''18. `tickRateHz` is clamped to `[1, 240]` by `server/config.ts`. The\n    `welcome.tickRate` value is the effective runtime tick rate after\n    clamping.\n\n### SensorSpec contract details\n\n19. `sensorSpec.layoutVersion` is `"v3"`. This is the only supported version.\n    Legacy layouts (`v2` and `legacy`) are no longer supported.\n20. `sensorSpec.sensorCount` always equals 19 + (4 * bins). The scalar count\n    is fixed at 19.\n21. All four binned channels (`food`, `hazard`, `wall`, `head`) are always\n    present in the v3 layout.\n22. All sensor values are clamped into `[-1, 1]` by `buildSensors`.\n''',
'''18. The Rust-authoritative simulation uses a fixed 60 Hz physics step. The\n    `welcome.tickRate` value is the client-facing cadence metadata and should\n    be used rather than assuming a retired configurable TypeScript tick loop.\n\n### SensorSpec contract details\n\n19. `welcome.sensorSpec` is the canonical sensor contract for the connected\n    server. Current servers can expose the frozen v3 layout or admitted v4\n    Basic/Motion layouts; clients must not infer semantics from vector width.\n20. `sensorSpec.sensorCount` and `sensorSpec.order` define the exact active\n    vector. PyRL validates the complete discovered contract before constructing\n    or resuming a model.\n21. V4 widths depend on profile and bin count, and current Motion/B16 runs use\n    264 inputs. Equal widths do not imply equal channel meanings; use the\n    server-provided contract identity and order.\n22. Sensor numeric semantics are owned by the Rust server contract. Training\n    code should consume the advertised order instead of duplicating a hard-coded\n    v3 formula.\n''')
text = text.replace(
'''6. Receive `sensors` each tick and send `action` updates.\n''',
'''6. Receive the ordered asynchronous `sensors` stream and send paced `action` updates.\n''')
text = text.replace(
'''- `sensorSpec` defines the exact sensor ordering and size.\n- `tickRate` is the effective runtime tick frequency in Hz after config\n  clamping.\n''',
'''- `sensorSpec` defines the exact sensor ordering, size, layout identity, and\n  any canonical v4 contract metadata supplied by the server.\n- `tickRate` is the server-advertised simulation cadence.\n''')
needle = '''- `sensors` is the observation vector for the assigned snake.\n- `meta` contains head position and heading for debugging or overlays.\n'''
replacement = '''- `sensors` is the observation vector for the assigned snake.\n- `meta` contains head position and heading for debugging or overlays.\n- Sensor packets are ordered asynchronous output from committed server steps.\n  The server does not wait for PyRL to acknowledge a packet or return an action\n  before computing the next ordinary step. `tick` identifies the source\n  boundary; PyRL may process several observations between sent actions when its\n  configured action pacing requires that.\n'''
if needle not in text:
    raise RuntimeError("API sensors bullet insertion point missing")
text = text.replace(needle, replacement, 1)
# Correct the deliberately stale welcome example enough to stop teaching a v2-only contract.
text = text.replace('''    "sensorCount": 55,\n    "order": ["heading_sin", "heading_cos"],\n    "layoutVersion": "v2"\n''', '''    "sensorCount": 264,\n    "order": ["heading_sin", "heading_cos"],\n    "layoutVersion": "v4"\n''')
write(path, text)

print("PyRL async observation documentation/tests patch applied")
