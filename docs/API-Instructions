# Server API Instructions for External Bot Clients

## Overview

This document explains how to connect a custom training program to the
simulation server and control a player snake. The external bot interface
exposes the same sensor vector and action space used by the built-in bots.
The server is authoritative: clients send turn and boost inputs, and the
server applies them during its tick loop.

## Endpoints and ports

- WebSocket: `ws://HOST:PORT` (default `ws://localhost:5174`).
- HTTP: `http://HOST:PORT` for REST endpoints on the same server.
- TLS: use `wss://` and `https://` if the server is behind TLS.

The defaults come from `server/config.toml`. Adjust `host` and `port` if you
run the server on a different interface.

## Protocol clarifications (from server code and defaults)

This section answers concrete integration questions by referencing the
current TypeScript server implementation and `server/config.toml`.

### WebSocket, handshake, and lifecycle

1. WebSocket URL format is `ws://HOST:PORT` with no path. Example:
   `ws://192.168.0.200:5174`. No query params are required or read by the
   server.
2. `PROTOCOL_VERSION` is `1`, defined in `server/protocol.ts`.
3. No auth headers, API keys, allowlists, or session tokens are required or
   supported for WebSocket connections. The server does not inspect custom
   headers for WS auth.
4. The server does not implement idle socket timeouts. Client `ping` is
   optional and acts only as a keepalive for intermediaries.
5. The server explicitly closes with code `1008` on protocol errors only.
   Other close codes may come from clients or the network; the server does
   not emit custom close codes beyond `1008`.

### Message schemas, field types, validation rules

6. `hello.clientType` must be exactly `"bot"` or `"ui"`. `hello.version` is
   strictly equal to `PROTOCOL_VERSION` (no range tolerance).
7. `join.name` is required only for `mode: "player"`. It must be a string
   of length <= 24. There are no character restrictions beyond length.
8. `assign` messages contain only `snakeId` and `controller`. The server
   does not emit extra fields. For external bot clients (`clientType: "bot"`)
   the `controller` field is always `"bot"`.
9. `sensors.meta` is optional in the schema but is always included by the
   current server build. No additional fields are emitted. Clients should
   ignore unknown fields if they ever appear.
10. `stats` broadcasts about once per second. `fitnessData` is included by
    default in the current build but should be treated as optional.
    `fitnessHistory` is omitted unless a new generation entry has been
    appended since the last broadcast. `viz` is included only when viz
    streaming is enabled and MT inference is off (`mtEnabled` in
    `server/config.toml`). `hofEntry` is included only when a new HoF entry
    is produced.
11. Protocol errors attempt to send `{ "type": "error", "message": "..." }`
    before closing. If sending fails, the server may close without a payload
    or with a minimal `{ "type": "error" }`. Clients should treat
    `error.message` as optional.
    Common error messages include:
    - `message too large`
    - `binary messages are not supported`
    - `invalid JSON`
    - `invalid message`
    - `duplicate hello`
    - `hello required before join`
    - `join required before action`
    - `action requires player mode`
    - `join required before view`
    - `join required before viz`
    - `join required before reset`
    - `reset requires ui client`

### Tick and action application semantics

12. `action.tick` is not validated against the server tick. The action is
    accepted regardless of tick value and updates the stored control input.
13. The server holds the latest action and samples it during the tick
    update. If multiple actions arrive in a tick, only the last one is used
    (subject to rate limits).
14. Actions are fire-and-forget. There is no acknowledgement, and dropped
    actions are silent.
15. When a snake dies, the server sends a new `assign` and the next tick's
    `sensors` correspond to the new `snakeId`. `assign` precedes the first
    `sensors` for the replacement snake.

### Rate limits and configuration

16. Defaults in `server/config.toml` are `maxActionsPerTick = 1` and
    `maxActionsPerSecond = 120`. These are read at startup (TOML, env, or
    CLI) and cannot be changed live.
17. Exceeding `maxActionsPerSecond` drops actions silently. There is no
    error message or stats counter exposed to clients.
18. `tickRateHz` is clamped to `[1, 240]` by `server/config.ts`. The
    `welcome.tickRate` value is the effective runtime tick rate after
    clamping.

### SensorSpec contract details

19. `sensorSpec.layoutVersion` is `"v3"`. This is the only supported version.
    Legacy layouts (`v2` and `legacy`) are no longer supported.
20. `sensorSpec.sensorCount` always equals 17 + (4 * bins). The scalar count
    is fixed at 17.
21. All four binned channels (`food`, `hazard`, `wall`, `head`) are always
    present in the v3 layout.
22. All sensor values are clamped into `[-1, 1]` by `buildSensors`.

### Optional UI-oriented messages and binary frames

23. `view` and `viz` are accepted after `join`. `view` supports `mode`,
    `viewW`, and `viewH`, but all are ignored by the simulation. `viz` can
    enable visualization data in `stats` even for bot clients (if MT
    inference is off).
24. `serializerVersion` is `1` in the current build. `frameByteLength` is
    derived from a sample frame and is primarily UI bookkeeping; actual
    frame buffers are variable length.

### HTTP endpoints alongside training clients

25. No HTTP endpoints require auth, special headers, or CSRF tokens. CORS is
    configured in `server/httpApi.ts` (`applyCors`) and defaults to allowing
    LAN origins with credentials, or `*` for non-credentialed requests.
26. `POST /api/import` accepts `force` as either a query param `?force=1` or
    a JSON body flag `{ "force": true }`. On `cfgHash` mismatch without
    force, it returns HTTP 409 with:

    ```json
    {
      "ok": false,
      "message": "cfgHash mismatch; pass force=true to override"
    }
    ```

    (HTTP 409).
27. Responses for `/health`, `/api/save`, `/api/export/latest`,
    `/api/resurrect` match the documented shapes and do not include extra
    fields in the current build. Clients should ignore unknown fields for
    forward compatibility.

## WebSocket protocol

### Connection sequence

1. Open a WebSocket connection.
2. Send `hello` with `clientType` and `version`.
3. Receive `welcome` (sensor spec, tick rate, versions).
4. Send `join` with `mode: "player"` and a `name`.
5. Receive `assign` with your `snakeId`.
6. Receive `sensors` each tick and send `action` updates.
7. Handle reassignments when the snake dies.

The server requires `hello` before `join`, and `join` before `action`, `view`,
`viz`, or `reset`.

### Encoding, size limits, and errors

- Client messages must be JSON text. Binary client messages are rejected.
- Max client payload size is 64 KiB by default.
- On protocol errors, the server sends `{ "type": "error" }` and closes the
  connection with code `1008`.

### Client-to-server messages

#### `hello`

```json
{
  "type": "hello",
  "clientType": "bot",
  "version": 1
}
```

- `clientType`: `"bot"` for training clients. `"ui"` is reserved for the
  browser UI.
- `version`: must equal `PROTOCOL_VERSION` (currently `1`).

#### `join`

```json
{
  "type": "join",
  "mode": "player",
  "name": "MyTrainer"
}
```

- `mode`: `"player"` to control a snake, or `"spectator"` to observe.
- `name`: required for `"player"` mode. Max length is 24 characters.

#### `action`

```json
{
  "type": "action",
  "tick": 12345,
  "snakeId": 42,
  "turn": -0.25,
  "boost": 1
}
```

- `tick`: typically echo the tick from the latest `sensors` message.
- `snakeId`: the id from the latest `assign` message.
- `turn`: float in `[-1, 1]` (left to right). Values are clamped.
- `boost`: float in `[0, 1]` (0 = off, 1 = on). Values are clamped.
- `tick` is not validated against the server tick; use it for traceability,
  not synchronization.

#### `ping`

```json
{ "type": "ping", "t": 123456 }
```

- Optional heartbeat. The server does not send a reply.

#### `view` (optional, UI-oriented)

```json
{ "type": "view", "mode": "overview", "viewW": 1920, "viewH": 1080 }
```

- `mode`, `viewW`, and `viewH` are optional and ignored by the simulation.

#### `viz` (optional, UI-oriented)

```json
{ "type": "viz", "enabled": true }
```

- Toggles visualizer streaming for UI clients.

#### `reset` (UI-only)

```json
{ "type": "reset", "settings": { "snakeCount": 300 } }
```

- Only accepted from `clientType: "ui"`.
- See `src/protocol/settings.ts` for allowed paths.

### Server-to-client messages

#### `welcome`

```json
{
  "type": "welcome",
  "sessionId": "abcd1234",
  "tickRate": 60,
  "worldSeed": 987654,
  "cfgHash": "...",
  "sensorSpec": {
    "sensorCount": 55,
    "order": ["heading_sin", "heading_cos"],
    "layoutVersion": "v2"
  },
  "serializerVersion": 1,
  "frameByteLength": 123456
}
```

- `sensorSpec` defines the exact sensor ordering and size.
- `tickRate` is the effective runtime tick frequency in Hz after config
  clamping.
- `serializerVersion` describes the binary frame layout (UI only).
- `frameByteLength` is the byte size of a sample frame buffer; real frames
  are variable length.

#### `assign`

```json
{ "type": "assign", "snakeId": 42, "controller": "bot" }
```

- Sent on initial control and whenever the assigned snake dies.
- Reset any per-snake state when a new `snakeId` arrives.
- External bot clients will only see `controller: "bot"`.

#### `sensors`

```json
{
  "type": "sensors",
  "tick": 12345,
  "snakeId": 42,
  "sensors": [0.12, -0.34, 0.7],
  "meta": { "x": 12.3, "y": -4.5, "dir": 1.57 }
}
```

- `sensors` is the observation vector for the assigned snake.
- `meta` contains head position and heading for debugging or overlays.

#### `stats`

```json
{
  "type": "stats",
  "tick": 12345,
  "gen": 3,
  "generationTime": 120.0,
  "generationSeconds": 60.0,
  "alive": 210,
  "aliveTotal": 300,
  "baselineBotsAlive": 210,
  "baselineBotsTotal": 220,
  "fps": 60
}
```

- Broadcast to all joined clients about once per second.
- Optional fields may be omitted. See the clarifications for when each
  field is present.

#### `error`

```json
{ "type": "error", "message": "join required before action" }
```

- Sent before the server closes the connection on protocol errors. The
  `message` field can be omitted if the send fails.

### Binary frame buffers (UI-only)

UI clients receive binary frames as `ArrayBuffer` payloads. Bot clients do not
receive frame buffers. If you want to render the world externally, open a
second connection with `clientType: "ui"` and `mode: "spectator"`.

Binary frame format is a `Float32Array` with this layout:

1. Header (7 floats)
   - `[generation, totalSnakes, aliveCount, worldRadius, cameraX,`
     `cameraY, zoom]`
2. Snake block (variable)
   - For each alive snake:
   - `[id, radius, skin, x, y, dir, boost, pointCount]`
   - Followed by `pointCount * 2` floats for body points `[x, y]`.
3. Pellet block (variable)
   - `[pelletCount]`
   - Followed by `pelletCount` entries of
     `[x, y, value, type, colorId]`.

The serializer version is included in `welcome.serializerVersion`.

## Sensor specification and layout

The server sends the exact sensor order in `welcome.sensorSpec.order`. Do not
assume a fixed layout; always build your input vector from this order.

### Scalar sensors (17 total)

The scalar sensors occupying indices 0-16 in the vector:

- `heading_sin`, `heading_cos`: sine/cosine of current heading.
- `size_norm`: snake size fraction in `[-1, 1]`.
- `boost_margin`: points relative to `minPointsToBoost`, in `[-1, 1]`.
- `points_pct`: log-scaled percentile vs best points this generation.
- `speed_norm`: speed relative to maximum boost speed.
- `boost_state`: current boost fuel status in `[0, 1]` mapped to `[-1, 1]`.
- `points_norm`: current points score normalized by generation best.
- `points_delta_norm`: change in points since the last simulation tick.
- `length_norm`: snake length relative to absolute max length.
- `boost_points_frac`: points available for boost relative to minimum cost.
- `boost_cost_norm`: current point loss rate from boosting (scales with size).
- `wall_dist_norm`: distance to the circular world boundary.
- `nearest_food_dist_norm`: distance to the nearest food pellet.
- `nearest_body_dist_norm`: distance to the nearest snake segment (any snake).
- `nearest_head_dist_norm`: distance to the nearest enemy snake head.
- `age_norm`: survival time normalized by the generation duration limit.

### Binned sensors

The binned channels follow the scalar sensors. With $N$ bins, there are $4 \times N$
total binned inputs:

- `food_0` ... `food_(N-1)`: local food density.
- `hazard_0` ... `hazard_(N-1)`: clearance to nearby bodies.
- `wall_0` ... `wall_(N-1)`: distance to the circular world wall.
- `head_0` ... `head_(N-1)`: pressure from nearby enemy heads.

All channels use **centered bin mapping**:
- Bin 0 is centered at $-\pi$ (directly behind).
- Bin $N/2$ is centered at $0$ (directly ahead).

## Action timing and rate limits

- Sensors are published at the start of each tick for controlled snakes.
- Actions are sampled once per tick; the server holds the last action until
  it is replaced.
- Expect one-tick latency between a `sensors` message and the action that it
  influences.
- Rate limits are enforced per connection:
  - `maxActionsPerTick` (default `1`).
  - `maxActionsPerSecond` (default `120`).
  - Extra actions are dropped silently.

## Multi-agent control and reassignment

- Each WebSocket connection controls one snake at a time.
- To control multiple snakes, open multiple connections.
- When a snake dies, the server assigns a new snake and sends a fresh
  `assign` message. Reset per-agent state on reassignment.

## HTTP API endpoints

All endpoints are served from the same host and port as the WebSocket server.

### `GET /health`

Returns server status:

```json
{ "ok": true, "tick": 12345, "clients": 3 }
```

### `POST /api/save`

Persists a snapshot and returns the snapshot id:

```json
{ "ok": true, "snapshotId": 7 }
```

### `GET /api/export/latest`

Returns the latest snapshot payload. The payload includes:

- `generation`, `archKey`, and `genomes`.
- `cfgHash` and `worldSeed`.
- Optional `settings` and `updates`.

### `POST /api/import`

Imports a snapshot payload. The body may be the payload directly or wrapped
as `{ "payload": { ... } }`.

- If `cfgHash` differs from the current server config, the request fails
  with `409` unless `force=true` (body flag or `?force=1`).
- Response:

```json
{ "ok": true, "used": 200, "total": 300 }
```

### `POST /api/resurrect`

Spawns a snake from a genome. Body is either a `GenomeJSON` object or
`{ "genome": { ... } }`.

```json
{ "ok": true, "snakeId": 42 }
```

### `GET /api/graph-presets?limit=50`

Lists graph presets:

```json
{ "ok": true, "presets": [ { "id": 1, "name": "MySpec" } ] }
```

### `GET /api/graph-presets/:id`

Loads a specific preset:

```json
{ "ok": true, "preset": { "id": 1, "name": "MySpec", "spec": {} } }
```

### `POST /api/graph-presets`

Saves a preset:

```json
{ "name": "MySpec", "spec": { "nodes": [], "edges": [] } }
```

### `GET /api/hof?limit=50`

Returns Hall of Fame entries:

```json
{ "ok": true, "hof": [ { "gen": 10, "fitness": 123 } ] }
```

### `POST /api/hof`

Saves Hall of Fame entries:

```json
{ "hof": [ { "gen": 10, "fitness": 123, "genome": { ... } } ] }
```

## Minimal Node bot client (ws)

This example uses the `ws` package in Node. It connects, joins as a player,
and sends an action for each `sensors` message.

Install dependency:

```bash
npm install ws
```

Example client:

```js
const WebSocket = require("ws");

const url = process.env.SLITHER_WS_URL ?? "ws://localhost:5174";
const name = process.env.SLITHER_BOT_NAME ?? "ExternalBot";

let snakeId = null;
let sensorOrder = [];
let sensorIndex = {};

const ws = new WebSocket(url);

ws.on("open", () => {
  const hello = { type: "hello", clientType: "bot", version: 1 };
  ws.send(JSON.stringify(hello));
});

ws.on("message", (data, isBinary) => {
  if (isBinary) return;
  const msg = JSON.parse(data.toString());

  if (msg.type === "welcome") {
    sensorOrder = msg.sensorSpec?.order ?? [];
    sensorIndex = Object.fromEntries(
      sensorOrder.map((label, i) => [label, i])
    );
    ws.send(JSON.stringify({ type: "join", mode: "player", name }));
    return;
  }

  if (msg.type === "assign") {
    snakeId = msg.snakeId;
    return;
  }

  if (msg.type === "sensors") {
    if (snakeId == null || msg.snakeId !== snakeId) return;
    const action = policy(msg.sensors, sensorIndex);
    ws.send(
      JSON.stringify({
        type: "action",
        tick: msg.tick,
        snakeId,
        turn: action.turn,
        boost: action.boost
      })
    );
    return;
  }

  if (msg.type === "error") {
    console.error("Server error:", msg.message);
  }
});

function policy(sensors, index) {
  const boostIdx = index.boost_margin ?? -1;
  const boostMargin = boostIdx >= 0 ? sensors[boostIdx] : -1;
  const boost = boostMargin > 0 ? 1 : 0;
  return { turn: 0, boost };
}
```

## Protocol TypeScript snippets

These are minimal client-side typings that match the server protocol.

```ts
export const PROTOCOL_VERSION = 1;

export type ClientType = "ui" | "bot";
export type JoinMode = "spectator" | "player";

export type HelloMsg = {
  type: "hello";
  clientType: ClientType;
  version: number;
};

export type JoinMsg = {
  type: "join";
  mode: JoinMode;
  name?: string;
};

export type PingMsg = {
  type: "ping";
  t?: number;
};

export type ActionMsg = {
  type: "action";
  tick: number;
  snakeId: number;
  turn: number;
  boost: number;
};

export type ViewMsg = {
  type: "view";
  viewW?: number;
  viewH?: number;
  mode?: "overview" | "follow" | "toggle";
};

export type VizMsg = {
  type: "viz";
  enabled: boolean;
};

export type SettingsUpdate = {
  path: string;
  value: number;
};

export type ResetMsg = {
  type: "reset";
  settings?: Record<string, unknown>;
  updates?: SettingsUpdate[];
  graphSpec?: unknown | null;
};

export type ClientMessage =
  | HelloMsg
  | JoinMsg
  | PingMsg
  | ActionMsg
  | ViewMsg
  | VizMsg
  | ResetMsg;

export type SensorSpec = {
  sensorCount: number;
  order: string[];
  layoutVersion: "v3";
};

export type WelcomeMsg = {
  type: "welcome";
  sessionId: string;
  tickRate: number;
  worldSeed: number;
  cfgHash: string;
  sensorSpec: SensorSpec;
  serializerVersion: number;
  frameByteLength: number;
};

export type AssignMsg = {
  type: "assign";
  snakeId: number;
  controller: "player" | "bot";
};

export type SensorsMsg = {
  type: "sensors";
  tick: number;
  snakeId: number;
  sensors: number[];
  meta?: { x: number; y: number; dir: number };
};

export type StatsMsg = {
  type: "stats";
  tick: number;
  gen: number;
  generationTime: number;
  generationSeconds: number;
  alive: number;
  aliveTotal: number;
  baselineBotsAlive: number;
  baselineBotsTotal: number;
  fps: number;
  fitnessData?: {
    gen: number;
    avgFitness: number;
    maxFitness: number;
    minFitness: number;
  };
  fitnessHistory?: Array<{
    gen: number;
    best: number;
    avg: number;
    min: number;
    speciesCount?: number;
    topSpeciesSize?: number;
    avgWeight?: number;
    weightVariance?: number;
  }>;
  viz?: {
    kind: string;
    layers: Array<{
      count: number;
      activations: ArrayLike<number> | null;
      isRecurrent?: boolean;
    }>;
  };
  hofEntry?: {
    gen: number;
    seed: number;
    fitness: number;
    points: number;
    length: number;
    genome: {
      archKey: string;
      brainType?: string;
      weights: number[];
      fitness?: number;
    };
  };
};

export type ErrorMsg = {
  type: "error";
  message?: string;
};

export type ServerMessage =
  | WelcomeMsg
  | AssignMsg
  | SensorsMsg
  | StatsMsg
  | ErrorMsg;
```

## Example control loop (pseudo code)

```text
connect ws
send hello
wait for welcome
send join player with name
wait for assign
loop:
  wait for sensors
  compute action from sensor vector
  send action using the same tick and snakeId
```

## Common integration pitfalls

- Missing `hello` or `join` causes an immediate protocol error.
- `name` is required for `player` mode.
- Ignore `assign` and your actions will be dropped silently.
- Spamming actions above the rate limits leads to dropped inputs.
- Do not assume a fixed sensor length; always read `sensorSpec`.
- Bot clients do not receive binary frame buffers.
