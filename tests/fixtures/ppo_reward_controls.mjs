/** Task-only placements using the real server's existing UI control protocol. */
import { once } from 'node:events';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const delay = milliseconds => new Promise(done => setTimeout(done, milliseconds));
async function until(predicate) {
  const deadline = performance.now() + 10_000;
  while (performance.now() < deadline) {
    if (await predicate()) return;
    await delay(10);
  }
  throw new Error('native smoke control did not reach its authoritative boundary');
}

/** Read a real frame pellet with clearance from the other complete bodies. */
function foodPosition(bytes, controlledId) {
  const read = index => bytes.readFloatLE(index * 4);
  const occupied = [];
  let cursor = 7;
  for (let snake = 0; snake < read(2); snake++) {
    const id = read(cursor), count = read(cursor + 7);
    for (let point = 0; point < count; point++) {
      if (id !== controlledId) occupied.push([read(cursor + 8 + point * 2),
        read(cursor + 9 + point * 2)]);
    }
    cursor += 8 + count * 2;
  }
  const count = read(cursor++);
  for (let pellet = 0; pellet < count; pellet++, cursor += 5) {
    const x = read(cursor), y = read(cursor + 1);
    if (Math.hypot(x, y) < read(3) - 300 &&
        occupied.every(body => Math.hypot(x - body[0], y - body[1]) > 300)) return { x, y };
  }
  throw new Error('native smoke world has no food with body clearance');
}

export async function createControls(port, root) {
  const require = createRequire(pathToFileURL(resolve(root, 'package.json')));
  const WebSocket = require('ws');
  const { buildStackGraphSpec } = await import(pathToFileURL(
    resolve(root, 'src/brains/stackBuilder.ts')).href);
  const socket = new WebSocket(`ws://127.0.0.1:${port}`);
  let frame;
  socket.on('message', (raw, binary) => { if (binary) frame = raw; });
  await once(socket, 'open');
  const exchange = (request, predicate) => new Promise((done, reject) => {
    const cleanup = () => { clearTimeout(timer); socket.off('message', receive); };
    const receive = (raw, binary) => {
      if (binary) return;
      const message = JSON.parse(raw.toString());
      if (message.type === 'error') { cleanup(); reject(new Error(message.message)); }
      else if (predicate(message)) { cleanup(); done(message); }
    };
    const timer = setTimeout(() => { cleanup(); reject(new Error('native control reply timeout')); }, 10_000);
    socket.on('message', receive);
    socket.send(JSON.stringify(request));
  });
  await exchange({ type: 'hello', clientType: 'ui', version: 2 }, value => value.type === 'welcome');
  socket.send(JSON.stringify({ type: 'join', mode: 'spectator' }));
  const notice = await exchange({ type: 'reset', settings: { snakeCount: 1, simSpeed: 0.1 },
    updates: [{ path: 'baselineBots.count', value: 0 }, { path: 'pelletCountTarget', value: 100 },
      { path: 'growPerFood', value: 4 }, { path: 'generationSeconds', value: 240 }],
    graphSpec: buildStackGraphSpec({ hiddenLayers: 1, neurons1: 2, neurons2: 2,
      neurons3: 2, neurons4: 2, neurons5: 2 },
    { brain: { inSize: 264, outSize: 2, useMlp: false } }) },
  value => value.type === 'stateReplaced' && value.reason === 'reset');
  const welcome = notice.welcome;
  const health = async () => (await fetch(`http://127.0.0.1:${port}/api/health`)).json();
  await until(async () => (await health()).retention.activeRunId === welcome.runId);
  frame = undefined;
  socket.send(JSON.stringify({ type: 'join', mode: 'spectator', rejoinToken: notice.rejoinToken }));
  await until(() => frame !== undefined);
  let sequence = 0;
  return {
    close() { socket.terminate(); },
    async command(request) {
      const id = `smoke-${++sequence}-${request.id}`;
      if (request.action === 'generation') {
        const generation = (await health()).generation;
        socket.send(JSON.stringify({ type: 'finishGeneration', requestId: id,
          expectedRunId: welcome.runId, expectedGeneration: Number(BigInt(`0x${generation}`)) }));
        await until(async () => (await health()).generation !== generation);
        return;
      }
      if (!['food', 'death'].includes(request.action)) throw new Error('unsupported fixture action');
      const selection = await exchange({ type: 'selectEntity', requestId: `${id}-select`,
        expectedSessionId: welcome.sessionId, expectedRunId: welcome.runId,
        expectedGeneration: frame.readFloatLE(0), snakeId: request.snakeId },
      value => value.requestId === `${id}-select`);
      if (!selection.found) throw new Error('controlled smoke snake was not found');
      const result = await exchange({ type: 'godMode', requestId: id,
        action: request.action === 'food' ? 'move' : 'kill', snakeId: request.snakeId,
        selectionToken: selection.entity.token,
        ...(request.action === 'food' ? foodPosition(frame, request.snakeId) : {}) },
      value => value.requestId === id);
      if (!result.applied) throw new Error(`native smoke placement rejected: ${JSON.stringify(result)}`);
    }
  };
}
