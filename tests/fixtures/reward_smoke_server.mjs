/** Real Slither process for bounded Python protocol/actor smoke tests. */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline';

const [slitherRoot, dbPath, mode] = process.argv.slice(2);
if (!slitherRoot || !dbPath) throw new Error('expected Slither root and task-owned database path');
const moduleUrl = relative => pathToFileURL(resolve(slitherRoot, relative)).href;
const [{ startRustServer }, { DEFAULT_CONFIG }] = await Promise.all([
  import(moduleUrl('server/rustServer.ts')),
  import(moduleUrl('server/config.ts')),
]);
const server = await startRustServer({ ...DEFAULT_CONFIG, port: 0, resume: 'fresh', seed: 42,
  dbPath: resolve(dbPath), rustCalculationWorkers: 1 });
if (server.startupFault) {
  await server.close();
  throw new Error(`native startup failed: ${JSON.stringify(server.startupFault)}`);
}
const controls = mode === 'ppo'
  ? await (await import('./ppo_reward_controls.mjs')).createControls(server.port, slitherRoot)
  : undefined;
process.stdout.write(`PYRL_TEST_SERVER ${JSON.stringify({ port: server.port })}\n`);
let stopping = false;
const stop = async () => {
  if (stopping) return;
  stopping = true;
  controls?.close();
  await server.close();
  process.exit(0);
};
process.on('SIGTERM', () => { void stop(); });
for await (const line of createInterface({ input: process.stdin })) {
  if (line === 'stop') break;
  try {
    const request = JSON.parse(line);
    if (!controls) throw new Error('training controls not enabled');
    await controls.command(request);
    process.stdout.write(`PYRL_TEST_RESULT ${JSON.stringify({ id: request.id, ok: true })}\n`);
  } catch (error) {
    process.stdout.write(`PYRL_TEST_RESULT ${JSON.stringify({ ok: false, error: String(error) })}\n`);
  }
}
await stop();
