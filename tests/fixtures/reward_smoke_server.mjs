/** Real Slither process for bounded Python protocol/actor smoke tests. */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [slitherRoot, dbPath] = process.argv.slice(2);
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
process.stdout.write(`PYRL_TEST_SERVER ${JSON.stringify({ port: server.port })}\n`);
let stopping = false;
const stop = async () => {
  if (stopping) return;
  stopping = true;
  await server.close();
  process.exit(0);
};
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { if (chunk.includes('stop')) void stop(); });
process.stdin.on('end', () => { void stop(); });
process.on('SIGTERM', () => { void stop(); });
