/* End-to-end test of the AI atelier loop using the diffusion server's --mock
 * backend: app captures fake-camera frames, streams them over the websocket,
 * receives processed frames, and displays them.
 *
 * Usage: node test/ai-mock.cjs   (needs python3 + `pip install websockets pillow`)
 */
'use strict';

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

function loadPlaywright() {
  try {
    return require('playwright');
  } catch {
    const globalRoot = execSync('npm root -g').toString().trim();
    return require(path.join(globalRoot, 'playwright'));
  }
}

const { chromium } = loadPlaywright();
const root = path.join(__dirname, '..');
const outDir = path.join(__dirname, 'out');
const HTTP_PORT = 8617;
const WS_PORT = 8767;

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const http = spawn('python3', ['-m', 'http.server', String(HTTP_PORT), '--bind', '127.0.0.1'],
    { cwd: root, stdio: 'ignore' });
  const server = spawn('python3', ['server/monet_server.py', '--mock', '--port', String(WS_PORT)],
    { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
  let serverLog = '';
  server.stdout.on('data', (d) => { serverLog += d; });
  server.stderr.on('data', (d) => { serverLog += d; });
  await new Promise((r) => setTimeout(r, 1500));

  const browser = await chromium.launch({
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--enable-unsafe-swiftshader',
    ],
  });

  try {
    if (server.exitCode !== null) throw new Error(`server died early:\n${serverLog}`);
    const page = await browser.newPage({ viewport: { width: 1380, height: 940 } });
    const consoleErrors = [];
    page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
    page.on('pageerror', (e) => consoleErrors.push(String(e)));

    await page.goto(`http://127.0.0.1:${HTTP_PORT}/index.html?low=1`);
    await page.waitForFunction(
      () => window.__monet && window.__monet.mode === 'camera' && window.__monet.frames > 4,
      null, { timeout: 120000 });

    await page.fill('#ai-url', `ws://127.0.0.1:${WS_PORT}`);
    await page.click('#b-ai');
    await page.waitForFunction(() => window.__monet.ai.connected, null, { timeout: 30000 });
    await page.waitForFunction(() => window.__monet.ai.roundtrips > 5, null, { timeout: 120000 });

    await page.evaluate(() => {
      window.__monet.sample = undefined;
      window.__monet.requestSample = true;
    });
    await page.waitForFunction(() => window.__monet.sample !== undefined, null, { timeout: 30000 });
    const dbg = await page.evaluate(() => ({
      roundtrips: window.__monet.ai.roundtrips,
      sample: +window.__monet.sample.toFixed(3),
      status: document.querySelector('#ai-status').textContent,
      errors: window.__monet.errors,
    }));
    console.log('[ai-mock]', JSON.stringify(dbg));

    const appErrors = dbg.errors.filter((e) => !e.startsWith('camera:'));
    if (appErrors.length) throw new Error(`app errors: ${appErrors.join('; ')}`);
    if (consoleErrors.length) throw new Error(`console errors: ${consoleErrors.join('; ')}`);
    if (!dbg.status.startsWith('connected')) throw new Error(`unexpected status: ${dbg.status}`);
    if (dbg.sample < 0.02) throw new Error('canvas appears black in AI mode');

    await page.screenshot({ path: path.join(outDir, 'ai-mock.png'), animations: 'disabled' });
    console.log('OK — screenshot in test/out/ai-mock.png');
  } finally {
    await browser.close();
    http.kill();
    server.kill();
  }
})().catch((e) => {
  console.error(e.message || e);
  process.exit(1);
});
