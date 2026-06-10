/* Headless smoke test: serves the app, opens it in Chromium with a fake
 * camera device, waits for the paint pipeline to produce frames, and saves
 * screenshots to test/out/. Fails on shader/console errors or a blank canvas.
 *
 * Usage: node test/screenshot.cjs
 * Requires: python3 on PATH, playwright + its chromium (global install is fine).
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
const PORT = 8612;

// ?low=1 keeps buffers small so software GL (SwiftShader) stays responsive
const SCENARIOS = [
  { name: 'camera', query: '?low=1', expectMode: 'camera' },
  { name: 'demo', query: '?demo=1&low=1', expectMode: 'demo' },
];

(async () => {
  fs.mkdirSync(outDir, { recursive: true });
  const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], {
    cwd: root,
    stdio: 'ignore',
  });
  await new Promise((r) => setTimeout(r, 800));

  const browser = await chromium.launch({
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      '--enable-unsafe-swiftshader',
    ],
  });

  try {
    for (const { name, query, expectMode } of SCENARIOS) {
      const page = await browser.newPage({ viewport: { width: 1380, height: 940 } });
      const consoleErrors = [];
      page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
      page.on('pageerror', (e) => consoleErrors.push(String(e)));

      await page.goto(`http://127.0.0.1:${PORT}/index.html${query}`);
      await page.waitForFunction(
        (mode) => window.__monet && window.__monet.frames > 8 && window.__monet.mode === mode,
        expectMode,
        { timeout: 120000 },
      );
      const dbg = await page.evaluate(() => ({
        frames: window.__monet.frames,
        fps: +window.__monet.fps.toFixed(1),
        mode: window.__monet.mode,
        errors: window.__monet.errors,
      }));
      console.log(`[${name}]`, JSON.stringify(dbg));

      const shaderErrors = dbg.errors.filter((e) => !e.startsWith('camera:'));
      if (shaderErrors.length) throw new Error(`[${name}] app errors: ${shaderErrors.join('; ')}`);
      if (consoleErrors.length) throw new Error(`[${name}] console errors: ${consoleErrors.join('; ')}`);

      for (const camo of [0, 1]) {
        await page.evaluate((v) => {
          const el = document.querySelector('#s-camo');
          el.value = v;
          el.dispatchEvent(new Event('input'));
          window.__monet.sample = undefined;
          window.__monet.requestSample = true;
        }, camo);
        await page.waitForFunction(() => window.__monet.sample !== undefined, null, { timeout: 30000 });
        const sample = await page.evaluate(() => window.__monet.sample);
        console.log(`[${name}] camo ${camo} centre brightness ${sample.toFixed(3)}`);
        if (sample < 0.02) throw new Error(`[${name}] canvas appears black at camo ${camo}`);
      }

      await page.screenshot({ path: path.join(outDir, `${name}.png`), animations: 'disabled' });
      await page.close();
    }
    console.log('OK — screenshots in test/out/');
  } finally {
    await browser.close();
    server.kill();
  }
})().catch((e) => {
  console.error(e.message || e);
  process.exit(1);
});
