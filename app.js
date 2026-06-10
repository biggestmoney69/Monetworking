/* Monetworking — live impressionism.
 * Camera (or demo garden) → anisotropic Kuwahara paint pipeline → framed canvas.
 */
'use strict';

(() => {
  // debug hook used by the headless test harness
  const dbg = (window.__monet = { frames: 0, errors: [], fps: 0, mode: 'demo' });

  const qsParams = new URLSearchParams(location.search);
  const FORCE_DEMO = qsParams.has('demo');
  const LOW = qsParams.has('low');          // small buffers for software GL / weak devices
  const MAX_INTERNAL_W = LOW ? 480 : 960;

  const $ = (s) => document.querySelector(s);
  const ui = {
    canvas: $('#view'), frame: $('#frame'), hold: $('#hold'),
    overlay: $('#overlay'), overlayMsg: $('#overlay-msg'),
    toast: $('#toast'), fps: $('#fps'), meta: $('#meta'),
    bFlip: $('#b-flip'), bPause: $('#b-pause'), bSave: $('#b-save'), bDemo: $('#b-demo'),
    bAi: $('#b-ai'), aiUrl: $('#ai-url'), aiPrompt: $('#ai-prompt'),
    aiStatus: $('#ai-status'), aiDot: $('#ai-dot'),
    sStrength: $('#s-strength'), oStrength: $('#o-strength'),
    sStructure: $('#s-structure'), oStructure: $('#o-structure'),
  };

  const state = {
    mode: 'demo', facing: 'user', mirror: false,
    paused: false, snap: false,
    stream: null,
    iw: 4, ih: 3,                  // internal paint resolution
    justReset: true,               // suppress temporal blend on fresh buffers
    lastPaint: 0,                  // index of most recently painted ping-pong target
    params: { camo: 0.7, brush: 5, q: 8, dream: 0.55, warm: 0.6, weave: 0.55, relief: 0.55, wet: 0.45 },
  };

  function fatal(msg) {
    dbg.errors.push(msg);
    ui.overlayMsg.textContent = msg;
    ui.overlay.classList.add('show');
  }

  let toastTimer = 0;
  function toast(msg) {
    ui.toast.textContent = msg;
    ui.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => ui.toast.classList.remove('show'), 4000);
  }

  // ---------------------------------------------------------------- GL setup
  const gl = ui.canvas.getContext('webgl2', {
    antialias: false, alpha: false, powerPreference: 'high-performance',
    preserveDrawingBuffer: true,   // screenshots & "Save painting" stay reliable
  });
  if (!gl) return fatal('This browser does not support WebGL2, which the painting pipeline needs.');

  const floatRenderable = !!(gl.getExtension('EXT_color_buffer_float') ||
                             gl.getExtension('EXT_color_buffer_half_float'));
  const tensorDefines = floatRenderable ? '' : '#define PACK_TENSOR\n';

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(gl.createVertexArray());   // fullscreen triangle uses gl_VertexID only

  function compile(type, src, name) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`${name}: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
  }

  function program(name, fsBody, defines = '') {
    const fsSrc = `#version 300 es\nprecision highp float;\n${defines}${fsBody}`;
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, SHADERS.vert, `${name}.vert`));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc, `${name}.frag`));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(`${name}: ${gl.getProgramInfoLog(p)}`);
    }
    p.u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      p.u[info.name.replace('[0]', '')] = gl.getUniformLocation(p, info.name);
    }
    return p;
  }

  let progSource, progTensor, progBlur, progKuwahara, progComposite, progBlend;
  try {
    progSource = program('source', SHADERS.source);
    progTensor = program('tensor', SHADERS.tensor, tensorDefines);
    progBlur = program('blur', SHADERS.blur);
    progKuwahara = program('kuwahara', SHADERS.kuwahara, tensorDefines);
    progComposite = program('composite', SHADERS.composite);
    progBlend = program('blend', SHADERS.blend);
  } catch (e) {
    return fatal(`Shader build failed — ${e.message}`);
  }

  function makeTexture(w, h, internalFormat) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (w) gl.texStorage2D(gl.TEXTURE_2D, 1, internalFormat, w, h);
    return tex;
  }

  function makeTarget(w, h, internalFormat) {
    const tex = makeTexture(w, h, internalFormat);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo, w, h };
  }

  const videoTex = makeTexture(0, 0, gl.RGBA8);   // storage allocated per-upload
  let rt = null;                                  // render targets, sized to internal res

  function allocTargets(w, h) {
    if (rt) {
      for (const t of Object.values(rt)) {
        gl.deleteTexture(t.tex);
        gl.deleteFramebuffer(t.fbo);
      }
    }
    const tensorFormat = floatRenderable ? gl.RGBA16F : gl.RGBA8;
    rt = {
      src: makeTarget(w, h, gl.RGBA8),
      tensorA: makeTarget(w, h, tensorFormat),
      tensorB: makeTarget(w, h, tensorFormat),
      paint0: makeTarget(w, h, gl.RGBA8),
      paint1: makeTarget(w, h, gl.RGBA8),
    };
  }

  function setInternalSize(w, h) {
    state.iw = w;
    state.ih = h;
    state.justReset = true;
    allocTargets(w, h);
    ui.frame.style.setProperty('--ar', `${w} / ${h}`);
    resizeCanvas();
    ui.meta.textContent = `WebGL2 · ${w}×${h}${floatRenderable ? '' : ' · 8-bit tensor'}`;
  }

  function resizeCanvas() {
    const r = ui.hold.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    const w = Math.round(r.width * dpr);
    const h = Math.round(r.height * dpr);
    if (ui.canvas.width !== w || ui.canvas.height !== h) {
      ui.canvas.width = w;
      ui.canvas.height = h;
    }
  }
  new ResizeObserver(resizeCanvas).observe(ui.hold);

  // ------------------------------------------------------------------ camera
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;

  function stopStream() {
    if (state.stream) {
      for (const t of state.stream.getTracks()) t.stop();
      state.stream = null;
    }
  }

  function useDemo() {
    stopStream();
    state.mode = 'demo';
    dbg.mode = 'demo';
    state.mirror = false;
    setInternalSize(LOW ? 480 : 880, LOW ? 360 : 660);
    ui.bDemo.textContent = 'Use camera';
  }

  const CAMERA_ERRORS = {
    NotAllowedError: 'Camera permission was declined — painting the demo garden instead.',
    NotFoundError: 'No camera was found — painting the demo garden instead.',
    NotReadableError: 'The camera is busy in another app — painting the demo garden instead.',
    NotSupportedError: 'Camera needs a secure context (https or localhost) — painting the demo garden.',
  };

  async function startCamera(facing) {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw Object.assign(new Error('insecure context'), { name: 'NotSupportedError' });
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      stopStream();
      state.stream = stream;
      video.srcObject = stream;
      await video.play();
      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;
      state.facing = facing;
      state.mirror = facing === 'user';
      state.mode = 'camera';
      dbg.mode = 'camera';
      const w = Math.min(MAX_INTERNAL_W, vw);
      setInternalSize(w, Math.round((w * vh) / vw));
      ui.bDemo.textContent = 'Demo scene';
      toast(`Painting from camera · ${vw}×${vh}`);
      return true;
    } catch (e) {
      dbg.errors.push(`camera: ${e.name || e.message}`);
      if (state.mode !== 'demo') useDemo();
      toast(CAMERA_ERRORS[e.name] || `Camera unavailable (${e.name || e.message}) — painting the demo garden.`);
      return false;
    }
  }

  // ------------------------------------------------------- AI atelier (WS)
  // Streams source frames to a local diffusion server (server/monet_server.py)
  // and shows the generated paintings, temporally blended to calm flicker.
  const ai = (state.ai = {
    on: false, ws: null, inflight: false, ready: false,
    bitmap: null, w: 0, h: 0, cur: 0, lastSend: 0,
    texNew: null, blendA: null, blendB: null,
  });
  dbg.ai = { connected: false, roundtrips: 0 };
  ai.texNew = makeTexture(0, 0, gl.RGBA8);
  const capFull = document.createElement('canvas');
  const capSmall = document.createElement('canvas');

  function aiStatus(msg, ok) {
    ui.aiStatus.textContent = msg;
    ui.aiDot.classList.toggle('on', !!ok);
  }

  function aiInert(on) {
    for (const id of ['#s-brush', '#s-q', '#s-camo']) {
      $(id).closest('.slider').classList.toggle('inert', on);
    }
  }

  let settingsTimer = 0;
  function aiSendSettings() {
    clearTimeout(settingsTimer);
    settingsTimer = setTimeout(() => {
      if (ai.ws?.readyState === 1) {
        ai.ws.send(JSON.stringify({
          type: 'settings',
          prompt: ui.aiPrompt.value,
          strength: +ui.sStrength.value,
          structure: +ui.sStructure.value,
        }));
      }
    }, 300);
  }

  // src framebuffer → upright ≤512px JPEG → websocket
  function aiCapture() {
    const { iw, ih } = state;
    const buf = new Uint8Array(iw * ih * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.src.fbo);
    gl.readPixels(0, 0, iw, ih, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    capFull.width = iw;
    capFull.height = ih;
    capFull.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(buf.buffer), iw, ih), 0, 0);
    const s = 512 / Math.max(iw, ih);
    const cw = Math.max(8, Math.round(iw * s));
    const ch = Math.max(8, Math.round(ih * s));
    capSmall.width = cw;
    capSmall.height = ch;
    const ctx = capSmall.getContext('2d');
    ctx.save();
    ctx.scale(1, -1);                       // readPixels rows are bottom-up
    ctx.drawImage(capFull, 0, -ch, cw, ch);
    ctx.restore();
    capSmall.toBlob((blob) => {
      if (blob && ai.ws?.readyState === 1) ai.ws.send(blob);
      else ai.inflight = false;
    }, 'image/jpeg', 0.82);
  }

  function aiDisconnect(msg) {
    if (ai.ws) {
      ai.ws.onclose = ai.ws.onerror = null;
      try { ai.ws.close(); } catch { /* already closed */ }
    }
    ai.ws = null;
    ai.on = false;
    ai.ready = false;
    ai.inflight = false;
    dbg.ai.connected = false;
    state.justReset = true;                 // don't wet-blend against stale paint
    ui.bAi.textContent = 'Connect AI';
    aiInert(false);
    aiStatus(msg || 'offline', false);
  }

  function aiConnect() {
    const url = ui.aiUrl.value.trim() || 'ws://localhost:8765';
    aiStatus('connecting…', false);
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      return aiStatus(`bad address (${e.message})`, false);
    }
    ai.ws = ws;
    ws.onopen = () => {
      ai.on = true;
      dbg.ai.connected = true;
      ui.bAi.textContent = 'Disconnect AI';
      aiInert(true);
      const camoSlider = $('#s-camo');
      camoSlider.value = 0;                 // the model paints the pond now
      camoSlider.dispatchEvent(new Event('input'));
      toast('AI atelier connected — the diffusion model takes the easel.');
    };
    ws.onmessage = async (ev) => {
      if (typeof ev.data === 'string') {
        const d = JSON.parse(ev.data);
        if (d.type === 'hello') {
          aiStatus(`connected — ${d.backend} on ${d.device}`, true);
          aiSendSettings();
        }
        return;
      }
      ai.inflight = false;
      dbg.ai.roundtrips++;
      const bmp = await createImageBitmap(ev.data);
      if (ai.bitmap) ai.bitmap.close();
      ai.bitmap = bmp;                      // consumed by the render loop
    };
    ws.onclose = () => aiDisconnect('offline — start server/monet_server.py');
    ws.onerror = () => aiDisconnect('connection failed — is the server running?');
  }

  // upload the newest generated frame and ease it into the blended state
  function aiConsume(now) {
    const bmp = ai.bitmap;
    ai.bitmap = null;
    if (bmp.width !== ai.w || bmp.height !== ai.h) {
      ai.w = bmp.width;
      ai.h = bmp.height;
      for (const t of [ai.blendA, ai.blendB].filter(Boolean)) {
        gl.deleteTexture(t.tex);
        gl.deleteFramebuffer(t.fbo);
      }
      ai.blendA = makeTarget(ai.w, ai.h, gl.RGBA8);
      ai.blendB = makeTarget(ai.w, ai.h, gl.RGBA8);
      ai.ready = false;
    }
    tex(0, ai.texNew);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    bmp.close();
    ai.cur = 1 - ai.cur;
    const dst = ai.cur ? ai.blendB : ai.blendA;
    const old = ai.cur ? ai.blendA : ai.blendB;
    const u = pass(progBlend, dst);
    tex(0, ai.texNew);
    tex(1, old.tex);
    gl.uniform1i(u.uNew, 0);
    gl.uniform1i(u.uOld, 1);
    gl.uniform2f(u.uRes, ai.w, ai.h);
    gl.uniform1f(u.uMix, ai.ready ? Math.min(state.params.wet, 0.85) : 0);
    draw();
    ai.ready = true;
  }

  // ------------------------------------------------------------------ render
  const tex = (unit, texture) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  };

  function pass(prog, target) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, target ? target.w : ui.canvas.width, target ? target.h : ui.canvas.height);
    gl.useProgram(prog);
    return prog.u;
  }

  const draw = () => gl.drawArrays(gl.TRIANGLES, 0, 3);
  const t0 = performance.now();
  let lastT = -1;
  let fpsEma = 0;
  let lastFpsText = 0;

  function render(now) {
    requestAnimationFrame(render);
    if (lastT >= 0) {
      const dt = Math.max(now - lastT, 0.1);
      fpsEma = fpsEma ? fpsEma * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
    }
    lastT = now;
    dbg.fps = fpsEma;
    if (now - lastFpsText > 500) {
      lastFpsText = now;
      ui.fps.textContent = `${Math.round(fpsEma)} fps`;
    }

    const p = state.params;
    const painting = !state.paused;

    if (painting) {
      if (state.mode === 'camera' && video.readyState >= 2) {
        tex(0, videoTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
      }

      let u = pass(progSource, rt.src);
      tex(0, videoTex);
      gl.uniform1i(u.uVideo, 0);
      gl.uniform2f(u.uRes, state.iw, state.ih);
      gl.uniform1f(u.uTime, (now - t0) / 1000);
      gl.uniform1i(u.uMode, state.mode === 'camera' ? 0 : 1);
      gl.uniform1i(u.uMirror, state.mirror ? 1 : 0);
      draw();

      // the shader painter rests while the diffusion model has the easel
      if (!ai.on) {
        const cur = 1 - state.lastPaint;
        const paintCur = cur ? rt.paint1 : rt.paint0;
        const paintPrev = cur ? rt.paint0 : rt.paint1;

        u = pass(progTensor, rt.tensorA);
        tex(0, rt.src.tex);
        gl.uniform1i(u.uSrc, 0);
        gl.uniform2f(u.uRes, state.iw, state.ih);
        draw();

        u = pass(progBlur, rt.tensorB);
        tex(0, rt.tensorA.tex);
        gl.uniform1i(u.uTex, 0);
        gl.uniform2f(u.uRes, state.iw, state.ih);
        gl.uniform2f(u.uDir, 1, 0);
        draw();

        u = pass(progBlur, rt.tensorA);
        tex(0, rt.tensorB.tex);
        gl.uniform1i(u.uTex, 0);
        gl.uniform2f(u.uRes, state.iw, state.ih);
        gl.uniform2f(u.uDir, 0, 1);
        draw();

        u = pass(progKuwahara, paintCur);
        tex(0, rt.src.tex);
        tex(1, rt.tensorA.tex);
        tex(2, paintPrev.tex);
        gl.uniform1i(u.uSrc, 0);
        gl.uniform1i(u.uTensor, 1);
        gl.uniform1i(u.uPrev, 2);
        gl.uniform2f(u.uRes, state.iw, state.ih);
        gl.uniform1f(u.uRadius, p.brush);
        gl.uniform1f(u.uQ, p.q);
        gl.uniform1f(u.uWet, state.justReset ? 0 : p.wet);
        draw();

        state.lastPaint = cur;
        state.justReset = false;
      }
    }

    if (ai.on) {
      if (ai.bitmap) aiConsume(now);
      if (painting && !ai.inflight && ai.ws?.readyState === 1 && now - ai.lastSend > 80) {
        ai.inflight = true;
        ai.lastSend = now;
        aiCapture();
      }
    }

    const aiLive = ai.on && ai.ready;
    const paintTex = aiLive
      ? (ai.cur ? ai.blendB : ai.blendA).tex
      : (state.lastPaint ? rt.paint1 : rt.paint0).tex;
    const u = pass(progComposite, null);
    tex(0, paintTex);
    gl.uniform1i(u.uPaint, 0);
    gl.uniform2f(u.uRes, ui.canvas.width, ui.canvas.height);
    gl.uniform2f(u.uPaintRes, aiLive ? ai.w : state.iw, aiLive ? ai.h : state.ih);
    gl.uniform1f(u.uDream, p.dream);
    gl.uniform1f(u.uWarm, p.warm);
    gl.uniform1f(u.uWeave, p.weave);
    gl.uniform1f(u.uRelief, p.relief);
    gl.uniform1f(u.uCamo, p.camo);
    gl.uniform1f(u.uTime, (now - t0) / 1000);
    draw();

    if (dbg.requestSample) {
      // test harness probe: mean brightness of a patch at canvas centre,
      // read in-frame because the drawing buffer is not preserved
      dbg.requestSample = false;
      const n = 8;
      const px = new Uint8Array(n * n * 4);
      gl.readPixels(
        Math.max((ui.canvas.width - n) >> 1, 0), Math.max((ui.canvas.height - n) >> 1, 0),
        n, n, gl.RGBA, gl.UNSIGNED_BYTE, px,
      );
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
      dbg.sample = sum / (n * n * 3 * 255);
    }

    if (state.snap) {
      state.snap = false;
      ui.canvas.toBlob((blob) => {
        if (!blob) return toast('Could not export the painting.');
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `monet-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
        toast('Painting saved.');
      });
    }

    dbg.frames++;
  }

  // --------------------------------------------------------------------- UI
  const fmt = {
    camo: (v) => `${Math.round(v * 100)}%`,
    brush: (v) => `${(+v).toFixed(1)} px`,
    q: (v) => (+v).toFixed(1),
    dream: (v) => `${Math.round(v * 100)}%`,
    warm: (v) => `${Math.round(v * 100)}%`,
    weave: (v) => `${Math.round(v * 100)}%`,
    relief: (v) => `${Math.round(v * 100)}%`,
    wet: (v) => `${Math.round(v * 100)}%`,
  };

  for (const key of Object.keys(state.params)) {
    const input = $(`#s-${key}`);
    const output = $(`#o-${key}`);
    if (LOW && key === 'brush') input.value = 3;
    state.params[key] = +input.value;
    output.textContent = fmt[key](input.value);
    input.addEventListener('input', () => {
      state.params[key] = +input.value;
      output.textContent = fmt[key](input.value);
    });
  }

  ui.bFlip.addEventListener('click', () => {
    startCamera(state.facing === 'user' ? 'environment' : 'user');
  });

  ui.bPause.addEventListener('click', () => {
    state.paused = !state.paused;
    ui.bPause.textContent = state.paused ? 'Resume' : 'Pause';
    toast(state.paused ? 'Paused — grade & canvas sliders still apply.' : 'Painting resumed.');
  });

  ui.bSave.addEventListener('click', () => { state.snap = true; });

  ui.bDemo.addEventListener('click', () => {
    if (state.mode === 'camera') {
      useDemo();
      toast('Painting the demo garden.');
    } else {
      startCamera(state.facing);
    }
  });

  ui.bAi.addEventListener('click', () => {
    if (ai.ws || ai.on) aiDisconnect('offline');
    else aiConnect();
  });

  for (const [input, output] of [[ui.sStrength, ui.oStrength], [ui.sStructure, ui.oStructure]]) {
    output.textContent = `${Math.round(input.value * 100)}%`;
    input.addEventListener('input', () => {
      output.textContent = `${Math.round(input.value * 100)}%`;
      aiSendSettings();
    });
  }
  ui.aiPrompt.addEventListener('input', aiSendSettings);

  // -------------------------------------------------------------------- boot
  useDemo();                      // paint something immediately
  if (!FORCE_DEMO) startCamera('user');
  requestAnimationFrame(render);
})();
