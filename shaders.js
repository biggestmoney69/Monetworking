/* GLSL sources for the Monetworking paint pipeline.
 *
 * Pipeline per frame (all fullscreen passes):
 *   1. source    — camera frame (or procedural demo garden) into the working buffer
 *   2. tensor    — structure tensor (E, G, F) from Sobel gradients
 *   3. blur ×2   — separable gaussian on the tensor → smooth local flow field
 *   4. kuwahara  — anisotropic Kuwahara filter with polynomial sector weights
 *                  (Kyprianidis, Kang & Döllner), elliptical kernels aligned to flow
 *   5. composite — Monet color grade, impasto relief, canvas weave, vignette
 *
 * Fragment bodies omit "#version"/precision; app.js prepends those plus defines.
 * PACK_TENSOR is defined when float render targets are unavailable and the
 * tensor must be squeezed into RGBA8.
 */
'use strict';

const SHADERS = {};

SHADERS.vert = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

SHADERS.source = `
uniform sampler2D uVideo;
uniform vec2  uRes;
uniform float uTime;
uniform int   uMode;    // 0 = camera, 1 = demo garden
uniform int   uMirror;
out vec4 o;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i),                 hash(i + vec2(1, 0)), u.x),
             mix(hash(i + vec2(0, 1)),    hash(i + vec2(1, 1)), u.x), u.y);
}

const float HORIZON = 0.46;
const vec2  SUN    = vec2(0.66, 0.70);

vec3 sky(vec2 uv, float t) {
  float h = clamp((uv.y - HORIZON) / (1.0 - HORIZON), 0.0, 1.0);
  vec3 c = mix(vec3(0.99, 0.80, 0.52), vec3(0.83, 0.63, 0.55), smoothstep(0.0, 0.45, h));
  c = mix(c, vec3(0.38, 0.46, 0.66), smoothstep(0.35, 1.0, h));
  float aspect = uRes.x / uRes.y;
  float d = distance(uv * vec2(aspect, 1.0), SUN * vec2(aspect, 1.0));
  c += vec3(1.00, 0.85, 0.60) * exp(-d * d * 180.0) * 0.90;
  c += vec3(1.00, 0.70, 0.45) * exp(-d * d * 9.0) * 0.35;
  float n = 0.6 * noise(vec2(uv.x * 5.0 + t * 0.015, uv.y * 16.0))
          + 0.4 * noise(vec2(uv.x * 11.0 - t * 0.010, uv.y * 30.0));
  float cloud = smoothstep(0.52, 0.78, n) * smoothstep(1.0, 0.35, h);
  return mix(c, vec3(0.93, 0.74, 0.70), cloud * 0.45);
}

vec3 garden(vec2 uv, float t) {
  if (uv.y > HORIZON) return sky(uv, t);

  float depth = (HORIZON - uv.y) / HORIZON;            // 0 at horizon → 1 at bottom
  float rip = 0.5 * sin(uv.x * 48.0 + t * 0.9 + depth * 22.0)
            + 0.5 * sin(uv.x * 23.0 - t * 0.6 + depth * 40.0);
  vec2 ruv = vec2(uv.x + rip * 0.006 * (0.3 + depth),
                  HORIZON + (HORIZON - uv.y) * 0.85 + rip * 0.012 * depth);
  vec3 c = mix(sky(ruv, t), vec3(0.13, 0.27, 0.33), 0.25 + 0.45 * depth);

  // column of sun glitter
  float gx = exp(-pow((uv.x - SUN.x) / (0.05 + depth * 0.16), 2.0));
  float sparkle = smoothstep(0.55, 1.0, 0.5 + 0.5 * sin(uv.y * 240.0 + rip * 3.0 + t * 2.2));
  c += vec3(1.00, 0.75, 0.45) * gx * sparkle * (0.55 - 0.25 * depth);

  // lily pads: (x, drop below horizon, radius)
  vec3 pads[5] = vec3[5](vec3(0.22, 0.18, 0.090), vec3(0.40, 0.10, 0.062),
                         vec3(0.62, 0.16, 0.078), vec3(0.80, 0.085, 0.050),
                         vec3(0.31, 0.055, 0.045));
  for (int i = 0; i < 5; i++) {
    float fi = float(i);
    vec2 pc = vec2(pads[i].x + 0.004 * sin(t * 0.4 + fi * 1.7),
                   HORIZON - pads[i].y + 0.003 * cos(t * 0.5 + fi * 2.1));
    vec2 d = (uv - pc) / vec2(pads[i].z, pads[i].z * 0.38);
    float m = smoothstep(1.0, 0.82, dot(d, d));
    float shade = clamp(0.55 + d.x * 0.45 + d.y * 0.18 + 0.15 * hash(pc * 40.0 + fi), 0.15, 1.0);
    c = mix(c, mix(vec3(0.13, 0.30, 0.19), vec3(0.44, 0.60, 0.28), shade), m);
  }
  // one blossom
  vec2 bp = uv - vec2(0.40 + 0.004 * sin(t * 0.4 + 1.7), HORIZON - 0.094);
  float bd = dot(bp, bp);
  c = mix(c, vec3(0.95, 0.62, 0.70), smoothstep(0.00028, 0.00012, bd));
  c += vec3(1.0, 0.9, 0.95) * exp(-bd * 24000.0) * 0.55;
  return c;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  if (uMirror == 1) uv.x = 1.0 - uv.x;
  if (uMode == 0) {
    o = vec4(texture(uVideo, vec2(uv.x, 1.0 - uv.y)).rgb, 1.0);  // video rows are top-down
  } else {
    o = vec4(garden(uv, uTime), 1.0);
  }
}`;

SHADERS.tensor = `
uniform sampler2D uSrc;
uniform vec2 uRes;
out vec4 o;

vec3 pack(vec3 t) {
#ifdef PACK_TENSOR
  return vec3(clamp(t.xy * 0.25, 0.0, 1.0), clamp(t.z * 0.25 + 0.5, 0.0, 1.0));
#else
  return t;
#endif
}

void main() {
  vec2 px = 1.0 / uRes;
  vec2 uv = gl_FragCoord.xy * px;
  vec3 tl = texture(uSrc, uv + px * vec2(-1,  1)).rgb;
  vec3 tc = texture(uSrc, uv + px * vec2( 0,  1)).rgb;
  vec3 tr = texture(uSrc, uv + px * vec2( 1,  1)).rgb;
  vec3 ml = texture(uSrc, uv + px * vec2(-1,  0)).rgb;
  vec3 mr = texture(uSrc, uv + px * vec2( 1,  0)).rgb;
  vec3 bl = texture(uSrc, uv + px * vec2(-1, -1)).rgb;
  vec3 bc = texture(uSrc, uv + px * vec2( 0, -1)).rgb;
  vec3 br = texture(uSrc, uv + px * vec2( 1, -1)).rgb;
  vec3 gx = (tr + 2.0 * mr + br - tl - 2.0 * ml - bl) * 0.25;
  vec3 gy = (tl + 2.0 * tc + tr - bl - 2.0 * bc - br) * 0.25;
  o = vec4(pack(vec3(dot(gx, gx), dot(gy, gy), dot(gx, gy))), 1.0);
}`;

// Packing is affine, so blurring packed values equals packing blurred values.
SHADERS.blur = `
uniform sampler2D uTex;
uniform vec2 uRes;
uniform vec2 uDir;
out vec4 o;

void main() {
  vec2 px = 1.0 / uRes;
  vec2 uv = gl_FragCoord.xy * px;
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int i = -4; i <= 4; i++) {
    float w = exp(-float(i * i) / 8.0);     // sigma = 2
    sum += w * texture(uTex, uv + uDir * px * float(i)).rgb;
    wsum += w;
  }
  o = vec4(sum / wsum, 1.0);
}`;

SHADERS.kuwahara = `
uniform sampler2D uSrc;
uniform sampler2D uTensor;
uniform sampler2D uPrev;
uniform vec2  uRes;
uniform float uRadius;   // brush size in px
uniform float uQ;        // sector sharpness
uniform float uWet;      // temporal blend with previous frame
out vec4 o;

vec3 unpack(vec3 p) {
#ifdef PACK_TENSOR
  return vec3(p.xy * 4.0, (p.z - 0.5) * 4.0);
#else
  return p;
#endif
}

void main() {
  vec2 px = 1.0 / uRes;
  vec2 uv = gl_FragCoord.xy * px;

  // local orientation + anisotropy from the smoothed structure tensor
  vec3 t = unpack(texture(uTensor, uv).rgb);
  float E = t.x, G = t.y, F = t.z;
  float D = sqrt(max((E - G) * (E - G) + 4.0 * F * F, 0.0));
  float lambda1 = 0.5 * (E + G + D);
  float lambda2 = 0.5 * (E + G - D);
  vec2 dir = vec2(lambda1 - E, -F);
  dir = (dot(dir, dir) > 1e-10) ? normalize(dir) : vec2(0.0, 1.0);
  float phi = atan(dir.y, dir.x);
  float A = (lambda1 + lambda2 > 1e-8) ? (lambda1 - lambda2) / (lambda1 + lambda2) : 0.0;

  // elliptical kernel stretched along the flow direction
  float r = max(uRadius, 1.0);
  float a = r * clamp(1.0 + A, 1.0, 1.8);
  float b = r * clamp(1.0 / (1.0 + A), 0.55, 1.0);
  float cp = cos(phi), sp = sin(phi);
  mat2 SR = mat2(0.5 / a, 0.0, 0.0, 0.5 / b) * mat2(cp, -sp, sp, cp);
  int maxX = int(sqrt(a * a * cp * cp + b * b * sp * sp)) + 1;
  int maxY = int(sqrt(a * a * sp * sp + b * b * cp * cp)) + 1;

  // polynomial sector weights (8 sectors)
  float zeta = clamp(2.0 / r, 0.18, 1.0);
  float zc = 0.58;
  float eta = (zeta + cos(zc)) / (sin(zc) * sin(zc));

  vec4 m[8];
  vec3 s[8];
  for (int k = 0; k < 8; k++) { m[k] = vec4(0.0); s[k] = vec3(0.0); }

  for (int j = -maxY; j <= maxY; j++) {
    for (int i = -maxX; i <= maxX; i++) {
      vec2 v = SR * vec2(float(i), float(j));
      float vv = dot(v, v);
      if (vv > 0.25) continue;
      vec3 c = texture(uSrc, uv + vec2(float(i), float(j)) * px).rgb;
      float w[8];
      float z, vxx, vyy, sum = 0.0;
      vxx = zeta - eta * v.x * v.x;
      vyy = zeta - eta * v.y * v.y;
      z = max(0.0,  v.y + vxx); w[0] = z * z; sum += w[0];
      z = max(0.0, -v.x + vyy); w[2] = z * z; sum += w[2];
      z = max(0.0, -v.y + vxx); w[4] = z * z; sum += w[4];
      z = max(0.0,  v.x + vyy); w[6] = z * z; sum += w[6];
      vec2 vr = 0.7071068 * vec2(v.x - v.y, v.x + v.y);
      vxx = zeta - eta * vr.x * vr.x;
      vyy = zeta - eta * vr.y * vr.y;
      z = max(0.0,  vr.y + vxx); w[1] = z * z; sum += w[1];
      z = max(0.0, -vr.x + vyy); w[3] = z * z; sum += w[3];
      z = max(0.0, -vr.y + vxx); w[5] = z * z; sum += w[5];
      z = max(0.0,  vr.x + vyy); w[7] = z * z; sum += w[7];
      float g = exp(-3.125 * vv) / max(sum, 1e-5);
      for (int k = 0; k < 8; k++) {
        float wk = w[k] * g;
        m[k] += vec4(c * wk, wk);
        s[k] += c * c * wk;
      }
    }
  }

  // blend sector means, favouring low-variance (homogeneous) sectors
  vec4 acc = vec4(0.0);
  for (int k = 0; k < 8; k++) {
    float wk = max(m[k].w, 1e-6);
    vec3 mean = m[k].rgb / wk;
    vec3 v2 = abs(s[k] / wk - mean * mean);
    float sigma2 = v2.r + v2.g + v2.b;
    float wf = 1.0 / (1.0 + pow(1000.0 * sigma2, 0.5 * uQ));
    acc += vec4(mean * wf, wf);
  }
  vec3 paint = acc.rgb / max(acc.a, 1e-6);
  o = vec4(mix(paint, texture(uPrev, uv).rgb, uWet), 1.0);
}`;

// Temporal blend for AI-generated frames: diffusion output flickers between
// frames, so each new result is eased into the previous blended state.
SHADERS.blend = `
uniform sampler2D uNew;
uniform sampler2D uOld;
uniform vec2  uRes;
uniform float uMix;   // how much of the old image survives
out vec4 o;

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec3 a = texture(uNew, vec2(uv.x, 1.0 - uv.y)).rgb;   // image rows are top-down
  vec3 b = texture(uOld, uv).rgb;
  o = vec4(mix(a, b, uMix), 1.0);
}`;

SHADERS.composite = `
uniform sampler2D uPaint;
uniform vec2  uRes;       // output resolution
uniform vec2  uPaintRes;
uniform float uDream;     // pastel Monet grade amount
uniform float uWarm;      // 0 cool … 1 warm (0.5 neutral)
uniform float uWeave;     // canvas texture strength
uniform float uRelief;    // impasto strength
uniform float uCamo;      // 0 plein-air … 1 hidden in the lily pond
uniform float uTime;
out vec4 o;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
vec2 hash2(vec2 p) {
  return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453123);
}
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i),              hash(i + vec2(1, 0)), u.x),
             mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

// plain-weave height field, q in thread units
float cloth(vec2 q) {
  vec2 cell = floor(q);
  vec2 f = fract(q);
  float over = mod(cell.x + cell.y, 2.0);
  float hx = sin(3.14159 * f.x) * (0.85 + 0.30 * hash(vec2(cell.x, 7.0)));
  float hy = sin(3.14159 * f.y) * (0.85 + 0.30 * hash(vec2(13.0, cell.y)));
  float h = mix(max(hx, hy * 0.92), max(hy, hx * 0.92), over);
  return h + (hash(q * 37.0) - 0.5) * 0.18;   // stray fibres
}

/* ---- lily-pond camouflage -------------------------------------------------
 * Semantic camouflage: the source's tonal structure is preserved but its
 * materials are re-painted as Monet's pond — dark water in the shadows,
 * lily-pad clusters in the midtones, sky reflections in the lights.
 */

// luminance → pond color, monotonic so the hidden image still reads
vec3 pondPalette(float l) {
  l = clamp(l, 0.0, 1.0);
  vec3 c = mix(vec3(0.09, 0.13, 0.22), vec3(0.16, 0.25, 0.46), smoothstep(0.0, 0.25, l));
  c = mix(c, vec3(0.42, 0.46, 0.70), smoothstep(0.22, 0.48, l));   // blue-violet
  c = mix(c, vec3(0.36, 0.54, 0.43), smoothstep(0.42, 0.64, l));   // pond green
  c = mix(c, vec3(0.63, 0.71, 0.56), smoothstep(0.60, 0.82, l));   // pale willow
  c = mix(c, vec3(0.86, 0.89, 0.83), smoothstep(0.80, 0.98, l));   // cool cream light
  return c;
}

// one grid layer of lily pads; returns (color, coverage).
// Pads carry the image: each inherits the painting's luminance at its centre,
// so the pad field itself forms the hidden picture (like canopy masses
// forming a car) while open water survives only in the deepest shadows.
vec4 lilyLayer(vec2 puv, float scale, float seed, float t, vec2 invAspect) {
  vec2 q = puv * scale;
  vec2 cell = floor(q);
  vec4 best = vec4(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 c = cell + vec2(i, j);
      vec2 rnd = hash2(c + seed);
      vec2 center = c + 0.5 + (rnd - 0.5) * 0.9
                  + 0.02 * vec2(sin(t * 0.5 + rnd.x * 6.28), cos(t * 0.4 + rnd.y * 6.28));
      float r = 0.40 + 0.34 * fract(rnd.x * 7.31);
      vec2 d = (q - center) / r;
      d.y *= 1.9;                                  // pads seen at an angle
      float e = dot(d, d);
      if (e > 1.0) continue;

      float lc = luma(textureLod(uPaint, (center / scale) * invAspect, 0.0).rgb);
      float gate = smoothstep(0.05, 0.16, lc);                     // deep shadow stays water
      gate *= 1.0 - smoothstep(0.86, 0.97, lc) * 0.75;             // sparkle shows in lights
      gate *= step(rnd.y, 0.96);                                   // a few cells empty
      gate *= 0.55 + 0.45 * smoothstep(0.30, 0.62, noise(c * 0.33 + seed)); // clustering
      if (gate <= 0.0) continue;

      // green strongest in the midtones; darks keep violet, lights keep cream
      float greenAmt = 0.55 * (1.0 - min(abs(lc - 0.45) * 2.4, 1.0)) + 0.12;
      vec3 pc = mix(pondPalette(lc), vec3(0.30, 0.47, 0.31), greenAmt);
      pc *= 0.84 + 0.34 * fract(rnd.y * 9.17);                     // broken color
      // baked impasto rim: lit top-left edge, shaded lower-right
      float rim = smoothstep(0.45, 1.0, e);
      float lit = clamp(dot(normalize(d), normalize(vec2(-0.6, 0.8))), 0.0, 1.0);
      pc += rim * lit * 0.20;
      pc *= 1.0 - rim * (1.0 - lit) * 0.22;
      // an occasional blossom on bright pads
      if (fract(rnd.x * 13.77) > 0.80 && lc > 0.45) {
        vec2 bd = (q - center - vec2(0.0, -0.08 * r)) / (r * 0.38);
        bd.y *= 1.6;
        float bm = 1.0 - smoothstep(0.45, 0.85, dot(bd, bd));
        vec3 bc = mix(vec3(0.88, 0.56, 0.66), vec3(0.96, 0.93, 0.86), fract(rnd.y * 5.13));
        pc = mix(pc, bc, bm);
        pc += bm * lit * 0.12;
      }
      float mask = (1.0 - smoothstep(0.74, 0.98, e)) * gate;
      if (mask > best.a) best = vec4(pc, mask);
    }
  }
  return best;
}

vec3 pond(vec2 uv, vec3 paintCol, float t) {
  float l = luma(paintCol);
  vec2 aspect = vec2(uPaintRes.x / uPaintRes.y, 1.0);
  vec2 invAspect = vec2(uPaintRes.y / uPaintRes.x, 1.0);
  vec2 puv = uv * aspect;

  // water: palette over rippled luminance, with horizontal reflection dabs
  float rip = noise(vec2(puv.x * 9.0 + t * 0.05, puv.y * 70.0));
  vec3 col = pondPalette(l * 0.94 + (rip - 0.5) * 0.14);
  col = mix(col, paintCol, 0.10);                  // a whisper of the real colors
  float streak = smoothstep(0.62, 0.78, rip) - smoothstep(0.78, 0.92, rip);
  col = mix(col, vec3(0.85, 0.87, 0.82), streak * 0.35 * smoothstep(0.25, 0.6, l));
  float shadowStreak = 1.0 - smoothstep(0.10, 0.30, rip);
  col = mix(col, vec3(0.10, 0.15, 0.26), shadowStreak * 0.30 * (1.0 - smoothstep(0.5, 0.8, l)));

  vec4 pads = lilyLayer(puv, 6.0, 0.0, t, invAspect);
  col = mix(col, pads.rgb, pads.a);
  vec4 small = lilyLayer(puv, 11.0, 7.7, t, invAspect);
  col = mix(col, small.rgb, small.a * 0.92);
  return col;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec3 raw = texture(uPaint, uv).rgb;
  vec3 col = raw;

  // -- Monet grade: cool lifted shadows, warm creamy lights, gentle pastel
  float l = luma(col);
  vec3 tinted = col * mix(vec3(0.52, 0.56, 0.78), vec3(1.06, 1.00, 0.88), smoothstep(0.05, 0.90, l));
  tinted = mix(tinted, vec3(luma(tinted)), 0.18);
  tinted += (vec3(0.30, 0.31, 0.42) - tinted) * 0.22 * (1.0 - smoothstep(0.0, 0.35, l));
  col = mix(col, tinted, uDream);
  col *= mix(vec3(0.93, 0.98, 1.07), vec3(1.07, 1.00, 0.90), uWarm);

  // -- dissolve into the lily pond
  if (uCamo > 0.001) col = mix(col, pond(uv, raw, uTime), uCamo);

  // -- impasto: treat paint luminance as thickness, light it.
  // Central differences + a saturating curve keep hard edges from turning
  // into embossed halos; Monet kept his contours soft.
  vec2 ppx = 1.0 / uPaintRes;
  float hL = luma(texture(uPaint, uv - vec2(ppx.x, 0.0)).rgb);
  float hR = luma(texture(uPaint, uv + vec2(ppx.x, 0.0)).rgb);
  float hD = luma(texture(uPaint, uv - vec2(0.0, ppx.y)).rgb);
  float hU = luma(texture(uPaint, uv + vec2(0.0, ppx.y)).rgb);
  vec2 pg = vec2(hR - hL, hU - hD) * 0.5;
  pg /= 1.0 + 9.0 * abs(pg);

  // -- canvas weave height + slope (same thread size at any window size)
  vec2 q = gl_FragCoord.xy * (190.0 / uRes.y);
  float h0 = cloth(q);
  vec2 cg = vec2(cloth(q + vec2(0.08, 0.0)) - h0, cloth(q + vec2(0.0, 0.08)) - h0) / 0.08;

  vec3 n = normalize(vec3(-(pg * uRelief * 14.0 + cg * uWeave * 0.10), 1.0));
  vec3 L = normalize(vec3(-0.35, 0.55, 0.76));
  float shade = clamp(1.0 + (dot(n, L) - L.z) * 1.1, 0.55, 1.4);
  float spec = pow(clamp(dot(n, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0, 1.0), 24.0)
             * 0.18 * (uRelief * 0.7 + uWeave * 0.5);

  col *= 1.0 + uWeave * 0.16 * (h0 - 0.55);   // threads catch a bit of pigment
  col = col * shade + spec;

  vec2 vc = uv - 0.5;
  col *= 1.0 - dot(vc, vc) * 0.45;            // soft vignette
  o = vec4(col, 1.0);
}`;
