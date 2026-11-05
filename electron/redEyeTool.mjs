/* ============================================================
   Red-eye correction math — used by:
     - the live WebGL preview in src/App.jsx (GLImage composites this
       CPU-side per texture upload, same as electron/stampTool.mjs)
     - the renderer's CPU bake paths (bakePreview/bakeCanvas/the
       histogram sampler in src/App.jsx)
     - the native export pipeline (bakeRaw in electron/main.cjs)

   .mjs under electron/ on purpose — see electron/colorAdjust.mjs's
   header for the rationale (ESM regardless of package.json, packaged
   by electron-builder's "electron/**" allowlist).

   Two steps, deliberately split:

   1. detectRedEye() runs ONCE, when the user drags a box around an eye
      (on whatever buffer the editor has at hand — the proxy). It finds
      the red pupil inside the box and fits an ellipse to it; that
      ellipse (center, per-axis spread, correlation) is what gets STORED,
      in original-image pixels. Re-detecting at every render instead
      would let the fit drift between the proxy, a deep-zoom tile and the
      full-res export (different pixels → slightly different blob), so
      the same edit would look different depending on where it was
      rendered.

      Detection: per-pixel "redness" = (R − max(G,B)) / R, thresholded
      relative to the box's own reddest pixel; then only the 8-connected
      component nearest the box's CENTER is kept (a flood fill), so a
      reddish patch of skin or a lip at the box's edge can't hijack the
      fit. The ellipse comes from that component's redness-weighted mean
      and covariance (second moments). For a filled disk of radius R the
      standard deviation along each axis is R/2, so "pupil edge" is at
      Mahalanobis distance 2.

   2. applyRedEyeToRGBBuffer() replays stored ellipses against any buffer.
      Weight per pixel = soft elliptical mask (logistic falloff around the
      pupil edge, scaled by the user's Size) × a redness gate (a pixel
      that isn't actually red — the white catchlight, the iris rim —
      stays as it is). Corrected color = a NEUTRAL dark gray built from
      the pixel's own green/blue light (the red channel is exactly what's
      contaminated by the flash reflection, so it can't be trusted for
      brightness), scaled down by Darken. Everything is blended in LINEAR
      light (IEC 61966-2-1 sRGB curve, same as colorAdjust.mjs) so the
      soft edge falls off the way a physical blend of two lights would.

   Entry shape (stored, original-image px):
     { x, y, sx, sy, rho, size, darken }
       x, y    — pupil center
       sx, sy  — per-axis standard deviation of the detected pupil
       rho     — x/y correlation (−1..1), lets the ellipse tilt
       size    — user multiplier on the pupil radius (default 1)
       darken  — 0..1, how dark the corrected pupil gets (default 0.5)
   ============================================================ */

export const REDEYE_DEFAULT_SIZE = 1;
export const REDEYE_DEFAULT_DARKEN = 0.5;
const REDEYE_MIN_PIXELS = 4;         // a "pupil" smaller than this is noise, not an eye
// Redness scale for reference: skin ≈ 0.2, a brown iris ≈ 0.3, a lip ≈
// 0.6, a flash-lit red pupil ≈ 0.7-0.85.
const REDEYE_MIN_REDNESS = 0.4;      // box's reddest pixel must be at least this red to count as a red eye at all
const REDEYE_THRESHOLD_FRAC = 0.45;  // pixel joins the blob if its redness ≥ this × the box's peak redness
const REDEYE_GATE_LO = 0.35;         // redness gate: below this a pixel is left alone (skin, brown iris)...
const REDEYE_GATE_HI = 0.6;          // ...above this it's fully corrected
const REDEYE_EDGE_CENTER = 1.35;     // mask is 50% at this × pupil radius — deliberately PAST the fitted edge:
                                     // the fit's own edge pixels are half-red/half-iris and still need fixing,
                                     // and the redness gate (not the mask) is what protects the iris
const REDEYE_EDGE_SHARPNESS = 12;    // logistic slope of the mask around that point (higher = harder edge);
                                     // with the center above: ≈0.985 at the fitted edge, ≈0.0004 at the reach
const REDEYE_MASK_REACH = 2.0;       // mask evaluated out to this × pupil radius, zero beyond
const REDEYE_RHO_LIMIT = 0.95;       // keeps the ellipse's (1 − rho²) denominator away from 0

export function isEmptyRedEye(list) {
  return !list || list.length === 0;
}

/* Drops anything malformed and fills defaults, so every read site can
   trust the shape (see feedback on persisted-data migration: normalize at
   every read). Number.isFinite rather than truthiness — 0 is a valid
   darken. */
export function normalizeRedEye(list) {
  if (!Array.isArray(list)) return [];
  const out = [];
  for (const e of list) {
    if (!e || !Number.isFinite(e.x) || !Number.isFinite(e.y) || !(e.sx > 0) || !(e.sy > 0)) continue;
    out.push({
      x: e.x, y: e.y, sx: e.sx, sy: e.sy,
      rho: Number.isFinite(e.rho) ? Math.max(-REDEYE_RHO_LIMIT, Math.min(REDEYE_RHO_LIMIT, e.rho)) : 0,
      size: Number.isFinite(e.size) && e.size > 0 ? e.size : REDEYE_DEFAULT_SIZE,
      darken: Number.isFinite(e.darken) ? Math.max(0, Math.min(1, e.darken)) : REDEYE_DEFAULT_DARKEN,
    });
  }
  return out;
}

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow(Math.max(0, (c + 0.055) / 1.055), 2.4);
}
function linearToSrgb(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(0, c), 1 / 2.4) - 0.055;
}
const SRGB_TO_LINEAR_LUT = (() => {
  const t = new Float64Array(256);
  for (let i = 0; i < 256; i++) t[i] = srgbToLinear(i / 255);
  return t;
})();

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function redness(r, g, b) {
  return r > 0 ? Math.max(0, (r - Math.max(g, b)) / r) : 0;
}

/* Finds the red pupil inside box {x0, y0, x1, y1} (buffer px, inclusive-
   exclusive, clamped to the buffer). Returns { x, y, sx, sy, rho } in
   buffer px, or null when there's no convincing red blob — the caller
   tells the user "no red eye found" rather than guessing. */
export function detectRedEye(buf, width, height, channels, box) {
  const x0 = Math.max(0, Math.floor(Math.min(box.x0, box.x1)));
  const y0 = Math.max(0, Math.floor(Math.min(box.y0, box.y1)));
  const x1 = Math.min(width, Math.ceil(Math.max(box.x0, box.x1)));
  const y1 = Math.min(height, Math.ceil(Math.max(box.y0, box.y1)));
  const bw = x1 - x0, bh = y1 - y0;
  if (bw < 3 || bh < 3) return null;

  const red = new Float32Array(bw * bh);
  let peak = 0;
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const o = ((y + y0) * width + x + x0) * channels;
      // very dark pixels have a meaningless redness ratio (tiny R over
      // tinier G/B) — require some actual red light before it counts
      const v = buf[o] < 40 ? 0 : redness(buf[o], buf[o + 1], buf[o + 2]);
      red[y * bw + x] = v;
      if (v > peak) peak = v;
    }
  }
  if (peak < REDEYE_MIN_REDNESS) return null;
  const thr = peak * REDEYE_THRESHOLD_FRAC;

  // seed = the above-threshold pixel closest to the box center, so the
  // fill starts in the eye the user boxed rather than at whatever red
  // thing happens to be strongest (a lip, a skin blotch at the edge)
  const cx = (bw - 1) / 2, cy = (bh - 1) / 2;
  let seed = -1, bestD = Infinity;
  for (let i = 0; i < red.length; i++) {
    if (red[i] < thr) continue;
    const dx = (i % bw) - cx, dy = Math.floor(i / bw) - cy;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; seed = i; }
  }
  if (seed < 0) return null;

  const seen = new Uint8Array(red.length);
  const stack = [seed];
  seen[seed] = 1;
  let sw = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, count = 0;
  while (stack.length) {
    const i = stack.pop();
    const px = i % bw, py = (i - px) / bw;
    const w = red[i];
    sw += w; sx += w * px; sy += w * py;
    sxx += w * px * px; syy += w * py * py; sxy += w * px * py;
    count++;
    for (let ny = py - 1; ny <= py + 1; ny++) {
      if (ny < 0 || ny >= bh) continue;
      for (let nx = px - 1; nx <= px + 1; nx++) {
        if (nx < 0 || nx >= bw) continue;
        const j = ny * bw + nx;
        if (seen[j] || red[j] < thr) continue;
        seen[j] = 1;
        stack.push(j);
      }
    }
  }
  if (count < REDEYE_MIN_PIXELS || sw <= 0) return null;

  const mx = sx / sw, my = sy / sw;
  // +1/12: variance of a single uniformly-covered pixel cell — keeps a
  // 1-2 px wide blob from collapsing to a zero-width ellipse
  const vx = Math.max(sxx / sw - mx * mx, 0) + 1 / 12;
  const vy = Math.max(syy / sw - my * my, 0) + 1 / 12;
  const cxy = sxy / sw - mx * my;
  const rho = Math.max(-REDEYE_RHO_LIMIT, Math.min(REDEYE_RHO_LIMIT, cxy / Math.sqrt(vx * vy)));
  return { x: mx + x0, y: my + y0, sx: Math.sqrt(vx), sy: Math.sqrt(vy), rho };
}

function applyOne(buf, width, height, channels, e) {
  const size = e.size, rho = e.rho;
  const sx = e.sx * size, sy = e.sy * size;
  const inv = 1 / (1 - rho * rho);
  // pupil edge at Mahalanobis distance 2 (see header) → u = 1 there
  const reach = 2 * REDEYE_MASK_REACH;
  const ex = Math.ceil(sx * reach), ey = Math.ceil(sy * reach);
  const bx0 = Math.max(0, Math.floor(e.x) - ex), bx1 = Math.min(width - 1, Math.ceil(e.x) + ex);
  const by0 = Math.max(0, Math.floor(e.y) - ey), by1 = Math.min(height - 1, Math.ceil(e.y) + ey);
  if (bx1 < bx0 || by1 < by0) return;
  const k = 1 - 0.8 * e.darken;
  for (let y = by0; y <= by1; y++) {
    const dy = (y - e.y) / sy;
    for (let x = bx0; x <= bx1; x++) {
      const dx = (x - e.x) / sx;
      const u = Math.sqrt((dx * dx - 2 * rho * dx * dy + dy * dy) * inv) / 2;
      if (u > REDEYE_MASK_REACH) continue;
      const m = 1 / (1 + Math.exp(REDEYE_EDGE_SHARPNESS * (u - REDEYE_EDGE_CENTER)));
      const o = (y * width + x) * channels;
      const r = buf[o], g = buf[o + 1], b = buf[o + 2];
      const w = m * smoothstep(REDEYE_GATE_LO, REDEYE_GATE_HI, redness(r, g, b));
      if (w <= 0.001) continue;
      const lr = SRGB_TO_LINEAR_LUT[r], lg = SRGB_TO_LINEAR_LUT[g], lb = SRGB_TO_LINEAR_LUT[b];
      const t = 0.5 * (lg + lb) * k;
      buf[o] = Math.round(linearToSrgb(lr + (t - lr) * w) * 255);
      buf[o + 1] = Math.round(linearToSrgb(lg + (t - lg) * w) * 255);
      buf[o + 2] = Math.round(linearToSrgb(lb + (t - lb) * w) * 255);
    }
  }
}

/* Maps stored entries (original-image px) into another buffer's pixel
   space: `project(x, y)` → [x', y'] for the center, plus the rotation
   (radians, same sense as `project`'s own rotation) and scale that space
   applies. The ellipse's shape goes through the matching covariance
   transform Σ' = s²·R·Σ·Rᵀ, so a rotated crop rotates the ellipse with it
   instead of keeping it axis-aligned. size/darken are dimensionless and
   pass through. */
export function mapRedEye(list, project, theta, s) {
  if (isEmptyRedEye(list)) return [];
  const cs = Math.cos(theta), sn = Math.sin(theta);
  return normalizeRedEye(list).map((e) => {
    const a = e.sx * e.sx, b = e.sy * e.sy, c = e.rho * e.sx * e.sy;
    const xx = s * s * (a * cs * cs - 2 * c * sn * cs + b * sn * sn);
    const yy = s * s * (a * sn * sn + 2 * c * sn * cs + b * cs * cs);
    const xy = s * s * ((a - b) * sn * cs + c * (cs * cs - sn * sn));
    const [x, y] = project(e.x, e.y);
    const sx = Math.sqrt(xx), sy = Math.sqrt(yy);
    return { ...e, x, y, sx, sy, rho: xy / (sx * sy) };
  });
}

/* Mutates buf in place (Node Buffer or Uint8ClampedArray) and returns it.
   `list` entries must already be in THIS buffer's pixel space (callers map
   from original-image px, like stamp strokes). No-op on an empty list. */
export function applyRedEyeToRGBBuffer(buf, width, height, channels, list) {
  if (isEmptyRedEye(list)) return buf;
  for (const e of normalizeRedEye(list)) applyOne(buf, width, height, channels, e);
  return buf;
}
