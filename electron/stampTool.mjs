/* ============================================================
   Clone stamp math — used by:
     - the live WebGL preview in src/App.jsx (GLImage composites this
       CPU-side, per texture upload, rather than reimplementing it in
       the shader — see GLImage's texture-upload effect)
     - the renderer's CPU bake paths (bakePreview/bakeCanvas in
       src/App.jsx, thumbnails + browser-fallback export)
     - the native export pipeline (bakeRaw in electron/main.cjs)

   .mjs on purpose — see electron/colorAdjust.mjs's header for the
   full rationale (Node always treats this as ESM regardless of
   package.json's missing "type": "module", so electron/main.cjs
   can `await import()` it while Vite bundles it for the renderer;
   lives under electron/ because electron-builder only packages
   "electron/**").

   Algorithm: a RAW 1:1 pixel copy along a swept "capsule" (a brush
   stroke's own polyline, thickened by radius with round ends) — the
   same thing real Photoshop's Clone Stamp does: every destination
   pixel gets the literal source pixel at a fixed offset (dx, dy),
   with NO automatic color/tone correction of any kind. `stamp` is an
   array of brush STROKES — { points: [{x,y,dx?,dy?}, ...], radius,
   dx, dy, feather, opacity } — a stroke is a path of brush POINTS
   sharing one radius, one rigid nominal source offset (dx,dy = source
   − destination, same as holding Alt/Option and clicking a source
   point in Photoshop), one feather amount (edge falloff width as a
   fraction of radius — Photoshop's brush "hardness", inverted) and
   one opacity (0..1, Photoshop's "Opacity"/"Flow"). A point's own
   dx/dy, when present, overrides the stroke's for that one point only
   — set once at paint time (StampOverlay's self-overlap search: a
   curved stroke's rigid offset can otherwise sample from a
   destination this same stroke already painted a moment earlier, an
   obviously-repeating seam), then replayed identically forever after
   like any other stored geometry. A plain circular dab is just a
   one-point stroke.

   This module previously did automatic destination-side color/tone
   matching (median ring sampling, then a fitted linear gradient —
   see feedback_stamp_planefit.md and feedback_ring_sampling_median.md
   in project memory for that whole history). It was removed
   deliberately, on explicit direction: real Photoshop's Clone Stamp
   does not do this either — it's a raw memory copy of RGB values,
   and avoiding a visible seam (matching source tone to the
   surrounding destination) is the USER's job — pick a well-matched
   source point, lower opacity, multiple passes — not the tool's. If
   a future session is tempted to reintroduce automatic tone
   correction, re-read that decision first rather than rebuilding it
   from scratch.

   Every destination PIXEL within a stroke's footprint is touched
   EXACTLY ONCE: for each pixel we find the closest point on the
   stroke's own polyline (iterating its line segments, not its stored
   points — the distance is to the nearest point on any segment, which
   is what actually makes the covered area a smooth capsule instead of
   a chain of separate circles), linearly interpolate that segment's
   two endpoints' source offset by the projection parameter, and copy
   from there. A chain of independently-blended circular dabs is
   deliberately NOT done, because it reads as a real, visible artifact:
   the covered silhouette scallops (bulges at each dab center, pinches
   between them) instead of reading as one smooth brush stroke, and a
   pixel inside more than one dab's radius — the normal case, since
   dabs are spaced well under 2×radius apart for continuous coverage —
   would get blended again by every subsequent overlapping dab,
   compounding opacity along the stroke.

   Read-before-write: a whole stroke's source region is snapshotted
   into a small local buffer BEFORE any of its destination pixels are
   written, exactly the same hazard a single overlapping dab always
   had to guard against — without it, a later segment of the SAME
   stroke could read pixels an earlier segment of that stroke had
   already overwritten a moment before, silently self-contaminating
   wherever source and destination circles overlap within one stroke.

   NEAREST-NEIGHBOR pixel copy, not bilinear: a real drag's offset
   (dx, dy) is essentially never an exact integer, so sampling the
   source with bilinear interpolation at a fractional coordinate
   averages 2-4 neighboring source pixels together on every copied
   pixel. On real photographic grain that's not subtle: tested with a
   realistic fractional offset against a uniformly-grainy synthetic
   scene, the disk interior's own variance dropped to ~28% of the
   surrounding grain's natural level — a heavily blurred patch is
   still visibly a separate patch against sharp surroundings, exactly
   the "frame outlining the clone" this fixed. sampleNearest is a
   literal, un-averaged copy — genuinely 1:1 — at the cost of the
   copied content potentially sitting up to half a source pixel off a
   perfectly continuous sub-pixel position; for photographic grain (no
   hard geometric edges to alias against) that's imperceptible.

   DITHERED feather, not a continuous cross-fade: this is NOT color
   correction (it never reads the destination's surrounding content,
   only the brush's own geometric falloff), so it stayed even after
   automatic tone matching was removed. Blending clone and original
   pixel VALUES continuously across the feather ring — the obvious
   `original*(1-a) + clone*a` a naive brush-hardness mask would do —
   averages two INDEPENDENT realizations of photographic grain/noise
   together, and averaging independent noise reduces its variance: the
   transition ring comes out visibly smoother/lower-grain than both
   the clone's own interior and the untouched surroundings, reading as
   an obvious "sticker" edge even when the copied color is fine.
   ditherThreshold instead makes every pixel in the ring commit FULLY
   to clone OR original, chosen by a fixed per-pixel hash against the
   geometric alpha at that point — over the whole ring this reproduces
   the exact same 0→1 falloff curve a soft brush hardness implies, but
   every individual OUTPUT pixel keeps its own side's full, un-averaged
   grain. Stroke opacity (the user-chosen "how fully the clone
   replaces the original", constant across the whole stroke, same as
   Photoshop's Opacity/Flow) is deliberately NOT dithered — it doesn't
   vary across the disk's own geometry, so it isn't a source of a
   ring-shaped artifact, and a user who picked partial opacity is
   asking for an actual blend.

   Coordinate space: stroke geometry (points[].x/y, radius, dx, dy)
   is always in *caller-supplied buffer* pixel units — callers scale
   from the stored original-image-pixel coordinates into whatever
   resolution they're baking at before calling these functions,
   exactly like bakePreview already does for qcx/qcy via its own
   proxy-scale factor.

   Strokes are always replayed in array order against the buffer
   they're given, mutating it as they go — so a later stroke's source
   legitimately picks up an earlier stroke's result when the two
   overlap (stack cloned-over pixels the same way a real clone stamp
   tool would). GLImage relies on exactly this: it recomposites the
   *entire* current stroke list from a pristine decode on every
   change, rather than rendering each stroke independently, so the
   live preview can never show a stroke sourcing from since-edited
   pixels as if they were still untouched.

   HEAL strokes (`mode: "heal"`) — a SEPARATE, opt-in stroke mode, not a
   change to Clone. Everything above about Clone (raw 1:1 copy, no
   automatic correction) still holds for every stroke without
   `mode: "heal"`; the user deliberately keeps that. A heal stroke has NO
   user-chosen source: its single (dx, dy) is found automatically
   (findHealSource, below) and per-point offset overrides are ignored.
   It shares the rest of Clone's geometry (capsule footprint, dithered
   feather, opacity, read-before-write snapshot) and only changes WHAT gets
   written inside the footprint: the source TEXTURE, re-toned to match the
   destination's surroundings — the classic "healing brush" idea
   (Georgiev, "Photoshop Healing Brush: a tool for seamless cloning",
   2004), done here with these specific, deliberately conservative choices:

   - Work in a SQRT domain per channel (v = sqrt(byte/255)), not on raw
     bytes and not in log: a perceptually-even-ish space where an additive
     correction behaves like a brightness ratio in the highlights and like
     an offset in the shadows, and that (unlike log) is defined at 0. The
     result is squared back, so it can never go negative.
   - Correction field δ = v(destination) − v(source), measured only on a
     thin RING just outside the footprint (where the destination is the
     real, untouched surroundings — see feedback_ring_sampling_median.md's
     throughline: the destination's target tone must come from what the
     footprint is NOT). Ring samples whose δ is an outlier (beyond
     HEAL_OUTLIER_K robust deviations from the ring's median) are dropped —
     the wire/pole-crossing-the-ring case that once left ghosts behind.
   - δ is then interpolated INTO the footprint as a smooth membrane:
     a push-pull pyramid (normalized-convolution downsample, blended
     upsample — Gortler et al., "The Lumigraph", 1996) for a fast global
     estimate, refined by plain Gauss-Seidel averaging sweeps (no
     over-relaxation). Both steps only ever take weighted AVERAGES of
     existing ring values, so by the discrete maximum principle every
     interpolated δ stays within the ring's own [min, max] — there is no
     way for this to overshoot into the corrupted/"rainbow" output the old
     SOR Poisson attempt produced on dense dab clusters
     (feedback_stamp_poisson_blend.md). Worst case it's a soft, wrong tone,
     never noise.
   - Output = (v(source) + δ)², composited through the same dithered
     feather + opacity as Clone. The source's own grain is untouched (δ is
     smooth), so the texture-preservation reasoning above still applies.
   ============================================================ */

import { removeFill, hashRegion, neuralState, neuralVersion, FILL_LOOK } from "./removeTool.mjs";

const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
const REMOVE_EDGE_IN = 0.15;      // Remove's automatic edge may fade up to this × radius inside the painted outline
const STROKE_LOOK = "v3";         // bump when a stroke's output changes for the same input (keys the stroke cache)
const DEFAULT_FEATHER = 0.3;      // default feather zone width, as a fraction of radius (Photoshop's brush "hardness", inverted)
const DEFAULT_OPACITY = 1;        // default strength — 1 = fully replace, less = blend with the original (Photoshop's Opacity/Flow)
const HEAL_RING_FRAC = 0.15;      // heal: boundary ring width, as a fraction of radius...
const HEAL_RING_MIN = 2;          // ...but never thinner than this many px
const HEAL_OUTLIER_K = 3;         // heal: ring δ further than this × robust spread (1.4826·MAD) from the median is dropped
const HEAL_OUTLIER_FLOOR = 0.02;  // heal: spread floor (sqrt-domain units) so a near-uniform ring doesn't reject ordinary grain
const HEAL_SWEEPS = 30;           // heal: Gauss-Seidel refinement sweeps after the push-pull estimate
const HEAL_RING_SMOOTH = 0.5;    // heal: ring δ is averaged along the ring over ± this × radius (see healSource)

export function isEmptyStamp(stamp) {
  return !stamp || stamp.length === 0;
}

/* Back-compat migration: photos edited before strokes existed (or before
   the "heal" -> "clone stamp" rename) have `stamp`/`heal` entries in the
   old single-circle shape ({dstX,dstY,srcX,srcY,radius}). Every entry
   point that reads stored stamp/heal edits (this module, and App.jsx's
   mapStampStrokes/Editor state init) runs it through this first, so old
   edits keep working — and re-saving them (any further stamp edit
   auto-commits the whole array) upgrades them to the new shape for good.
   Already-new-shape entries pass through untouched; anything unrecognized
   is dropped rather than crashing the pipeline. */
export function normalizeStamp(stamp) {
  if (!stamp) return [];
  const out = [];
  for (const entry of stamp) {
    if (entry && Array.isArray(entry.points) && entry.points.length && entry.radius > 0) {
      out.push({
        ...entry,
        // Number.isFinite, not `> 0`: 0 is a deliberate, valid choice for
        // both (a hard edge / a fully transparent stroke) — `> 0` treated it
        // as falsy-and-unset and silently substituted the default instead,
        // e.g. a feather of 0 (hard edge) silently became DEFAULT_FEATHER's
        // soft 30% ring. Only genuinely missing/non-numeric values should
        // fall back.
        feather: Number.isFinite(entry.feather) ? entry.feather : DEFAULT_FEATHER,
        opacity: Number.isFinite(entry.opacity) ? entry.opacity : DEFAULT_OPACITY,
      });
    } else if (entry && entry.dstX !== undefined && entry.srcX !== undefined && entry.radius > 0) {
      out.push({
        points: [{ x: entry.dstX, y: entry.dstY }],
        radius: entry.radius,
        dx: entry.srcX - entry.dstX,
        dy: entry.srcY - entry.dstY,
        feather: DEFAULT_FEATHER,
        opacity: DEFAULT_OPACITY,
      });
    }
  }
  return out;
}

/* Nearest-neighbor sample at fractional (x, y) — no interpolation, just
   the single closest source pixel, exactly as stored. See the module
   header's "NEAREST-NEIGHBOR pixel copy" section for why: this is the
   ONLY sampling this module does now (there's no separate area-context
   sampling anymore, since there's no automatic color match left to
   compute one for). */
function sampleNearest(buf, width, height, channels, x, y) {
  const cx = Math.min(Math.max(Math.round(x), 0), width - 1);
  const cy = Math.min(Math.max(Math.round(y), 0), height - 1);
  const o = (cy * width + cx) * channels;
  return [buf[o], buf[o + 1], buf[o + 2]];
}

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/* Deterministic, well-distributed pseudo-random value in [0,1) for an
   integer pixel coordinate — a standard 32-bit hash-mix, the same shape
   as the "stochastic/dithered alpha" technique GPU shaders use to avoid
   banding. Same (x,y) always gives the same value (stable across
   re-renders of the same stroke, no temporal flicker/shimmer), and
   neighboring pixels land far apart in the output range (no visible
   large-scale pattern within the dithered feather ring — see its call
   site in applyStrokeToBuffer). */
function ditherThreshold(x, y) {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}

/* Closest point to (px,py) on the segment a→b, plus the projection
   parameter t (0 at a, 1 at b) used to interpolate per-endpoint data
   (the source offset) smoothly along the segment. Degenerates cleanly
   to "distance to a" when a===b (a one-point stroke's single "segment"). */
function closestOnSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax, aby = by - ay;
  const lenSq = abx * abx + aby * aby;
  let t = lenSq > 0 ? ((px - ax) * abx + (py - ay) * aby) / lenSq : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return { x: ax + abx * t, y: ay + aby * t, t };
}

function median(arr) {
  const s = Float32Array.from(arr).sort();
  const n = s.length;
  return n === 0 ? 0 : n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/* Heal stroke's re-toned source colors — see the module header's HEAL
   section for the why. `best` is the stroke's footprint (pixel key →
   {dist, dx, dy}) extended out to the boundary ring; `snap` the stroke's
   read-before-write snapshot. Returns { get(key) → [r, g, b] } (0-255
   floats) for every footprint pixel (dist ≤ outer).

   Everything runs on a dense grid over the footprint's bounding box — the
   pyramid and the sweeps need neighbors. Grid cells: 1 = footprint (δ
   unknown, solved for), 2 = boundary ring (δ measured, fixed), 3 = ring
   sample rejected as an outlier (solved for, like the footprint), 0 = not
   part of the domain (outside the stroke, or off the buffer — treated as
   "no neighbor", i.e. the membrane simply doesn't see it). */
function healSource(best, snap, rx0, ry0, rw, rh, width, outer, ringR) {
  let gx0 = Infinity, gy0 = Infinity, gx1 = -Infinity, gy1 = -Infinity;
  for (const key of best.keys()) {
    const x = key % width, y = (key - x) / width;
    if (x < gx0) gx0 = x; if (x > gx1) gx1 = x;
    if (y < gy0) gy0 = y; if (y > gy1) gy1 = y;
  }
  const gw = gx1 - gx0 + 1, gh = gy1 - gy0 + 1, n = gw * gh;
  const type = new Uint8Array(n);
  const sv = new Float32Array(n * 3);   // source, sqrt domain (footprint cells)
  const d = new Float32Array(n * 3);    // δ: measured on the ring, solved elsewhere
  const ringIdx = [];
  const snapV = (x, y, c) => Math.sqrt(snap[(y * rw + x) * 3 + c] / 255);

  for (const [key, v] of best) {
    const x = key % width, y = (key - x) / width;
    const g = (y - gy0) * gw + (x - gx0);
    // nearest source pixel, clamped to the snapshot — identical rule to
    // sampleNearest, so a heal stroke picks exactly the texels Clone would
    const sx = Math.min(Math.max(Math.round(x + v.dx - rx0), 0), rw - 1);
    const sy = Math.min(Math.max(Math.round(y + v.dy - ry0), 0), rh - 1);
    if (v.dist <= outer) {
      type[g] = 1;
      for (let c = 0; c < 3; c++) sv[g * 3 + c] = snapV(sx, sy, c);
    } else {
      type[g] = 2;
      const lx = x - rx0, ly = y - ry0;
      for (let c = 0; c < 3; c++) d[g * 3 + c] = snapV(lx, ly, c) - snapV(sx, sy, c);
      ringIdx.push(g);
    }
  }

  // robust outlier rejection on the ring (per channel; one bad channel
  // rejects the sample)
  if (ringIdx.length >= 8) {
    for (let c = 0; c < 3; c++) {
      const vals = ringIdx.map((g) => d[g * 3 + c]);
      const med = median(vals);
      const spread = Math.max(1.4826 * median(vals.map((x) => Math.abs(x - med))), HEAL_OUTLIER_FLOOR);
      for (const g of ringIdx) if (Math.abs(d[g * 3 + c] - med) > HEAL_OUTLIER_K * spread) type[g] = 3;
    }
  }
  /* Then smooth the surviving ring δ ALONG the ring (mean of valid ring
     samples within ringR of each one). On structured texture — brick and
     mortar, tiles — a source offset that isn't an exact multiple of the
     pattern puts mortar against brick at alternating spots around the
     ring, so per-sample δ flips between "+orange" and "+white". That
     split is bimodal, so the rejection above can't tell which half is
     wrong, and the membrane would carry each flip into a color blob near
     the edge (seen on a real brick-wall photo). Averaging over a window
     wider than the pattern cancels the flips, while samples already
     rejected (a wire crossing the ring) stay out of every average — which
     is why this runs AFTER rejection rather than replacing it with
     blurred inputs (tried: a wide blur folds the wire back in). */
  if (ringR > 0 && ringIdx.length) {
    const SW = gw + 1, sat = new Float64Array(SW * (gh + 1) * 4);
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
      const g = j * gw + i, q = ((j + 1) * SW + i + 1) * 4, up = (j * SW + i + 1) * 4, lf = ((j + 1) * SW + i) * 4, ul = (j * SW + i) * 4;
      const ok = type[g] === 2 ? 1 : 0;
      for (let k = 0; k < 4; k++) sat[q + k] = (k < 3 ? d[g * 3 + k] * ok : ok) + sat[up + k] + sat[lf + k] - sat[ul + k];
    }
    const sm = new Float32Array(ringIdx.length * 3);
    ringIdx.forEach((g, t) => {
      if (type[g] !== 2) return;
      const i = g % gw, j = (g - i) / gw;
      const x0 = Math.max(0, i - ringR), y0 = Math.max(0, j - ringR), x1 = Math.min(gw, i + ringR + 1), y1 = Math.min(gh, j + ringR + 1);
      const at = (y, x, k) => sat[(y * SW + x) * 4 + k];
      const cnt = at(y1, x1, 3) - at(y0, x1, 3) - at(y1, x0, 3) + at(y0, x0, 3);
      for (let c = 0; c < 3; c++) sm[t * 3 + c] = (at(y1, x1, c) - at(y0, x1, c) - at(y1, x0, c) + at(y0, x0, c)) / cnt;
    });
    ringIdx.forEach((g, t) => { if (type[g] === 2) for (let c = 0; c < 3; c++) d[g * 3 + c] = sm[t * 3 + c]; });
  }

  const known = (t) => t === 2;
  let anyKnown = false;
  for (const g of ringIdx) if (known(type[g])) { anyKnown = true; break; }

  if (anyKnown) {
    // ---- push: normalized-convolution pyramid ([1,2,1]/4 per axis, stride 2)
    const levels = [{ w: gw, h: gh, A: new Float32Array(n * 3), W: new Float32Array(n) }];
    for (let g = 0; g < n; g++) {
      if (!known(type[g])) continue;
      levels[0].W[g] = 1;
      for (let c = 0; c < 3; c++) levels[0].A[g * 3 + c] = d[g * 3 + c];
    }
    const K = [0.25, 0.5, 0.25];
    while (levels[levels.length - 1].w > 1 || levels[levels.length - 1].h > 1) {
      const f = levels[levels.length - 1];
      const w = Math.ceil(f.w / 2), h = Math.ceil(f.h / 2);
      const A = new Float32Array(w * h * 3), W = new Float32Array(w * h);
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        let a0 = 0, a1 = 0, a2 = 0, ww = 0;
        for (let b = -1; b <= 1; b++) {
          const y = 2 * j + b; if (y < 0 || y >= f.h) continue;
          for (let a = -1; a <= 1; a++) {
            const x = 2 * i + a; if (x < 0 || x >= f.w) continue;
            const k = K[a + 1] * K[b + 1], q = y * f.w + x;
            ww += k * f.W[q];
            a0 += k * f.A[q * 3]; a1 += k * f.A[q * 3 + 1]; a2 += k * f.A[q * 3 + 2];
          }
        }
        const q = j * w + i;
        W[q] = ww; A[q * 3] = a0; A[q * 3 + 1] = a1; A[q * 3 + 2] = a2;
      }
      levels.push({ w, h, A, W });
    }
    // ---- pull: coarse estimate, upsampled bilinearly and blended under
    // each level's own (weight-saturated) measurement
    let F = null;
    for (let l = levels.length - 1; l >= 0; l--) {
      const L = levels[l];
      const out = new Float32Array(L.w * L.h * 3);
      const prev = F, P = l + 1 < levels.length ? levels[l + 1] : null;
      for (let j = 0; j < L.h; j++) for (let i = 0; i < L.w; i++) {
        const q = j * L.w + i;
        let u0 = 0, u1 = 0, u2 = 0;
        if (prev) {
          const fx = Math.min(Math.max((i + 0.5) / 2 - 0.5, 0), P.w - 1);
          const fy = Math.min(Math.max((j + 0.5) / 2 - 0.5, 0), P.h - 1);
          const x0 = Math.floor(fx), y0 = Math.floor(fy);
          const x1 = Math.min(x0 + 1, P.w - 1), y1 = Math.min(y0 + 1, P.h - 1);
          const tx = fx - x0, ty = fy - y0;
          const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
          const q00 = (y0 * P.w + x0) * 3, q10 = (y0 * P.w + x1) * 3, q01 = (y1 * P.w + x0) * 3, q11 = (y1 * P.w + x1) * 3;
          u0 = w00 * prev[q00] + w10 * prev[q10] + w01 * prev[q01] + w11 * prev[q11];
          u1 = w00 * prev[q00 + 1] + w10 * prev[q10 + 1] + w01 * prev[q01 + 1] + w11 * prev[q11 + 1];
          u2 = w00 * prev[q00 + 2] + w10 * prev[q10 + 2] + w01 * prev[q01 + 2] + w11 * prev[q11 + 2];
        }
        const wt = L.W[q];
        // a cell with ≥25% real coverage trusts its own measurement; the
        // finest level's known cells (W = 1) keep their exact δ
        const alpha = prev ? Math.min(1, wt * 4) : 1;
        const inv = wt > 1e-6 ? 1 / wt : 0;
        out[q * 3] = alpha * L.A[q * 3] * inv + (1 - alpha) * u0;
        out[q * 3 + 1] = alpha * L.A[q * 3 + 1] * inv + (1 - alpha) * u1;
        out[q * 3 + 2] = alpha * L.A[q * 3 + 2] * inv + (1 - alpha) * u2;
      }
      F = out;
    }
    for (let g = 0; g < n; g++) {
      if (type[g] === 1 || type[g] === 3) for (let c = 0; c < 3; c++) d[g * 3 + c] = F[g * 3 + c];
    }
    // ---- refine: plain Gauss-Seidel averaging over the free cells (4-
    // neighbors within the domain). No over-relaxation: every update is a
    // convex average, so δ can't leave the ring's own range.
    for (let it = 0; it < HEAL_SWEEPS; it++) {
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
        const g = j * gw + i;
        if (type[g] !== 1 && type[g] !== 3) continue;
        let s0 = 0, s1 = 0, s2 = 0, cnt = 0;
        if (i > 0 && type[g - 1]) { s0 += d[(g - 1) * 3]; s1 += d[(g - 1) * 3 + 1]; s2 += d[(g - 1) * 3 + 2]; cnt++; }
        if (i < gw - 1 && type[g + 1]) { s0 += d[(g + 1) * 3]; s1 += d[(g + 1) * 3 + 1]; s2 += d[(g + 1) * 3 + 2]; cnt++; }
        if (j > 0 && type[g - gw]) { s0 += d[(g - gw) * 3]; s1 += d[(g - gw) * 3 + 1]; s2 += d[(g - gw) * 3 + 2]; cnt++; }
        if (j < gh - 1 && type[g + gw]) { s0 += d[(g + gw) * 3]; s1 += d[(g + gw) * 3 + 1]; s2 += d[(g + gw) * 3 + 2]; cnt++; }
        if (!cnt) continue;
        d[g * 3] = s0 / cnt; d[g * 3 + 1] = s1 / cnt; d[g * 3 + 2] = s2 / cnt;
      }
    }
  } else {
    // no usable boundary (stroke entirely surrounded by off-buffer area or
    // rejected samples) — degrade to a plain clone rather than guess
    d.fill(0);
  }

  return {
    get(key) {
      const x = key % width, y = (key - x) / width;
      const g = ((y - gy0) * gw + (x - gx0)) * 3;
      const out = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        const v = Math.min(1, Math.max(0, sv[g + c] + d[g + c]));
        // rounded: a Node Buffer (bakeRaw) truncates float writes
        out[c] = Math.round(v * v * 255);
      }
      return out;
    },
  };
}

/* Composites one stroke — its whole polyline as a single smooth capsule —
   onto buf in place. `stroke` = { points:[{x,y,dx?,dy?}], radius, dx, dy,
   feather, opacity } (already normalized).

   Two passes, at whole-stroke granularity: the first only READS from buf
   (copies the region this stroke could possibly read from into a small
   local snapshot); the second only WRITES, copying each destination
   pixel exactly once from whichever polyline segment is closest to it.
   See the module header's "Read-before-write" section for why the read
   has to fully finish before any write starts.

   "Touched exactly once" also means the covered region is tracked in a
   Map keyed by pixel offset rather than a dense array sized to the
   stroke's full bounding box — a long, spread-out stroke (say, a wire
   traced corner-to-corner) would otherwise force allocating (and
   scanning) an array covering its entire bounding box even though the
   pixels actually near its path are a small fraction of that area. The
   Map only ever holds entries for pixels within `radius*(1+feather)` of
   some segment, i.e. proportional to the stroke's own length. */
function applyStrokeToBuffer(buf, width, height, channels, stroke) {
  const radius = stroke.radius;
  const pts = stroke.points;
  if (!(radius > 0) || !pts || !pts.length) return;
  // Number.isFinite, not `> 0` — see normalizeStamp's own note: 0 is a
  // deliberate, valid feather (hard edge) or opacity (fully transparent),
  // not a missing value. This is the path the live in-progress draft
  // stroke goes through directly (GLImage folds it straight into the
  // composite without normalizeStamp first), so it's the one that actually
  // mattered for the reported bug — a committed stroke is already
  // normalized by the time it gets here.
  const feather = Number.isFinite(stroke.feather) ? stroke.feather : DEFAULT_FEATHER;
  const opacity = Number.isFinite(stroke.opacity) ? Math.min(1, stroke.opacity) : DEFAULT_OPACITY;
  const outer = radius * (1 + feather);
  // heal also needs the thin boundary ring just outside the footprint (see
  // the module header's HEAL section); for Clone `ring` is 0 and `reach`
  // is exactly `outer`, so everything below is unchanged for it.
  const heal = stroke.mode === "heal";
  // per-point offset overrides belong to Clone's manual source; a heal
  // stroke has one automatic offset (findHealSource) — a stroke switched
  // from Clone to Heal may still carry overrides, which must not apply
  const dxs = pts.map((p) => (heal ? stroke.dx : (p.dx ?? stroke.dx)));
  const dys = pts.map((p) => (heal ? stroke.dy : (p.dy ?? stroke.dy)));
  const ring = heal ? Math.max(HEAL_RING_MIN, radius * HEAL_RING_FRAC) : 0;
  const reach = outer + ring;
  const ringR = heal ? Math.max(1, Math.round(radius * HEAL_RING_SMOOTH)) : 0;

  const box = readBox(pts, dxs, dys, reach, width, height);
  // whole read footprint falls off-buffer (e.g. a stroke that's off-frame
  // in a cropped preview, or outside a deep-zoom tile's crop window) —
  // bail before allocating the snapshot below.
  if (!box) return;
  const [rx0, ry0, rx1, ry1] = box;
  // same stroke over the same pixels as last time → write what it wrote
  const cacheKey = strokeCacheKey(stroke, width, height, buf, channels, rx0, ry0, rx1, ry1);
  // a Remove stroke still waiting for its network result is cached too,
  // but only until the next result lands anywhere (neuralVersion)
  const waitKey = stroke.mode === "remove" ? `${cacheKey}|nv${neuralVersion()}` : null;
  const hitKey = strokeCache.has(cacheKey) ? cacheKey : waitKey && strokeCache.has(waitKey) ? waitKey : null;
  const hit = hitKey && strokeCache.get(hitKey);
  if (hit) {
    strokeCache.delete(hitKey); strokeCache.set(hitKey, hit); // LRU: most recent last
    replayStroke(hit, buf, channels);
    if (hitKey === waitKey) pendingStrokes++; // its network result is still to come
    return;
  }
  const rw = rx1 - rx0 + 1, rh = ry1 - ry0 + 1;
  const snap = new Float32Array(rw * rh * 3);
  for (let y = 0; y < rh; y++) {
    const srcRow = (y + ry0) * width;
    for (let x = 0; x < rw; x++) {
      const o = (srcRow + x + rx0) * channels;
      const so = (y * rw + x) * 3;
      snap[so] = buf[o]; snap[so + 1] = buf[o + 1]; snap[so + 2] = buf[o + 2];
    }
  }

  // segments = consecutive point pairs; a single-point stroke gets one
  // degenerate (zero-length) segment so the exact same math applies.
  const segCount = pts.length > 1 ? pts.length - 1 : 1;
  // nearest segment per pixel, in flat arrays over the read box — a Map
  // with an object per pixel per segment cost ~1 s per big stroke on every
  // recomposite; the footprint Map below is built once, from the winners
  const bestD = new Float64Array(rw * rh).fill(Infinity);
  const bestDx = new Float32Array(rw * rh), bestDy = new Float32Array(rw * rh);
  const reachSq = reach * reach;
  for (let s = 0; s < segCount; s++) {
    const i0 = s, i1 = pts.length > 1 ? s + 1 : 0;
    const ax = pts[i0].x, ay = pts[i0].y, bx = pts[i1].x, by = pts[i1].y;
    const sx0 = Math.max(0, Math.floor(Math.min(ax, bx) - reach));
    const sx1 = Math.min(width - 1, Math.ceil(Math.max(ax, bx) + reach));
    const sy0 = Math.max(0, Math.floor(Math.min(ay, by) - reach));
    const sy1 = Math.min(height - 1, Math.ceil(Math.max(ay, by) + reach));
    if (sx1 < sx0 || sy1 < sy0) continue;
    // closestOnSegment, inlined
    const abx = bx - ax, aby = by - ay, lenSq = abx * abx + aby * aby;
    for (let y = sy0; y <= sy1; y++) {
      const row = (y - ry0) * rw - rx0;
      for (let x = sx0; x <= sx1; x++) {
        let t = lenSq > 0 ? ((x - ax) * abx + (y - ay) * aby) / lenSq : 0;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const ex = x - (ax + abx * t), ey = y - (ay + aby * t);
        const dSq = ex * ex + ey * ey;
        if (dSq > reachSq) continue;
        const i = row + x, dist = Math.sqrt(dSq);
        if (bestD[i] <= dist) continue;
        bestD[i] = dist;
        bestDx[i] = dxs[i0] + (dxs[i1] - dxs[i0]) * t;
        bestDy[i] = dys[i0] + (dys[i1] - dys[i0]) * t;
      }
    }
  }
  const best = new Map();
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    const i = y * rw + x;
    if (bestD[i] !== Infinity) best.set((y + ry0) * width + x + rx0, { dist: bestD[i], dx: bestDx[i], dy: bestDy[i] });
  }

  const healed = heal ? healSource(best, snap, rx0, ry0, rw, rh, width, outer, ringR) : null;
  // Remove: content-aware fill of the whole footprint, no source at all
  // (electron/removeTool.mjs). Reads buf before anything below writes it.
  let removed = null;
  if (stroke.mode === "remove") {
    const holeKeys = [];
    for (const [key, v] of best) if (v.dist <= outer) holeKeys.push(key);
    const strokeKey = `${radius.toFixed(2)}|${feather}|${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(";")}`;
    // the edge is automatic (removeTool's textureFill): the band from the
    // footprint's edge to a little inside the outline the user painted
    const band = Math.max(2, outer - radius * (1 - REMOVE_EDGE_IN));
    if (holeKeys.length) removed = removeFill(buf, width, height, channels, holeKeys, strokeKey, band);
    // still being filled in the background (removeTool's setAsyncFill):
    // leave the area as it is until the result lands and re-renders
    if (!removed) { if (holeKeys.length) pendingStrokes++; return; }
    if (!removed.final) pendingStrokes++;
  }

  const written = [];
  for (const [key, v] of best) {
    // ring pixels (heal only, dist > outer) land at alphaGeom 0 here and
    // are never written — they're read-only boundary samples
    const rem = removed && removed.get(key);
    if (removed && !rem) continue;
    // Remove: its own automatic edge, and no partial opacity (half a
    // removed object is never what's wanted)
    const alphaGeom = rem ? rem[3] : 1 - smoothstep(radius, outer, v.dist);
    if (alphaGeom <= 0) continue;
    const x = key % width, y = (key - x) / width;
    // DITHERED feather — see the module header's own section on this.
    // Not color correction: this never reads the destination's own
    // surrounding content, only the brush's geometric falloff.
    const dithered = alphaGeom >= 1 ? 1 : ditherThreshold(x, y) < alphaGeom ? 1 : 0;
    const alpha = rem ? dithered : dithered * opacity;
    if (alpha <= 0) continue;
    // Clone: raw 1:1 copy — no correction added to sr/sg/sb, matching real
    // Photoshop's Clone Stamp (see module header). Heal: the same source
    // pixel, re-toned by the interpolated boundary correction.
    const [sr, sg, sb] = rem || (healed ? healed.get(key) : sampleNearest(snap, rw, rh, 3, x + v.dx - rx0, y + v.dy - ry0));
    const o = key * channels;
    // clamped by hand: a Node Buffer (export) wraps 256 → 0 instead
    buf[o] = clamp255(buf[o] * (1 - alpha) + sr * alpha);
    buf[o + 1] = clamp255(buf[o + 1] * (1 - alpha) + sg * alpha);
    buf[o + 2] = clamp255(buf[o + 2] * (1 - alpha) + sb * alpha);
    written.push(key);
  }
  const keys = Int32Array.from(written), rgb = new Uint8Array(keys.length * 3);
  for (let i = 0; i < keys.length; i++) {
    const o = keys[i] * channels;
    rgb[i * 3] = buf[o]; rgb[i * 3 + 1] = buf[o + 1]; rgb[i * 3 + 2] = buf[o + 2];
  }
  if (removed && !removed.final) strokeCachePut(waitKey, { keys, rgb }, false);
  else strokeCachePut(cacheKey, { keys, rgb });
}

/* the box a stroke reads (footprint + heal ring + clone source), clamped
   to the buffer; null when it's entirely off-buffer */
function readBox(pts, dxs, dys, reach, width, height) {
  const readMargin = reach + 1;
  let rx0 = Infinity, ry0 = Infinity, rx1 = -Infinity, ry1 = -Infinity;
  for (let i = 0; i < pts.length; i++) {
    const sx = pts[i].x + dxs[i], sy = pts[i].y + dys[i];
    rx0 = Math.min(rx0, pts[i].x - readMargin, sx - readMargin);
    ry0 = Math.min(ry0, pts[i].y - readMargin, sy - readMargin);
    rx1 = Math.max(rx1, pts[i].x + readMargin, sx + readMargin);
    ry1 = Math.max(ry1, pts[i].y + readMargin, sy + readMargin);
  }
  rx0 = Math.max(0, Math.floor(rx0)); ry0 = Math.max(0, Math.floor(ry0));
  rx1 = Math.min(width - 1, Math.ceil(rx1)); ry1 = Math.min(height - 1, Math.ceil(ry1));
  return rx1 < rx0 || ry1 < ry0 ? null : [rx0, ry0, rx1, ry1];
}
function replayStroke(entry, buf, channels) {
  for (let i = 0; i < entry.keys.length; i++) {
    const o = entry.keys[i] * channels;
    buf[o] = entry.rgb[i * 3]; buf[o + 1] = entry.rgb[i * 3 + 1]; buf[o + 2] = entry.rgb[i * 3 + 2];
  }
}

/* Per-stroke output cache. A recomposite (every edit, every background
   fill landing) re-applies ALL strokes; a stroke whose parameters and
   underlying pixels (its whole read box: footprint, heal ring, clone
   source) are unchanged writes exactly what it wrote before, so that's
   replayed instead of recomputed (~300 ms → a few ms per big stroke).
   Remove strokes are keyed with the AI mode, which changes their look,
   and until final (see removeFill's `final`) also with neuralVersion. */
const STROKE_CACHE_MAX_BYTES = 96 * 1024 * 1024;
const strokeCache = new Map();
let strokeCacheBytes = 0;
function strokeCacheKey(stroke, width, height, buf, channels, rx0, ry0, rx1, ry1) {
  const pts = stroke.points.map((p) => `${p.x},${p.y}${p.dx != null ? `,${p.dx},${p.dy}` : ""}`).join(";");
  const ai = `${stroke.mode === "remove" ? `${neuralState()}|${FILL_LOOK}` : ""}|${STROKE_LOOK}`;
  // buffer type too: a Uint8ClampedArray rounds the blend, a Buffer truncates it
  return `${stroke.mode}|${stroke.radius}|${stroke.feather}|${stroke.opacity}|${stroke.dx},${stroke.dy}|${pts}|${width}x${height}x${channels}|${buf.constructor.name}|${ai}|${hashRegion(buf, width, channels, rx0, ry0, rx1, ry1)}`;
}
function strokeCachePut(key, entry, final = true) {
  if (final && strokeCacheSink) strokeCacheSink(key, entry);
  const old = strokeCache.get(key);
  if (old) strokeCacheBytes -= old.keys.byteLength + old.rgb.byteLength;
  strokeCache.delete(key);
  strokeCache.set(key, entry);
  strokeCacheBytes += entry.keys.byteLength + entry.rgb.byteLength;
  for (const [k, e] of strokeCache) {
    if (strokeCacheBytes <= STROKE_CACHE_MAX_BYTES) break;
    strokeCache.delete(k);
    strokeCacheBytes -= e.keys.byteLength + e.rgb.byteLength;
  }
}

/* ============================================================
   Automatic source for HEAL strokes — Heal has no user-chosen source
   (Lightroom's Heal picks it for you, the same way: Camera Raw's own
   "FindRetouchSource" step), so the stroke's (dx, dy) is computed here
   from the photo instead of dragged by hand. Clone keeps its manual,
   dragged source — this is only ever called for mode "heal".

   Candidates: HEAL_SRC_ANGLES directions × HEAL_SRC_DIST_MULTS distances
   around the stroke. A candidate is rejected if its copied area would
   leave the buffer, overlap the stroke's own footprint (it would copy the
   very defect being removed), or overlap another stroke (`obstacles`).
   Survivors are scored, lowest wins:
     context  — around the ring: destination − source, minus the best-fit
                PLANE per channel (the heal blend itself removes any tone
                offset and smooth gradient, so those must not count), then
                the MEDIAN size of what's left. Low = the source's
                surrounding STRUCTURE matches the destination's: a source
                taken along a shadow band, a fold, a horizon continues it
                (a plain squared difference preferred flat, uniform areas
                instead, and on a spot inside the shadow under a lip pasted
                a light "window" into the shadow). Median, so a wire or a
                spire crossing part of the ring can't dominate.
     patch    — a light penalty (HEAL_SRC_PATCH_W) on the candidate's
                interior differing from the median color of the
                destination's surroundings, against copying in an obvious
                foreign object
     distance — a small tie-breaker toward nearer sources
   The heal blend then removes whatever tone difference remains, so the
   search only has to get texture and content right.
   ============================================================ */
const HEAL_SRC_ANGLES = 24;
const HEAL_SRC_DIST_MULTS = [1.15, 1.5, 2, 2.7, 3.6, 5, 7]; // × the footprint's WIDTH (2·outer) — not its length: a long
                                                       // wire needs a small sideways shift, and along-the-stroke
                                                       // candidates are rejected by the overlap check anyway
const HEAL_SRC_BAND = 0.6;                             // context band width, × radius
const HEAL_SRC_SAMPLES = 360;
const HEAL_SRC_PATCH_W = 0.25;                          // target sample count per region (subsampled on a grid)

/* Least-squares plane v ≈ a + b·x + c·y through the samples; returns the
   residuals (falls back to "minus the mean" if the fit is degenerate). */
function planeResiduals(xs, ys, vs) {
  const n = vs.length;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sv = 0, sxv = 0, syv = 0;
  for (let i = 0; i < n; i++) { const x = xs[i], y = ys[i], v = vs[i]; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sv += v; sxv += x * v; syv += y * v; }
  const A = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]], b = [sv, sxv, syv];
  const det3 = (M) => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det3(A);
  if (Math.abs(D) < 1e-9) { const m = sv / n; return vs.map((v) => v - m); }
  const p = [0, 1, 2].map((col) => det3(A.map((r, i) => r.map((x, j) => (j === col ? b[i] : x)))) / D);
  return vs.map((v, i) => v - (p[0] + p[1] * xs[i] + p[2] * ys[i]));
}

function distToPolyline(pts, x, y) {
  let best = Infinity;
  const segs = pts.length > 1 ? pts.length - 1 : 1;
  for (let s = 0; s < segs; s++) {
    const a = pts[s], b = pts[pts.length > 1 ? s + 1 : 0];
    const c = closestOnSegment(x, y, a.x, a.y, b.x, b.y);
    const d = Math.hypot(x - c.x, y - c.y);
    if (d < best) best = d;
  }
  return best;
}

/* pts/radius/obstacles in this buffer's pixel space. Returns { dx, dy },
   or null when no candidate fits (tiny buffer, stroke covering most of it). */
export function findHealSource(buf, width, height, channels, pts, radius, feather, obstacles = []) {
  if (!pts || !pts.length || !(radius > 0)) return null;
  const outer = radius * (1 + (Number.isFinite(feather) ? feather : DEFAULT_FEATHER));
  const band = outer + Math.max(3, radius * HEAL_SRC_BAND);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of pts) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  // sample the footprint (hole) and the context band on a grid sized to
  // land near HEAL_SRC_SAMPLES points each
  const bandArea = Math.PI * (band * band - outer * outer) + 2 * (band - outer) * Math.hypot(x1 - x0, y1 - y0);
  const step = Math.max(1, Math.sqrt(bandArea / HEAL_SRC_SAMPLES));
  const hole = [], ring = [];
  for (let y = Math.floor(y0 - band); y <= y1 + band; y += step) {
    for (let x = Math.floor(x0 - band); x <= x1 + band; x += step) {
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= width || yi >= height) continue;
      const d = distToPolyline(pts, xi, yi);
      if (d <= outer) hole.push(xi, yi);
      else if (d <= band) ring.push(xi, yi);
    }
  }
  if (hole.length === 0 || ring.length < 6) return null;
  const px = (x, y, c) => buf[(y * width + x) * channels + c];

  // background reference: per-channel median of the destination's band
  const med = [0, 1, 2].map((c) => {
    const v = [];
    for (let i = 0; i < ring.length; i += 2) v.push(px(ring[i], ring[i + 1], c));
    return median(v);
  });

  // footprint occupancy mask (radius outer·1.05) over its own bounding box,
  // rasterized segment by segment — makes the per-candidate overlap test
  // O(1) per sample instead of a distance-to-polyline walk (which made a
  // long wire stroke cost ~50 ms per search)
  const keep = outer * 1.05;
  const mx0 = Math.floor(x0 - keep), my0 = Math.floor(y0 - keep);
  const mw = Math.ceil(x1 + keep) - mx0 + 1, mh = Math.ceil(y1 + keep) - my0 + 1;
  const mask = new Uint8Array(mw * mh);
  const segs = pts.length > 1 ? pts.length - 1 : 1;
  for (let sgi = 0; sgi < segs; sgi++) {
    const a = pts[sgi], b = pts[pts.length > 1 ? sgi + 1 : 0];
    const sx0 = Math.max(mx0, Math.floor(Math.min(a.x, b.x) - keep)), sx1 = Math.min(mx0 + mw - 1, Math.ceil(Math.max(a.x, b.x) + keep));
    const sy0 = Math.max(my0, Math.floor(Math.min(a.y, b.y) - keep)), sy1 = Math.min(my0 + mh - 1, Math.ceil(Math.max(a.y, b.y) + keep));
    for (let y = sy0; y <= sy1; y++) for (let x = sx0; x <= sx1; x++) {
      const c = closestOnSegment(x, y, a.x, a.y, b.x, b.y);
      if (Math.hypot(x - c.x, y - c.y) <= keep) mask[(y - my0) * mw + (x - mx0)] = 1;
    }
  }
  const inFootprint = (x, y) => {
    const ix = Math.round(x) - mx0, iy = Math.round(y) - my0;
    return ix >= 0 && iy >= 0 && ix < mw && iy < mh && mask[iy * mw + ix] === 1;
  };

  const diam = 2 * outer;
  const cxs = 0.5 * (x0 + x1), cys = 0.5 * (y0 + y1);
  let best = null;
  for (const mult of HEAL_SRC_DIST_MULTS) {
    const dist = diam * mult;
    for (let k = 0; k < HEAL_SRC_ANGLES; k++) {
      const ang = (k / HEAL_SRC_ANGLES) * Math.PI * 2;
      const dx = Math.cos(ang) * dist, dy = Math.sin(ang) * dist;
      // reject: copied area off-buffer, or overlapping this stroke / others
      let ok = true;
      for (let i = 0; i < hole.length && ok; i += 2) {
        const sx = hole[i] + dx, sy = hole[i + 1] + dy;
        if (sx < 0 || sy < 0 || sx >= width - 1 || sy >= height - 1) ok = false;
        else if (inFootprint(sx, sy)) ok = false;
        else for (const o of obstacles) if (Math.hypot(sx - o.x, sy - o.y) <= o.radius * 1.05) { ok = false; break; }
      }
      if (!ok) continue;
      // fit: dest − source around the ring, minus the best PLANE per channel
      // (heal's own correction removes any tone offset and smooth gradient),
      // scored by the robust (median) size of what's left — i.e. "does the
      // source's surrounding STRUCTURE match", independent of its tone
      const dl = [[], [], []], xs = [], ys = [];
      for (let i = 0; i < ring.length; i += 2) {
        const sx = Math.round(ring[i] + dx), sy = Math.round(ring[i + 1] + dy);
        if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue;
        xs.push(ring[i] - cxs); ys.push(ring[i + 1] - cys);
        for (let c = 0; c < 3; c++) dl[c].push((px(ring[i], ring[i + 1], c) - px(sx, sy, c)) / 255);
      }
      const nc = xs.length;
      if (nc < 6) continue;
      let context = 0;
      for (let c = 0; c < 3; c++) context += median(planeResiduals(xs, ys, dl[c]).map(Math.abs));
      let patch = 0;
      for (let i = 0; i < hole.length; i += 2) {
        const sx = Math.round(hole[i] + dx), sy = Math.round(hole[i + 1] + dy);
        for (let c = 0; c < 3; c++) { const e = (px(sx, sy, c) - med[c]) / 255; patch += e * e; }
      }
      const score = context + HEAL_SRC_PATCH_W * patch / (hole.length / 2) + 0.0005 * mult;
      if (!best || score < best.score) best = { score, dx, dy };
    }
  }
  return best ? { dx: best.dx, dy: best.dy } : null;
}

/* Mutates buf in place (a Node Buffer or a Uint8ClampedArray alike) and
   returns it. `stamp` is an array of strokes (see header); strokes are
   composited in array order, so a later stroke's source/blend can
   legitimately read pixels an earlier stroke in the same call already
   modified (matches stacking clone-stamp edits in Photoshop). No-ops
   when isEmptyStamp(stamp). */
/* strokes the last applyStampToRGBBuffer left unfinished: a Remove stroke
   still waiting for its background fill or its network result. 0 = what
   it drew is final (every stroke's output is in the stroke cache). */
let pendingStrokes = 0;
export function stampPending() { return pendingStrokes; }
export function applyStampToRGBBuffer(buf, width, height, channels, stamp) {
  pendingStrokes = 0;
  if (isEmptyStamp(stamp)) return buf;
  for (const stroke of normalizeStamp(stamp)) {
    applyStrokeToBuffer(buf, width, height, channels, stroke);
  }
  return buf;
}

/* ---- persisting the stroke cache (renderer only) ----
   A fresh app session has an empty stroke cache, so reopening a photo
   with many Remove strokes used to redo all of them, in a chain (each
   waits for the strokes under it). setStrokeCacheSink receives every final
   stroke output (to store on disk); restoreStrokeCache walks a stroke
   list over a copy of the pristine pixels, asking `load(key)` for each
   stroke's stored output and replaying it, so its hash for the next
   stroke is right. A stroke that isn't stored leaves its read box unknown,
   so later strokes reading any of it are skipped too (their keys hash
   those pixels); strokes elsewhere still restore. After that, a normal
   composite only computes what's actually missing. */
let strokeCacheSink = null;
export function setStrokeCacheSink(fn) { strokeCacheSink = fn; }
export function packStrokeEntry(entry) {
  const n = entry.keys.length, out = new Uint8Array(4 + n * 7);
  new DataView(out.buffer).setUint32(0, n, true);
  out.set(new Uint8Array(entry.keys.buffer, entry.keys.byteOffset, n * 4), 4);
  out.set(entry.rgb, 4 + n * 4);
  return out;
}
function unpackStrokeEntry(bytes) {
  if (!bytes || bytes.length < 4) return null;
  const n = new DataView(bytes.buffer, bytes.byteOffset, bytes.length).getUint32(0, true);
  if (bytes.length !== 4 + n * 7) return null;
  const keys = new Int32Array(n);
  new Uint8Array(keys.buffer).set(bytes.subarray(4, 4 + n * 4));
  return { keys, rgb: bytes.slice(4 + n * 4) };
}
export async function restoreStrokeCache(buf, width, height, channels, stamp, load) {
  let restored = 0;
  const unknown = []; // read boxes of strokes not restored: their pixels aren't final here
  const touches = (b) => unknown.some((u) => b[0] <= u[2] && u[0] <= b[2] && b[1] <= u[3] && u[1] <= b[3]);
  for (const stroke of normalizeStamp(stamp)) {
    const radius = stroke.radius, pts = stroke.points;
    if (!(radius > 0) || !pts || !pts.length) continue;
    const feather = Number.isFinite(stroke.feather) ? stroke.feather : DEFAULT_FEATHER;
    const heal = stroke.mode === "heal";
    const dxs = pts.map((p) => (heal ? stroke.dx : (p.dx ?? stroke.dx)));
    const dys = pts.map((p) => (heal ? stroke.dy : (p.dy ?? stroke.dy)));
    const outer = radius * (1 + feather);
    const reach = outer + (heal ? Math.max(HEAL_RING_MIN, radius * HEAL_RING_FRAC) : 0);
    const box = readBox(pts, dxs, dys, reach, width, height);
    if (!box) continue;
    if (touches(box)) { unknown.push(box); continue; }
    const key = strokeCacheKey(stroke, width, height, buf, channels, ...box);
    let entry = strokeCache.get(key);
    if (!entry) {
      entry = unpackStrokeEntry(await load(key));
      if (!entry) { unknown.push(box); continue; }
      strokeCachePut(key, entry, false);
      restored++;
    }
    replayStroke(entry, buf, channels);
  }
  return restored;
}
