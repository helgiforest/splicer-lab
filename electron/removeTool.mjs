/* ============================================================
   Content-aware REMOVE — the fill behind the stamp tool's "Remove" mode
   (stampTool.mjs, `mode: "remove"`): paint over something, and the area is
   rebuilt from many small pieces of the photo around it, with no source to
   pick. This is the same family of algorithm as Lightroom's default
   Content-Aware Remove (Camera Raw's PatchMatch fill: a nearest-neighbor
   field refined coarse-to-fine, see its "upsampleNNF"/"PatchMatchFillHole"
   stages), written here from the published papers, not from anyone's code:
     - PatchMatch: Barnes, Shechtman, Finkelstein, Goldman, "PatchMatch: a
       randomized correspondence algorithm for structural image editing",
       SIGGRAPH 2009 — the fast nearest-neighbor-field search
     - Wexler, Shechtman, Irani, "Space-Time Completion of Video", 2007 —
       the fill itself: iterate "find each patch's best match elsewhere" /
       "re-estimate every hole pixel from all the matches covering it",
       coarse-to-fine over an image pyramid

   Lessons from this project's earlier, removed fill tool (see project
   memory, project_heal_tool.md) are built in on purpose:
     - The search runs on a CROP around the stroke at the buffer's own
       resolution — never on a downsampled whole photo, which averaged real
       grain away before the search even saw it ("swirl"). A crop is only
       downsampled when the hole itself is huge (REMOVE_MAX_HOLE_PX).
     - The FINEST level takes each pixel straight from its own match (no
       color voting, no offset smoothing) — averaging there turned grain
       into mush. Voting is only used on the coarser levels, where it's
       what makes the structure converge.
     - Deterministic: a seeded RNG per stroke, so the same stroke on the
       same pixels gives the same result every time it's re-rendered.
   ============================================================ */

const PR = 3;                       // patch radius → 7×7 patches
const REMOVE_PAD_FACTOR = 2.2;      // crop padding around the hole, × the hole's effective diameter
const REMOVE_PAD_MIN = 40;          // px
const REMOVE_MAX_HOLE_PX = 60000;   // hole pixels above this → work on a downsampled crop (speed cap)
const REMOVE_COARSE_HOLE = 10;      // stop building the pyramid once the hole is this small (px across)
const EM_COARSE = 5, EM_FINE = 2;   // EM iterations on coarse levels / on the finest level
const PM_PASSES = 2;                // PatchMatch propagation + random-search passes per EM iteration
const NPIX = (2 * PR + 1) ** 2;
// tone ("bias") compensation, as in Camera Raw's PatchMatch gain/bias
// parameters: a source patch may be matched — and copied — shifted in
// tone by the difference of the two patches' means, so a piece of texture
// from a lighter or darker spot can still be used where it fits
// structurally. Penalized (BIAS_PENALTY) so same-tone matches still win
// when they exist, and capped (BIAS_MAX, 0-255 units per channel).
const BIAS_PENALTY = 0.95;
const BIAS_MAX = 6;

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

function downsample(level) {
  const { w, h, img, hole } = level;
  const w2 = Math.max(1, Math.ceil(w / 2)), h2 = Math.max(1, Math.ceil(h / 2));
  const img2 = new Float32Array(w2 * h2 * 3), hole2 = new Uint8Array(w2 * h2);
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    let r = 0, g = 0, b = 0, n = 0, anyHole = 0;
    for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
      const sx = 2 * x + i, sy = 2 * y + j;
      if (sx >= w || sy >= h) continue;
      const q = sy * w + sx;
      if (hole[q]) { anyHole = 1; continue; }
      r += img[q * 3]; g += img[q * 3 + 1]; b += img[q * 3 + 2]; n++;
    }
    const q2 = y * w2 + x;
    hole2[q2] = anyHole;
    if (n) { img2[q2 * 3] = r / n; img2[q2 * 3 + 1] = g / n; img2[q2 * 3 + 2] = b / n; }
  }
  return { w: w2, h: h2, img: img2, hole: hole2 };
}

/* where a patch may be COPIED FROM: fully inside the crop, touching no hole */
function validSources(w, h, hole) {
  const sat = new Int32Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) sat[(y + 1) * (w + 1) + x + 1] = hole[y * w + x] + sat[y * (w + 1) + x + 1] + sat[(y + 1) * (w + 1) + x] - sat[y * (w + 1) + x];
  const ok = new Uint8Array(w * h);
  const list = [];
  for (let y = PR; y < h - PR; y++) for (let x = PR; x < w - PR; x++) {
    const x0 = x - PR, y0 = y - PR, x1 = x + PR + 1, y1 = y + PR + 1;
    const s = sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0] + sat[y0 * (w + 1) + x0];
    if (s === 0) { ok[y * w + x] = 1; list.push(y * w + x); }
  }
  return { ok, list };
}

/* patches to fill: every patch center whose 7×7 window touches the hole
   (so boundary patches anchor the fill to what's really there) */
function targetsOf(w, h, hole) {
  const t = [];
  for (let y = PR; y < h - PR; y++) for (let x = PR; x < w - PR; x++) {
    let touches = false;
    for (let j = -PR; j <= PR && !touches; j++) for (let i = -PR; i <= PR; i++) if (hole[(y + j) * w + x + i]) { touches = true; break; }
    if (touches) t.push(y * w + x);
  }
  return t;
}

/* per-pixel 7×7 window means (3 channels) — the patch means bias uses */
function patchMeans(img, w, h) {
  const tmp = new Float32Array(w * h * 3), out = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    let s = 0, n = 0;
    for (let i = Math.max(0, x - PR); i <= Math.min(w - 1, x + PR); i++) { s += img[(y * w + i) * 3 + c]; n++; }
    tmp[(y * w + x) * 3 + c] = s / n;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, y - PR); j <= Math.min(h - 1, y + PR); j++) { s += tmp[(j * w + x) * 3 + c]; n++; }
    out[(y * w + x) * 3 + c] = s / n;
  }
  return out;
}

const clampBias = (v) => (v > BIAS_MAX ? BIAS_MAX : v < -BIAS_MAX ? -BIAS_MAX : v);

/* bias-compensated patch distance: Σ((T − S) − Δ)² + penalty·n·ΣΔ², with
   Δ = clamp(mean(T) − mean(S)) per channel. Since Σ(T − S) = n·(mean
   difference), the compensated SSD is plain SSD minus a closed-form term,
   so the early-exit cap just moves by that amount. */
function biasDist(img, means, w, a, b, cap) {
  let corr = 0;
  for (let c = 0; c < 3; c++) {
    const dm = means[a * 3 + c] - means[b * 3 + c], d = clampBias(dm);
    corr += NPIX * (2 * d * dm - d * d) - BIAS_PENALTY * NPIX * d * d;
  }
  return patchDist(img, w, a, b, cap + corr) - corr;
}

function patchDist(img, w, a, b, cap) {
  let d = 0;
  for (let j = -PR; j <= PR; j++) {
    const ra = (a + j * w) * 3, rb = (b + j * w) * 3;
    for (let i = -PR * 3; i <= PR * 3 + 2; i++) {
      const e = img[ra + i] - img[rb + i];
      d += e * e;
    }
    if (d >= cap) return d;
  }
  return d;
}

/* coarsest-level starting guess: peel the hole inward, each pixel the
   average of its already-known neighbors */
function onionFill(level) {
  const { w, h, img, hole } = level;
  const known = new Uint8Array(w * h);
  for (let q = 0; q < w * h; q++) known[q] = hole[q] ? 0 : 1;
  let left = true;
  while (left) {
    left = false;
    const next = [];
    for (let q = 0; q < w * h; q++) {
      if (known[q]) continue;
      const x = q % w, y = (q - x) / w;
      let r = 0, g = 0, b = 0, n = 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const p = ny * w + nx;
        if (!known[p]) continue;
        r += img[p * 3]; g += img[p * 3 + 1]; b += img[p * 3 + 2]; n++;
      }
      if (n) next.push([q, r / n, g / n, b / n]); else left = true;
    }
    if (!next.length) break;
    for (const [q, r, g, b] of next) { img[q * 3] = r; img[q * 3 + 1] = g; img[q * 3 + 2] = b; known[q] = 1; }
  }
}

/* PatchMatch: improve nnf (target → source center) in place */
function patchMatch(level, targets, nnf, nnd, src, rand, passes, means) {
  const { w, h, img } = level;
  const tIndex = new Int32Array(w * h).fill(-1);
  targets.forEach((t, k) => { tIndex[t] = k; });
  const tryCand = (k, t, s) => {
    if (s < 0 || s >= w * h || !src.ok[s]) return;
    const d = biasDist(img, means, w, t, s, nnd[k]);
    if (d < nnd[k]) { nnd[k] = d; nnf[k] = s; }
  };
  for (let pass = 0; pass < passes; pass++) {
    const fwd = pass % 2 === 0;
    for (let n = 0; n < targets.length; n++) {
      const k = fwd ? n : targets.length - 1 - n;
      const t = targets[k];
      const x = t % w;
      // propagation: a neighbor's match, shifted by one
      const step = fwd ? -1 : 1;
      const kl = x + step >= 0 && x + step < w ? tIndex[t + step] : -1;
      if (kl >= 0) tryCand(k, t, nnf[kl] - step);
      const ku = tIndex[t + step * w] ?? -1;
      if (ku >= 0) tryCand(k, t, nnf[ku] - step * w);
      // random search, exponentially shrinking window around the current match
      const sx = nnf[k] % w, sy = (nnf[k] - sx) / w;
      for (let R = Math.max(w, h); R >= 1; R = Math.floor(R / 2)) {
        const cx = Math.round(sx + (rand() * 2 - 1) * R), cy = Math.round(sy + (rand() * 2 - 1) * R);
        if (cx < PR || cy < PR || cx >= w - PR || cy >= h - PR) continue;
        tryCand(k, t, cy * w + cx);
      }
    }
  }
}

/* EM "M" step: every hole pixel = similarity-weighted average of what all
   the matches covering it say it should be */
function vote(level, targets, nnf, nnd, means) {
  const { w, h, img, hole } = level;
  const acc = new Float64Array(w * h * 3), ws = new Float64Array(w * h);
  const sorted = Float64Array.from(nnd).sort();
  const sigma2 = Math.max(1, sorted[Math.floor(sorted.length * 0.75)] / ((2 * PR + 1) ** 2 * 3));
  for (let k = 0; k < targets.length; k++) {
    const t = targets[k], s = nnf[k];
    const wt = Math.exp(-Math.max(0, nnd[k]) / ((2 * PR + 1) ** 2 * 3) / (2 * sigma2));
    const b0 = clampBias(means[t * 3] - means[s * 3]), b1 = clampBias(means[t * 3 + 1] - means[s * 3 + 1]), b2 = clampBias(means[t * 3 + 2] - means[s * 3 + 2]);
    for (let j = -PR; j <= PR; j++) for (let i = -PR; i <= PR; i++) {
      const q = t + j * w + i;
      if (!hole[q]) continue;
      const p = s + j * w + i;
      acc[q * 3] += wt * (img[p * 3] + b0); acc[q * 3 + 1] += wt * (img[p * 3 + 1] + b1); acc[q * 3 + 2] += wt * (img[p * 3 + 2] + b2);
      ws[q] += wt;
    }
  }
  for (let q = 0; q < w * h; q++) if (hole[q] && ws[q] > 0) { img[q * 3] = acc[q * 3] / ws[q]; img[q * 3 + 1] = acc[q * 3 + 1] / ws[q]; img[q * 3 + 2] = acc[q * 3 + 2] / ws[q]; }
}

/* Fills the hole of a w×h RGB crop (Float32, 0-255) in place.
   hole: Uint8Array (1 = to fill). seed: deterministic RNG seed. */
export function inpaint(img, w, h, hole, seed = 1) {
  const rand = rng(seed);
  // pyramid until the hole is small
  const levels = [{ w, h, img, hole }];
  const holeSpan = (lv) => { let n = 0; for (const v of lv.hole) n += v; return Math.sqrt(n); };
  while (holeSpan(levels[levels.length - 1]) > REMOVE_COARSE_HOLE && levels[levels.length - 1].w > 4 * (2 * PR + 1) && levels[levels.length - 1].h > 4 * (2 * PR + 1)) {
    levels.push(downsample(levels[levels.length - 1]));
  }
  let prev = null; // { level, targets, nnf }
  for (let l = levels.length - 1; l >= 0; l--) {
    const lv = levels[l];
    const src = validSources(lv.w, lv.h, lv.hole);
    if (!src.list.length) { onionFill(lv); prev = null; continue; }
    const targets = targetsOf(lv.w, lv.h, lv.hole);
    const nnf = new Int32Array(targets.length), nnd = new Float64Array(targets.length).fill(Infinity);
    if (!prev) {
      onionFill(lv);
      for (let k = 0; k < targets.length; k++) nnf[k] = src.list[Math.floor(rand() * src.list.length)];
    } else {
      // upsample the coarser field: same match, scaled ×2, same sub-offset
      const P = prev.level, pIndex = new Int32Array(P.w * P.h).fill(-1);
      prev.targets.forEach((t, k) => { pIndex[t] = k; });
      for (let k = 0; k < targets.length; k++) {
        const t = targets[k], x = t % lv.w, y = (t - x) / lv.w;
        const px = Math.min(P.w - 1, x >> 1), py = Math.min(P.h - 1, y >> 1);
        const pk = pIndex[py * P.w + px];
        let s = -1;
        if (pk >= 0) {
          const ps = prev.nnf[pk], psx = ps % P.w, psy = (ps - psx) / P.w;
          const sx = 2 * psx + (x & 1), sy = 2 * psy + (y & 1);
          if (sx >= 0 && sy >= 0 && sx < lv.w && sy < lv.h && src.ok[sy * lv.w + sx]) s = sy * lv.w + sx;
        }
        nnf[k] = s >= 0 ? s : src.list[Math.floor(rand() * src.list.length)];
      }
      // starting colors for this level's hole: vote with the upsampled
      // field over a smooth estimate (onion fill — gives bias a sane mean)
      onionFill(lv);
      const m0 = patchMeans(lv.img, lv.w, lv.h);
      for (let k = 0; k < targets.length; k++) nnd[k] = biasDist(lv.img, m0, lv.w, targets[k], nnf[k], Infinity);
      vote(lv, targets, nnf, nnd, m0);
    }
    let means = patchMeans(lv.img, lv.w, lv.h);
    for (let k = 0; k < targets.length; k++) nnd[k] = biasDist(lv.img, means, lv.w, targets[k], nnf[k], Infinity);
    const iters = l === 0 ? EM_FINE : EM_COARSE;
    for (let it = 0; it < iters; it++) {
      patchMatch(lv, targets, nnf, nnd, src, rand, PM_PASSES, means);
      if (l > 0 || it < iters - 1) {
        vote(lv, targets, nnf, nnd, means);
        means = patchMeans(lv.img, lv.w, lv.h);
        for (let k = 0; k < targets.length; k++) nnd[k] = biasDist(lv.img, means, lv.w, targets[k], nnf[k], Infinity);
      }
    }
    lv.means = means;
    if (l === 0) {
      // finest level: each hole pixel straight from its own patch's match
      // (no averaging — keeps the photo's real grain; see header)
      const own = new Int32Array(lv.w * lv.h).fill(-1);
      targets.forEach((t, k) => { own[t] = nnf[k]; });
      for (let q = 0; q < lv.w * lv.h; q++) {
        if (!lv.hole[q] || own[q] < 0) continue;
        const s = own[q], m = lv.means;
        // the match's own pixel, shifted by its patch's tone bias
        for (let c = 0; c < 3; c++) lv.img[q * 3 + c] = lv.img[s * 3 + c] + clampBias(m[q * 3 + c] - m[s * 3 + c]);
      }
    }
    prev = { level: lv, targets, nnf };
  }
  return img;
}

/* ---- neural shape + matched grain (the hybrid) ----
   When a local neural fill for the same crop is available (MI-GAN or
   LaMa, run by electron/neuralFill.mjs in its own utility process), the result is
     the network's fill  — the SHAPE: how a shadow, a wall, a fence
                           continues through the hole, which the network
                           gets right where patch copying can't; scaled to
                           the buffer bilinearly (a big hole is worked on a
                           downsampled crop — nearest-neighbor scaling made
                           visible blocks)
   + synthetic GRAIN     — noise matched to the real grain right around the
                           hole (per-channel strength, channel correlation,
                           pixel-to-pixel correlation), only as much as the
                           network's smoother fill is missing.
   The grain used to be PatchMatch's high frequencies; on a large hole
   that carried whole copied OBJECTS into the fill (a red-white barrier
   pasted over a removed vehicle), so it's measured, not copied, now.

   Neural inference is async and this function is sync, so it's two-step:
   without a result, this returns the PatchMatch fill and queues a job
   (takeNeuralJobs); whoever can run the network — GLImage over IPC, or
   bakeRaw — does so, calls putNeuralResult, and re-renders. The
   job key is the same key the fill is cached under (stroke geometry +
   crop + pixel hash), so a result can only ever land on the exact pixels
   it was computed from. */
// engine per hole, like Camera Raw choosing among its inpainters: MI-GAN
// (fast) for ordinary spots, LaMa (stronger on structure, ~7× slower) once
// the hole covers more than this fraction of the buffer — a FRACTION, not
// pixels, so the proxy preview and the full-res export pick the same one
const LAMA_HOLE_FRACTION = 0.002;
// mode: the user's opt-in choice — "off" (PatchMatch only), "fast" (MI-GAN
// for every hole) or "max" (LaMa for big holes once downloaded); null =
// never asked = off. Results are keyed per engine (`${key}|${engine}`), so
// switching modes re-renders with the right one instead of a stale result.
const neural = { enabled: false, mode: null, lama: false, version: 0, results: new Map(), pending: new Map(), inflight: new Set(), failed: new Set() };
const NEURAL_RESULTS_MAX = 32, NEURAL_PENDING_MAX = 16;

/* Turn queuing on only where something will actually run the jobs. */
export function setNeuralEnabled(on) { neural.enabled = !!on; }
export function setNeuralMode(mode, lamaReady) { neural.mode = mode; neural.lama = !!lamaReady; neural.version++; }
/* what a Remove stroke's final look depends on besides its pixels */
export function neuralState() { return `${neural.mode}|${neural.lama}`; }
/* bumped whenever a network result lands (or the mode changes): a stroke
   still waiting for one may look different after it */
export function neuralVersion() { return neural.version; }
function engineFor(holePx, bufferPx) {
  if (neural.mode === "fast") return "migan";
  if (neural.mode === "max") return neural.lama && holePx / bufferPx > LAMA_HOLE_FRACTION ? "lama" : "migan";
  return null;
}
export function takeNeuralJobs() {
  const jobs = [...neural.pending.values()];
  neural.pending.clear();
  for (const j of jobs) neural.inflight.add(j.key);
  return jobs;
}
/* rgb: Uint8Array w·h·3 (the network's output for job.key), or null on
   failure — remembered, so a failing network isn't re-asked on every redraw */
export function putNeuralResult(key, rgb) {
  neural.inflight.delete(key);
  neural.version++;
  if (!rgb) { neural.failed.add(key); return; }
  neural.results.set(key, rgb);
  if (neural.results.size > NEURAL_RESULTS_MAX) neural.results.delete(neural.results.keys().next().value);
}
/* how many Remove fills are still being computed (PatchMatch or network) */
export function fillsInProgress() {
  return fill.pending.size + fill.inflight.size + neural.pending.size + neural.inflight.size;
}

function sampleBilinear(src, w, h, fx, fy, out) {
  const x = Math.min(Math.max(fx, 0), w - 1), y = Math.min(Math.max(fy, 0), h - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const tx = x - x0, ty = y - y0;
  for (let c = 0; c < 3; c++) {
    const a = src[(y0 * w + x0) * 3 + c], b = src[(y0 * w + x1) * 3 + c];
    const d = src[(y1 * w + x0) * 3 + c], e = src[(y1 * w + x1) * 3 + c];
    const top = a + (b - a) * tx;
    out[c] = top + (d + (e - d) * tx - top) * ty;
  }
}

/* deterministic standard-normal value for an integer cell + channel */
function gauss(x, y, c, seed) {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(c + 1, 2246822519) ^ seed;
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16;
  let g = Math.imul(h, 2654435761); g = Math.imul(g ^ (g >>> 15), 2246822519); g ^= g >>> 13;
  const u1 = ((h >>> 0) + 1) / 4294967297, u2 = (g >>> 0) / 4294967296;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const GRAIN_RING = 14;   // px around the hole whose grain is measured
const GRAIN_SD_MAX = 8;  // levels: grain, not texture — busy rings (foliage) cap here

/* robust σ of a zero-mean sample: 1.4826 × median |v| (ignores the few
   large values an edge in the ring contributes) */
function robustSigma(vals) {
  if (!vals.length) return 0;
  const a = new Float32Array(vals.length);
  for (let i = 0; i < a.length; i++) a[i] = Math.abs(vals[i]);
  return 1.4826 * kthSmallest(a, a.length >> 1);
}
/* the k-th smallest value (what a[k] would be after a full sort), by
   in-place quickselect — O(n) instead of sorting a whole big hole's worth */
function kthSmallest(a, k) {
  let lo = 0, hi = a.length - 1;
  while (hi > lo) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j; else if (k >= i) lo = i; else break;
  }
  return a[k];
}

/* base + synthetic grain matched to the grain just outside the hole (see
   the hybrid's header). hf = pixel − its 3×3 mean, measured only where the
   whole 3×3 is real (outside the hole) / entirely inside the hole (base).
   The ring usually crosses edges (a wall's top, a kerb, the next object):
   plain variance read those as "grain" and sprayed loud colored noise into
   the fill, so every statistic is robust — σ from the median, correlation
   and lag only over samples within 3σ. */
function addMatchedGrain(buf, width, height, channels, holeKeys, base, key) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const k of holeKeys) { const x = k % width, y = (k - x) / width; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  const bx0 = Math.max(0, x0 - GRAIN_RING), by0 = Math.max(0, y0 - GRAIN_RING);
  const bx1 = Math.min(width - 1, x1 + GRAIN_RING), by1 = Math.min(height - 1, y1 + GRAIN_RING);
  const bw = bx1 - bx0 + 1, bh = by1 - by0 + 1;
  const inHole = new Int32Array(bw * bh).fill(-1);
  holeKeys.forEach((k, idx) => { const x = k % width, y = (k - x) / width; inHole[(y - by0) * bw + (x - bx0)] = idx; });
  const val = (x, y, c) => { const i = inHole[(y - by0) * bw + (x - bx0)]; return i >= 0 ? base[i * 3 + c] : buf[(y * width + x) * channels + c]; };
  // hf samples: ring (real pixels) with their scan position, and base
  const ring = [[], [], []], ringPos = [], fill = [[], [], []];
  const hf = new Float32Array(3);
  for (let y = by0 + 1; y < by1; y++) {
    for (let x = bx0 + 1; x < bx1; x++) {
      let holes = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) if (inHole[(y + j - by0) * bw + (x + i - bx0)] >= 0) holes++;
      if (holes !== 0 && holes !== 9) continue;
      for (let c = 0; c < 3; c++) {
        let m = 0;
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) m += val(x + i, y + j, c);
        hf[c] = val(x, y, c) - m / 9;
      }
      const dst = holes === 0 ? ring : fill;
      dst[0].push(hf[0]); dst[1].push(hf[1]); dst[2].push(hf[2]);
      if (holes === 0) ringPos.push(y * width + x);
    }
  }
  const out = Float32Array.from(base);
  const n = ringPos.length;
  if (n < 30) return out;
  const sr = [0, 1, 2].map((c) => robustSigma(ring[c]));
  const sf = [0, 1, 2].map((c) => robustSigma(fill[c]));
  // correlation and lag-1 over the inlier samples only
  const inl = new Uint8Array(n);
  let cov = [0, 0, 0], v = [0, 0, 0], m = 0;
  for (let t = 0; t < n; t++) {
    const a = ring[0][t], b = ring[1][t], c = ring[2][t];
    if (Math.abs(a) > 3 * sr[0] || Math.abs(b) > 3 * sr[1] || Math.abs(c) > 3 * sr[2]) continue;
    inl[t] = 1; m++;
    v[0] += a * a; v[1] += b * b; v[2] += c * c;
    cov[0] += a * b; cov[1] += b * c; cov[2] += a * c;
  }
  if (m < 30) return out;
  v = v.map((x) => x / m); cov = cov.map((x) => x / m);
  const rho = Math.min(0.95, Math.max(0, (cov[0] / Math.sqrt(v[0] * v[1] + 1e-9) + cov[1] / Math.sqrt(v[1] * v[2] + 1e-9) + cov[2] / Math.sqrt(v[0] * v[2] + 1e-9)) / 3));
  const lumVar = (v[0] + v[1] + v[2] + 2 * (cov[0] + cov[1] + cov[2])) / 9;
  let lag = 0, lagN = 0;
  for (let t = 1; t < n; t++) {
    if (!inl[t] || !inl[t - 1] || ringPos[t] !== ringPos[t - 1] + 1) continue;
    lag += ((ring[0][t] + ring[1][t] + ring[2][t]) / 3) * ((ring[0][t - 1] + ring[1][t - 1] + ring[2][t - 1]) / 3);
    lagN++;
  }
  const lag1 = lagN && lumVar > 1e-6 ? lag / lagN / lumVar : 0;
  const sd = [0, 1, 2].map((c) => Math.min(GRAIN_SD_MAX, Math.sqrt(Math.max(0, sr[c] * sr[c] - sf[c] * sf[c]))));
  if (sd[0] + sd[1] + sd[2] < 0.05) return out;
  // hf = pixel − 3×3 mean removes 1/9 of a white field's variance per pixel
  // and more of a correlated one — compensate so the ADDED noise's hf
  // matches the measured hf
  const smooth = lag1 > 0.2;
  const hfGain = smooth ? 1 / 0.62 : 1 / Math.sqrt(8 / 9);
  let seed = 2166136261;
  for (let i = 0; i < key.length; i++) seed = Math.imul(seed ^ key.charCodeAt(i), 16777619);
  const noise = (x, y, c) => {
    if (!smooth) return gauss(x, y, c, seed);
    // 3×3 binomial-correlated noise, renormalized to unit variance (÷0.375)
    let a = 0;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) a += (i ? 1 : 2) * (j ? 1 : 2) * gauss(x + i, y + j, c, seed);
    return a / 16 / 0.375;
  };
  const a = Math.sqrt(rho), b = Math.sqrt(1 - rho);
  holeKeys.forEach((k, idx) => {
    const x = k % width, y = (k - x) / width;
    const shared = noise(x, y, 3);
    for (let c = 0; c < 3; c++) out[idx * 3 + c] = base[idx * 3 + c] + hfGain * sd[c] * (a * shared + b * noise(x, y, c));
  });
  return out;
}

/* ---- the stamp-tool entry point, with a small result cache ---- */
const cache = new Map(); // key → { w, h, pm: crop (PatchMatch), input, hole, nnOut?: final colors with the neural fill }
const CACHE_MAX = 32;

export function hashRegion(buf, width, channels, x0, y0, x1, y1) {
  let h = 2166136261 >>> 0;
  const step = Math.max(1, Math.floor(((x1 - x0 + 1) * (y1 - y0 + 1)) / 4096));
  let k = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++, k++) {
    if (k % step) continue;
    const o = (y * width + x) * channels;
    h = Math.imul(h ^ buf[o], 16777619) ^ Math.imul(buf[o + 1], 31) ^ Math.imul(buf[o + 2], 131);
  }
  return h >>> 0;
}

/* `holeKeys`: buffer pixel indices (y·width + x) to fill. Returns
   Map(key → [r, g, b]). Reads `buf` only. */
/* ---- PatchMatch in the background ----
   PatchMatch takes up to ~2 s on a big stroke (M-series; slower PCs
   several times that), so the on-screen composite doesn't run it inline:
   with setAsyncFill(true) a cache miss queues a job (takeFillJobs) for a
   Web Worker that runs `inpaint`, and putFillResult stores what it
   returns. Export, thumbnails and the histogram stay synchronous. */
// waiting: crops of this composite's strokes still being filled
const fill = { async: false, waiting: [], pending: new Map(), inflight: new Map() };
export function setAsyncFill(on) { fill.async = !!on; if (on) fill.waiting = []; }
/* after a background-mode composite: did it leave any Remove stroke out? */
export function fillDeferred() { return fill.waiting.length > 0; }
export function takeFillJobs() {
  const jobs = [...fill.pending.values()];
  fill.pending.clear();
  for (const j of jobs) fill.inflight.set(j.key, j);
  return jobs;
}
/* pm: the filled crop (w·h·3 bytes, the job's own w/h), or null on failure */
export function putFillResult(key, pm) {
  const job = fill.inflight.get(key);
  fill.inflight.delete(key);
  if (job && pm && pm.length === job.w * job.h * 3) makeEntry(key, job.w, job.h, Uint8ClampedArray.from(pm), job.hole);
}

function seedOf(key) {
  let seed = 2166136261;
  for (let i = 0; i < key.length; i++) seed = Math.imul(seed ^ key.charCodeAt(i), 16777619);
  return seed;
}

/* the work crop: hole pixels flagged, the rest averaged down by `s` */
function fillCrop(buf, width, channels, holeKeys, cx0, cy0, cx1, cy1, s, w, h) {
  const img = new Float32Array(w * h * 3), cnt = new Float32Array(w * h), hole = new Uint8Array(w * h);
  const fw = cx1 - cx0 + 1, inHole = new Uint8Array(fw * (cy1 - cy0 + 1));
  for (const k of holeKeys) { const x = k % width; inHole[((k - x) / width - cy0) * fw + x - cx0] = 1; }
  for (let y = cy0; y <= cy1; y++) for (let x = cx0; x <= cx1; x++) {
    const gx = Math.min(w - 1, Math.floor((x - cx0) * s)), gy = Math.min(h - 1, Math.floor((y - cy0) * s));
    const g = gy * w + gx, o = (y * width + x) * channels;
    if (inHole[(y - cy0) * fw + x - cx0]) { hole[g] = 1; continue; }
    img[g * 3] += buf[o]; img[g * 3 + 1] += buf[o + 1]; img[g * 3 + 2] += buf[o + 2]; cnt[g]++;
  }
  for (let g = 0; g < w * h; g++) if (cnt[g]) { img[g * 3] /= cnt[g]; img[g * 3 + 1] /= cnt[g]; img[g * 3 + 2] /= cnt[g]; } else hole[g] = 1;
  return { img, hole };
}

/* the network gets the PatchMatch-filled crop, not a crop with an empty
   hole: MI-GAN's pipeline does NOT fully ignore pixels under the mask — a
   black hole came back as a dark fill (center 45,49,32 where ~150,170,170
   was right); any sensible first guess there gives ~1-level accuracy */
function makeEntry(key, w, h, pm, hole) {
  const entry = { w, h, pm, input: new Uint8Array(pm.buffer), hole, nnOut: {} };
  cache.set(key, entry);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
  return entry;
}

export function removeFill(buf, width, height, channels, holeKeys, strokeKey) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const key of holeKeys) {
    const x = key % width, y = (key - x) / width;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const eff = Math.sqrt(holeKeys.length);
  const pad = Math.max(REMOVE_PAD_MIN, Math.round(REMOVE_PAD_FACTOR * eff)) + PR + 2;
  const cx0 = Math.max(0, x0 - pad), cy0 = Math.max(0, y0 - pad);
  const cx1 = Math.min(width - 1, x1 + pad), cy1 = Math.min(height - 1, y1 + pad);
  const key = `${strokeKey}|${width}x${height}|${cx0},${cy0},${cx1},${cy1}|${hashRegion(buf, width, channels, cx0, cy0, cx1, cy1)}`;
  // work scale: shrink only a huge hole (speed cap)
  const s = Math.min(1, Math.sqrt(REMOVE_MAX_HOLE_PX / Math.max(1, holeKeys.length)));
  const fw = cx1 - cx0 + 1, fh = cy1 - cy0 + 1;
  const w = Math.max(1, Math.round(fw * s)), h = Math.max(1, Math.round(fh * s));

  let entry = cache.get(key);
  if (!entry && fill.async) {
    // background mode (on-screen composite): never run PatchMatch here —
    // queue it for the worker and leave this stroke unfilled until it
    // lands. A stroke whose crop overlaps one that's still waiting waits
    // too: its pixels will change once that one is filled, so computing
    // it now would be wasted. Separate strokes are filled in parallel.
    const overlaps = fill.waiting.some((r) => cx0 <= r[2] && r[0] <= cx1 && cy0 <= r[3] && r[1] <= cy1);
    if (!overlaps && !fill.pending.has(key) && !fill.inflight.has(key)) {
      const job = fillCrop(buf, width, channels, holeKeys, cx0, cy0, cx1, cy1, s, w, h);
      fill.pending.set(key, { key, w, h, img: job.img, hole: job.hole, seed: seedOf(key) });
    }
    fill.waiting.push([cx0, cy0, cx1, cy1]);
    return null;
  }
  if (!entry) {
    const { img, hole } = fillCrop(buf, width, channels, holeKeys, cx0, cy0, cx1, cy1, s, w, h);
    inpaint(img, w, h, hole, seedOf(key));
    entry = makeEntry(key, w, h, Uint8ClampedArray.from(img), hole);
  }

  const engine = engineFor(holeKeys.length, width * height);
  const nkey = engine && `${key}|${engine}`;
  const nn = engine ? neural.results.get(nkey) : null;
  if (engine && !nn && neural.enabled && !neural.inflight.has(nkey) && !neural.pending.has(nkey) && !neural.failed.has(nkey)) {
    neural.pending.set(nkey, { key: nkey, w, h, rgb: entry.input, hole: entry.hole, engine });
    if (neural.pending.size > NEURAL_PENDING_MAX) neural.pending.delete(neural.pending.keys().next().value);
  }

  // base: the network's fill if there is one, else PatchMatch's — sampled
  // bilinearly from the (possibly downsampled) work crop
  const src = nn || entry.pm;
  const base = new Float32Array(holeKeys.length * 3), px = new Float32Array(3);
  holeKeys.forEach((k, idx) => {
    const x = k % width, y = (k - x) / width;
    sampleBilinear(src, w, h, (x - cx0 + 0.5) * s - 0.5, (y - cy0 + 0.5) * s - 0.5, px);
    base[idx * 3] = px[0]; base[idx * 3 + 1] = px[1]; base[idx * 3 + 2] = px[2];
  });
  if (nn && !entry.nnOut[engine]) entry.nnOut[engine] = addMatchedGrain(buf, width, height, channels, holeKeys, base, key);
  const final = nn ? entry.nnOut[engine] : base;

  const out = new Map();
  holeKeys.forEach((k, idx) => {
    out.set(k, [Math.round(final[idx * 3]), Math.round(final[idx * 3 + 1]), Math.round(final[idx * 3 + 2])]);
  });
  // final = nothing better is coming for this exact input (no network
  // result still to land) — only then may the caller cache what it drew
  out.final = !engine || !!nn;
  return out;
}
