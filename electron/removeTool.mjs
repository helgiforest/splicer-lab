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
   + the photo's DETAIL  — its grain and fine texture, copied from around
                           the hole (textureFill below); synthetic grain
                           matched to the ring (addMatchedGrain) only when
                           there's too little photo around to copy from.
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
// part of a network result's cache key (memory and disk) — bump when what
// the network is shown changes (neuralFill's LAMA_CONTEXT 0.15 → 0.6;
// e: the letterbox margin beyond a hole at the photo's edge is hole too)
const NET_LOOK = { lama: "|c0.6e" };
// part of a Remove stroke's cached OUTPUT key — bump when textureFill changes
export const FILL_LOOK = "tt5";
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
export function neuralState() { return `${neural.mode}|${neural.lama}${NET_LOOK.lama}`; }
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

/* ---- texture transfer: the network's shape, the photo's own detail ----
   The network (and PatchMatch on a downsampled crop) returns a SMOOTH fill:
   LaMa works at 512² and is scaled back up, and neither model makes film
   grain. Synthetic grain (addMatchedGrain) fixed the noise level but not
   the look — real grain is clumpy and signal-dependent, and a soft kerb
   edge stays soft under any noise. So the fill's detail is COPIED instead:
     L = the fill (hole) + the photo (around it), low-passed at σ — what
         the network got right; σ ≈ its blur scale, so L holds no
         detail the network didn't really have
     a guided PatchMatch (on a sparse 5×5 grid of L) finds, for the hole,
     real places around it whose L-neighborhood looks the same; the most
     common displacements win (plus each pixel's own match as a handicapped
     fallback — near the frame's edge most common ones point off it), and
     the hole is rebuilt band by band from the photo at those places:
       below σ    — grain: always the photo's (Σw·H / √Σw², so mixing two
                    spots keeps its strength)
       σ … 2.5σ   — fine texture: the photo's, except on the network's
                    own edges (a horizon, a kerb stay exactly where it put
                    them) and where no displacement really matches
       coarser    — the network's, except where both it and the photo are
                    nearly flat there: then the photo's evenness (the
                    network's colors drift in big blotches over a sky)
     Copying whole pieces per displacement, not per-pixel matches: those
     left seams ("worms") as soon as the detail had any size. The coarse
     bands are never copied over strong texture — doing it pixel by pixel
     made a patchwork of the ripples. Finally the fill's tone is matched to
     the photo along the hole's edge and spread inward.
   It also gives the stroke its edge automatically (the stroke's own
   feather/opacity no longer apply to Remove): within `band` of the
   footprint's edge, wherever the original pixels already agree with the
   fill (background the brush overlapped), the fill fades out into them
   (dithered by the caller, like the other modes' feather); where they
   don't (the object's rim, its shadow), the fill stays at full strength
   right up to the edge. */
const TT_PAD_MIN = 40, TT_PAD_MAX = 256; // px of real photo around the hole to copy detail from
const TT_ITERS = 5;                      // guided PatchMatch passes
const TT_GRID = 2;                       // sparse patch: (2·TT_GRID+1)² samples, `step` apart
const TT_SIGMA_STEPS = [1, 1.5, 2, 3, 4, 6];  // σ only goes up to TT_SIGMA_MAX; coarser steps can still be boosted
const TT_SIGMA_MAX = 3;      // at 6 a sky got sea-colored swirls copied in (yacht removal)
const TT_BOOST_FROM = 0.4, TT_BOOST_UNTIL = 0.9, TT_BOOST_MAX = 1.6; // washed-out bands above σ: boosted when under this share of the photo's energy, by at most ×
const TT_DISPS = 12;                     // displacements the detail is copied along
const OWN_PENALTY = 2;                   // log-cost handicap of a pixel's own match among them

/* box radii whose 3 passes approximate a Gaussian of σ (Kovesi) */
function boxRadii(sigma) {
  const n = 3, wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
  const m = Math.round((12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
  return [0, 1, 2].map((i) => ((i < m ? wl : wl + 2) - 1) >> 1);
}
function boxPass(src, dst, len, count, stride, step, r) {
  // `count` lines of `len` samples; sample k of line j at j·stride + k·step
  const inv = 1 / (2 * r + 1);
  for (let j = 0; j < count; j++) {
    const o = j * stride;
    let acc = 0;
    for (let i = -r; i <= r; i++) acc += src[o + Math.min(len - 1, Math.max(0, i)) * step];
    for (let k = 0; k < len; k++) {
      dst[o + k * step] = acc * inv;
      acc += src[o + Math.min(len - 1, k + r + 1) * step] - src[o + Math.max(0, k - r) * step];
    }
  }
}
function gaussBlur(plane, w, h, sigma) {
  let a = Float32Array.from(plane), b = new Float32Array(plane.length);
  for (const r of boxRadii(sigma)) {
    if (r < 1) continue;
    boxPass(a, b, w, h, w, 1, r); boxPass(b, a, h, w, 1, w, r);
  }
  return a;
}
/* chamfer (3-4) distance, in px, of every cell to the nearest cell where
   `isSeed` — two raster passes */
function chamfer(w, h, isSeed) {
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = isSeed(i) ? 0 : 1e9;
  const A = 1, B = Math.SQRT2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x; let v = d[i];
    if (x > 0) v = Math.min(v, d[i - 1] + A);
    if (y > 0) { v = Math.min(v, d[i - w] + A); if (x > 0) v = Math.min(v, d[i - w - 1] + B); if (x < w - 1) v = Math.min(v, d[i - w + 1] + B); }
    d[i] = v;
  }
  for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
    const i = y * w + x; let v = d[i];
    if (x < w - 1) v = Math.min(v, d[i + 1] + A);
    if (y < h - 1) { v = Math.min(v, d[i + w] + A); if (x < w - 1) v = Math.min(v, d[i + w + 1] + B); if (x > 0) v = Math.min(v, d[i + w - 1] + B); }
    d[i] = v;
  }
  return d;
}

/* returns { rgb: Float32Array n·3, alpha: Float32Array n } for holeKeys'
   order, or null when there's too little real photo around to copy from */
function textureFill(buf, width, height, channels, holeKeys, base, key, sigma, band, transfer = true) {
  const n = holeKeys.length;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const k of holeKeys) { const x = k % width, y = (k - x) / width; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  // the box is sized for the largest σ it may come to (TT_SIGMA_MAX)
  const Rmax = TT_GRID * Math.round(TT_SIGMA_MAX * 0.75);
  const pad = Math.min(TT_PAD_MAX, Math.max(TT_PAD_MIN, Math.round(Math.sqrt(n)))) + Rmax + Math.ceil(3 * TT_SIGMA_MAX);
  // the box always reaches `pad` past the hole, also beyond the photo's
  // edge (a hole cut by the frame): every patch offset stays inside it —
  // flat indices that ran off one side used to wrap onto the other, and
  // water got copied into the sky. Out there it's the edge pixel repeated,
  // never a source and never "original" for the automatic edge.
  const bx0 = x0 - pad, by0 = y0 - pad, bw = x1 - x0 + 1 + 2 * pad, bh = y1 - y0 + 1 + 2 * pad;
  const N = bw * bh;
  const hole = new Int32Array(N).fill(-1), outside = new Uint8Array(N);
  holeKeys.forEach((k, i) => { const x = k % width, y = (k - x) / width; hole[(y - by0) * bw + x - bx0] = i; });
  // I: fill inside, photo outside; P: the photo everywhere
  const I = [0, 1, 2].map(() => new Float32Array(N)), P = [0, 1, 2].map(() => new Float32Array(N));
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const gx = Math.min(width - 1, Math.max(0, x + bx0)), gy = Math.min(height - 1, Math.max(0, y + by0));
    const i = y * bw + x, o = (gy * width + gx) * channels, hi = hole[(gy - by0) * bw + gx - bx0];
    if (gx !== x + bx0 || gy !== y + by0) outside[i] = 1;
    for (let c = 0; c < 3; c++) { P[c][i] = buf[o + c]; I[c][i] = hi >= 0 ? base[hi * 3 + c] : buf[o + c]; }
  }
  // σ MEASURED, not guessed from the crop geometry: the scale up to which
  // the fill is missing the photo's detail. Band by band (between two
  // successive blurs of the luminance), the fill's energy deep inside the
  // hole against the photo's around it; σ = the coarsest band of an
  // unbroken run, from the finest up, where the fill has under half of
  // it. A network that drew brick crisply keeps its brick (σ ≈ 1: only
  // grain is copied); a soft LaMa over a wide hole gets more replaced.
  const boosts = [];
  if (transfer) {
    const lum = new Float32Array(N);
    for (let i = 0; i < N; i++) lum[i] = (I[0][i] + I[1][i] + I[2][i]) / 3;
    const dDeep = chamfer(bw, bh, (i) => hole[i] < 0);
    const dOut0 = chamfer(bw, bh, (i) => hole[i] >= 0 || outside[i]);
    let prevB = lum, found = 1, run = true, prevSc = 0;
    for (const sc of TT_SIGMA_STEPS) {
      const B = gaussBlur(lum, bw, bh, sc);
      let eh = 0, nh = 0, ep = 0, np = 0;
      for (let i = 0; i < N; i += 2) {
        const v = prevB[i] - B[i];
        if (hole[i] >= 0) { if (dDeep[i] > 2 * sc + 1) { eh += v * v; nh++; } }
        else if (!outside[i] && dOut0[i] > 2 * sc + 1) { ep += v * v; np++; }
      }
      prevB = B;
      if (nh < 50 || np < 200) break;
      const ratio = eh / nh / (ep / np);
      if (run && ratio < 0.5 && sc <= TT_SIGMA_MAX) found = sc;
      else {
        run = false;
        // a band the network drew, but washed out (water ripples at half
        // their contrast): boosted back toward the photo's, never past it
        // (only a band the network really drew: one it all but flattened
        // holds mostly leftovers — boosted, the removed yacht's hull came
        // back as a ghost on the horizon)
        if (ratio >= TT_BOOST_FROM && ratio < TT_BOOST_UNTIL) boosts.push({ lo: prevSc, hi: sc, g: Math.min(TT_BOOST_MAX, Math.sqrt(1 / ratio)) - 1 });
      }
      prevSc = sc;
    }
    sigma = Math.min(TT_SIGMA_MAX, found);
  }
  const step = Math.max(1, Math.round(sigma * 0.75)), R = TT_GRID * step;
  const L = I.map((p) => gaussBlur(p, bw, bh, sigma));
  // sources: real pixels whose whole sparse patch is real and inside the box
  const dOut = chamfer(bw, bh, (i) => hole[i] >= 0);
  const srcs = [];
  const sT0 = Math.ceil(Math.min(8, sigma * 2.5) * 0.75) + 1; // the coarse band's patch reach (below)
  const rim = Math.max(R, sT0) + 1;
  for (let y = rim; y < bh - rim; y++) for (let x = rim; x < bw - rim; x++) if (!outside[y * bw + x] && dOut[y * bw + x] > R + 1) srcs.push(y * bw + x);
  if (srcs.length < 256) return null;
  const L0 = L[0], L1 = L[1], L2 = L[2];
  // the photo's own detail level, for how close a match has to be
  const hf = new Float32Array(Math.min(srcs.length, 20000));
  for (let k = 0; k < hf.length; k++) {
    const q = srcs[Math.floor((k * srcs.length) / hf.length)];
    hf[k] = (P[0][q] - L0[q] + P[1][q] - L1[q] + P[2][q] - L2[q]) / 3;
  }
  const grain = Math.max(1, robustSigma(hf));
  // the low-pass of grain σ_g at σ keeps ≈ σ_g/(2√π σ) of it per sample;
  // a difference well above that is a real one
  const lpNoise = grain / (3.5 * sigma) + 0.75;

  // the edge: agreement of the original with the fill, on the low-pass of both
  const alpha = new Float32Array(n).fill(1);
  if (band > 0) {
    const LP = P.map((p) => gaussBlur(p, bw, bh, sigma));
    const dIn = chamfer(bw, bh, (i) => hole[i] < 0 && !outside[i]);
    const agreeTol = 3 * lpNoise + 2.5;
    for (let p = 0; p < N; p++) {
      const i = hole[p];
      if (i < 0) continue;
      const e = dIn[p] - 0.5;
      if (e >= band) continue;
      const g = e <= 0 ? 0 : (e / band) * (e / band) * (3 - 2 * (e / band));
      const diff = (Math.abs(LP[0][p] - L0[p]) + Math.abs(LP[1][p] - L1[p]) + Math.abs(LP[2][p] - L2[p])) / 3;
      alpha[i] = 1 - (1 - g) * Math.exp(-((diff / agreeTol) ** 2));
    }
  }
  if (!transfer) return { rgb: base, alpha };

  const offs = [];
  for (let j = -TT_GRID; j <= TT_GRID; j++) for (let i = -TT_GRID; i <= TT_GRID; i++) offs.push(j * step * bw + i * step);
  const K = offs.length;
  const dist = (p, q, cap) => {
    let d = 0;
    for (let k = 0; k < K; k++) {
      const a = p + offs[k], b = q + offs[k];
      const e0 = L0[a] - L0[b], e1 = L1[a] - L1[b], e2 = L2[a] - L2[b];
      d += e0 * e0 + e1 * e1 + e2 * e2;
      if (d >= cap) return d;
    }
    return d;
  };
  // the same comparison on every other grid sample (3×3 of the 5×5): the
  // per-pixel choice below is smoothed over a few px anyway
  const lite = offs.filter((_, k) => (k % (2 * TT_GRID + 1)) % 2 === 0 && Math.floor(k / (2 * TT_GRID + 1)) % 2 === 0);
  const distLite = (p, q) => {
    let d = 0;
    for (let k = 0; k < lite.length; k++) {
      const a = p + lite[k], b = q + lite[k];
      const e0 = L0[a] - L0[b], e1 = L1[a] - L1[b], e2 = L2[a] - L2[b];
      d += e0 * e0 + e1 * e1 + e2 * e2;
    }
    return d * (K / lite.length);
  };
  // all hole pixels in scan order; the search itself only needs to find
  // the common displacements, so it runs on every g-th of them per axis
  const tgt = new Int32Array(n);
  let t = 0;
  for (let i = 0; i < N; i++) if (hole[i] >= 0) tgt[t++] = i;
  const g = n > 20000 ? 3 : n > 4000 ? 2 : 1;
  const grid = [];
  for (let k = 0; k < n; k++) { const p = tgt[k], x = p % bw; if (x % g === 0 && ((p - x) / bw) % g === 0) grid.push(p); }
  const m = grid.length;
  const gOf = new Int32Array(N).fill(-1);
  for (let k = 0; k < m; k++) gOf[grid[k]] = k;
  const ok = new Uint8Array(N);
  for (const q of srcs) ok[q] = 1;
  const rand = rng(seedOf(key) ^ 0x5bd1e995);
  const nnf = new Int32Array(m), nnd = new Float32Array(m);
  for (let k = 0; k < m; k++) { nnf[k] = srcs[Math.floor(rand() * srcs.length)]; nnd[k] = dist(grid[k], nnf[k], Infinity); }
  const maxR = Math.max(bw, bh);
  const tryQ = (k, p, q) => {
    if (q < 0 || q >= N || !ok[q]) return;
    const d = dist(p, q, nnd[k]);
    if (d < nnd[k]) { nnd[k] = d; nnf[k] = q; }
  };
  for (let it = 0; it < TT_ITERS; it++) {
    const fwd = (it & 1) === 0, dir = fwd ? g : -g;
    for (let kk = 0; kk < m; kk++) {
      const k = fwd ? kk : m - 1 - kk, p = grid[k];
      // propagation: the neighbor's match, shifted along (targets sit
      // ≥ pad inside the box, so p ± g and p ± g·bw are always in it)
      const nh = gOf[p - dir], nv = gOf[p - dir * bw];
      if (nh >= 0) tryQ(k, p, nnf[nh] + dir);
      if (nv >= 0) tryQ(k, p, nnf[nv] + dir * bw);
      // random search around the current match, shrinking
      const q0 = nnf[k], qx = q0 % bw, qy = (q0 - qx) / bw;
      for (let r = maxR; r >= 1; r >>= 1) {
        const x = qx + Math.round((rand() * 2 - 1) * r), y = qy + Math.round((rand() * 2 - 1) * r);
        if (x >= 0 && y >= 0 && x < bw && y < bh) tryQ(k, p, y * bw + x);
      }
    }
  }
  // Copying each pixel from its own match leaves seams between the
  // matched pieces — invisible in fine grain, ugly as soon as the detail
  // has any size (worms). So the matches only VOTE for a few displacements
  // (the most common ones), and the detail is copied whole from those:
  // each pixel takes a soft, spatially smoothed choice among them by how
  // well each one's low-pass matches here (a kerb gets the kerb's
  // displacement, the asphalt beside it another), mixed so the grain keeps
  // its strength where two overlap (Σw·H / √Σw², not Σw·H).
  const votes = new Map();
  for (let k = 0; k < m; k++) { const d = nnf[k] - grid[k]; votes.set(d, (votes.get(d) || 0) + 1); }
  const ranked = [...votes.entries()].sort((u, v) => v[1] - u[1]);
  const disp = [];
  for (const [d] of ranked) {
    if (disp.length >= TT_DISPS) break;
    const dy = Math.round(d / bw), dx = d - dy * bw;
    // near-duplicates copy the same grain twice
    if (disp.some((e) => Math.abs(e.dx - dx) < 3 && Math.abs(e.dy - dy) < 3)) continue;
    disp.push({ d, dx, dy });
  }
  const D = disp.length, BIG = 1e12;
  // Two bands. The network's own texture is invented (and washed out by
  // its 512² grid): ripples, cloud grain, cobbles. So away from edges the
  // whole band below σT is replaced by the photo's — T = Lt + (photo − Lt)
  // at the chosen displacement, Lt the network's fill blurred at σT. On
  // EDGES of that coarse fill (a horizon, a kerb) copying a band that wide
  // would drag the copied edge a few px off the network's — there only the
  // fine band below σ is copied (F = L + (photo − L)), the edge stays the
  // network's. Edge = gradient of Lt well above the photo's own texture.
  const sT = Math.min(8, sigma * 2.5);
  const Lt = I.map((pl) => gaussBlur(pl, bw, bh, sT));
  const stepT = Math.max(1, Math.round(sT * 0.75));
  const liteT = [];
  for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) liteT.push(j * stepT * bw + i * stepT);
  const distT = (p, q) => {
    let d = 0;
    for (let k = 0; k < liteT.length; k++) {
      const a = p + liteT[k], b = q + liteT[k];
      if (b < 0 || b >= N) return BIG;
      const e0 = Lt[0][a] - Lt[0][b], e1 = Lt[1][a] - Lt[1][b], e2 = Lt[2][a] - Lt[2][b];
      d += e0 * e0 + e1 * e1 + e2 * e2;
    }
    return d * (K / liteT.length);
  };
  // per displacement: patch distance (fine + coarse) at every hole pixel,
  // box-smoothed over the hole so the choice changes gradually
  const cost = disp.map(({ d }) => {
    const c = new Float32Array(N).fill(BIG);
    for (let k = 0; k < n; k++) { const p = tgt[k], q = p + d; if (q >= 0 && q < N && ok[q]) c[p] = distLite(p, q) + distT(p, q); }
    return c;
  });
  const sm = Math.max(2, Math.round(sigma * 1.5));
  for (const c of cost) {
    const tmp = new Float32Array(N);
    // log-domain smoothing: one invalid neighbor must not poison a pixel
    for (let i = 0; i < N; i++) c[i] = Math.log1p(c[i]);
    boxPass(c, tmp, bw, bh, bw, 1, sm); boxPass(tmp, c, bh, bw, 1, bw, sm);
  }
  // smoothed log-cost: a ratio of e^0.6 ≈ 1.8 between two displacements → weight 1 : 0.37
  const temp = 0.6;
  // Bands: L (σ) ⊃ Lt (σT) ⊃ Lm ⊃ Lc, all low-passes of the filled image.
  // The output is the network's coarsest one plus every finer band, each
  // either COPIED from the chosen photo spots or kept from the network:
  // the network's where it has an edge at that scale (a horizon, a kerb —
  // its structure), the network's too where the spot copied from has an
  // edge at that scale (a horizon must not be copied into the sky), the
  // photo's everywhere else — the network's invented mid-scale blotches
  // and soft ripples replaced by the real thing.
  const LB = [L, Lt, ...[6, 15, 35].map((f) => I.map((pl) => gaussBlur(pl, bw, bh, sigma * f)))];
  const NB = LB.length;
  const gradAt = (A, p) => {
    let gsum = 0;
    for (let c = 0; c < 3; c++) gsum += Math.abs(A[c][p + 1] - A[c][p - 1]) + Math.abs(A[c][p + bw] - A[c][p - bw]);
    return gsum / 6;
  };
  // each band's typical (texture) gradient, over the photo around the hole
  const gMed = LB.map((A) => {
    const gs = new Float32Array(Math.min(srcs.length, 20000));
    for (let k = 0; k < gs.length; k++) gs[k] = gradAt(A, srcs[Math.floor((k * srcs.length) / gs.length)]);
    return Math.max(0.05, kthSmallest(gs, gs.length >> 1));
  });
  const edgeOf = (b, p) => { const r = gradAt(LB[b], p) / gMed[b]; return r <= 2.5 ? 0 : r >= 5 ? 1 : ((r - 2.5) / 2.5) ** 2 * (3 - 2 * ((r - 2.5) / 2.5)); };
  // Detail energy around every source (box-smoothed |photo − Lt|²): a
  // spot far busier than what's typical around the hole (a window grille
  // next to plain brick) matches at the low-pass yet would paste its
  // lines in as "texture" — such candidates are handicapped by how much
  // busier they are.
  const Eh = new Float32Array(N), tmpEh = new Float32Array(N);
  for (let i = 0; i < N; i++) { let e = 0; for (let c = 0; c < 3; c++) { const v = P[c][i] - Lt[c][i]; e += v * v; } Eh[i] = e / 3; }
  const er = Math.max(2, Math.round(sT));
  boxPass(Eh, tmpEh, bw, bh, bw, 1, er); boxPass(tmpEh, Eh, bh, bw, 1, bw, er);
  const es = new Float32Array(Math.min(srcs.length, 20000));
  for (let k = 0; k < es.length; k++) es[k] = Eh[srcs[Math.floor((k * srcs.length) / es.length)]];
  const eMax = 2.5 * Math.max(1, kthSmallest(es, es.length >> 1));
  const busy = (q) => (Eh[q] > eMax ? 1.5 * Math.log(Eh[q] / eMax) : 0);
  // what a real match may cost: both patches' low-pass noise, with a wide
  // margin — textures like water never match closely, and only a plainly
  // wrong match (sea for sky) should lose its coarse band
  const tolLog = Math.log1p(8 * K * 3 * (3 * lpNoise) ** 2);
  const rgb = new Float32Array(n * 3), w = new Float32Array(D + 1), dd = new Int32Array(D + 1), c0 = new Float64Array(D + 1);
  for (let k = 0; k < n; k++) {
    const p = tgt[k], i = hole[p];
    // candidates: the common displacements, plus this spot's own match
    // (the nearest search sample's) — near the photo's edge most common
    // ones point off it, and the few left may be the wrong thing entirely
    // (sea under sky); its own match is at least the right kind
    const x = p % bw, y = (p - x) / bw, gk = gOf[(y - (y % g)) * bw + x - (x % g)];
    let best = Infinity;
    for (let j = 0; j <= D; j++) {
      dd[j] = j < D ? disp[j].d : gk >= 0 ? nnf[gk] - grid[gk] : 0;
      const q = p + dd[j];
      if (j === D && gk < 0) { c0[j] = Infinity; continue; }
      // (its own match: unsmoothed, so it's held back by OWN_PENALTY — a
      // fallback, not a rival; on its own it patchworks the fill)
      c0[j] = q >= 0 && q < N && ok[q] ? (j < D ? cost[j][p] : Math.log1p(distLite(p, q) + distT(p, q)) + OWN_PENALTY) + busy(q) : Infinity;
      if (c0[j] < best) best = c0[j];
    }
    if (best === Infinity) { for (let c = 0; c < 3; c++) rgb[i * 3 + c] = L[c][p]; continue; }
    let sw = 0, sw2 = 0;
    for (let j = 0; j <= D; j++) { w[j] = c0[j] === Infinity ? 0 : Math.exp(-(c0[j] - best) / temp); sw += w[j]; }
    for (let j = 0; j <= D; j++) { w[j] /= sw; sw2 += w[j] * w[j]; }
    const norm = 1 / Math.sqrt(sw2);
    // no candidate really matches: copy only the grain
    const eBad = best <= tolLog ? 0 : Math.min(1, (best - tolLog) / 2);
    // per band (1…NB−1): keep the network's — where it has an edge there,
    // or (coarse bands) where the spot copied from has one
    const keep = new Float32Array(NB);
    keep[1] = Math.max(edgeOf(1, p), eBad);
    for (let b = 2; b < NB; b++) {
      let es = 0;
      for (let j = 0; j <= D; j++) if (w[j] > 1e-3) es += w[j] * edgeOf(b, p + dd[j]);
      keep[b] = Math.max(edgeOf(b, p), es, eBad);
    }
    // The coarse bands: copied only where the photo is FLAT at that scale
    // (a sky, a wall) — the network's drifting blotches there are replaced
    // by the photo's evenness. Where the photo has texture that coarse
    // (ripples, waves) the network's own is kept: copying it from
    // different spots pixel by pixel left a patchwork of seams.
    let jb = 0;
    for (let j = 1; j <= D; j++) if (w[j] > w[jb]) jb = j;
    const qb = p + dd[jb];
    const flat = new Float32Array(NB);
    for (let b = 2; b < NB; b++) {
      // …and only where the network's own band there is faint (drift):
      // a strong one is texture it drew on purpose (ripples)
      let m = 0, mf = 0;
      for (let c = 0; c < 3; c++) { m += Math.abs(LB[b - 1][c][qb] - LB[b][c][qb]); mf += Math.abs(LB[b - 1][c][p] - LB[b][c][p]); }
      m /= 3; mf /= 3;
      flat[b] = (1 - keep[b]) * (m <= 1.5 ? 1 : m >= 4 ? 0 : (4 - m) / 2.5) * (mf <= 3 ? 1 : mf >= 6 ? 0 : (6 - mf) / 3);
    }
    for (let c = 0; c < 3; c++) {
      let v = LB[NB - 1][c][p], h0 = 0, h1 = 0;
      for (let j = 0; j <= D; j++) {
        if (w[j] <= 1e-3) continue;
        const q = p + dd[j];
        h0 += w[j] * (P[c][q] - L[c][q]);
        h1 += w[j] * (LB[0][c][q] - LB[1][c][q]);
      }
      // the grain-like bands mixed variance-preserving (see above)
      v += h0 * norm + keep[1] * (LB[0][c][p] - LB[1][c][p]) + (1 - keep[1]) * h1 * norm;
      for (let b = 2; b < NB; b++) v += (1 - flat[b]) * (LB[b - 1][c][p] - LB[b][c][p]) + flat[b] * (LB[b - 1][c][qb] - LB[b][c][qb]);
      rgb[i * 3 + c] = v;
    }
  }
  // the boosts (see σ above), on the network's own bands — not on its
  // edges, where it would only sharpen into halos
  for (const bst of boosts) {
    if (bst.lo < sigma) continue;
    const lo = bst.lo === sigma ? L : I.map((pl) => gaussBlur(pl, bw, bh, bst.lo));
    const hi = I.map((pl) => gaussBlur(pl, bw, bh, bst.hi));
    for (let k = 0; k < n; k++) {
      const p = tgt[k], i = hole[p], wgt = bst.g * (1 - edgeOf(1, p));
      if (wgt <= 0) continue;
      for (let c = 0; c < 3; c++) rgb[i * 3 + c] += wgt * (lo[c][p] - hi[c][p]);
    }
  }
  // Tone: the network's colors drift a little at large scale (a lilac
  // patch in a blue sky) — invisible as texture, obvious as a blotch. The
  // difference between the photo's and the fill's local means along the
  // hole's edge is spread into the hole (inverse-distance), so the fill
  // meets its surroundings in tone everywhere. Edge samples that differ a
  // lot are left out: that's the removed object's rim, not drift.
  const sB = Math.max(6, sT * 2);
  const realM = new Float32Array(N), fillM = new Float32Array(N);
  for (let i = 0; i < N; i++) { realM[i] = hole[i] < 0 && !outside[i] ? 1 : 0; fillM[i] = hole[i] >= 0 ? 1 : 0; }
  const nRealW = gaussBlur(realM, bw, bh, sB), nFillW = gaussBlur(fillM, bw, bh, sB);
  const realC = [0, 1, 2].map((c) => { const a = new Float32Array(N); for (let i = 0; i < N; i++) a[i] = realM[i] * P[c][i]; return gaussBlur(a, bw, bh, sB); });
  const fillC = [0, 1, 2].map((c) => { const a = new Float32Array(N); for (let i = 0; i < N; i++) if (hole[i] >= 0) a[i] = rgb[hole[i] * 3 + c]; return gaussBlur(a, bw, bh, sB); });
  const edgeD = chamfer(bw, bh, (i) => hole[i] < 0 && !outside[i]);
  const samp = [];
  for (let k = 0; k < n; k++) {
    const p = tgt[k];
    if (edgeD[p] > 2 || nRealW[p] < 0.15 || nFillW[p] < 0.15) continue;
    samp.push([p % bw, Math.floor(p / bw), ...[0, 1, 2].map((c) => realC[c][p] / nRealW[p] - fillC[c][p] / nFillW[p])]);
  }
  if (samp.length >= 8) {
    const mag = new Float32Array(samp.length);
    samp.forEach((t, j) => { mag[j] = (Math.abs(t[2]) + Math.abs(t[3]) + Math.abs(t[4])) / 3; });
    const med = kthSmallest(Float32Array.from(mag), mag.length >> 1);
    const keep = samp.filter((_, j) => mag[j] <= 3 * med + 4);
    const stride = Math.max(1, Math.ceil(keep.length / 300));
    const use = keep.filter((_, j) => j % stride === 0);
    for (let k = 0; k < n && use.length; k++) {
      const p = tgt[k], x = p % bw, y = (p - x) / bw, i = hole[p];
      let sw = 0, d0 = 0, d1 = 0, d2 = 0;
      for (const t of use) {
        const dx = t[0] - x, dy = t[1] - y, wgt = 1 / (dx * dx + dy * dy + 1) ** 1.5;
        sw += wgt; d0 += wgt * t[2]; d1 += wgt * t[3]; d2 += wgt * t[4];
      }
      rgb[i * 3] += d0 / sw; rgb[i * 3 + 1] += d1 / sw; rgb[i * 3 + 2] += d2 / sw;
    }
  }
  return { rgb, alpha };
}

/* the network's blur scale, in buffer px: how many buffer pixels one of
   its 512² cells covers (LaMa and MI-GAN both work at 512) */
function netSigma(holeW, holeH, w, h, s) {
  const side = Math.min(Math.max(w, h), Math.round(Math.max(holeW, holeH) * s * 1.3) + 32);
  const perCell = Math.max(side, 1) / 512 / s;
  return Math.min(4, Math.max(1.5, 1.1 * perCell));
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

/* `holeKeys`: buffer pixel indices (y·width + x) to fill; `band`: px from
   the footprint's edge where the fill may fade into matching original
   pixels (textureFill). Returns Map(key → [r, g, b, alpha]). Reads `buf`
   only. */
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

export function removeFill(buf, width, height, channels, holeKeys, strokeKey, band = 0) {
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
  const nkey = engine && `${key}|${engine}${NET_LOOK[engine] || ""}`;
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
  // final = nothing better is coming for this exact input (no network
  // result still to land) — only then may the caller cache what it drew
  const isFinal = !engine || !!nn;
  const tag = `${nn ? engine : "pm"}|${band}`;
  let done = isFinal ? entry.nnOut[tag] : null;
  if (isFinal && !done) {
    // detail is missing wherever the base came through a resampling: the
    // network's 512² grid, or PatchMatch's downsampled crop (at full scale
    // PatchMatch copies real pixels and already has it)
    const sigma = nn ? netSigma(x1 - x0 + 1, y1 - y0 + 1, w, h, s) : s < 0.9 ? Math.min(4, Math.max(1.5, 1.1 / s)) : 0;
    done = textureFill(buf, width, height, channels, holeKeys, base, key, sigma || 1.5, band, !!sigma)
      // too little photo around to copy from: the old matched noise
      || { rgb: nn ? addMatchedGrain(buf, width, height, channels, holeKeys, base, key) : base, alpha: null };
    entry.nnOut[tag] = done;
  }
  const final = done ? done.rgb : base, alpha = done ? done.alpha : null;

  const out = new Map();
  holeKeys.forEach((k, idx) => {
    // clamped: added grain/detail can overshoot 0..255, and a Node Buffer
    // (export) WRAPS instead of clamping — 256 became 0 in one channel,
    // cyan/magenta specks over bright brick
    const v = (c) => { const x = Math.round(final[idx * 3 + c]); return x < 0 ? 0 : x > 255 ? 255 : x; };
    out.set(k, [v(0), v(1), v(2), alpha ? alpha[idx] : 1]);
  });
  out.final = isFinal;
  return out;
}
