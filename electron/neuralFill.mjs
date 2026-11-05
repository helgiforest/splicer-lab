/* ============================================================
   Local neural inpainting for the stamp tool's Remove mode — runs in its
   own utility process (electron/neuralWorker.cjs), never in the main
   process: an inference blocks its thread for up to seconds (LaMa ~3.5 s,
   plus ~4.5 s to load), which in main froze every photo/tile/thumbnail
   request meanwhile. main.cjs proxies the renderer's "neural-inpaint" IPC
   and the export path (bakeRaw) to it. Nothing leaves the machine.

   Model: MI-GAN (Sargsyan et al., Picsart AI Research, "MI-GAN: A Simple
   Baseline for Image Inpainting on Mobile Devices", ICCV 2023) — the
   published ONNX "pipeline" export migan_pipeline_v2.onnx (MI-GAN-512,
   trained on Places2), 28.1 MB, code and weights MIT-licensed (see
   electron/models/LICENSE-MIGAN). The pipeline does its own crop around the
   mask, resize to 512×512, inference, resize back and paste — inputs are a
   uint8 RGB image [1,3,H,W] and a uint8 mask [1,1,H,W] (255 = keep, 0 =
   fill), output uint8 RGB [1,3,H,W].

   Runtime: onnxruntime-web's WebAssembly backend (MIT), the same single
   ~15 MB .wasm on every OS — no per-platform native binaries to package.
   Up to 8 threads (MI-GAN ≈2× faster than 1 thread on a 420² crop);
   falls back to a single thread if threads can't start (e.g. inside a
   packaged asar).

   This module only answers "give me MI-GAN's fill for this crop". How
   that's combined with the PatchMatch fill (shape from the network, grain
   from PatchMatch) lives in electron/removeTool.mjs.
   ============================================================ */
import { createRequire } from "module";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";

const require = createRequire(import.meta.url);
const MODEL_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), "models", "migan_pipeline_v2.onnx");
// shared by both models: onnxruntime-web fixes the WASM thread count at its
// first initialization, process-wide (8 vs 4 threads: LaMa 3.4 s vs 4.1 s)
const MAX_THREADS = 8;

let ort = null;
let sessionP = null;

async function createSession(threads) {
  ort = ort || require("onnxruntime-web");
  ort.env.wasm.numThreads = threads;
  return ort.InferenceSession.create(fs.readFileSync(MODEL_PATH));
}

function session() {
  if (!sessionP) {
    const threads = Math.max(1, Math.min(MAX_THREADS, (os.cpus() || []).length || 1));
    sessionP = createSession(threads).catch(() => createSession(1));
    // a failed load must not poison every later call
    sessionP.catch(() => { sessionP = null; });
  }
  return sessionP;
}

/* ---- LaMa: the larger, stronger model, for big holes ----
   big-lama (Suvorov et al., "Resolution-robust Large Mask Inpainting with
   Fourier Convolutions", WACV 2022), ONNX port Carve/LaMa-ONNX
   lama_fp32.onnx — 208 MB, Apache-2.0 (electron/models/LICENSE-LAMA).
   Much better than MI-GAN at continuing STRUCTURE through a large hole
   (façade lines, fences, repeating patterns), ~7× slower (~3.4 s at 8
   threads on Apple M-series). Fixed 512×512 input: float32 RGB 0..1
   [1,3,512,512] + float32 mask [1,1,512,512] (1 = fill); output float32
   RGB 0..255. The crop is letterboxed into 512² (aspect kept; the margin
   is edge-replicated and marked KNOWN), and the result scaled back. */
// downloaded on demand into userData (main.cjs passes the path in
// SPLICER_LAMA_PATH); a dev checkout may also have it next to MI-GAN
const LAMA_PATHS = [process.env.SPLICER_LAMA_PATH, path.join(path.dirname(fileURLToPath(import.meta.url)), "models", "lama_fp32.onnx")];
const lamaPath = () => LAMA_PATHS.find((p) => p && fs.existsSync(p)) || null;
const LAMA = 512;
const LAMA_DILATE = 6; // mask growth, in 512-grid cells
let lamaP = null;
function lamaSession() {
  if (!lamaP) {
    // MI-GAN's session first: it's what initializes the shared WASM runtime
    // (thread count, single-thread fallback)
    lamaP = session().then(() => ort.InferenceSession.create(fs.readFileSync(lamaPath())));
    lamaP.catch(() => { lamaP = null; });
  }
  return lamaP;
}

/* bilinear sample of a w×h interleaved-RGB float/uint8 image at (fx, fy) */
function bilinear(src, w, h, fx, fy, out, o) {
  const x = Math.min(Math.max(fx, 0), w - 1), y = Math.min(Math.max(fy, 0), h - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const tx = x - x0, ty = y - y0;
  for (let c = 0; c < 3; c++) {
    const a = src[(y0 * w + x0) * 3 + c], b = src[(y0 * w + x1) * 3 + c];
    const d = src[(y1 * w + x0) * 3 + c], e = src[(y1 * w + x1) * 3 + c];
    out[o + c] = (a + (b - a) * tx) + ((d + (e - d) * tx) - (a + (b - a) * tx)) * ty;
  }
}

/* The crop handed over is sized for PatchMatch's search (lots of context);
   LaMa only needs a margin around the hole, and every extra pixel of
   context costs resolution once it's squeezed into 512² — so it runs on
   a tighter SQUARE around the hole (LAMA_CONTEXT × the hole's size on each
   side), pasted back into the full crop. Tight on purpose: with more
   context a hole next to a similar object (a car beside the removed car)
   gets that object continued into it; 0.15 was the cleanest on both test
   photos (0.35 left a car-shaped ghost). */
const LAMA_CONTEXT = 0.15;
async function lamaRun(rgb, hole, w, h) {
  let hx0 = w, hy0 = h, hx1 = -1, hy1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (hole[y * w + x]) { if (x < hx0) hx0 = x; if (x > hx1) hx1 = x; if (y < hy0) hy0 = y; if (y > hy1) hy1 = y; }
  if (hx1 < 0) return Uint8Array.from(rgb);
  const side = Math.min(Math.max(w, h), Math.round(Math.max(hx1 - hx0, hy1 - hy0) * (1 + 2 * LAMA_CONTEXT)) + 32);
  const cx = (hx0 + hx1) / 2, cy = (hy0 + hy1) / 2;
  const sx0 = Math.max(0, Math.min(w - Math.min(side, w), Math.round(cx - side / 2)));
  const sy0 = Math.max(0, Math.min(h - Math.min(side, h), Math.round(cy - side / 2)));
  const sw = Math.min(side, w), sh = Math.min(side, h);
  const subRgb = new Uint8Array(sw * sh * 3), subHole = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    const q = (y + sy0) * w + x + sx0, t = y * sw + x;
    subRgb[t * 3] = rgb[q * 3]; subRgb[t * 3 + 1] = rgb[q * 3 + 1]; subRgb[t * 3 + 2] = rgb[q * 3 + 2];
    subHole[t] = hole[q];
  }
  const sub = await lamaRunSquare(subRgb, subHole, sw, sh);
  const result = Uint8Array.from(rgb);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    const q = (y + sy0) * w + x + sx0, t = y * sw + x;
    if (!hole[q]) continue;
    result[q * 3] = sub[t * 3]; result[q * 3 + 1] = sub[t * 3 + 1]; result[q * 3 + 2] = sub[t * 3 + 2];
  }
  return result;
}

async function lamaRunSquare(rgb, hole, w, h) {
  const s = await lamaSession();
  const k = LAMA / Math.max(w, h);               // letterbox scale
  const sw = Math.max(1, Math.round(w * k)), sh = Math.max(1, Math.round(h * k));
  const N = LAMA * LAMA;
  const img = new Float32Array(3 * N), mask = new Float32Array(N), px = new Float32Array(3);
  for (let y = 0; y < LAMA; y++) for (let x = 0; x < LAMA; x++) {
    const q = y * LAMA + x;
    // outside the scaled crop: clamp = edge replication; always "known"
    const fx = (Math.min(x, sw - 1) + 0.5) / k - 0.5, fy = (Math.min(y, sh - 1) + 0.5) / k - 0.5;
    bilinear(rgb, w, h, fx, fy, px, 0);
    img[q] = px[0] / 255; img[N + q] = px[1] / 255; img[2 * N + q] = px[2] / 255;
    if (x < sw && y < sh) {
      // a 512-cell is "hole" if the crop pixel nearest its center is
      const hx = Math.min(w - 1, Math.max(0, Math.round(fx))), hy = Math.min(h - 1, Math.max(0, Math.round(fy)));
      mask[q] = hole[hy * w + hx] ? 1 : 0;
    }
  }
  // Grow the mask (8-neighbour, LAMA_DILATE cells): the brush edge leaves
  // the object's rim and shadow just outside the hole, and LaMa happily
  // continues them inward — a removed car came back as a dark car-shaped
  // ghost. The grown band is only hidden from the network; the paste
  // below still writes the original hole only.
  for (let d = 0; d < LAMA_DILATE; d++) {
    const prev = Float32Array.from(mask);
    for (let y = 0; y < LAMA; y++) for (let x = 0; x < LAMA; x++) {
      const q = y * LAMA + x;
      if (prev[q]) continue;
      for (let j = -1; j <= 1 && !mask[q]; j++) for (let i = -1; i <= 1; i++) {
        const xx = x + i, yy = y + j;
        if (xx >= 0 && yy >= 0 && xx < LAMA && yy < LAMA && prev[yy * LAMA + xx]) { mask[q] = 1; break; }
      }
    }
  }
  // this export does NOT blank the masked pixels itself (big-lama's trainer
  // did, outside the generator): whatever is under the mask leaks into the
  // fill — e.g. PatchMatch's first guess, which here held a copy of the jeep
  // parked next to the removed car. Blank it, as in training.
  for (let q = 0; q < N; q++) if (mask[q]) img[q] = img[N + q] = img[2 * N + q] = 0;
  const out = await s.run({ image: new ort.Tensor("float32", img, [1, 3, LAMA, LAMA]), mask: new ort.Tensor("float32", mask, [1, 1, LAMA, LAMA]) });
  const res = out.output.data;
  const inter = new Float32Array(3 * N);
  for (let q = 0; q < N; q++) { inter[q * 3] = res[q]; inter[q * 3 + 1] = res[N + q]; inter[q * 3 + 2] = res[2 * N + q]; }
  // back to crop size: hole pixels from the network (bilinear), the rest untouched
  const result = Uint8Array.from(rgb);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const q = y * w + x;
    if (!hole[q]) continue;
    bilinear(inter, LAMA, LAMA, (x + 0.5) * k - 0.5, (y + 0.5) * k - 0.5, px, 0);
    for (let c = 0; c < 3; c++) result[q * 3 + c] = Math.max(0, Math.min(255, Math.round(px[c])));
  }
  return result;
}

/* rgb: Uint8Array w·h·3 interleaved; hole: Uint8Array w·h (1 = fill);
   engine: "migan" (default, fast) or "lama" (large holes).
   Returns Uint8Array w·h·3 interleaved. Calls are serialized (one
   inference at a time — the WASM backend is single-instance anyway). */
let queue = Promise.resolve();
export function neuralInpaint(rgb, hole, w, h, engine = "migan") {
  const job = queue.then(async () => {
    // LaMa is an optional download (see main.cjs) — without it, MI-GAN
    if (engine === "lama" && lamaPath()) return lamaRun(rgb, hole, w, h);
    const s = await session();
    const n = w * h;
    const planar = new Uint8Array(3 * n), mask = new Uint8Array(n);
    for (let q = 0; q < n; q++) {
      planar[q] = rgb[q * 3]; planar[n + q] = rgb[q * 3 + 1]; planar[2 * n + q] = rgb[q * 3 + 2];
      mask[q] = hole[q] ? 0 : 255;
    }
    const out = await s.run({
      image: new ort.Tensor("uint8", planar, [1, 3, h, w]),
      mask: new ort.Tensor("uint8", mask, [1, 1, h, w]),
    });
    const res = out.result.data;
    const inter = new Uint8Array(3 * n);
    for (let q = 0; q < n; q++) { inter[q * 3] = res[q]; inter[q * 3 + 1] = res[n + q]; inter[q * 3 + 2] = res[2 * n + q]; }
    return inter;
  });
  queue = job.catch(() => {});
  return job;
}

/* Loads a model ahead of its first use (the Remove tool being picked), so
   the first stroke doesn't also pay for the load (LaMa: ~4.5 s). */
export function warm(engine) {
  return engine === "lama" ? (lamaPath() ? lamaSession() : Promise.resolve()) : session();
}
