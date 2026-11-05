const { app, BrowserWindow, ipcMain, dialog, protocol, net, utilityProcess } = require("electron");
const os = require("os");
const path = require("path");
const fs = require("fs");
const fsp = fs.promises;
const { pathToFileURL } = require("url");

/* Explicit, not left to package.json field-resolution order: app.getName()
   (and everything derived from it — app.getPath('userData'), the macOS
   dock/notification identity) should say "Splicer Lab" deterministically,
   in both an unpackaged dev run and a real build, rather than depending on
   Electron reading productName vs. name correctly in every context. Must
   run before the very first app.getPath('userData') call below. Note: this
   does NOT rename what macOS shows in the top menu bar during an
   unpackaged `electron .` dev run — that specific label is read from the
   generic Electron.app bundle's own Info.plist, a known Electron dev-mode
   limitation with no code-level fix; only a real packaged build (`npm run
   dist:mac`, using this project's own productName/appId) shows "Splicer
   Lab" there. */
app.setName("Splicer Lab");

/* colorAdjust.mjs is ESM (see the file for why); cache the dynamic import
   so every bake doesn't re-import it */
let colorAdjustP = null;
function getColorAdjust() {
  if (!colorAdjustP) colorAdjustP = import(pathToFileURL(path.join(__dirname, "colorAdjust.mjs")).href);
  return colorAdjustP;
}

/* redEyeTool.mjs — same ESM/dynamic-import rationale as colorAdjust.mjs */
let redEyeToolP = null;
function getRedEyeTool() {
  if (!redEyeToolP) redEyeToolP = import(pathToFileURL(path.join(__dirname, "redEyeTool.mjs")).href);
  return redEyeToolP;
}

/* ---------------- AI for Remove ----------------
   The networks (neuralFill.mjs) run in a utility process of their own
   (neuralWorker.cjs), forked on first use — an inference blocks its thread
   for up to seconds, which here in main would freeze every photo, tile and
   thumbnail request meanwhile. Opt-in: the renderer sends the user's choice
   ("off" | "fast" | "max", null = not asked yet = off) via ai-set-mode, and
   LaMa (the "max" model, 208 MB) is downloaded only when the user picks it. */
let aiMode = null;
const LAMA_MODEL = {
  url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx",
  size: 208044816,
  sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
};
const lamaUserPath = () => path.join(app.getPath("userData"), "models", "lama_fp32.onnx");
// a dev checkout may have it next to MI-GAN; never packaged (build.files)
const lamaDevPath = path.join(__dirname, "models", "lama_fp32.onnx");
const lamaReady = () => fs.existsSync(lamaUserPath()) || fs.existsSync(lamaDevPath);

let neuralProc = null;
let neuralSeq = 0;
const neuralCalls = new Map();
function neuralProcess() {
  if (neuralProc) return neuralProc;
  neuralProc = utilityProcess.fork(path.join(__dirname, "neuralWorker.cjs"), [], {
    serviceName: "Splicer Lab AI",
    env: { ...process.env, SPLICER_LAMA_PATH: lamaUserPath() },
  });
  neuralProc.on("message", (m) => {
    const call = neuralCalls.get(m.id);
    if (!call) return;
    neuralCalls.delete(m.id);
    if (m.ok) call.resolve(m.out); else call.reject(new Error(m.error));
  });
  neuralProc.on("exit", (code) => {
    neuralProc = null;
    for (const call of neuralCalls.values()) call.reject(new Error(`AI process exited (${code})`));
    neuralCalls.clear();
  });
  return neuralProc;
}
function callNeural(msg) {
  return new Promise((resolve, reject) => {
    const id = ++neuralSeq;
    neuralCalls.set(id, { resolve, reject });
    neuralProcess().postMessage({ ...msg, id });
  });
}

/* Disk cache for Remove fills — PatchMatch crops (from the renderer's
   worker) and network outputs — so reopening an edited photo, or zooming
   into it again, doesn't recompute strokes that were already filled. Keys
   are content keys (stroke geometry + crop + pixel hash, see removeTool's
   removeFill), so an entry can only ever match the exact same input. LRU
   by mtime, capped at FILL_CACHE_MAX_BYTES. */
const FILL_CACHE_VERSION = 1; // bump when a fill's output for the same input changes
const FILL_CACHE_MAX_BYTES = 512 * 1024 * 1024;
const fillCacheDir = () => path.join(app.getPath("userData"), "fill-cache");
const fillCacheFile = (key) => path.join(fillCacheDir(), crypto.createHash("sha1").update(`${FILL_CACHE_VERSION}|${key}`).digest("hex") + ".bin");
async function fillCacheGet(key, len) {
  try {
    const file = fillCacheFile(key);
    const buf = await fsp.readFile(file);
    if (len && buf.length !== len) return null;
    const now = new Date();
    fsp.utimes(file, now, now).catch(() => {});
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.length);
  } catch {
    return null;
  }
}
let fillCachePuts = 0;
async function fillCachePut(key, bytes) {
  try {
    await fsp.mkdir(fillCacheDir(), { recursive: true });
    const file = fillCacheFile(key);
    await fsp.writeFile(file + ".tmp", bytes);
    await fsp.rename(file + ".tmp", file);
    if (fillCachePuts++ % 20 === 0) await pruneFillCache();
  } catch {}
}
async function pruneFillCache() {
  const dir = fillCacheDir();
  const files = [];
  for (const name of await fsp.readdir(dir)) {
    try { const st = await fsp.stat(path.join(dir, name)); files.push({ name, size: st.size, mtime: st.mtimeMs }); } catch {}
  }
  files.sort((a, b) => b.mtime - a.mtime);
  let total = 0;
  for (const f of files) {
    total += f.size;
    if (total > FILL_CACHE_MAX_BYTES) await fsp.rm(path.join(dir, f.name), { force: true });
  }
}

/* One network fill (job: removeTool's neural job), through the disk cache. */
async function neuralInpaintCached(job) {
  const engine = job.engine === "lama" && !lamaReady() ? "migan" : job.engine;
  const cacheKey = job.key && `nn|${engine}|${job.w}x${job.h}|${job.key}`;
  if (cacheKey) {
    const hit = await fillCacheGet(cacheKey, job.w * job.h * 3);
    if (hit) return hit;
  }
  const out = await callNeural({ op: "inpaint", rgb: job.rgb, hole: job.hole, w: job.w, h: job.h, engine });
  if (cacheKey && out) await fillCachePut(cacheKey, out);
  return out;
}

/* LaMa, downloaded once into userData when the user picks "max": pinned
   upstream revision, SHA-256 checked, written to .part and renamed, so a
   cancelled or corrupt download never leaves a model that loads. Progress
   goes to the window as lama-download-progress. */
let lamaDownloadP = null;
function downloadLama(sender) {
  if (lamaReady()) return Promise.resolve(true);
  if (lamaDownloadP) return lamaDownloadP;
  lamaDownloadP = (async () => {
    const dest = lamaUserPath(), tmp = dest + ".part";
    await fsp.mkdir(path.dirname(dest), { recursive: true });
    // Chromium's network stack first (system proxy settings), Node's as a
    // fallback where that one can't connect
    let res;
    try { res = await net.fetch(LAMA_MODEL.url); } catch { res = await fetch(LAMA_MODEL.url); }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const hash = crypto.createHash("sha256");
    const out = fs.createWriteStream(tmp);
    let received = 0, lastPct = -1;
    try {
      for await (const chunk of res.body) {
        hash.update(chunk);
        received += chunk.length;
        if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
        const pct = Math.floor((received / LAMA_MODEL.size) * 100);
        if (pct !== lastPct && !sender.isDestroyed()) { lastPct = pct; sender.send("lama-download-progress", { received, total: LAMA_MODEL.size }); }
      }
    } finally {
      await new Promise((r) => out.end(r));
    }
    if (received !== LAMA_MODEL.size || hash.digest("hex") !== LAMA_MODEL.sha256) {
      await fsp.rm(tmp, { force: true });
      throw new Error("the downloaded model failed its checksum");
    }
    await fsp.rename(tmp, dest);
    return true;
  })().finally(() => { lamaDownloadP = null; });
  return lamaDownloadP;
}

/* ---------------- file logging ----------------
   One plain-text file per day under userData/logs, so a bug report can
   include a log file instead of a DevTools screenshot. Covers main-process
   crashes here, plus renderer errors (including CSP violations, which
   don't throw — they only fire a securitypolicyviolation event) reported
   through the log-error IPC channel registered further down. */
function appendLog(line) {
  try {
    const dir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `${new Date().toISOString().slice(0, 10)}.log`);
    fs.appendFileSync(file, `[${new Date().toISOString()}] ${line}\n`);
  } catch {}
}
process.on("uncaughtException", (err) => appendLog(`[main] uncaughtException: ${err?.stack || err}`));
process.on("unhandledRejection", (reason) => appendLog(`[main] unhandledRejection: ${reason?.stack || reason}`));

/* one-time migration: userData lives at Application Support/<app name>, so
   any rename of the app would otherwise point every user at a fresh, empty
   folder — losing their library.json (watched folders, ratings, crops,
   presets) and forcing every RAW/HEIC file to reconvert. If the CURRENT
   folder has no library yet but an old-name one exists alongside it,
   adopt its contents instead of leaving them orphaned. Tries every prior
   name this app has ever resolved to, oldest first, stopping at the first
   one that actually has a library to adopt. */
function adoptOldUserData(oldName) {
  const newDir = app.getPath("userData");
  if (fs.existsSync(path.join(newDir, "library.json"))) return;
  const oldDir = path.join(path.dirname(newDir), oldName);
  if (!fs.existsSync(path.join(oldDir, "library.json"))) return;
  try {
    fs.mkdirSync(newDir, { recursive: true });
    for (const name of fs.readdirSync(oldDir)) {
      const dest = path.join(newDir, name);
      if (!fs.existsSync(dest)) fs.renameSync(path.join(oldDir, name), dest);
    }
  } catch {}
}
adoptOldUserData("Meridian"); // pre-Splicer name
// this file didn't call app.setName() before now, so Electron's own
// default (package.json's "name" field, NOT "productName") is what any
// existing dev install's userData folder actually used — adopt from there
// now that setName("Splicer Lab") above changes the resolved folder.
adoptOldUserData("splicer-lab");

const NATIVE_RE = /\.(jpe?g|png|webp|gif|avif|bmp)$/i;   // Chromium decodes these itself
const TIFF_RE = /\.(tif|tiff)$/i;
const HEIC_RE = /\.(heic|heif)$/i;
const RAW_RE = /\.(cr2|cr3|nef|nrw|arw|srf|sr2|orf|rw2|raf|dng|pef|srw|erf|kdc|3fr|mrw|x3f)$/i;
const IMG_RE = new RegExp(
  `(${NATIVE_RE.source}|${TIFF_RE.source}|${HEIC_RE.source}|${RAW_RE.source})`, "i"
);
const isImg = (name) => IMG_RE.test(name);
const libFile = () => path.join(app.getPath("userData"), "library.json");
const convCacheDir = () => path.join(app.getPath("userData"), "converted");
const thumbCacheDir = () => path.join(app.getPath("userData"), "thumbs");
const proxyCacheDir = () => path.join(app.getPath("userData"), "proxies");
const previewCacheDir = () => path.join(app.getPath("userData"), "previews");

/* single-photo export folder, remembered across restarts (not just within
   a session) so re-exporting the same photo later suggests an already-
   unique name instead of colliding and triggering the OS's native
   "replace?" prompt */
const exportStateFile = () => path.join(app.getPath("userData"), "export-state.json");
let lastExportDir = null;
try {
  const s = JSON.parse(fs.readFileSync(exportStateFile(), "utf8"));
  if (s && typeof s.lastExportDir === "string") lastExportDir = s.lastExportDir;
} catch {}
function setLastExportDir(dir) {
  lastExportDir = dir;
  try { fs.writeFileSync(exportStateFile(), JSON.stringify({ lastExportDir: dir })); } catch {}
}


/* editor proxy: ~3200px working copy. The editor never touches the
   full-resolution original directly, so a 200-MP file pans and zooms
   exactly as fluidly as a 12-MP one; deep-zoom sharpness is restored
   by on-demand tiles (see render-tile). */
const proxyInflight = new Map();
const PROXY_LONG = 3200;
async function proxyFor(p) {
  const st = await fsp.stat(p);
  const key = crypto.createHash("md5").update(`px|${p}|${st.size}|${st.mtimeMs}`).digest("hex");
  await fsp.mkdir(proxyCacheDir(), { recursive: true });
  const out = path.join(proxyCacheDir(), key + ".jpg");
  if (fs.existsSync(out)) return out;
  if (proxyInflight.has(out)) return proxyInflight.get(out);
  const job = (async () => {
    const src = needsConvert(p) ? await convertOnce(p) : p;
    await acquireSlot();
    try {
      const sharp = require("sharp");
      await sharp(src, { limitInputPixels: false })
        .rotate()
        .resize(PROXY_LONG, PROXY_LONG, { fit: "inside", withoutEnlargement: true })
        .jpeg({ quality: 90, chromaSubsampling: "4:4:4" })
        .withMetadata()
        .toFile(out);
      return out;
    } finally {
      releaseSlot();
    }
  })().finally(() => proxyInflight.delete(out));
  proxyInflight.set(out, job);
  return job;
}

/* small on-disk thumbnails so the library grid never loads 45-MP originals */
const thumbInflight = new Map();
async function thumbFor(p) {
  const st = await fsp.stat(p);
  const key = crypto.createHash("md5").update(`t|${p}|${st.size}|${st.mtimeMs}`).digest("hex");
  await fsp.mkdir(thumbCacheDir(), { recursive: true });
  const out = path.join(thumbCacheDir(), key + ".jpg");
  if (fs.existsSync(out)) return out;
  if (thumbInflight.has(out)) return thumbInflight.get(out);
  const job = (async () => {
    // convertOnce has its own queue slot — resolve it first to avoid
    // two thumb jobs holding both slots while waiting for conversion
    const src = needsConvert(p) ? await convertOnce(p) : p;
    await acquireSlot();
    try {
      const sharp = require("sharp");
      await sharp(src, { limitInputPixels: false })
        .rotate()
        .resize(512, 512, { fit: "inside", withoutEnlargement: true })
        .jpeg({ quality: 80, mozjpeg: true })
        .toFile(out);
      return out;
    } finally {
      releaseSlot();
    }
  })().finally(() => thumbInflight.delete(out));
  thumbInflight.set(out, job);
  return job;
}

/* on-disk cache for the small edited (crop/color/stamp) preview JPEG the
   renderer already bakes for the library grid + filmstrip thumbnail
   (App.jsx's bakePreview). That bake itself stays in the renderer (canvas
   API, already correct, no reason to duplicate it in Node) — this just
   persists its output keyed the same way as thumbFor/proxyFor (original
   file's path+size+mtime), so a fresh app launch can show it immediately
   instead of every edited-but-not-yet-reopened photo falling back to its
   plain unedited thumbnail until the user reopens it. */
function previewCacheKey(p, st) {
  return crypto.createHash("md5").update(`pv|${p}|${st.size}|${st.mtimeMs}`).digest("hex");
}

/* ---------------- TIFF / RAW / HEIC → displayable JPEG/PNG ----------------
   Chromium can't decode these, so the photo:// protocol converts them in
   the main process and caches the result on disk (keyed by path+size+mtime,
   so an edited-on-disk file re-converts automatically). Heavy decodes run
   through a small queue so a folder of RAWs doesn't eat all the RAM. */

const crypto = require("crypto");

/* dcraw worker (see raw-worker.cjs): keeps the main process responsive */
const { Worker } = require("worker_threads");
let rawWorker = null, rawSeq = 0;
const rawPending = new Map();
function dcrawInWorker(op, filePath) {
  if (!rawWorker) {
    rawWorker = new Worker(path.join(__dirname, "raw-worker.cjs"));
    rawWorker.on("message", (m) => {
      const pend = rawPending.get(m.id);
      if (!pend) return;
      rawPending.delete(m.id);
      m.error ? pend.reject(new Error(m.error)) : pend.resolve(m.data ? Buffer.from(m.data) : null);
    });
    rawWorker.on("error", () => {
      for (const pend of rawPending.values()) pend.reject(new Error("raw worker crashed"));
      rawPending.clear();
      rawWorker = null;
    });
  }
  const id = ++rawSeq;
  return new Promise((resolve, reject) => {
    rawPending.set(id, { resolve, reject });
    rawWorker.postMessage({ id, op, path: filePath });
  });
}
const inflight = new Map(); // path -> Promise<convertedPath>
let convActive = 0;
const convWaiters = [];
const acquireSlot = () =>
  new Promise((res) => {
    if (convActive < 2) { convActive++; res(); }
    else convWaiters.push(res);
  });
const releaseSlot = () => {
  convActive--;
  const next = convWaiters.shift();
  if (next) { convActive++; next(); }
};

/* generic rescue: many RAWs (incl. CR3) carry a full-size JPEG preview;
   find the largest well-formed JPEG segment in the raw bytes */
function largestEmbeddedJpeg(buf) {
  let best = null;
  for (let i = 0; i < buf.length - 3; i++) {
    if (buf[i] === 0xff && buf[i + 1] === 0xd8 && buf[i + 2] === 0xff) {
      const end = buf.indexOf(Buffer.from([0xff, 0xd9]), i + 3);
      if (end > i) {
        const len = end + 2 - i;
        if (!best || len > best.len) best = { start: i, len };
        i = end + 1;
      }
    }
  }
  return best && best.len > 50 * 1024 ? buf.subarray(best.start, best.start + best.len) : null;
}

async function convertToDisplayable(p) {
  const st = await fsp.stat(p);
  const key = crypto.createHash("md5").update(`${p}|${st.size}|${st.mtimeMs}`).digest("hex");
  await fsp.mkdir(convCacheDir(), { recursive: true });
  const jpgPath = path.join(convCacheDir(), key + ".jpg");
  const pngPath = path.join(convCacheDir(), key + ".png");
  if (fs.existsSync(jpgPath)) return jpgPath;
  if (fs.existsSync(pngPath)) return pngPath;

  await acquireSlot();
  try {
    const buf = await fsp.readFile(p);

    if (HEIC_RE.test(p)) {
      const heicConvert = require("heic-convert");
      const out = await heicConvert({ buffer: buf, format: "JPEG", quality: 0.92 });
      await fsp.writeFile(jpgPath, Buffer.from(out));
      return jpgPath;
    }

    if (TIFF_RE.test(p)) {
      const sharp = require("sharp");
      const meta = await sharp(buf, { limitInputPixels: false }).metadata();
      if (meta.hasAlpha) {
        await sharp(buf, { limitInputPixels: false }).png().withMetadata().toFile(pngPath);
        return pngPath;
      }
      await sharp(buf, { limitInputPixels: false }).jpeg({ quality: 97, chromaSubsampling: "4:4:4" }).withMetadata().toFile(jpgPath);
      return jpgPath;
    }

    if (RAW_RE.test(p)) {
      // 1) embedded full-size preview via dcraw in a worker thread
      try {
        const thumb = await dcrawInWorker("thumb", p);
        if (thumb && thumb.length > 20 * 1024 && thumb[0] === 0xff && thumb[1] === 0xd8) {
          await fsp.writeFile(jpgPath, thumb);
          return jpgPath;
        }
      } catch {}
      // 2) full raw development (worker thread), stored as JPEG q97 —
      //    a 45-MP PNG decodes painfully slowly, JPEG keeps things fluid
      try {
        const tiff = await dcrawInWorker("tiff", p);
        if (tiff && tiff.length) {
          const sharp = require("sharp");
          await sharp(tiff, { limitInputPixels: false })
            .jpeg({ quality: 97, chromaSubsampling: "4:4:4" })
            .withMetadata()
            .toFile(jpgPath);
          return jpgPath;
        }
      } catch {}
      // 3) last resort (covers CR3 and other formats dcraw predates):
      //    scan the file for its largest embedded JPEG
      const jpeg = largestEmbeddedJpeg(buf);
      if (jpeg) {
        await fsp.writeFile(jpgPath, jpeg);
        return jpgPath;
      }
      throw new Error("raw decode failed");
    }

    throw new Error("unsupported format");
  } finally {
    releaseSlot();
  }
}

function convertOnce(p) {
  if (inflight.has(p)) return inflight.get(p);
  const job = convertToDisplayable(p).finally(() => inflight.delete(p));
  inflight.set(p, job);
  return job;
}

let mainWindow = null;

/* Custom protocol so the renderer can display photos straight from disk */
protocol.registerSchemesAsPrivileged([
  {
    scheme: "photo",
    privileges: { secure: true, supportFetchAPI: true, stream: true, corsEnabled: true },
  },
]);

async function scanDir(dir, folderLabel, out) {
  let entries;
  try {
    entries = await fsp.readdir(dir, { withFileTypes: true });
  } catch {
    return; // no permission for a subfolder — skip it
  }
  for (const e of entries) {
    if (e.name.startsWith(".")) continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      await scanDir(full, `${folderLabel}/${e.name}`, out);
    } else if (e.isFile() && isImg(e.name)) {
      let size = 0, mtime = 0;
      try {
        const st = await fsp.stat(full);
        size = st.size; mtime = st.mtimeMs;
      } catch {}
      out.push({ path: full, name: e.name, folder: folderLabel, size, mtime });
    }
  }
}

/* ---------------- folder watching ---------------- */

const watchers = new Map();   // root -> { watcher, timer, poll }
const RECURSIVE_OK = process.platform !== "linux";

async function rescanAndNotify(root) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const records = [];
  await scanDir(root, path.basename(root), records);
  mainWindow.webContents.send("folder-scan", { root, records });
}

function scheduleScan(root) {
  const entry = watchers.get(root);
  if (!entry) return;
  clearTimeout(entry.timer);
  entry.timer = setTimeout(() => rescanAndNotify(root), 1200);
}

function watchRoot(root) {
  if (watchers.has(root) || !fs.existsSync(root)) return;
  const entry = { watcher: null, timer: null, poll: null };
  try {
    entry.watcher = fs.watch(root, { recursive: RECURSIVE_OK }, () => scheduleScan(root));
    entry.watcher.on("error", () => {});
  } catch {
    /* watching unavailable — polling below still covers it */
  }
  // Linux fs.watch is not recursive; poll as a safety net
  if (!RECURSIVE_OK) {
    entry.poll = setInterval(() => rescanAndNotify(root), 15000);
  }
  watchers.set(root, entry);
}

function unwatchRoot(root) {
  const entry = watchers.get(root);
  if (!entry) return;
  clearTimeout(entry.timer);
  if (entry.poll) clearInterval(entry.poll);
  try { entry.watcher?.close(); } catch {}
  watchers.delete(root);
}

/* ---------------- path allowlist ----------------
   The renderer is trusted (contextIsolation + a narrow contextBridge API,
   no remote content ever loads), but the photo:// protocol and the IPC
   handlers below take a raw path and read/convert whatever's there. This
   allowlist keeps that scoped to files the app actually knows about —
   watched folders, plus individually-imported files that aren't under
   one — so a compromised renderer (e.g. a supply-chain-poisoned
   dependency) can't use them as a generic arbitrary-file-read primitive. */
const allowedFiles = new Set();
function isAllowedPath(p) {
  if (allowedFiles.has(p)) return true;
  for (const root of watchers.keys()) {
    const rel = path.relative(root, p);
    if (rel && !rel.startsWith("..") && !path.isAbsolute(rel)) return true;
  }
  return false;
}

/* ---------------- window ---------------- */

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1300,
    height: 840,
    minWidth: 940,
    minHeight: 620,
    backgroundColor: "#0B0C0F",
    title: "Splicer Lab",
    icon: path.join(__dirname, "..", "build", "icon.png"),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));

  // the app never links out; block any navigation/new-window attempt
  // (e.g. from a stray target="_blank") instead of letting it hijack
  // the window or spawn an unsandboxed popup
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  mainWindow.webContents.on("will-navigate", (e) => e.preventDefault());

  // Catch changes that happened while the app was in the background
  mainWindow.on("focus", () => {
    for (const root of watchers.keys()) scheduleScan(root);
  });
}

app.whenReady().then(() => {
  appendLog(`[main] app ready, version ${app.getVersion()}, platform ${process.platform}`);

  protocol.handle("photo", async (request) => {
    const rest = request.url.replace(/^photo:\/\//, "");
    const [encPath, query] = rest.split("?");
    const p = decodeURIComponent(encPath);
    if (!isAllowedPath(p)) return new Response("not found", { status: 404 });
    try {
      if (query && query.includes("thumb")) {
        const t = await thumbFor(p);
        return net.fetch(pathToFileURL(t).toString());
      }
      if (query && query.includes("proxy")) {
        const pr = await proxyFor(p);
        return net.fetch(pathToFileURL(pr).toString());
      }
      if (query && query.includes("preview")) {
        // unlike thumb/proxy, never generate one here — this only ever
        // serves a preview a past editing session already baked and saved
        // via cache-edit-preview below; if none exists, the caller (load-
        // library) simply doesn't set previewUrl and the UI falls back to
        // the plain thumbnail.
        const st = await fsp.stat(p);
        const cached = path.join(previewCacheDir(), previewCacheKey(p, st) + ".jpg");
        if (fs.existsSync(cached)) return net.fetch(pathToFileURL(cached).toString());
        return new Response("no cached preview", { status: 404 });
      }
      if (NATIVE_RE.test(p)) return net.fetch(pathToFileURL(p).toString());
      const converted = await convertOnce(p);
      return net.fetch(pathToFileURL(converted).toString());
    } catch {
      // thumbnail failed (e.g. exotic format) — fall back to the original
      try {
        if (NATIVE_RE.test(p)) return net.fetch(pathToFileURL(p).toString());
      } catch {}
      return new Response("decode failed", { status: 415 });
    }
  });

  if (process.platform === "darwin" && !app.isPackaged) {
    // packaged builds get the Dock icon from the .icns bundle automatically;
    // an unpackaged `npm start` run needs it set explicitly to match.
    // build/ isn't in the electron-builder "files" list, so this path
    // doesn't exist in packaged builds — gate on isPackaged or this throws
    // before createWindow() runs and the app silently never shows a window.
    app.dock.setIcon(path.join(__dirname, "..", "build", "icon.png"));
  }
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

/* ---------------- IPC: importing ---------------- */

/* One dialog, either kind of pick: macOS lets the user select files AND
   folders side by side (openFile + openDirectory combined); the renderer
   feeds whatever comes back into import-paths below, which already knows
   how to sort files from folders (same code path as drag-and-drop). */
ipcMain.handle("pick-files-or-folder", async () => {
  const res = await dialog.showOpenDialog({
    title: "Import photos or a folder",
    properties: ["openFile", "openDirectory", "multiSelections"],
    filters: [{
      name: "Images",
      extensions: [
        "jpg", "jpeg", "png", "webp", "gif", "avif", "bmp",
        "tif", "tiff", "heic", "heif",
        "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "orf", "rw2",
        "raf", "dng", "pef", "srw", "erf", "kdc", "3fr", "mrw", "x3f",
      ],
    }],
  });
  if (res.canceled) return [];
  return res.filePaths;
});

/* Turn syncing on for a set of existing folder roots (e.g. folders that
   were imported before watching existed, or re-enabling from the sidebar) */
ipcMain.handle("watch-roots", (_e, roots) => {
  for (const r of roots || []) watchRoot(r);
  return [...watchers.keys()];
});

/* Dropped items: mix of files and folders (absolute paths from the renderer) */
ipcMain.handle("import-paths", async (_e, paths) => {
  const records = [];
  const roots = [];
  for (const p of paths || []) {
    try {
      const st = await fsp.stat(p);
      if (st.isDirectory()) {
        await scanDir(p, path.basename(p), records);
        watchRoot(p);
        roots.push(p);
      } else if (st.isFile() && isImg(p)) {
        allowedFiles.add(p);
        records.push({ path: p, name: path.basename(p), folder: "", size: st.size, mtime: st.mtimeMs });
      }
    } catch {
      /* skip unreadable */
    }
  }
  return { records, roots };
});

ipcMain.handle("unwatch-root", (_e, root) => {
  unwatchRoot(root);
  return true;
});

/* ---------------- IPC: persistent library ---------------- */

ipcMain.handle("load-library", async () => {
  try {
    const raw = await fsp.readFile(libFile(), "utf8");
    const parsed = JSON.parse(raw);
    // legacy format was a plain array of items
    const items = Array.isArray(parsed) ? parsed : parsed.items || [];
    const savedRoots = Array.isArray(parsed) ? [] : parsed.watchedRoots || [];
    const alive = [];
    let missing = 0;
    for (const it of items) {
      if (!it || !it.path) { missing++; continue; }
      try {
        const st = fs.statSync(it.path);
        it.sizeBytes = st.size;      // always refresh: the file may have
        it.size = st.size;           // been replaced on disk since import
        it.mtime = st.mtimeMs;
        delete it.previewUrl;        // blob/data: URL from a past session — dead on arrival
        if (it.edits) {
          // if a previous session cached this photo's baked preview to disk
          // (cache-edit-preview below), serve it immediately instead of
          // leaving previewUrl unset — otherwise every edited photo the
          // user hasn't reopened yet in THIS session shows its plain
          // unedited thumbnail until they do. `v=` busts the renderer's own
          // image cache if the cached file is later overwritten with a
          // newer edit while pointing at this same key.
          try {
            const key = previewCacheKey(it.path, st);
            const cached = path.join(previewCacheDir(), key + ".jpg");
            const cst = fs.statSync(cached);
            it.previewUrl = `photo://${encodeURIComponent(it.path)}?preview&k=${key}&v=${cst.mtimeMs}`;
          } catch {}
        }
        allowedFiles.add(it.path);
        alive.push(it);
      } catch {
        missing++;
      }
    }
    for (const root of savedRoots) watchRoot(root);
    const settings = Array.isArray(parsed) ? {} : parsed.settings || {};
    return { items: alive, missing, watchedRoots: [...watchers.keys()], settings };
  } catch {
    return { items: [], missing: 0, watchedRoots: [], settings: {} };
  }
});

ipcMain.handle("save-library", async (_e, items, settings) => {
  try {
    const payload = { items, watchedRoots: [...watchers.keys()], settings: settings || {} };
    await fsp.writeFile(libFile(), JSON.stringify(payload), "utf8");
    return true;
  } catch {
    return false;
  }
});

/* persist App.jsx's bakePreview() output (a data:image/jpeg;base64,... URL)
   to disk, keyed like thumbFor/proxyFor — see previewCacheKey above. Called
   fire-and-forget every time the renderer computes a fresh preview, so
   load-library can serve it back on the next launch without recomputing
   anything. dataUrl === null means edits were cleared — drop the stale
   cached file rather than leave an orphaned one behind. */
ipcMain.handle("cache-edit-preview", async (_e, filePath, dataUrl) => {
  try {
    const st = await fsp.stat(filePath);
    const out = path.join(previewCacheDir(), previewCacheKey(filePath, st) + ".jpg");
    if (!dataUrl) {
      await fsp.unlink(out).catch(() => {});
      return true;
    }
    const m = /^data:image\/jpeg;base64,(.+)$/.exec(dataUrl);
    if (!m) return false;
    await fsp.mkdir(previewCacheDir(), { recursive: true });
    await fsp.writeFile(out, Buffer.from(m[1], "base64"));
    return true;
  } catch {
    return false;
  }
});

/* ---------------- IPC: export to disk ---------------- */

/* --------------------------------------------------------------
   Export engine v2 — sharp + MozJPEG.
   The final encode no longer uses the browser's canvas codec:
   sharp applies the crop/rotation losslessly on raw pixels and
   encodes with MozJPEG (best quality-per-byte JPEG encoder in
   production use) or high-effort WebP. Quality is binary-searched
   natively, which is much faster than canvas re-encodes.
   -------------------------------------------------------------- */

const clampN = (v, a, b) => Math.min(b, Math.max(a, v));
const needsConvert = (p) => TIFF_RE.test(p) || HEIC_RE.test(p) || RAW_RE.test(p);

/* bake one photo (with its non-destructive edits) to raw RGB at the
   requested output scale; returns { data, info } */
async function bakeRaw(item, scalePct) {
  const sharp = require("sharp");
  const src = needsConvert(item.path) ? await convertOnce(item.path) : item.path;
  // stage 1: decode + EXIF auto-orient
  const oriented = await sharp(src, { limitInputPixels: false })
    .rotate()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const e = item.edits;
  // Clone stamp, then color adjust — same full-res, EXIF-oriented,
  // pre-rotate/crop buffer, the one frame stable regardless of whatever
  // crop/rotate/scale choices follow, exactly mirroring how qcx/qcy/cw/ch
  // are themselves defined relative to this same pre-rotate frame. `e.heal`
  // is read as a fallback for photos saved before the heal->clone-stamp
  // rename (old data under the old field name, not the removed automatic
  // Heal tool's own `e.fill`, which is simply never read anymore — any
  // leftover `edits.fill` on an old photo is inert dead data now).
  const stampData = e && (e.stamp || e.heal);
  if (stampData && stampData.length) {
    // in the AI utility process (neuralWorker.cjs's bakeStamp): PatchMatch,
    // the networks and heal at full resolution take seconds, which here
    // would freeze every other request until the export finished
    const { data, info } = oriented;
    const out = await callNeural({
      op: "bakeStamp", data, width: info.width, height: info.height, channels: info.channels,
      stamp: stampData, mode: aiMode, lamaReady: lamaReady(),
    });
    data.set(out);
  }
  // red eye: same frame as the stamp (original-image px, EXIF-oriented,
  // pre-rotate/crop), so stored entries apply as-is — no mapping needed
  if (e && e.redeye && e.redeye.length) {
    const { applyRedEyeToRGBBuffer } = await getRedEyeTool();
    applyRedEyeToRGBBuffer(oriented.data, oriented.info.width, oriented.info.height, oriented.info.channels, e.redeye);
  }
  let base = oriented;
  let cw = base.info.width, ch = base.info.height, cx = cw / 2, cy = ch / 2;
  if (e) {
    if (Math.abs(e.theta) > 0.01) {
      base = await sharp(oriented.data, {
        raw: { width: oriented.info.width, height: oriented.info.height, channels: oriented.info.channels },
        limitInputPixels: false,
      })
        .rotate(e.theta, { background: { r: 0, g: 0, b: 0 } })
        .raw()
        .toBuffer({ resolveWithObject: true });
    }
    const t = (e.theta * Math.PI) / 180;
    cx = base.info.width / 2 + (e.qcx * Math.cos(t) - e.qcy * Math.sin(t));
    cy = base.info.height / 2 + (e.qcx * Math.sin(t) + e.qcy * Math.cos(t));
    cw = Math.round(e.cw);
    ch = Math.round(e.ch);
  }
  let left = Math.round(cx - cw / 2), top = Math.round(cy - ch / 2);
  cw = clampN(cw, 1, base.info.width);
  ch = clampN(ch, 1, base.info.height);
  left = clampN(left, 0, base.info.width - cw);
  top = clampN(top, 0, base.info.height - ch);
  let pipe = sharp(base.data, {
    raw: { width: base.info.width, height: base.info.height, channels: base.info.channels },
    limitInputPixels: false,
  }).extract({ left, top, width: cw, height: ch });
  const outW = Math.max(1, Math.round(cw * (scalePct / 100)));
  if (outW !== cw) pipe = pipe.resize(outW);
  const rawObj = await pipe.raw().toBuffer({ resolveWithObject: true });
  if (e && e.adjust) {
    const { applyAdjustToRGBBuffer } = await getColorAdjust();
    // scale/origDiag: local sliders (Texture/Clarity/Dehaze, pv-2
    // Highlights/Shadows) size their filters off the ORIGINAL image, so the
    // export filters the same physical area the proxy preview did
    applyAdjustToRGBBuffer(rawObj.data, rawObj.info.width, rawObj.info.height, rawObj.info.channels, e.adjust, {
      scale: rawObj.info.width / cw,
      origDiag: Math.hypot(oriented.info.width, oriented.info.height),
    });
  }
  return rawObj;
}

function encoderFor(rawObj, format, meta) {
  const sharp = require("sharp");
  const opts = { raw: { width: rawObj.info.width, height: rawObj.info.height, channels: rawObj.info.channels } };
  // withMetadata() always attaches an sRGB ICC profile (correct colors on
  // wide-gamut displays); meta.exif optionally carries key camera fields
  const md = meta && meta.exif ? { exif: meta.exif } : {};
  return (q) =>
    format === "webp"
      ? sharp(rawObj.data, opts).webp({ quality: q, effort: 5 }).withMetadata(md).toBuffer()
      : sharp(rawObj.data, opts)
          .jpeg({ quality: q, mozjpeg: true, chromaSubsampling: q >= 90 ? "4:4:4" : "4:2:0" })
          .withMetadata(md)
          .toBuffer();
}

/* binary-search encoder quality to fit targetBytes; quality can reach 100 —
   if even that is under the target, the file simply cannot get better */
async function encodeToTarget(rawObj, format, targetBytes, meta) {
  const enc = encoderFor(rawObj, format, meta);
  let lo = 5, hi = 100;
  let buf = await enc(hi);
  if (buf.length <= targetBytes) return { buf, quality: hi, over: false, maxed: true };
  let bufLo = await enc(lo);
  if (bufLo.length > targetBytes) return { buf: bufLo, quality: lo, over: true, maxed: false };
  let best = { buf: bufLo, quality: lo, over: false, maxed: false };
  for (let i = 0; i < 7 && lo < hi - 1; i++) {
    const mid = Math.round((lo + hi) / 2);
    const b = await enc(mid);
    if (b.length <= targetBytes) { best = { buf: b, quality: mid, over: false, maxed: false }; lo = mid; }
    else hi = mid;
  }
  return best;
}

/* small cache so the estimate dialog doesn't re-bake on every keystroke */
const bakeCache = new Map(); // key -> Promise<rawObj>
function bakeCached(item, scalePct) {
  const key = `${item.path}|${JSON.stringify(item.edits)}|${scalePct}`;
  if (bakeCache.has(key)) return bakeCache.get(key);
  const job = bakeRaw(item, scalePct);
  bakeCache.set(key, job);
  if (bakeCache.size > 4) bakeCache.delete(bakeCache.keys().next().value);
  return job;
}


/* deep-zoom tile: exact visible region of the ORIENTED original at the
   requested pixel density — snaps the proxy view to pixel sharpness */
let tileGen = 0;
/* renderer → main: one Remove-mode neural job (see removeTool.mjs /
   GLImage). job = { w, h, rgb: Uint8Array w·h·3, hole: Uint8Array w·h,
   engine: "migan" | "lama" } */
ipcMain.handle("neural-inpaint", async (_e, job) => {
  try {
    return await neuralInpaintCached(job);
  } catch (err) {
    appendLog(`[main] neural inpaint failed: ${err?.stack || err}`);
    return null;
  }
});
ipcMain.handle("neural-warm", async (_e, engine) => {
  try {
    await callNeural({ op: "warm", engine: engine === "lama" ? "lama" : "migan" });
    return true;
  } catch (err) {
    appendLog(`[main] neural warm-up failed: ${err?.stack || err}`);
    return false;
  }
});
ipcMain.handle("fill-cache-get", (_e, key, len) => fillCacheGet(String(key), len));
ipcMain.handle("fill-cache-put", (_e, key, bytes) => fillCachePut(String(key), bytes));
ipcMain.handle("ai-info", () => ({
  cpus: (os.cpus() || []).length,
  memGB: os.totalmem() / 2 ** 30,
  lamaReady: lamaReady(),
  lamaBytes: LAMA_MODEL.size,
}));
ipcMain.handle("ai-set-mode", (_e, mode) => { aiMode = ["off", "fast", "max"].includes(mode) ? mode : null; });
ipcMain.handle("lama-download", async (e) => {
  try {
    return { ok: await downloadLama(e.sender) };
  } catch (err) {
    appendLog(`[main] LaMa download failed: ${err?.stack || err}`);
    return { ok: false, error: String(err?.message || err) };
  }
});

ipcMain.handle("render-tile", async (_e, p, t) => {
  if (!isAllowedPath(p)) return null;
  const myGen = ++tileGen;
  try {
    await acquireSlot();
    try {
      if (myGen !== tileGen) return null; // a newer tile request superseded this one while queued
      const sharp = require("sharp");
      const src = needsConvert(p) ? await convertOnce(p) : p;
      const sx = Math.max(0, Math.round(t.sx));
      const sy = Math.max(0, Math.round(t.sy));
      const sw = Math.max(4, Math.round(t.sw));
      const sh = Math.max(4, Math.round(t.sh));
      const dw = Math.min(4096, Math.max(8, Math.round(t.dw)));
      const buf = await sharp(src, { limitInputPixels: false })
        .rotate()
        .extract({ left: sx, top: sy, width: sw, height: sh })
        .resize({ width: dw })
        .jpeg({ quality: 90, chromaSubsampling: "4:4:4" })
        .withMetadata()
        .toBuffer();
      return myGen === tileGen ? buf : null;
    } finally {
      releaseSlot();
    }
  } catch {
    return null;
  }
});

ipcMain.handle("log-error", (_e, message) => {
  appendLog(`[renderer] ${String(message).slice(0, 2000)}`);
  return true;
});

ipcMain.handle("read-exif", async (_e, p) => {
  if (!isAllowedPath(p)) return null;
  try {
    const exifr = require("exifr");
    const d = await exifr.parse(p, {
      pick: ["Make", "Model", "LensModel", "ISO", "ExposureTime", "FNumber", "FocalLength", "FocalLengthIn35mmFormat", "DateTimeOriginal"],
    });
    if (!d) return null;
    return {
      make: d.Make || null,
      model: d.Model || null,
      lens: d.LensModel || null,
      iso: d.ISO || null,
      exposure: d.ExposureTime || null,
      fnumber: d.FNumber || null,
      focal: d.FocalLength || null,
      // 35 mm equivalent — Auto Straighten's camera model (electron/autoStraighten.mjs)
      focal35: d.FocalLengthIn35mmFormat || null,
      taken: d.DateTimeOriginal ? new Date(d.DateTimeOriginal).getTime() : null,
    };
  } catch {
    return null;
  }
});


/* minimal EXIF carry-over for "Keep EXIF" exports */
async function exifMetaFor(p) {
  try {
    const exifr = require("exifr");
    const d = await exifr.parse(p, { pick: ["Make", "Model", "DateTimeOriginal", "LensModel"] });
    if (!d) return null;
    const ifd0 = {};
    if (d.Make) ifd0.Make = String(d.Make);
    if (d.Model) ifd0.Model = String(d.Model);
    const exifIFD = {};
    if (d.DateTimeOriginal) {
      const dt = new Date(d.DateTimeOriginal);
      const pad = (n) => String(n).padStart(2, "0");
      exifIFD.DateTimeOriginal = `${dt.getFullYear()}:${pad(dt.getMonth() + 1)}:${pad(dt.getDate())} ${pad(dt.getHours())}:${pad(dt.getMinutes())}:${pad(dt.getSeconds())}`;
    }
    if (d.LensModel) exifIFD.LensModel = String(d.LensModel);
    const exif = {};
    if (Object.keys(ifd0).length) exif.IFD0 = ifd0;
    if (Object.keys(exifIFD).length) exif.IFD2 = exifIFD;
    return Object.keys(exif).length ? { exif } : null;
  } catch {
    return null;
  }
}

/* file name from template: {name}, {date} (file date, YYYY-MM-DD) */
function buildExportName(item, template, ext) {
  const base = item.name.replace(/\.[^.]+$/, "");
  let date = "";
  try {
    date = new Date(fs.statSync(item.path).mtimeMs).toISOString().slice(0, 10);
  } catch {}
  const tpl = (template || "{name}").trim() || "{name}";
  let name = tpl.split("{name}").join(base).split("{date}").join(date);
  if (tpl === "{name}" && item.edits) name += "-edited";
  name = name.replace(/[\\/:*?"<>|]/g, "-");
  return `${name}.${ext}`;
}

function uniquePath(dir, fileName) {
  let candidate = path.join(dir, fileName);
  if (!fs.existsSync(candidate)) return candidate;
  const ext = path.extname(fileName);
  const stem = fileName.slice(0, -ext.length);
  for (let i = 2; i < 1000; i++) {
    candidate = path.join(dir, `${stem} (${i})${ext}`);
    if (!fs.existsSync(candidate)) return candidate;
  }
  return candidate;
}

ipcMain.handle("estimate-photo", async (_e, item, opts) => {
  if (!isAllowedPath(item.path)) return { error: "not found" };
  try {
    const targetBytes = opts.targetMB * 1024 * 1024;
    const fullRaw = await bakeCached(item, opts.scale);
    const meta = opts.keepExif ? await exifMetaFor(item.path) : null;
    const fullPixels = fullRaw.info.width * fullRaw.info.height;

    // For large images, search quality on a much smaller preview (long edge
    // capped ~1400px) and scale the resulting byte count back up by the
    // pixel-count ratio. JPEG/WebP size is close to linear in pixel count at
    // a fixed quality, so this gives a near-instant, still-accurate estimate.
    // The real export always encodes at full resolution — this only speeds
    // up the live preview number in the dialog.
    const PREVIEW_CAP = 1400 * 1400;
    let raw = fullRaw, sizeRatio = 1;
    if (fullPixels > PREVIEW_CAP * 1.3) {
      const previewScale = Math.sqrt(PREVIEW_CAP / fullPixels) * 100;
      raw = await bakeRaw(item, opts.scale * (previewScale / 100));
      sizeRatio = fullPixels / (raw.info.width * raw.info.height);
    }

    const res = await encodeToTarget(raw, opts.format, targetBytes / sizeRatio, meta);
    const scaledSize = Math.round(res.buf.length * sizeRatio);
    return {
      size: sizeRatio > 1 ? Math.min(scaledSize, res.over ? scaledSize : Math.max(scaledSize, 1)) : res.buf.length,
      quality: res.quality, over: res.over, maxed: res.maxed,
      w: fullRaw.info.width, h: fullRaw.info.height,
    };
  } catch (err) {
    return { error: String(err) };
  }
});

ipcMain.handle("pick-dir", async () => {
  const res = await dialog.showOpenDialog({
    title: "Choose export folder",
    properties: ["openDirectory", "createDirectory"],
  });
  return res.canceled || !res.filePaths.length ? null : res.filePaths[0];
});

ipcMain.handle("export-photos", async (event, items, opts) => {
  items = items.filter((it) => isAllowedPath(it.path));
  if (!items.length) return { canceled: true };
  let dir = opts.dir || null;
  const ext = opts.format === "webp" ? "webp" : "jpg";
  if (!dir) {
    if (items.length === 1) {
      const name = buildExportName(items[0], opts.template, ext);
      const defaultPath = lastExportDir ? uniquePath(lastExportDir, name) : name;
      const res = await dialog.showSaveDialog({ title: "Export photo", defaultPath });
      if (res.canceled || !res.filePath) return { canceled: true };
      const raw = await bakeCached(items[0], opts.scale);
      const meta = opts.keepExif ? await exifMetaFor(items[0].path) : null;
      const enc = await encodeToTarget(raw, opts.format, opts.targetMB * 1024 * 1024, meta);
      await fsp.writeFile(res.filePath, enc.buf);
      setLastExportDir(path.dirname(res.filePath));
      return { ok: true, dir: lastExportDir, exported: 1 };
    }
    const res = await dialog.showOpenDialog({
      title: "Choose export folder",
      properties: ["openDirectory", "createDirectory"],
    });
    if (res.canceled || !res.filePaths.length) return { canceled: true };
    dir = res.filePaths[0];
  }
  await fsp.mkdir(dir, { recursive: true });
  let exported = 0, failed = 0;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    event.sender.send("export-progress", { done: i, total: items.length, name: it.name });
    try {
      const raw = await bakeCached(it, opts.scale);
      const meta = opts.keepExif ? await exifMetaFor(it.path) : null;
      const enc = await encodeToTarget(raw, opts.format, opts.targetMB * 1024 * 1024, meta);
      await fsp.writeFile(uniquePath(dir, buildExportName(it, opts.template, ext)), enc.buf);
      exported++;
    } catch {
      failed++;
    }
  }
  event.sender.send("export-progress", { done: items.length, total: items.length });
  return { ok: true, dir, exported, failed };
});

/* legacy path used by the browser-fallback renderer */

ipcMain.handle("export-files", async (_e, files) => {
  if (!files || !files.length) return { canceled: true };

  if (files.length === 1) {
    const res = await dialog.showSaveDialog({
      title: "Export photo",
      defaultPath: files[0].name,
    });
    if (res.canceled || !res.filePath) return { canceled: true };
    await fsp.writeFile(res.filePath, Buffer.from(files[0].data));
    return { ok: true, dir: path.dirname(res.filePath) };
  }

  const res = await dialog.showOpenDialog({
    title: "Choose export folder",
    properties: ["openDirectory", "createDirectory"],
  });
  if (res.canceled || !res.filePaths.length) return { canceled: true };
  const dir = res.filePaths[0];
  for (const f of files) {
    await fsp.writeFile(path.join(dir, f.name), Buffer.from(f.data));
  }
  return { ok: true, dir };
});
