/* Downloads the LaMa inpainting model (electron/models/lama_fp32.onnx,
   208 MB) — too big for git (GitHub refuses files over 100 MB), so it is
   fetched on npm install instead, from a pinned revision of the upstream
   Carve/LaMa-ONNX repo and checked against its SHA-256. Runs as
   postinstall: skips when the file is already there and intact. Offline,
   it only warns (Remove then uses MI-GAN for every hole) — except on CI,
   where a missing model fails the build so no release ships without it. */
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MODELS = [
  {
    file: "lama_fp32.onnx",
    url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx",
    size: 208044816,
    sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
  },
];

const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "electron", "models");

function sha256(file) {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash("sha256");
    fs.createReadStream(file).on("data", (d) => h.update(d)).on("end", () => resolve(h.digest("hex"))).on("error", reject);
  });
}

async function fetchModel(m) {
  const dest = path.join(dir, m.file);
  if (fs.existsSync(dest) && fs.statSync(dest).size === m.size && (await sha256(dest)) === m.sha256) return;
  console.log(`fetch-models: downloading ${m.file} (${(m.size / 1e6).toFixed(0)} MB)…`);
  const res = await fetch(m.url);
  if (!res.ok) throw new Error(`${m.file}: HTTP ${res.status}`);
  const tmp = dest + ".part";
  const out = fs.createWriteStream(tmp);
  for await (const chunk of res.body) if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
  await new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve())));
  const got = await sha256(tmp);
  if (got !== m.sha256) {
    fs.rmSync(tmp, { force: true });
    throw new Error(`${m.file}: checksum mismatch (${got})`);
  }
  fs.renameSync(tmp, dest);
  console.log(`fetch-models: ${m.file} ok`);
}

try {
  for (const m of MODELS) await fetchModel(m);
} catch (err) {
  console.warn(`fetch-models: ${err.message}`);
  if (process.env.CI) process.exit(1);
  console.warn("fetch-models: Remove will use MI-GAN only; run `node scripts/fetch-models.mjs` to retry.");
}
