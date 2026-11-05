/* Utility-process entry for local neural inpainting (electron/neuralFill.mjs),
   forked by main.cjs — see neuralFill.mjs's header for why it isn't run in
   the main process. Messages: { id, op: "inpaint", rgb, hole, w, h, engine }
   → { id, ok, out }; { id, op: "warm", engine } → { id, ok }; and
   { id, op: "bakeStamp", data, width, height, channels, stamp, mode,
   lamaReady } → { id, ok, out } — an export's whole stamp composite
   (PatchMatch, networks, heal — seconds at full resolution), done here so
   the main process never blocks on it. */
const path = require("path");
const { pathToFileURL } = require("url");

const lazy = (file) => { let p = null; return () => (p = p || import(pathToFileURL(path.join(__dirname, file)).href)); };
const neuralFill = lazy("neuralFill.mjs"), stampTool = lazy("stampTool.mjs"), removeTool = lazy("removeTool.mjs");

/* Strokes one at a time, each Remove stroke with its network result before
   the next stroke reads its pixels — exactly what the on-screen composite
   settles on, with one network run per stroke (re-running the whole stack
   until nothing new was queued redid every later overlapping stroke). */
async function bakeStamp(nf, msg) {
  const [st, rt] = await Promise.all([stampTool(), removeTool()]);
  const { data: buf, width, height, channels } = msg;
  rt.setNeuralMode(msg.mode, msg.lamaReady);
  for (const stroke of st.normalizeStamp(msg.stamp)) {
    const before = stroke.mode === "remove" ? Uint8Array.from(buf) : null;
    rt.setNeuralEnabled(true);
    st.applyStampToRGBBuffer(buf, width, height, channels, [stroke]);
    rt.setNeuralEnabled(false);
    const jobs = rt.takeNeuralJobs();
    if (!jobs.length) continue;
    for (const j of jobs) {
      try { rt.putNeuralResult(j.key, await nf.neuralInpaint(j.rgb, j.hole, j.w, j.h, j.engine)); } catch { rt.putNeuralResult(j.key, null); }
    }
    buf.set(before);
    st.applyStampToRGBBuffer(buf, width, height, channels, [stroke]);
  }
  return buf;
}

process.parentPort.on("message", async ({ data }) => {
  const { id, op } = data;
  try {
    const nf = await neuralFill();
    if (op === "warm") {
      await nf.warm(data.engine);
      process.parentPort.postMessage({ id, ok: true });
    } else if (op === "bakeStamp") {
      process.parentPort.postMessage({ id, ok: true, out: await bakeStamp(nf, data) });
    } else {
      const out = await nf.neuralInpaint(data.rgb, data.hole, data.w, data.h, data.engine);
      process.parentPort.postMessage({ id, ok: true, out });
    }
  } catch (err) {
    process.parentPort.postMessage({ id, ok: false, error: String(err?.stack || err) });
  }
});
