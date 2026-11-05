/* Web Worker for the editor's heavy pixel work, off the UI thread:
   - Remove's PatchMatch fill (electron/removeTool.mjs `inpaint`, see its
     setAsyncFill): in { key, w, h, img, hole, seed }, out { key, pm }
     (w·h·3 bytes)
   - Auto Straighten (electron/autoStraighten.mjs): in { straighten: id,
     pixels, width, height, focal35 }, out { straighten: id, angle } */
import { inpaint } from "../electron/removeTool.mjs";
import { autoStraightenAngle } from "../electron/autoStraighten.mjs";

self.onmessage = ({ data: job }) => {
  if (job.straighten != null) {
    let angle = null;
    try { angle = autoStraightenAngle(job.pixels, job.width, job.height, 4, job.focal35); } catch {}
    self.postMessage({ straighten: job.straighten, angle });
    return;
  }
  let pm = null;
  try {
    inpaint(job.img, job.w, job.h, job.hole, job.seed);
    pm = Uint8Array.from(Uint8ClampedArray.from(job.img));
  } catch {}
  self.postMessage({ key: job.key, pm }, pm ? [pm.buffer] : []);
};
