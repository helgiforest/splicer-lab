import { useState, useRef, useEffect, useLayoutEffect, useCallback, useMemo, cloneElement, Children } from "react";
import { createPortal } from "react-dom";
import {
  CURRENT_PV, defaultAdjust, normalizeAdjust, isDefaultAdjust, compileAdjust, needsGuides, computeGuides, neutralizeWhiteBalance,
  applyAdjustToRGBBuffer, DEHAZE_T_MIN, DEHAZE_OMEGA,
} from "../electron/colorAdjust.mjs";
import { isEmptyStamp, applyStampToRGBBuffer, normalizeStamp, findHealSource } from "../electron/stampTool.mjs";
import { setNeuralEnabled, takeNeuralJobs, putNeuralResult } from "../electron/removeTool.mjs";
import {
  isEmptyRedEye, normalizeRedEye, detectRedEye, applyRedEyeToRGBBuffer, mapRedEye,
  REDEYE_DEFAULT_SIZE, REDEYE_DEFAULT_DARKEN,
} from "../electron/redEyeTool.mjs";

/* ============================================================
   SPLICER LAB — a focused photo tool (desktop build)
   Library (Photomator-style) + best-in-class crop, a full
   develop module, clone stamp retouching, and target-file-size
   export.

   Runs inside Electron (persistent library, real folders on
   disk) and falls back to pure-browser mode when the native
   bridge is absent.
   ============================================================ */

const NATIVE = typeof window !== "undefined" && !!window.meridian;
const photoURL = (absPath) => `photo://${encodeURIComponent(absPath)}`;
const thumbURL = (p) => (NATIVE && p.path ? `${photoURL(p.path)}?thumb` : p.url);
const proxyURL = (p) => (NATIVE && p.path ? `${photoURL(p.path)}?proxy` : p.url);
const PROXY_LONG = 3200;
// Max magnitude an ANIMATED refit (double-click zoom-to-fit, aspect switch,
// rotate90) is allowed to jump before it skips the CSS transition and just
// snaps instead — see refit's own comment.
const FLIP_MAX_JUMP = 8;
// Stable empty-array identity for Before/After's "no stamp" state — reused
// every render (not a fresh []) so holding compare doesn't itself trigger
// GLImage's own [stamp, …] redraw effect on every single render tick.
const EMPTY_STAMP = [];

/* Photos at or under this file size skip the proxy + deep-zoom-tile system
   entirely in the editor: the whole original loads straight away and the
   editor works against it directly, so there's no lower-resolution proxy
   ceiling for a deep-zoom tile to ever need to patch in — which also means
   no tile ever swaps in, sidestepping that swap-in's inherent glimpse-of-
   the-old-tile glitch for anything this size or smaller (see
   feedback_stale_pristine_buffer_race.md — that race is fixed, but a
   proxy/native resolution jump on tile arrival is still an inherent part of
   the tile system, not a bug). Sized in MB (not megapixels) on purpose —
   scanner output is what this threshold is tuned against, and different
   scanners land at different file sizes for similar resolutions depending
   on their own JPEG settings; 30 MB comfortably covers this library's
   scanners (~9-11 MB and ~26 MB) without asking a modest machine to keep a
   much larger original decoded and resident during editing. */
const FULL_RES_SIZE_LIMIT = 30 * 1024 * 1024;
const isFullResEligible = (p) => p.sizeBytes > 0 && p.sizeBytes <= FULL_RES_SIZE_LIMIT;
const editSrcURL = (p) => {
  if (!NATIVE || !p.path) return p.url;
  return isFullResEligible(p) ? photoURL(p.path) : `${photoURL(p.path)}?proxy`;
};

/* Clone-stamp stroke geometry (electron/stampTool.mjs) is stored in absolute
   original-image pixel coordinates. bakePreview/bakeCanvas bake the crop/
   rotate/scale transform directly into the output pixel buffer via a single
   canvas-transform + drawImage (no separate "pre-rotate" raster stage the
   way main.cjs's bakeRaw has) — so stamp strokes have to be forward-projected
   through that same transform before applyStampToRGBBuffer can run on the
   final buffer. `s` is output-px per original-crop-px (== outW / e.cw,
   already computed by both callers); origW/origH is the full original
   image's own pixel size, matching the frame qcx/qcy are defined in. Each
   stroke's (dx, dy) is a relative offset vector (source − destination), not
   a point, so it only goes through rotation + scale, never the translation
   part of the transform. */
function mapStampStrokes(stamp, origW, origH, e, s, outW, outH) {
  if (isEmptyStamp(stamp)) return stamp;
  const t = rad(e.theta);
  const project = (x, y) => {
    const relX = x - origW / 2 - e.qcx;
    const relY = y - origH / 2 - e.qcy;
    const [ox, oy] = rot(relX, relY, t);
    return [outW / 2 + s * ox, outH / 2 + s * oy];
  };
  return normalizeStamp(stamp).map((stroke) => {
    const [ndx, ndy] = rot(stroke.dx, stroke.dy, t);
    return {
      dx: ndx * s, dy: ndy * s,
      radius: stroke.radius * s,
      feather: stroke.feather,
      opacity: stroke.opacity,
      mode: stroke.mode,
      points: stroke.points.map((p) => {
        const [x, y] = project(p.x, p.y);
        if (p.dx === undefined) return { x, y };
        // per-dab offset override (see stampTool.mjs header) — same
        // rotation+scale treatment as the stroke's own dx/dy, since it's
        // also a relative offset vector, not a point.
        const [pdx, pdy] = rot(p.dx, p.dy, t);
        return { x, y, dx: pdx * s, dy: pdy * s };
      }),
    };
  });
}

/* Red-eye corrections, through the same crop/rotate/scale forward
   projection as mapStampStrokes above (centers as points; each ellipse's
   shape rotated + scaled by mapRedEye itself). */
function mapRedEyeEdits(redeye, origW, origH, e, s, outW, outH) {
  if (isEmptyRedEye(redeye)) return [];
  const t = rad(e.theta);
  return mapRedEye(redeye, (x, y) => {
    const [ox, oy] = rot(x - origW / 2 - e.qcx, y - origH / 2 - e.qcy, t);
    return [outW / 2 + s * ox, outH / 2 + s * oy];
  }, t, s);
}

/* bake a small edited preview from the PROXY (fast even for 200-MP files) */
async function bakePreview(srcUrl, origW, origH, edits, maxDim = 560) {
  // best-effort: a canvas read failure here (e.g. a still-loading proxy file)
  // should just skip the live preview, never hand back a broken image
  try {
    const img = await loadImage(srcUrl);
    const s0 = img.naturalWidth / origW; // proxy px per original px
    const e = edits || { theta: 0, qcx: 0, qcy: 0, cw: origW, ch: origH };
    let outW = e.cw * s0, outH = e.ch * s0;
    const cap = maxDim / Math.max(outW, outH);
    if (cap < 1) { outW *= cap; outH *= cap; }
    outW = Math.max(1, Math.round(outW));
    outH = Math.max(1, Math.round(outH));
    const s = outW / e.cw;
    const canvas = document.createElement("canvas");
    canvas.width = outW; canvas.height = outH;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, outW, outH);
    ctx.translate(outW / 2, outH / 2);
    ctx.scale(s / s0, s / s0);
    ctx.rotate(rad(e.theta));
    ctx.translate(-e.qcx * s0, -e.qcy * s0);
    ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
    const stampData = e.stamp || e.heal;
    if (!isEmptyStamp(stampData) || !isEmptyRedEye(e.redeye) || !isDefaultAdjust(e.adjust)) {
      const id = ctx.getImageData(0, 0, outW, outH);
      if (!isEmptyStamp(stampData)) {
        applyStampToRGBBuffer(id.data, outW, outH, 4, mapStampStrokes(stampData, origW, origH, e, s, outW, outH));
      }
      if (!isEmptyRedEye(e.redeye)) {
        applyRedEyeToRGBBuffer(id.data, outW, outH, 4, mapRedEyeEdits(e.redeye, origW, origH, e, s, outW, outH));
      }
      if (!isDefaultAdjust(e.adjust)) {
        applyAdjustToRGBBuffer(id.data, outW, outH, 4, e.adjust, { scale: s, origDiag: Math.hypot(origW, origH) });
      }
      ctx.putImageData(id, 0, 0);
    }
    return canvas.toDataURL("image/jpeg", 0.82);
  } catch {
    return null;
  }
}

/* persist bakePreview()'s output to disk (main.cjs's previews/ cache dir,
   keyed like thumbFor/proxyFor) so the NEXT app launch can show an edited
   photo's real preview immediately, instead of every edited-but-not-yet-
   reopened photo falling back to its plain unedited thumbnail until the
   user opens it again this session. Fire-and-forget: a failed write just
   means next launch falls back to the plain thumbnail, same as today. */
function cachePreviewToDisk(path, dataUrl) {
  if (NATIVE) window.meridian.cacheEditPreview(path, dataUrl).catch(() => {});
}
const isUnder = (p, root) => {
  if (!p || !root) return false;
  const np = p.replace(/\\/g, "/");
  const nr = root.replace(/\\/g, "/");
  return np.startsWith(nr) && (np.length === nr.length || np[nr.length] === "/");
};
const baseName = (p) => (p || "").split(/[\\/]/).filter(Boolean).pop() || p;

const CSS = `
:root {
  --bg: #0B0C0F;
  --surface: #14161B;
  --raised: #1C1F26;
  --stroke: #2A2E37;
  --stroke-soft: #22252d;
  --text: #E9EBEF;
  --muted: #8B909C;
  --faint: #5B606C;
  --accent: #F3C34F;
  --accent-ink: #241A03;
  --danger: #F26D6D;
  --hover: #232733;
  --seg-on: #262b36;
  --strip: #101217;
  --dim: rgba(8,9,11,.72);
  /* floating-tools look: canvas, glass rail/HUD, slider tracks */
  --accent-text: #F3C34F;
  --accent-soft: rgba(243,195,79,.14);
  --focus: rgba(243,195,79,.6);
  --canvas: #08090B;
  --glass: rgba(27,30,37,.72);
  --glass-stroke: rgba(255,255,255,.09);
  --glass-hi: rgba(255,255,255,.06);
  --glass-hover: rgba(255,255,255,.07);
  --glass-shadow: 0 12px 34px rgba(0,0,0,.45), 0 1px 2px rgba(0,0,0,.35);
  --well: rgba(0,0,0,.28);
  --track: #2B2F39;
  --thumb: #F4F5F7;
  --thumb-ring: rgba(0,0,0,.35);
  --tip-bg: #2A2E38;
  --tip-text: #F2F3F5;
  --kbd-bg: rgba(255,255,255,.07);
  --kbd-stroke: rgba(255,255,255,.12);
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
.app {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}
.app.light {
  --bg: #F2F2F5;
  --surface: #FFFFFF;
  --raised: #EBECEF;
  --stroke: #D6D9E0;
  --stroke-soft: #E5E7EC;
  --text: #1C1E24;
  --muted: #5C6270;
  --faint: #979DA9;
  --accent: #E0A21F;
  --accent-ink: #241A03;
  --danger: #D64545;
  --hover: #E2E4E9;
  --seg-on: #FFFFFF;
  --strip: #E9EAEE;
  --dim: rgba(238,239,242,.8);
  --accent-text: #9A6500;
  --accent-soft: rgba(224,162,31,.16);
  --focus: rgba(224,162,31,.7);
  --canvas: #E2E3E8;
  --glass: rgba(255,255,255,.78);
  --glass-stroke: rgba(18,20,26,.09);
  --glass-hi: rgba(255,255,255,.9);
  --glass-hover: rgba(18,20,26,.06);
  --glass-shadow: 0 12px 34px rgba(30,34,45,.16), 0 1px 2px rgba(30,34,45,.10);
  --well: rgba(18,20,26,.06);
  --track: #DDE0E6;
  --thumb: #FFFFFF;
  --thumb-ring: rgba(18,20,26,.22);
  --tip-bg: #1F2128;
  --tip-text: #F4F5F7;
  --kbd-bg: rgba(18,20,26,.06);
  --kbd-stroke: rgba(18,20,26,.14);
}
button { font-family: inherit; cursor: pointer; }
/* a button clicked with a mouse shouldn't keep a visible focus ring
   afterward — :focus-visible already limits the BROWSER's own ring to
   keyboard navigation in modern engines, but belt-and-suspenders here
   since this app also blurs buttons after a mouse click in JS (see App's
   mount effect) specifically so a stray Space/Enter afterward can't
   silently re-trigger whatever was last clicked. */
button:focus:not(:focus-visible) { outline: none; }
input, select { font-family: inherit; }

/* ---------- top bar ---------- */
.topbar {
  position: relative;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 16px;
  border-bottom: 1px solid var(--stroke-soft);
  background: var(--surface);
  flex-shrink: 0;
}
.topbar-progress {
  position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
  max-width: 46%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-size: 13px; color: var(--text); background: var(--raised);
  border: 1px solid var(--stroke); border-radius: 8px; padding: 6px 14px;
  pointer-events: none; animation: progressin .15s ease;
}
@keyframes progressin { from { opacity: 0; } to { opacity: 1; } }
.brand {
  display: flex; align-items: baseline; gap: 8px;
  margin-right: 8px; user-select: none;
}
.brand .name { font-weight: 700; letter-spacing: 0.14em; font-size: 13px; }
.brand .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); align-self: center; }
.spacer { flex: 1; }

.btn {
  display: inline-flex; align-items: center; gap: 7px;
  background: var(--raised);
  color: var(--text);
  border: 1px solid var(--stroke);
  border-radius: 8px;
  padding: 7px 12px;
  font-size: 13px;
  transition: background .15s, border-color .15s, opacity .15s;
  white-space: nowrap;
}
.btn:hover { background: var(--hover); }
.btn:disabled { opacity: .4; cursor: default; }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: transparent; font-weight: 600; }
.btn.primary:hover { background: #f8cf6e; }
.btn.ghost { background: transparent; border-color: transparent; color: var(--muted); }
.btn.ghost:hover { color: var(--text); background: var(--raised); }
.btn.active { border-color: var(--accent); color: var(--accent); }
.btn.icon { width: 34px; height: 34px; padding: 0; justify-content: center; flex-shrink: 0; }
.btn svg { flex-shrink: 0; }

.tip-bubble {
  position: fixed;
  z-index: 1000; pointer-events: none;
  background: var(--raised); color: var(--text); border: 1px solid var(--stroke);
  border-radius: 6px; padding: 5px 9px; font-size: 12px; font-family: var(--sans);
  white-space: nowrap; max-width: 280px; text-overflow: ellipsis; overflow: hidden;
  box-shadow: 0 4px 14px rgba(0,0,0,.35);
  animation: tipin .1s ease;
}
@keyframes tipin {
  from { opacity: 0; transform: translateY(-2px); }
  to { opacity: 1; transform: translateY(0); }
}

.thumbslider { display: flex; align-items: center; gap: 8px; color: var(--faint); }

/* ---------- layout ---------- */
.body { flex: 1; display: flex; min-height: 0; }
.sidebar {
  width: 208px; flex-shrink: 0;
  border-right: 1px solid var(--stroke-soft);
  background: var(--surface);
  padding: 14px 10px;
  overflow-y: auto;
}
.sidebar-resize {
  flex-shrink: 0; width: 6px; margin-left: -3px;
  cursor: ew-resize; position: relative; z-index: 1;
  touch-action: none;
}
.sidebar-resize::after {
  content: ""; position: absolute; top: 50%; left: 50%;
  height: 32px; width: 3px; border-radius: 2px;
  background: var(--stroke-soft); transform: translate(-50%, -50%);
}
.sidebar-resize:hover::after, .sidebar-resize:active::after { background: var(--accent); }
.side-label {
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--faint); padding: 14px 10px 6px;
}
.side-item {
  display: flex; align-items: center; gap: 9px;
  width: 100%; text-align: left;
  padding: 7px 10px; border-radius: 7px;
  background: transparent; border: none; color: var(--muted);
  font-size: 13px;
}
.side-item:hover { background: var(--raised); color: var(--text); }
.side-item.active { background: var(--raised); color: var(--text); }
.side-item .count { margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--faint); }
.side-rating { display: flex; align-items: center; gap: 1px; padding: 2px 10px 4px; }
.side-rating button {
  background: none; border: none; padding: 2px; font-size: 15px; line-height: 1;
  color: var(--faint); transition: color .1s;
}
.side-rating button:hover { color: var(--muted); }
.side-rating button.lit { color: var(--accent); }
.side-rating .clear { margin-left: 4px; font-size: 11px; display: inline-flex; }
.side-item svg { flex-shrink: 0; opacity: .8; }

.main { flex: 1; min-width: 0; display: flex; flex-direction: column; position: relative; }
.gridwrap { flex: 1; overflow-y: auto; padding: 18px; }
.grid { display: grid; gap: 6px; grid-template-columns: repeat(auto-fill, minmax(var(--cell), 1fr)); }

.cell {
  position: relative; border-radius: 6px; overflow: hidden;
  background: linear-gradient(110deg, var(--raised) 40%, var(--hover) 50%, var(--raised) 60%);
  background-size: 200% 100%;
  animation: shimmer 1.5s linear infinite;
  cursor: pointer;
  outline: 2px solid transparent; outline-offset: -2px;
  transition: outline-color .12s;
}
.cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cell.selected { outline-color: var(--accent); }
.cell .check {
  position: absolute; top: 8px; left: 8px; width: 21px; height: 21px;
  border-radius: 50%; border: 1.5px solid rgba(255,255,255,.85);
  background: rgba(0,0,0,.35); display: flex; align-items: center; justify-content: center;
  opacity: 0; transition: opacity .12s;
}
.cell:hover .check, .cell.selectmode .check { opacity: 1; }
.cell.selected .check { background: var(--accent); border-color: var(--accent); }
.badge {
  position: absolute; bottom: 7px; right: 7px;
  background: rgba(0,0,0,.55); backdrop-filter: blur(4px);
  border-radius: 5px; padding: 3px 6px;
  font-size: 9px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); font-weight: 600;
}

.empty {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 14px; color: var(--muted); text-align: center; padding: 40px;
}
.empty h2 { font-weight: 600; font-size: 17px; color: var(--text); }
.empty p { max-width: 340px; line-height: 1.55; font-size: 13px; }
.empty .row { display: flex; gap: 10px; margin-top: 6px; }

.dropveil {
  position: absolute; inset: 10px; border: 2px dashed var(--accent); border-radius: 14px;
  background: rgba(243,195,79,.06); z-index: 40; pointer-events: none;
  display: flex; align-items: center; justify-content: center;
  color: var(--accent); font-weight: 600; letter-spacing: .04em;
}

/* ---------- editor ---------- */
.editor { position: fixed; inset: 0; background: var(--bg); display: flex; flex-direction: column; z-index: 100; }
.editor .topbar { border-bottom-color: var(--stroke-soft); }
.ed-body { flex: 1; display: flex; min-height: 0; }
/* canvas column: the stage, with the floating-look HUD docked under it
   (in the flow, at a fixed height — the stage only resizes when a mode
   starts/ends, exactly like the old toolbelt, so the mode-driven refit
   suppression and crop's refit-on-enter keep working unchanged) */
.workspace { flex: 1; display: flex; flex-direction: column; min-width: 0; position: relative; background: var(--canvas); }
.stage { flex: 1; position: relative; overflow: hidden; min-height: 0; cursor: grab; }
.stage:active { cursor: grabbing; }

/* ---------- metadata panel ---------- */
.meta {
  width: 296px; flex-shrink: 0;
  border-left: 1px solid var(--stroke-soft); background: var(--surface);
  padding: 16px 16px 12px; overflow-y: auto; font-size: 12px;
}
.meta h4 {
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--faint); font-weight: 600; margin: 16px 0 8px;
}
.meta h4:first-child { margin-top: 0; }
.meta .mrow { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; }
.meta .mrow .k { color: var(--faint); flex-shrink: 0; }
.meta .mrow .v { color: var(--text); font-family: var(--mono); font-size: 11.5px; text-align: right; overflow-wrap: anywhere; }
.meta .divider { height: 1px; background: var(--stroke-soft); margin: 14px 0; }

/* ---------- develop panel: histogram + adjust sliders ---------- */
/* fixed dark background regardless of theme: the chart's additive RGB
   blend needs a dark backdrop for contrast (washes out near-white), same
   convention real photo apps use for the histogram specifically */
/* Background is a fixed dark charcoal regardless of app theme — deliberate,
   same convention every other photo editor's histogram uses, since the
   additive red/green/blue curves read cleanly against dark and would wash
   out against a light panel. On the light theme specifically, though, a
   plain background color with no border let the box's own edges (bottom
   edge included) blend into the light theme's own DevelopPanel background
   with no visible boundary — the border makes the box's extent legible in
   either theme instead of relying on a background-color contrast that
   only exists in one of them. */
.histogram { width: 100%; height: 88px; display: block; border-radius: 10px; background: #0F1115; border: 1px solid var(--stroke-soft); box-sizing: border-box; }
.pv-note {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  margin: 10px 0 2px; font-size: 12px; color: var(--muted);
}
.pv-note .chip { padding: 3px 11px; }
.wbpick-overlay { position: absolute; inset: 0; cursor: crosshair; touch-action: none; }

/* small icon button (section heads, HUD chips) */
.ib {
  width: 26px; height: 26px; border-radius: 7px; display: inline-grid; place-items: center; flex-shrink: 0;
  background: none; border: 0; color: var(--faint); transition: background .12s, color .12s, transform .08s;
}
.ib:hover { background: var(--hover); color: var(--text); }
.ib:active { transform: scale(.94); }
.ib:disabled { opacity: .32; pointer-events: none; }
.ib.on, .ib[aria-pressed="true"] { color: var(--accent-text); background: var(--accent-soft); }

/* side panel tabs */
.side-tabs { display: flex; padding: 2px; border-radius: 9px; background: var(--raised); margin-bottom: 14px; }
.side-tabs button {
  flex: 1; height: 28px; border-radius: 7px; border: 0; background: none; font-size: 12px; font-weight: 600;
  color: var(--muted); transition: background .12s, color .12s;
}
.side-tabs button:hover { color: var(--text); }
.side-tabs button[aria-selected="true"] { background: var(--seg-on); color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,.22), 0 0 0 .5px var(--stroke-soft); }

/* collapsible develop sections */
.sec { border-top: 1px solid var(--stroke-soft); margin-top: 12px; }
.sec-head { display: flex; align-items: center; gap: 2px; height: 44px; }
.sec-toggle {
  flex: 1; display: flex; align-items: center; gap: 8px; height: 30px; margin-left: -8px; padding: 0 8px;
  border-radius: 8px; border: 0; background: none; font-size: 12.5px; font-weight: 650; color: var(--text); text-align: left;
  transition: background .12s;
}
.sec-toggle:hover { background: var(--hover); }
.sec-toggle .chev { color: var(--faint); transition: transform .2s; }
.sec.closed .sec-toggle .chev { transform: rotate(-90deg); }
.sec-count {
  min-width: 18px; height: 18px; padding: 0 6px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center;
  background: var(--accent-soft); color: var(--accent-text); font-size: 10.5px; font-weight: 700; font-variant-numeric: tabular-nums;
}
.sec-body { padding-bottom: 6px; }

/* develop slider row */
.sl { padding: 3px 0 5px; }
.sl-head { display: flex; align-items: center; justify-content: space-between; height: 20px; }
.sl-label { font-size: 12.5px; color: var(--muted); transition: color .12s; }
.sl:hover .sl-label { color: var(--text); }
.sl-val {
  min-width: 40px; text-align: right; font-size: 12px; font-weight: 600;
  font-variant-numeric: tabular-nums; color: var(--faint);
}
.sl.mod .sl-val { color: var(--accent-text); }

/* RangeTrack: drawn track/fill/thumb under a transparent native range
   input whose own thumb is the same 14px wide, so the thumb drawn at
   7px + (100% − 14px)·p sits exactly where the real one is */
.track { position: relative; height: 22px; }
.track .line {
  position: absolute; left: 0; right: 0; top: 50%; height: 4px; margin-top: -2px; border-radius: 99px;
  background: var(--track); transition: height .14s, margin .14s; pointer-events: none;
}
.track:hover .line, .track:active .line, .track:hover .fill, .track:active .fill { height: 6px; margin-top: -3px; }
.track.grad .line { background: var(--g); opacity: .85; }
.track.grad:hover .line, .track.grad:active .line { opacity: 1; }
.track .fill {
  position: absolute; top: 50%; height: 4px; margin-top: -2px; border-radius: 99px; background: var(--accent);
  transition: height .14s, margin .14s; pointer-events: none;
  left: calc(7px + (100% - 14px) * var(--a)); width: calc((100% - 14px) * (var(--b) - var(--a)));
}
.track .mid {
  position: absolute; top: 50%; width: 2px; height: 10px; margin: -5px 0 0 -1px; border-radius: 1px;
  background: var(--faint); opacity: .55; pointer-events: none;
}
.track.grad .mid { background: var(--surface); opacity: .9; }
.track .thumb {
  position: absolute; top: 50%; left: calc(7px + (100% - 14px) * var(--p)); width: 14px; height: 14px; margin: -7px 0 0 -7px;
  border-radius: 50%; background: var(--thumb); box-shadow: 0 0 0 1px var(--thumb-ring), 0 1px 4px rgba(0,0,0,.3);
  transition: transform .14s, box-shadow .14s; pointer-events: none;
}
.track:hover .thumb { transform: scale(1.12); }
.track:active .thumb { transform: scale(1.22); box-shadow: 0 0 0 1px var(--thumb-ring), 0 0 0 6px var(--accent-soft), 0 2px 6px rgba(0,0,0,.35); }
.track:has(input:focus-visible) .thumb { box-shadow: 0 0 0 1px var(--thumb-ring), 0 0 0 4px var(--focus); }
.track .bubble {
  position: absolute; bottom: 21px; left: calc(7px + (100% - 14px) * var(--p)); transform: translate(-50%, 4px) scale(.9);
  min-width: 30px; height: 22px; padding: 0 7px; border-radius: 7px; display: grid; place-items: center;
  background: var(--tip-bg); color: var(--tip-text); font-size: 11.5px; font-weight: 650; font-variant-numeric: tabular-nums;
  box-shadow: 0 6px 16px rgba(0,0,0,.3); opacity: 0; pointer-events: none; transition: opacity .12s, transform .12s; z-index: 2;
}
.track .bubble::after {
  content: ""; position: absolute; left: 50%; bottom: -4px; width: 8px; height: 8px; margin-left: -4px;
  background: var(--tip-bg); transform: rotate(45deg); border-radius: 0 0 2px 0;
}
.track:active .bubble { opacity: 1; transform: translate(-50%, 0) scale(1); }
.track input[type="range"] {
  -webkit-appearance: none; appearance: none; position: absolute; inset: 0; width: 100%; height: 100%;
  margin: 0; padding: 0; background: transparent; opacity: 0; cursor: pointer; outline: none;
}
.track input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 22px; cursor: pointer; }
.track input[type="range"]::-moz-range-thumb { width: 14px; height: 22px; border: 0; background: transparent; cursor: pointer; }

.stars { display: inline-flex; gap: 2px; }
.stars button {
  background: none; border: none; padding: 0 1px; font-size: 17px; line-height: 1;
  color: var(--faint); transition: color .1s, transform .1s;
}
.stars button:hover { transform: scale(1.15); }
.stars button.lit { color: var(--accent); }
.stars.readonly { pointer-events: none; }
.stars.readonly button { font-size: 9px; padding: 0; }

/* ---------- filmstrip ---------- */
.filmstrip-resize {
  flex-shrink: 0; height: 6px; margin-top: -3px;
  cursor: ns-resize; position: relative; z-index: 1;
  touch-action: none;
}
.filmstrip-resize::after {
  content: ""; position: absolute; left: 50%; top: 50%;
  width: 32px; height: 3px; border-radius: 2px;
  background: var(--stroke-soft); transform: translate(-50%, -50%);
}
.filmstrip-resize:hover::after, .filmstrip-resize:active::after { background: var(--accent); }
.filmstrip {
  flex-shrink: 0; height: 92px;
  border-top: 1px solid var(--stroke-soft); background: var(--strip);
  display: flex; align-items: center; gap: 6px;
  overflow-x: auto; overflow-y: hidden; padding: 8px 12px;
  scrollbar-width: thin;
}
.filmstrip::-webkit-scrollbar { height: 7px; }
.strip-item {
  position: relative; height: 100%; aspect-ratio: 1; flex-shrink: 0;
  border-radius: 5px; overflow: hidden; cursor: pointer;
  outline: 2px solid transparent; outline-offset: -2px;
  transition: outline-color .12s; background: var(--raised);
}
.strip-item img { width: 100%; height: 100%; object-fit: cover; display: block; }
.strip-item.current { outline-color: var(--accent); }
.strip-item.sel { outline-color: #4B9BFF; }
.strip-item.sel img { opacity: .75; }
.strip-item .scheck {
  position: absolute; top: 4px; left: 4px; width: 17px; height: 17px;
  border-radius: 50%; background: #4B9BFF;
  display: flex; align-items: center; justify-content: center;
}
.strip-item .scheck svg path { stroke: #fff; }
.strip-item .sinfo {
  position: absolute; left: 0; right: 0; bottom: 0;
  background: linear-gradient(transparent, rgba(0,0,0,.75));
  padding: 10px 5px 3px; display: flex; justify-content: space-between; align-items: flex-end;
  /* scales with the resizable filmstrip's own height (--strip-h, set on
     .filmstrip) instead of a flat 9px — otherwise a bigger panel just
     makes the thumbnails bigger while this text stays pinned at its
     smallest size, reading as proportionally MORE cramped, not less. */
  font-size: clamp(9px, calc(var(--strip-h, 92px) * 0.1), 16px);
  font-family: var(--mono); color: rgba(255,255,255,.85);
}
.strip-item .sstars { color: var(--accent); letter-spacing: -1px; font-size: clamp(9px, calc(var(--strip-h, 92px) * 0.1), 16px); }
.strip-item .sedit {
  position: absolute; top: 4px; right: 4px; width: 6px; height: 6px;
  border-radius: 50%; background: var(--accent);
  /* a dark ring so the dot still reads as its own badge on the currently-
     open photo, whose whole thumbnail already has an accent-colored
     outline right next to this same corner */
  box-shadow: 0 0 0 1.5px rgba(0,0,0,.6);
}

/* sidebar sync toggle */
.side-item .synctoggle {
  display: none; margin-left: 4px; flex-shrink: 0;
  background: none; border: none; color: var(--faint); padding: 2px;
  border-radius: 4px; line-height: 0;
}
.side-item:hover .synctoggle { display: inline-flex; }
.side-item .synctoggle:hover { color: var(--accent); background: rgba(243,195,79,.12); }
.side-item .synctoggle.danger:hover { color: var(--danger); background: rgba(242,109,109,.12); }

/* rating badge on library cells */
.rbadge {
  position: absolute; bottom: 7px; left: 7px;
  background: rgba(0,0,0,.55); backdrop-filter: blur(4px);
  border-radius: 5px; padding: 2px 5px;
  font-size: 9px; color: var(--accent); letter-spacing: -0.5px;
}
.imgspace {
  position: absolute; left: 0; top: 0; transform-origin: center;
  user-select: none; pointer-events: none;
}
.imgspace .base {
  position: absolute; inset: 0; width: 100%; height: 100%;
  user-select: none; pointer-events: none; max-width: none;
}
.imgspace .tileimg {
  position: absolute; user-select: none; pointer-events: none; max-width: none;
  transition: opacity .18s ease;
}
/* promoted to its own GPU layer only WHILE zooming/panning: buttery motion
   during the gesture, full-quality re-raster the moment it ends */
.imgspace.gpu { will-change: transform; }
.cropframe {
  position: absolute; border: 1px solid rgba(255,255,255,.9);
  box-shadow: 0 0 0 9999px var(--dim);
  cursor: grab; touch-action: none;
}
.cropframe.grabbing { cursor: grabbing; }
.viewclip {
  position: absolute; overflow: hidden;
  box-shadow: 0 12px 48px rgba(0,0,0,.35);
}

.world { position: absolute; inset: 0; transform-origin: 0 0; }

.stamp-overlay { position: absolute; inset: 0; cursor: none; touch-action: none; }
.redeye-overlay { position: absolute; inset: 0; cursor: crosshair; touch-action: none; }
.stamp-svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; }
.stamp-cursor {
  position: absolute; border-radius: 50%; box-sizing: border-box;
  border: 1.5px solid rgba(255,255,255,.9); box-shadow: 0 0 0 1px rgba(0,0,0,.5);
  pointer-events: none; transform: translate(-50%, -50%);
}
.stamp-hit { position: absolute; border-radius: 50%; cursor: move; }
.stamp-handle {
  position: absolute; width: 10px; height: 10px; border-radius: 50%;
  background: #f3c34f; border: 1px solid rgba(0,0,0,.5); cursor: nesw-resize;
}


.thirds { position: absolute; inset: 0; pointer-events: none; }
.thirds .v, .thirds .h { position: absolute; background: rgba(255,255,255,.35); }
.thirds .v { top: 0; bottom: 0; width: 1px; }
.thirds .h { left: 0; right: 0; height: 1px; }

.handle { position: absolute; touch-action: none; }
.handle.corner { width: 22px; height: 22px; }
.handle.corner::before, .handle.corner::after {
  content: ""; position: absolute; background: #fff; border-radius: 2px;
}
.handle.corner::before { width: 16px; height: 3px; }
.handle.corner::after { width: 3px; height: 16px; }
.h-nw { left: -5px; top: -5px; cursor: nwse-resize; } .h-nw::before { left: 5px; top: 5px; } .h-nw::after { left: 5px; top: 5px; }
.h-ne { right: -5px; top: -5px; cursor: nesw-resize; } .h-ne::before { right: 5px; top: 5px; } .h-ne::after { right: 5px; top: 5px; }
.h-sw { left: -5px; bottom: -5px; cursor: nesw-resize; } .h-sw::before { left: 5px; bottom: 5px; } .h-sw::after { left: 5px; bottom: 5px; }
.h-se { right: -5px; bottom: -5px; cursor: nwse-resize; } .h-se::before { right: 5px; bottom: 5px; } .h-se::after { right: 5px; bottom: 5px; }
.handle.edge { background: transparent; }
.h-n { left: 24px; right: 24px; top: -6px; height: 12px; cursor: ns-resize; }
.h-s { left: 24px; right: 24px; bottom: -6px; height: 12px; cursor: ns-resize; }
.h-e { top: 24px; bottom: 24px; right: -6px; width: 12px; cursor: ew-resize; }
.h-w { top: 24px; bottom: 24px; left: -6px; width: 12px; cursor: ew-resize; }
.h-n::after, .h-s::after { content: ""; position: absolute; left: 50%; transform: translateX(-50%); width: 26px; height: 3px; background: #fff; border-radius: 2px; }
.h-n::after { top: 4px; } .h-s::after { bottom: 4px; }
.h-e::after, .h-w::after { content: ""; position: absolute; top: 50%; transform: translateY(-50%); width: 3px; height: 26px; background: #fff; border-radius: 2px; }
.h-e::after { right: 4px; } .h-w::after { left: 4px; }

.chip {
  background: transparent; border: 1px solid var(--stroke); color: var(--muted);
  border-radius: 999px; padding: 5px 13px; font-size: 12px;
  transition: all .12s;
}
.chip:hover { color: var(--text); border-color: var(--faint); }
.chip.on { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }

/* ---------- floating tools: glass rail + HUD ---------- */
.glass {
  background: var(--glass);
  backdrop-filter: blur(22px) saturate(1.6); -webkit-backdrop-filter: blur(22px) saturate(1.6);
  border: 1px solid var(--glass-stroke);
  box-shadow: var(--glass-shadow), inset 0 1px 0 var(--glass-hi);
}
.rail-dock { width: 64px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; background: var(--canvas); }
.rail { display: flex; flex-direction: column; gap: 2px; padding: 5px; border-radius: 14px; }
.rail-sep { height: 1px; margin: 4px 7px; background: var(--glass-stroke); }
.rail-group { display: flex; flex-direction: column; gap: 2px; border-radius: 10px; transition: background .18s, box-shadow .18s; }
.rail-group.engaged { background: var(--glass-hover); box-shadow: inset 0 0 0 1px var(--glass-stroke); }
.tool {
  position: relative; width: 36px; height: 36px; border-radius: 9px; display: grid; place-items: center;
  background: none; border: 0; color: var(--muted); transition: background .12s, color .12s, transform .08s, opacity .12s;
}
.tool:hover { background: var(--glass-hover); color: var(--text); }
.tool:active { transform: scale(.93); }
.tool:disabled { opacity: .3; pointer-events: none; }
.tool[aria-pressed="true"] {
  background: var(--accent); color: var(--accent-ink);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.35), 0 2px 8px rgba(243,195,79,.25);
}
.tool svg { width: 18px; height: 18px; }

.hud-dock { height: 76px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; padding: 0 12px; }
.hud {
  display: flex; align-items: center; gap: 2px; padding: 6px; border-radius: 16px;
  max-width: 100%; white-space: nowrap; animation: hudin .22s cubic-bezier(.2,.9,.25,1.1);
}
@keyframes hudin { from { opacity: 0; transform: translateY(10px) scale(.98); } to { opacity: 1; transform: none; } }
.hud-float { position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%); z-index: 6; animation: none; }
.hsep { width: 1px; height: 26px; background: var(--glass-stroke); margin: 0 5px; flex-shrink: 0; }
.hud-chip {
  display: flex; align-items: center; gap: 9px; height: 40px; padding: 0 12px 0 8px; border-radius: 10px;
  border: 0; background: none; color: var(--text); white-space: nowrap; flex-shrink: 0;
}
button.hud-chip { transition: background .12s; }
button.hud-chip:hover { background: var(--glass-hover); }
.hud-chip .chip-ico {
  width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center;
  background: var(--accent-soft); color: var(--accent-text);
}
.hud-chip .chip-ico svg { width: 16px; height: 16px; }
.hud-chip .chip-text { display: flex; flex-direction: column; gap: 4px; align-items: flex-start; }
.hud-chip .chip-name { font-size: 13px; font-weight: 650; line-height: 1; }
.dots { display: flex; gap: 3px; }
.dots i { width: 4px; height: 4px; border-radius: 2px; background: var(--faint); opacity: .7; transition: width .2s, background .2s, opacity .2s; }
.dots i.on { width: 12px; background: var(--accent); opacity: 1; }
.hud-hint { font-size: 12.5px; color: var(--muted); padding: 0 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.kbd {
  display: inline-grid; place-items: center; min-width: 18px; height: 18px; padding: 0 5px;
  border-radius: 5px; font: 600 10.5px/1 var(--mono); color: var(--muted);
  background: var(--kbd-bg); border: 1px solid var(--kbd-stroke);
}
.brush-prev {
  width: 40px; height: 40px; border-radius: 10px; background: var(--well); display: grid; place-items: center;
  flex-shrink: 0; margin-right: 4px; overflow: hidden;
}
.brush-prev i { display: block; border-radius: 50%; }
.hfield { width: 138px; flex-shrink: 1; min-width: 96px; padding: 3px 10px 0; display: flex; flex-direction: column; }
.hfield.wide { width: 210px; }
.hf-head { display: flex; align-items: baseline; justify-content: space-between; height: 15px; }
.hf-head .lab {
  font-size: 10.5px; font-weight: 600; color: var(--muted); letter-spacing: .02em;
  display: flex; align-items: center; gap: 5px;
}
.hf-head .sdot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
.hf-head .val { font-size: 12.5px; font-weight: 650; color: var(--text); font-variant-numeric: tabular-nums; }
.hf-head .val em { font-style: normal; font-weight: 500; font-size: 10.5px; color: var(--muted); margin-left: 1px; }
.hf-head .val.zero { color: var(--faint); }
.hfield .track { height: 20px; }
.selchip {
  display: flex; align-items: center; gap: 2px; height: 32px; padding: 0 3px 0 10px; border-radius: 9px;
  background: var(--accent-soft); color: var(--accent-text); font-size: 12px; font-weight: 600; white-space: nowrap;
  margin-left: 4px; flex-shrink: 0;
}
.selchip .sdot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); margin-right: 5px; }
.selchip .ib { color: var(--accent-text); }
.selchip .ib:hover { background: var(--accent-soft); }
.hud-seg { display: flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--well); flex-shrink: 0; }
.hud-seg button {
  height: 32px; padding: 0 10px; border-radius: 7px; border: 0; background: none; font-size: 12px; font-weight: 600; color: var(--muted);
  transition: background .12s, color .12s; white-space: nowrap;
}
.hud-seg button:hover { color: var(--text); background: var(--glass-hover); }
.hud-seg button[aria-pressed="true"] { background: var(--seg-on); color: var(--text); box-shadow: 0 1px 3px rgba(0,0,0,.25); }
.hud .ib { width: 36px; height: 36px; border-radius: 10px; color: var(--muted); }
.hud .ib:hover { background: var(--glass-hover); color: var(--text); }
.hud .ib[aria-pressed="true"] { color: var(--accent-text); background: var(--accent-soft); }
.hud-btn {
  height: 40px; padding: 0 12px; border-radius: 10px; border: 0; background: none; font-size: 13px; font-weight: 550; color: var(--muted);
  white-space: nowrap; transition: background .12s, color .12s; display: flex; align-items: center; gap: 8px;
}
.hud-btn:hover { background: var(--glass-hover); color: var(--text); }
.hud-done {
  height: 40px; padding: 0 8px 0 14px; border-radius: 10px; border: 0; background: var(--accent); color: var(--accent-ink);
  font-size: 13px; font-weight: 650; display: flex; align-items: center; gap: 9px; white-space: nowrap; flex-shrink: 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.35); transition: filter .12s, transform .08s;
}
.hud-done:hover { filter: brightness(1.07); }
.hud-done:active { transform: translateY(.5px); }
.hud-done .kbd { color: rgba(36,26,3,.7); background: rgba(36,26,3,.1); border-color: rgba(36,26,3,.16); }

/* topbar: primary Export */
.btn-export {
  height: 32px; padding: 0 14px 0 12px; border-radius: 8px; border: 0; background: var(--accent); color: var(--accent-ink);
  font-weight: 650; font-size: 13px; display: inline-flex; align-items: center; gap: 7px; margin-left: 4px; flex-shrink: 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.35), 0 1px 2px rgba(0,0,0,.25); transition: filter .12s, transform .08s;
}
.btn-export:hover { filter: brightness(1.07); }
.btn-export:active { transform: translateY(.5px); }
.crumbs { display: flex; align-items: center; gap: 7px; font-size: 13px; white-space: nowrap; min-width: 0; overflow: hidden; }
.crumbs .roll { color: var(--muted); overflow: hidden; text-overflow: ellipsis; }
.crumbs .sep { color: var(--faint); }
.crumbs .file { color: var(--text); font-weight: 600; overflow: hidden; text-overflow: ellipsis; }

/* ---------- dialog ---------- */
.veil {
  position: fixed; inset: 0; background: rgba(5,6,8,.66); backdrop-filter: blur(3px);
  display: flex; align-items: center; justify-content: center; z-index: 200;
}
.dialog {
  width: 380px; max-width: calc(100vw - 32px);
  background: var(--surface); border: 1px solid var(--stroke);
  border-radius: 14px; padding: 22px; box-shadow: 0 24px 60px rgba(0,0,0,.5);
}
.dialog h3 { font-size: 16px; font-weight: 600; margin-bottom: 3px; letter-spacing: -.01em; }
.dialog .sub { color: var(--faint); font-size: 12px; margin-bottom: 18px; font-family: var(--mono); }
.field { margin-bottom: 13px; }
.field label { display: block; font-size: 10.5px; letter-spacing: .07em; text-transform: uppercase; color: var(--faint); margin-bottom: 7px; }
.field .controls { display: flex; gap: 8px; }
.seg { display: flex; background: var(--raised); border: 1px solid var(--stroke); border-radius: 8px; overflow: hidden; flex: 1; }
.seg button {
  flex: 1; background: transparent; border: none; color: var(--muted);
  padding: 7px 0; font-size: 13px;
}
.seg button.on { background: var(--seg-on); color: var(--text); font-weight: 600; }
.numinput {
  display: flex; align-items: center; gap: 8px;
  background: var(--raised); border: 1px solid var(--stroke); border-radius: 8px;
  padding: 7px 12px; flex: 1;
}
.numinput input {
  background: transparent; border: none; outline: none; color: var(--text);
  width: 100%; font-size: 13.5px; font-family: var(--mono);
}
.numinput .unit { color: var(--faint); font-size: 11.5px; flex-shrink: 0; }
.estimate {
  background: var(--raised); border: 1px solid var(--stroke-soft); border-radius: 10px;
  padding: 11px 14px; margin: 14px 0; font-size: 12.5px; color: var(--muted);
  min-height: 40px; display: flex; align-items: center; gap: 10px; line-height: 1.4;
}
.estimate strong { color: var(--text); font-family: var(--mono); font-weight: 600; }
.estimate.warn { color: var(--danger); border-color: rgba(242,109,109,.35); }
.dialog .actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 8px; }
.chk {
  display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
  font-size: 12px; color: var(--muted); cursor: pointer; padding: 0 4px;
}
.chk-row { display: flex; margin-top: 10px; }
.chk input { accent-color: var(--accent); width: 14px; height: 14px; cursor: pointer; margin: 0 8px 0 0; }
.preset-list { display: flex; flex-direction: column; gap: 6px; }
.preset-row {
  display: flex; flex-direction: column; align-items: flex-start; gap: 2px;
  background: var(--raised); border: 1px solid var(--stroke); border-radius: 8px;
  padding: 9px 12px; text-align: left; transition: border-color .12s, background .12s;
}
.preset-row:hover { border-color: var(--accent); background: var(--hover); }
.preset-row:disabled { opacity: .5; }
.preset-row .cm-name { font-size: 13px; color: var(--text); font-weight: 500; }
.preset-row .cm-sub { font-size: 11px; color: var(--faint); font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; }

.toast {
  position: fixed; bottom: 26px; left: 50%; transform: translateX(-50%);
  background: var(--raised); border: 1px solid var(--stroke);
  border-radius: 10px; padding: 10px 18px; font-size: 13px; z-index: 300;
  box-shadow: 0 10px 30px rgba(0,0,0,.45); color: var(--text);
  animation: toastin .22s ease;
}
.ctxmenu {
  position: fixed; z-index: 400; min-width: 240px; max-width: 320px;
  background: var(--surface); border: 1px solid var(--stroke);
  border-radius: 10px; padding: 6px; box-shadow: 0 16px 48px rgba(0,0,0,.45);
  animation: menuin .13s ease;
}
@keyframes menuin { from { opacity: 0; } to { opacity: 1; } }
.ctxmenu .cm-title {
  font-size: 10px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--faint); padding: 6px 10px 4px;
}
.ctxmenu .cm-item {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: 7px; cursor: pointer;
}
.ctxmenu .cm-item:hover { background: var(--raised); }
.ctxmenu .cm-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px; }
.ctxmenu .cm-name { font-size: 13px; color: var(--text); }
.ctxmenu .cm-sub { font-size: 10.5px; color: var(--faint); font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ctxmenu .cm-del {
  background: none; border: none; color: var(--faint); padding: 4px; line-height: 0; border-radius: 5px;
}
.ctxmenu .cm-del:hover { color: var(--danger); background: rgba(242,109,109,.12); }
.ctxmenu .cm-sep { height: 1px; background: var(--stroke-soft); margin: 5px 6px; }
.ctxmenu .cm-empty { padding: 8px 10px; font-size: 12px; color: var(--faint); }
@keyframes shimmer { to { background-position: -200% 0; } }
@keyframes toastin { from { opacity: 0; transform: translate(-50%, 8px); } to { opacity: 1; transform: translate(-50%, 0); } }
.toast { display: flex; align-items: center; gap: 12px; }
.toast .toast-act {
  background: none; border: none; color: var(--accent); font-weight: 600;
  font-size: 13px; padding: 0; text-transform: uppercase; letter-spacing: .05em;
}
.toast .toast-act:hover { text-decoration: underline; }

.spin { width: 14px; height: 14px; border: 2px solid var(--stroke); border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; flex-shrink: 0; }
@keyframes spin { to { transform: rotate(360deg); } }

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: var(--stroke); border-radius: 5px; border: 2px solid var(--bg); }
::-webkit-scrollbar-track { background: transparent; }
@media (prefers-reduced-motion: reduce) {
  .toast { transition: none !important; animation: none !important; }
}
`;

/* ============ small utils ============ */
let __id = 0;
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `p${++__id}_${Date.now()}`);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const rad = (d) => (d * Math.PI) / 180;
const rot = (x, y, t) => [x * Math.cos(t) - y * Math.sin(t), x * Math.sin(t) + y * Math.cos(t)];
const irot = (x, y, t) => rot(x, y, -t);
const fmtMB = (bytes) => (bytes / (1024 * 1024)).toFixed(2) + " MB";
const fmtSize = (bytes) => {
  if (!bytes && bytes !== 0) return "";
  if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
};

/* Reconstruct the folder root that a photo was imported from:
   photo.path = <prefix>/<rootName>/<sub...>/<name>, photo.folder = <rootName>/<sub...> */
function deriveRoot(photo) {
  if (!photo?.path || !photo.folder) return null;
  const norm = photo.path.replace(/\\/g, "/");
  const tail = `${photo.folder}/${photo.name}`;
  if (!norm.endsWith(tail)) return null;
  const prefix = norm.slice(0, norm.length - tail.length); // ends with '/'
  return prefix + photo.folder.split("/")[0];
}

const imgCache = new Map();
const exifCache = new Map();
function loadImage(url) {
  if (imgCache.has(url)) return imgCache.get(url);
  const p = new Promise((res, rej) => {
    const im = new Image();
    im.crossOrigin = "anonymous"; // avoid tainting the canvas when drawing photo:// images
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = url;
  });
  imgCache.set(url, p);
  return p;
}

function toBlobAsync(canvas, type, quality) {
  return new Promise((res) => canvas.toBlob(res, type, quality));
}

/* ------------------------------------------------------------------
   Fast target-size solver.
   Encoding a full-resolution canvas 8-9 times (naive binary search)
   is what made the estimate slow. Instead:
     1. binary-search quality on a small proxy copy (encodes in ms),
        scaling the byte target by the pixel ratio;
     2. verify with 1-3 encodes of the real canvas, nudging quality
        with a secant step if the first guess overshoots;
     3. cache every measurement, so changing the MB value reuses
        already-known (quality → size) points instead of re-encoding.
   ------------------------------------------------------------------ */
const Q_MIN = 0.05, Q_MAX = 0.97;

function makeSizeSolver(canvas, mime) {
  const proxyPts = new Map(); // quality(fixed3) -> bytes
  const fullPts = new Map();  // quality(fixed3) -> { size, blob }
  let proxy = null;

  const ensureProxy = () => {
    if (proxy) return proxy;
    const long = Math.max(canvas.width, canvas.height);
    if (long <= 1400) { proxy = canvas; return proxy; }
    const f = 1400 / long;
    const p = document.createElement("canvas");
    p.width = Math.max(1, Math.round(canvas.width * f));
    p.height = Math.max(1, Math.round(canvas.height * f));
    const ctx = p.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(canvas, 0, 0, p.width, p.height);
    proxy = p;
    return proxy;
  };

  const measureProxy = async (q) => {
    const key = q.toFixed(3);
    if (proxyPts.has(key)) return proxyPts.get(key);
    const b = await toBlobAsync(ensureProxy(), mime, q);
    const size = b ? b.size : Infinity;
    proxyPts.set(key, size);
    return size;
  };

  const measureFull = async (q) => {
    const key = q.toFixed(3);
    if (fullPts.has(key)) return fullPts.get(key);
    const blob = await toBlobAsync(canvas, mime, q);
    const res = { size: blob ? blob.size : Infinity, blob };
    fullPts.set(key, res);
    return res;
  };

  return async function solve(targetBytes) {
    const p = ensureProxy();
    const ratio = (canvas.width * canvas.height) / (p.width * p.height);
    const proxyTarget = targetBytes / ratio;

    // 1) cheap binary search on the proxy
    let lo = Q_MIN, hi = Q_MAX;
    if ((await measureProxy(Q_MAX)) <= proxyTarget) {
      lo = Q_MAX;
    } else if ((await measureProxy(Q_MIN)) > proxyTarget) {
      lo = Q_MIN; // even minimum quality is too big — verify on full below
    } else {
      for (let i = 0; i < 7; i++) {
        const mid = (lo + hi) / 2;
        if ((await measureProxy(mid)) <= proxyTarget) lo = mid; else hi = mid;
      }
    }

    // 2) verify on the real canvas, nudging if needed
    let q = lo;
    let best = null;
    for (let i = 0; i < 4; i++) {
      const m = await measureFull(q);
      if (m.size <= targetBytes) {
        if (!best || m.size > best.size) best = { blob: m.blob, quality: q, over: false };
        break;
      }
      if (q <= Q_MIN + 1e-6) {
        return { blob: m.blob, quality: Q_MIN, over: true };
      }
      // secant-style step toward the target
      q = clamp(q * Math.pow(targetBytes / m.size, 0.9), Q_MIN, q - 0.02);
    }
    if (!best) {
      const m = await measureFull(Q_MIN);
      best = { blob: m.blob, quality: Q_MIN, over: m.size > targetBytes };
    }
    return best;
  };
}

/* Bake a photo (with its non-destructive edits) onto a canvas */
async function bakeCanvas(photo, edits, scalePct = 100, maxDim = Infinity) {
  const img = await loadImage(photo.url);
  const iw = img.naturalWidth, ih = img.naturalHeight;
  const e = edits || { theta: 0, qcx: 0, qcy: 0, cw: iw, ch: ih };
  let outW = e.cw * (scalePct / 100);
  let outH = e.ch * (scalePct / 100);
  const cap = maxDim / Math.max(outW, outH);
  if (cap < 1) { outW *= cap; outH *= cap; }
  outW = Math.max(1, Math.round(outW));
  outH = Math.max(1, Math.round(outH));
  const s = outW / e.cw;
  const canvas = document.createElement("canvas");
  canvas.width = outW; canvas.height = outH;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, outW, outH);
  ctx.translate(outW / 2, outH / 2);
  ctx.scale(s, s);
  ctx.rotate(rad(e.theta));
  ctx.translate(-e.qcx, -e.qcy);
  ctx.drawImage(img, -iw / 2, -ih / 2);
  const stampData = e.stamp || e.heal;
  if (!isEmptyStamp(stampData) || !isEmptyRedEye(e.redeye) || !isDefaultAdjust(e.adjust)) {
    const id = ctx.getImageData(0, 0, outW, outH);
    if (!isEmptyStamp(stampData)) {
      applyStampToRGBBuffer(id.data, outW, outH, 4, mapStampStrokes(stampData, iw, ih, e, s, outW, outH));
    }
    if (!isEmptyRedEye(e.redeye)) {
      applyRedEyeToRGBBuffer(id.data, outW, outH, 4, mapRedEyeEdits(e.redeye, iw, ih, e, s, outW, outH));
    }
    if (!isDefaultAdjust(e.adjust)) {
      applyAdjustToRGBBuffer(id.data, outW, outH, 4, e.adjust, { scale: s, origDiag: Math.hypot(iw, ih) });
    }
    ctx.putImageData(id, 0, 0);
  }
  return canvas;
}

/* ============ browser-mode file import (fallback) ============ */
// in the desktop app TIFF/RAW/HEIC are converted by the main process;
// in a plain browser undecodable files are skipped on load automatically
const IMG_RE = /\.(jpe?g|png|webp|gif|avif|bmp|tif|tiff|heic|heif|cr2|cr3|nef|nrw|arw|srf|sr2|orf|rw2|raf|dng|pef|srw|erf|kdc|3fr|mrw|x3f)$/i;
const isImageFile = (f) => (f.type && f.type.startsWith("image/")) || IMG_RE.test(f.name);

async function walkEntry(entry, parentPath, out) {
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej));
    if (isImageFile(file)) out.push({ file, folder: parentPath });
  } else if (entry.isDirectory) {
    const childPath = parentPath ? `${parentPath}/${entry.name}` : entry.name;
    const reader = entry.createReader();
    let batch;
    do {
      batch = await new Promise((res, rej) => reader.readEntries(res, rej));
      for (const e of batch) await walkEntry(e, childPath, out);
    } while (batch.length > 0);
  }
}

async function filesFromDataTransfer(dt) {
  const out = [];
  const items = dt.items ? Array.from(dt.items) : [];
  const entries = items
    .map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null))
    .filter(Boolean);
  if (entries.length) {
    for (const e of entries) await walkEntry(e, "", out);
  } else {
    for (const f of Array.from(dt.files || []))
      if (isImageFile(f)) out.push({ file: f, folder: "" });
  }
  return out;
}

/* ============ icons ============ */
const I = {
  photos: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="10" r="1.6"/><path d="M21 15l-5-5-9 9"/></svg>,
  folder: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/></svg>,
  upload: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 16V4m0 0l-4 4m4-4l4 4"/><path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/></svg>,
  export: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 4v12m0 0l-4-4m4 4l4-4"/><path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/></svg>,
  back: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9"><path d="M15 5l-7 7 7 7"/></svg>,
  level: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"><circle cx="12" cy="12" r="9"/><path d="M3.5 14.5l17-5"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/></svg>,
  flip: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="4" y="8" width="16" height="10" rx="1.5"/><path d="M8 4h10a2 2 0 0 1 2 2" opacity=".5"/></svg>,
  rotateLeft: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M9 4 6 7l3 3"/><path d="M6 7h8a5 5 0 1 1 0 10H8"/></svg>,
  rotateRight: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M15 4l3 3-3 3"/><path d="M18 7h-8a5 5 0 1 0 0 10h6"/></svg>,
  check: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#241A03" strokeWidth="3.2"><path d="M5 13l4 4 10-10"/></svg>,
  sort: <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M7 4v14m0 0l-3-3m3 3l3-3M17 20V6m0 0l-3 3m3-3l3 3"/></svg>,
  sun: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>,
  moon: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>,
  compare: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none"/></svg>,
  reset: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3.5 9a9 9 0 1 1-1 6.5"/><path d="M3.5 3.5v5.5h5.5"/></svg>,
  done: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round"><path d="M5 13l4 4 10-10"/></svg>,
  select: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3.5" y="3.5" width="17" height="17" rx="4"/><path d="M8 12.5l2.5 2.5L16 9.5"/></svg>,
  close: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>,
};

/* Custom hover tooltip: the browser's native `title` attribute has an
   inconsistent, engine-controlled delay (worse inside Electron) — this
   gives every hint the same reliable 1s-hold-to-show behavior instead.
   Clones the single child to attach a measuring ref + mouse handlers, so
   it adds no wrapper element/box to the layout; the bubble itself renders
   through a portal so it always sits above everything, positioned from
   the trigger's live bounding box rather than any local stacking context. */
function Tip({ text, children, side }) {
  const [pos, setPos] = useState(null);   // trigger anchor {x, y} once shown
  const [box, setBox] = useState(null);   // measured, viewport-clamped {left, top}
  const nodeRef = useRef(null);
  const bubbleRef = useRef(null);
  const timerRef = useRef(null);

  const cancel = () => { clearTimeout(timerRef.current); timerRef.current = null; };
  const onEnter = () => {
    cancel();
    timerRef.current = setTimeout(() => {
      const r = nodeRef.current?.getBoundingClientRect();
      if (!r) return;
      setBox(null);
      setPos(side === "right" ? { x: r.right + 10, y: r.top + r.height / 2, right: true } : { x: r.left + r.width / 2, y: r.bottom + 8 });
    }, 1000);
  };
  const onLeave = () => { cancel(); setPos(null); setBox(null); };
  useEffect(() => cancel, []);

  // runs before paint, so the un-clamped first-pass position (below) is
  // never actually visible — this only matters once the bubble's real
  // rendered width is known, which is exactly what was missing before:
  // a fixed 54px guess clipped any tooltip wider than ~110px at the edge.
  useLayoutEffect(() => {
    if (!pos || !bubbleRef.current) return;
    const r = bubbleRef.current.getBoundingClientRect();
    const margin = 8;
    if (pos.right) { setBox({ left: pos.x, top: pos.y - r.height / 2 }); return; }
    const left = Math.max(margin, Math.min(pos.x - r.width / 2, window.innerWidth - margin - r.width));
    setBox({ left, top: pos.y });
  }, [pos]);

  if (!text) return children;
  const child = Children.only(children);
  const merged = cloneElement(child, {
    ref: (node) => {
      nodeRef.current = node;
      const r = child.ref;
      if (typeof r === "function") r(node);
      else if (r && typeof r === "object") r.current = node;
    },
    onMouseEnter: (e) => { child.props.onMouseEnter?.(e); onEnter(); },
    onMouseLeave: (e) => { child.props.onMouseLeave?.(e); onLeave(); },
    onPointerDown: (e) => { child.props.onPointerDown?.(e); onLeave(); },
  });

  return (
    <>
      {merged}
      {pos && createPortal(
        <div ref={bubbleRef} className="tip-bubble" style={{ left: box ? box.left : pos.x, top: box ? box.top : pos.y }}>
          {text}
        </div>,
        document.body
      )}
    </>
  );
}

/* ============================================================
   RATING STARS / METADATA PANEL / FILMSTRIP
   ============================================================ */
function Stars({ value = 0, onChange, readonly }) {
  return (
    <div className={`stars ${readonly ? "readonly" : ""}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <Tip key={n} text={`${n}`}>
          <button
            className={n <= value ? "lit" : ""}
            onClick={() => onChange && onChange(n === value ? 0 : n)}
          >
            ★
          </button>
        </Tip>
      ))}
    </div>
  );
}

/* ---------- live RGB histogram ----------
   Caches one small (~480px-long-edge) downsampled RGBA buffer per photo
   PER CROP (drawn from the proxy image with the exact same crop/rotate
   canvas transform bakePreview uses — see sampleCroppedRGBBuffer — so the
   histogram reflects what the crop rectangle actually contains, not the
   whole original frame), then on every adjust change re-runs the shared
   CPU applyAdjustToRGBBuffer over that small buffer and rebins — cheap
   enough (tens of thousands of pixels) to redo on every slider tick. */

/* A real Gaussian kernel, not a fixed small hand-written tap list — lets
   sigma/radius be tuned directly rather than guessing a bigger tap array.
   A raw per-value histogram of GAIN-STRETCHED 8-bit data is genuinely
   comb-shaped (stretching 256 discrete input levels by, say, exposureGain
   > 1 means some output bins are mathematically unreachable by any input
   — real empty gaps, not noise, worse the harder any slider pushes) —
   every serious histogram display (Lightroom, Photoshop, Capture One
   included) smooths its bins before drawing for exactly this reason. An
   earlier, narrower 5-tap kernel here (effectively sigma≈1) turned out to
   still look "scattered" at even a small +5% exposure push per direct
   user testing — sigma=2.2/radius=6 is wide enough to read as a clean,
   continuous curve at any slider position while still preserving the
   distribution's real macro-shape (it doesn't reach far enough to blur
   separate major peaks into each other). Real clipping at 0/255 is
   reported separately, from the UNsmoothed raw counts (see clipFrac
   below) — smoothing would otherwise mute a hard clip spike into looking
   like "some pixels are bright," not "these are actually clipped." */
function gaussianKernel(sigma, radius) {
  const k = new Float64Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}
const HIST_SMOOTH_KERNEL = gaussianKernel(2.2, 6);

// Edge-clamped (repeats the boundary bin) rather than treating out-of-
// range taps as zero, so the 0 and 255 bins don't get artificially
// shrunk purely from having fewer neighbors to average with.
function smoothHistBins(arr) {
  const out = new Float64Array(256);
  const k = HIST_SMOOTH_KERNEL, radius = (k.length - 1) / 2;
  for (let i = 0; i < 256; i++) {
    let sum = 0;
    for (let j = -radius; j <= radius; j++) sum += arr[Math.min(255, Math.max(0, i + j))] * k[j + radius];
    out[i] = sum;
  }
  return out;
}

const HIST_SAMPLE_DIM = 480; // long-edge px — wide enough that per-bin counts aren't sparse/noisy before any smoothing even runs
// a clipped edge bin needs to hold a REAL, visible fraction of the sample
// to warn about — a handful of naturally pure-black/white pixels (a deep
// shadow, a specular highlight) exists in almost every photo and isn't
// what "clipping" means here.
const HIST_CLIP_FRACTION = 0.004;

/* The histogram's own equivalent of bakePreview's canvas crop/rotate
   transform (src/App.jsx, same qcx/qcy/theta/cw/ch math) — deliberately a
   SEPARATE small function, not a shared call, because bakePreview also
   needs to encode a JPEG for the commit/export path (load-bearing, kept
   untouched here) while this only ever needs raw pixels for statistics.
   Keep the transform math hand-in-sync with bakePreview if either changes.
   `crop` may be null (rect/view not ready yet) — falls back to the whole
   original frame, same default bakePreview itself uses. */
async function sampleCroppedRGBBuffer(srcUrl, origW, origH, crop, maxDim) {
  const img = await loadImage(srcUrl);
  const s0 = img.naturalWidth / origW;
  const e = crop || { theta: 0, qcx: 0, qcy: 0, cw: origW, ch: origH, stamp: [] };
  let outW = e.cw * s0, outH = e.ch * s0;
  const cap = maxDim / Math.max(outW, outH);
  if (cap < 1) { outW *= cap; outH *= cap; }
  outW = Math.max(1, Math.round(outW));
  outH = Math.max(1, Math.round(outH));
  const s = outW / e.cw;
  const canvas = document.createElement("canvas");
  canvas.width = outW; canvas.height = outH;
  const ctx = canvas.getContext("2d");
  ctx.translate(outW / 2, outH / 2);
  ctx.scale(s / s0, s / s0);
  ctx.rotate(rad(e.theta));
  ctx.translate(-e.qcx * s0, -e.qcy * s0);
  ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
  const id = ctx.getImageData(0, 0, outW, outH);
  if (!isEmptyStamp(e.stamp)) {
    applyStampToRGBBuffer(id.data, outW, outH, 4, mapStampStrokes(e.stamp, origW, origH, e, s, outW, outH));
  }
  if (!isEmptyRedEye(e.redeye)) {
    applyRedEyeToRGBBuffer(id.data, outW, outH, 4, mapRedEyeEdits(e.redeye, origW, origH, e, s, outW, outH));
  }
  // guides: computed lazily by Histogram's draw() the first time a guide-
  // reading slider is in use, then reused for every later tick on this
  // same sample (they depend only on these pixels, not on slider values)
  return { data: id.data, w: outW, h: outH, adjustOpts: { scale: s, origDiag: Math.hypot(origW, origH) }, guides: null };
}

function Histogram({ srcUrl, iw, ih, crop, adjust }) {
  const canvasRef = useRef(null);
  const bufRef = useRef(null);
  const rafRef = useRef(0);

  const draw = useCallback(() => {
    const cached = bufRef.current;
    const canvas = canvasRef.current;
    if (!cached || !canvas) return;
    const buf = Uint8ClampedArray.from(cached.data);
    if (!cached.guides && needsGuides(compileAdjust(adjust))) {
      cached.guides = computeGuides(cached.data, cached.w, cached.h, 4, cached.adjustOpts);
    }
    applyAdjustToRGBBuffer(buf, cached.w, cached.h, 4, adjust, { ...cached.adjustOpts, guides: cached.guides });
    const rRaw = new Uint32Array(256), gRaw = new Uint32Array(256), bRaw = new Uint32Array(256);
    for (let i = 0; i < buf.length; i += 4) { rRaw[buf[i]]++; gRaw[buf[i + 1]]++; bRaw[buf[i + 2]]++; }
    const r = smoothHistBins(rRaw), g = smoothHistBins(gRaw), b = smoothHistBins(bRaw);
    let max = 1;
    for (let i = 0; i < 256; i++) max = Math.max(max, r[i], g[i], b[i]);
    const cw = canvas.width, ch = canvas.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, cw, ch);
    ctx.globalCompositeOperation = "lighter";
    const drawChannel = (arr, color) => {
      ctx.beginPath();
      ctx.moveTo(0, ch);
      for (let i = 0; i < 256; i++) {
        // sqrt scale: raw counts are extremely peaky, this keeps the
        // low bins visible instead of one spike and a flat line
        const v = Math.sqrt(arr[i] / max);
        ctx.lineTo((i / 255) * cw, ch - v * ch);
      }
      ctx.lineTo(cw, ch);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    };
    drawChannel(r, "rgba(255,90,90,.75)");
    drawChannel(g, "rgba(70,220,120,.75)");
    drawChannel(b, "rgba(90,150,255,.75)");
    ctx.globalCompositeOperation = "source-over";

    // clipping indicators: a solid red bar flush against the edge whose
    // bin (0 = black point, 255 = white point) holds a real fraction of
    // the sample — from the RAW (pre-smoothing) counts specifically, so
    // smoothing can't mute a genuine clip spike into invisibility.
    const total = cached.w * cached.h;
    const clipBarW = Math.max(3, cw * 0.015);
    if (rRaw[0] + gRaw[0] + bRaw[0] > total * HIST_CLIP_FRACTION) {
      ctx.fillStyle = "rgba(255,40,40,.9)";
      ctx.fillRect(0, 0, clipBarW, ch);
    }
    if (rRaw[255] + gRaw[255] + bRaw[255] > total * HIST_CLIP_FRACTION) {
      ctx.fillStyle = "rgba(255,40,40,.9)";
      ctx.fillRect(cw - clipBarW, 0, clipBarW, ch);
    }
  }, [adjust]);

  useEffect(() => {
    let alive = true;
    bufRef.current = null;
    if (!srcUrl || !iw || !ih) return undefined;
    sampleCroppedRGBBuffer(srcUrl, iw, ih, crop, HIST_SAMPLE_DIM).then((buf) => {
      if (!alive) return;
      bufRef.current = buf;
      draw();
    }).catch(() => {});
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [srcUrl, iw, ih, crop, draw]);

  useEffect(() => {
    if (rafRef.current) return undefined;
    rafRef.current = requestAnimationFrame(() => { rafRef.current = 0; draw(); });
    return () => { if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = 0; } };
  }, [adjust, draw]);

  return <canvas ref={canvasRef} className="histogram" width={232} height={84} />;
}

/* ---------- develop panel: histogram + White Balance / Basic sliders ----------
   No ML auto-enhance toggle. The White Balance Selector (eyedropper, W) sits
   in the Color group's header — see WBPickOverlay. */
const ADJUST_GROUPS = [
  {
    title: "Color",
    fields: [
      { key: "temp", label: "Temperature", grad: "linear-gradient(90deg,#4f8fd9,#e8b23d)" },
      { key: "tint", label: "Tint", grad: "linear-gradient(90deg,#43b16a,#d6519b)" },
      { key: "vibrance", label: "Vibrance" },
      { key: "saturation", label: "Saturation" },
    ],
  },
  {
    title: "Light & Shadows",
    fields: [
      { key: "exposure", label: "Exposure" },
      { key: "contrast", label: "Contrast" },
      { key: "highlights", label: "Highlights" },
      { key: "shadows", label: "Shadows" },
      // stored as whitePoint/blackPoint (unchanged keys, so older edits keep
      // working); same direction in both process versions
      { key: "whitePoint", label: "Whites" },
      { key: "blackPoint", label: "Blacks" },
    ],
  },
  {
    title: "Presence",
    fields: [
      { key: "texture", label: "Texture" },
      { key: "clarity", label: "Clarity" },
      { key: "dehaze", label: "Dehaze" },
    ],
  },
];

/* A styled native range input: the track, fill and thumb are drawn
   underneath and the real <input type="range"> sits on top, transparent,
   with a thumb of the same 14px width — so dragging, clicking the track,
   keyboard nudges and every onPointerDown/onChange/onDoubleClick handler
   behave exactly as the bare input did; only the look changes. `origin`
   is where the accent fill starts (0 for ±100 sliders, else the min). */
function RangeTrack({ value, min, max, step = 1, origin = min, grad, bubble, ...inputProps }) {
  const span = max - min || 1;
  const p = clamp((value - min) / span, 0, 1);
  const o = clamp((origin - min) / span, 0, 1);
  return (
    <div
      className={`track ${grad ? "grad" : ""}`}
      style={{ "--p": p, "--a": Math.min(p, o), "--b": Math.max(p, o), ...(grad ? { "--g": grad } : null) }}
    >
      <div className="line" />
      {!grad && <div className="fill" />}
      {o > 0 && o < 1 && <div className="mid" style={{ left: `calc(7px + (100% - 14px) * ${o})` }} />}
      <div className="thumb" />
      {bubble != null && <div className="bubble">{bubble}</div>}
      <input type="range" min={min} max={max} step={step} value={value} {...inputProps} />
    </div>
  );
}

const fmtSigned = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(Math.round(v))}`;

const CHEVRON = <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" className="chev"><path d="M6 9l6 6 6-6"/></svg>;
const EYEDROPPER = <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4.5l5 5"/><path d="M17 7l2.6-2.6a1.9 1.9 0 0 0-2.7-2.7L14.3 4.3"/><path d="M15.5 8.5L6 18l-3 1 1-3 9.5-9.5"/></svg>;
const SECTIONS_KEY = "splicer.develop.closed";

function DevelopPanel({ photo, iw, ih, crop, adjust, onAdjustChange, onSliderStart, wbPicking, onToggleWBPick }) {
  const setField = (key, v) => onAdjustChange({ ...adjust, [key]: v });
  const legacy = (adjust.pv || 1) < CURRENT_PV;
  // which sections are folded — a per-viewer convenience, so localStorage
  const [closed, setClosed] = useState(() => {
    try { return JSON.parse(localStorage.getItem(SECTIONS_KEY)) || {}; } catch { return {}; }
  });
  const toggle = (title) => setClosed((c) => {
    const next = { ...c, [title]: !c[title] };
    try { localStorage.setItem(SECTIONS_KEY, JSON.stringify(next)); } catch { /* storage off */ }
    return next;
  });
  return (
    <>
      <Histogram srcUrl={proxyURL(photo)} iw={iw} ih={ih} crop={crop} adjust={adjust} />
      {legacy && (
        /* An older edit renders with the process version it was made in
           (see colorAdjust.mjs's PROCESS VERSIONS) — offered, never forced:
           updating changes how Exposure/Contrast/Highlights/Shadows/Whites/
           Blacks look on this photo, so it's the user's call. Undoable
           (onSliderStart snapshots first, same as a slider drag). */
        <div className="pv-note">
          <span>Older processing</span>
          <button
            className="chip"
            onClick={() => { onSliderStart(); onAdjustChange({ ...adjust, pv: CURRENT_PV }); }}
          >
            Update
          </button>
        </div>
      )}
      {ADJUST_GROUPS.map((group) => {
        const changed = group.fields.filter((f) => (adjust[f.key] || 0) !== 0).length;
        const isClosed = !!closed[group.title];
        return (
          <div className={`sec ${isClosed ? "closed" : ""}`} key={group.title}>
            <div className="sec-head">
              <button className="sec-toggle" onClick={() => toggle(group.title)} aria-expanded={!isClosed}>
                {CHEVRON}
                <span>{group.title}</span>
                {changed > 0 && <span className="sec-count">{changed}</span>}
              </button>
              {changed > 0 && (
                <Tip text={`Reset ${group.title}`}>
                  <button
                    className="ib"
                    onClick={() => {
                      onSliderStart();
                      onAdjustChange({ ...adjust, ...Object.fromEntries(group.fields.map((f) => [f.key, 0])) });
                    }}
                    aria-label={`Reset ${group.title}`}
                  >
                    {I.reset}
                  </button>
                </Tip>
              )}
              {group.title === "Color" && (
                <Tip text="White balance selector (W) — click something that should be neutral gray or white">
                  <button className={`ib ${wbPicking ? "on" : ""}`} onClick={onToggleWBPick} aria-label="White balance selector">
                    {EYEDROPPER}
                  </button>
                </Tip>
              )}
            </div>
            {!isClosed && (
              <div className="sec-body">
                {group.fields.map((f) => {
                  const v = adjust[f.key] || 0;
                  return (
                    <div className={`sl ${v !== 0 ? "mod" : ""}`} key={f.key}>
                      <div className="sl-head">
                        <span className="sl-label">{f.label}</span>
                        <span className="sl-val">{fmtSigned(v)}</span>
                      </div>
                      <RangeTrack
                        min={-100} max={100} origin={0} value={v} grad={f.grad} bubble={fmtSigned(v)}
                        className="adj"
                        onPointerDown={onSliderStart}
                        onChange={(e) => setField(f.key, parseFloat(e.target.value))}
                        onDoubleClick={() => setField(f.key, 0)}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

function MetaPanel({ photo, outW, outH, theta, exif }) {
  const ext = (photo.name.split(".").pop() || "").toUpperCase();
  const mp = ((photo.w * photo.h) / 1e6).toFixed(1);
  const fmtShutter = (v) => (v >= 1 ? `${v}s` : `1/${Math.round(1 / v)}s`);
  const hasExif = exif && (exif.model || exif.iso || exif.exposure || exif.fnumber);
  return (
    <>
      <h4>File</h4>
      <div className="mrow"><span className="k">Name</span><span className="v">{photo.name}</span></div>
      {photo.folder && (
        <div className="mrow"><span className="k">Folder</span><span className="v">{photo.folder}</span></div>
      )}
      <div className="mrow"><span className="k">Format</span><span className="v">{ext}</span></div>
      <div className="mrow"><span className="k">Dimensions</span><span className="v">{photo.w}×{photo.h}</span></div>
      <div className="mrow"><span className="k">Megapixels</span><span className="v">{mp} MP</span></div>
      {photo.sizeBytes > 0 && (
        <div className="mrow"><span className="k">Size</span><span className="v">{fmtSize(photo.sizeBytes)}</span></div>
      )}
      {photo.mtime > 0 && (
        <div className="mrow"><span className="k">Modified</span><span className="v">{new Date(photo.mtime).toLocaleDateString()}</span></div>
      )}
      {hasExif && (
        <>
          <div className="divider" />
          <h4>Camera</h4>
          {(exif.make || exif.model) && (
            <div className="mrow"><span className="k">Camera</span><span className="v">{[exif.make, exif.model].filter(Boolean).join(" ")}</span></div>
          )}
          {exif.lens && <div className="mrow"><span className="k">Lens</span><span className="v">{exif.lens}</span></div>}
          {exif.iso && <div className="mrow"><span className="k">ISO</span><span className="v">{exif.iso}</span></div>}
          {exif.exposure && <div className="mrow"><span className="k">Shutter</span><span className="v">{fmtShutter(exif.exposure)}</span></div>}
          {exif.fnumber && <div className="mrow"><span className="k">Aperture</span><span className="v">f/{exif.fnumber}</span></div>}
          {exif.focal && <div className="mrow"><span className="k">Focal</span><span className="v">{Math.round(exif.focal)} mm</span></div>}
          {exif.taken && <div className="mrow"><span className="k">Captured</span><span className="v">{new Date(exif.taken).toLocaleDateString()}</span></div>}
        </>
      )}
      <div className="divider" />
      <h4>Crop</h4>
      <div className="mrow"><span className="k">Output</span><span className="v">{outW}×{outH} px</span></div>
      {outH > 0 && (
        <div className="mrow"><span className="k">Ratio</span><span className="v">{(outW / outH).toFixed(2)}:1</span></div>
      )}
      <div className="mrow"><span className="k">Angle</span><span className="v">{theta > 0 ? "+" : ""}{theta.toFixed(1)}°</span></div>
    </>
  );
}

function Filmstrip({ items, currentId, onSelect, selectedIds, onToggleSel, onRangeSel, onContext, height }) {
  const ref = useRef(null);
  const anchor = useRef(null);
  useEffect(() => {
    const el = ref.current?.querySelector(".strip-item.current");
    el?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [currentId]);
  if (!items || items.length < 2) return null;
  const click = (p, idx, e) => {
    if (e.shiftKey) {
      // no anchor yet? the current photo is the natural range start
      const from = anchor.current ?? items.findIndex((x) => x.id === currentId);
      const [a, b] = [Math.min(from, idx), Math.max(from, idx)];
      onRangeSel(items.slice(Math.max(0, a), b + 1).map((x) => x.id));
      anchor.current = idx;
      return;
    }
    if (e.ctrlKey || e.metaKey) {
      onToggleSel(p.id);
      anchor.current = idx;
      return;
    }
    anchor.current = idx;
    onSelect(p.id);
  };
  return (
    <div className="filmstrip" ref={ref} style={height ? { height, "--strip-h": `${height}px` } : undefined}>
      {items.map((p, idx) => {
        const isSel = selectedIds?.has(p.id);
        return (
          <Tip key={p.id} text={`${p.name} — Ctrl+click to select, Shift+click for a range, right-click to export`}>
            <div
              className={`strip-item ${p.id === currentId ? "current" : ""} ${isSel ? "sel" : ""}`}
              onClick={(e) => click(p, idx, e)}
              onContextMenu={(e) => { e.preventDefault(); onContext && onContext(p.id, e.clientX, e.clientY); }}
              onMouseDown={(e) => { if (e.shiftKey) e.preventDefault(); }}
            >
              <Thumb src={p.previewUrl || thumbURL(p)} />
              {p.edits && <div className="sedit" />}
              {isSel && <div className="scheck">{I.check}</div>}
              <div className="sinfo">
                <span>{fmtSize(p.sizeBytes)}</span>
                {p.rating > 0 && <span className="sstars">{"★".repeat(p.rating)}</span>}
              </div>
            </div>
          </Tip>
        );
      })}
    </div>
  );
}

/* ============================================================
   EDITOR
   ============================================================ */
const ASPECTS = [
  { key: "free", label: "Free" },
  { key: "orig", label: "Original" },
  { key: "1:1", label: "1:1", v: 1 },
  { key: "3:2", label: "3:2", v: 3 / 2 },
  { key: "4:3", label: "4:3", v: 4 / 3 },
  { key: "16:9", label: "16:9", v: 16 / 9 },
  { key: "4:5", label: "4:5", v: 4 / 5 },
];
const swapLabel = (label) => {
  const m = label.match(/^(\d+):(\d+)$/);
  return m ? `${m[2]}:${m[1]}` : label;
};
const PAD = 36;
const MIN_CROP = 64;

/* ============================================================
   GLImage — WebGL-backed drop-in replacement for a plain <img>.
   Renders the same bitmap `<img>` would, but through a shader that
   applies the live tone/color adjustment (see electron/colorAdjust.mjs
   for the CPU twin these uniforms are compiled from). Sized and
   positioned purely by CSS (className/style), exactly like the <img>
   it replaces — the parent .imgspace transform math is untouched.
   ============================================================ */
const GL_VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

/* Live-paint budget: total brush DABS across every stroke on the photo
   (not stroke count) — a stroke is a path of small circular dabs, so this
   caps at, e.g., ~6 medium strokes rather than 64 clicks. GLImage recomposites
   the whole stamp list on the CPU on every change (see below), so this bound
   keeps that recomposite cheap enough to stay smooth while painting; the CPU
   export path has no such limit. Enforced at paint time (StampOverlay). */
const MAX_STAMP_DABS = 300;

const GL_FRAG = `
// highp, not mediump: this shader round-trips through pow() twice
// (sRGB<->linear, see below) — on genuine mediump precision (~10-bit
// mantissa) that round-trip alone can band visibly, especially in
// shadow detail where the sRGB curve is steepest. Desktop GPUs (this is
// an Electron app, not a mobile WebView) reliably implement highp in
// fragment shaders even though the GLSL ES spec only guarantees it in
// vertex shaders.
precision highp float;
uniform sampler2D u_tex;
// guide planes (colorAdjust.mjs computeGuides): r = fine base, g = mid
// base, b = coarse base, a = smoothed dark channel (sRGB-encoded linear).
// Same extent as u_tex, LINEAR-filtered — the GPU twin of sampleGuide().
uniform sampler2D u_guide;
uniform float u_useGuide;
// white balance as one 3×3 on linear sRGB: pv 1's per-channel gains are
// its diagonal, pv 2's is the Bradford white-balance matrix
uniform mat3 u_wbMat;
uniform float u_satGain;
uniform float u_exposureGain;
uniform float u_contrastAmt;
uniform float u_highlightGain;
uniform float u_shadowGain;
uniform float u_whiteShift;
uniform float u_blackShift;
uniform float u_pv;
uniform float u_vibrance, u_textureAmt, u_clarityAmt, u_dehazeAmt, u_airlight;
uniform float u_shoulderKnee, u_hiStops, u_shStops, u_contrastS, u_whitesAmt, u_blacksAmt;
varying vec2 v_uv;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

float smoothstepClamp(float e0, float e1, float x) {
  float t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

/* Standard sRGB (IEC 61966-2-1) transfer function — see
   electron/colorAdjust.mjs's header for why white balance/exposure/
   saturation happen in LINEAR light rather than directly on the
   gamma-encoded texture values, and why contrast/highlights/shadows/
   levels deliberately don't move too. KEEP IN SYNC WITH colorAdjust.mjs
   (srgbToLinear/linearToSrgb/adjustPixel). */
vec3 srgbToLinear(vec3 c) {
  vec3 lo = c / 12.92;
  vec3 hi = pow(max((c + 0.055) / 1.055, 0.0), vec3(2.4));
  return mix(lo, hi, step(0.04045, c));
}
vec3 linearToSrgb(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(max(c, 0.0), vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(0.0031308, c));
}
float srgbToLinear1(float c) { return c <= 0.04045 ? c / 12.92 : pow(max((c + 0.055) / 1.055, 0.0), 2.4); }
float linearToSrgb1(float c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055; }

// texture + clarity luminance shift (colorAdjust.mjs detailShift)
float detailShift(vec3 c, vec4 gd, float lumIn) {
  float lum = clamp(dot(c, LUMA), 0.0, 1.0);
  float mid = 1.0 - (2.0 * lum - 1.0) * (2.0 * lum - 1.0);
  return u_textureAmt * (lumIn - gd.r) + u_clarityAmt * (lumIn - gd.g) * (0.3 + 0.7 * mid);
}

void main() {
  vec3 tex = texture2D(u_tex, v_uv).rgb;
  vec4 gd = texture2D(u_guide, v_uv);
  bool useGuide = u_useGuide > 0.5;

  vec3 lin = srgbToLinear(tex);
  float lumIn = linearToSrgb1(dot(lin, LUMA));

  if (useGuide && u_dehazeAmt != 0.0) {
    float A = u_airlight;
    float t = max(${DEHAZE_T_MIN.toFixed(6)}, 1.0 - ${DEHAZE_OMEGA.toFixed(6)} * srgbToLinear1(gd.a) / A);
    if (u_dehazeAmt > 0.0) {
      lin += ((lin - A) / t + A - lin) * u_dehazeAmt;
    } else {
      lin += (A - lin) * (-u_dehazeAmt * 0.5 * (0.4 + 0.6 * (1.0 - t)));
    }
    lin = max(lin, 0.0);
  }

  lin = max(u_wbMat * lin, 0.0);
  lin *= u_exposureGain;

  if (u_vibrance != 0.0) {
    float mx = max(lin.r, max(lin.g, lin.b)), mn = min(lin.r, min(lin.g, lin.b));
    float sat = mx > 0.0 ? (mx - mn) / mx : 0.0;
    float skin = (lin.r >= lin.g && lin.g >= lin.b) ? 0.5 : 1.0;
    float gain = 1.0 + u_vibrance * (1.0 - sat) * (1.0 - sat) * skin;
    float grayV = dot(lin, LUMA);
    lin = grayV + (lin - grayV) * gain;
  }

  float grayLin = dot(lin, LUMA);
  lin = grayLin + (lin - grayLin) * u_satGain;

  if (useGuide && (u_hiStops != 0.0 || u_shStops != 0.0)) {
    float base = max(1e-4, srgbToLinear1(gd.b) * u_exposureGain);
    float lb2 = log2(base);
    float stops = u_hiStops * smoothstepClamp(-2.5, 0.0, lb2) + u_shStops * (1.0 - smoothstepClamp(-6.5, -2.0, lb2));
    lin *= exp2(stops);
  }
  if (u_shoulderKnee < 1.0) {
    float y = dot(lin, LUMA);
    float kn = u_shoulderKnee;
    if (y > kn) lin *= (kn + (1.0 - kn) * (1.0 - exp(-(y - kn) / (1.0 - kn)))) / y;
  }

  vec3 c = linearToSrgb(max(lin, 0.0));

  if (u_pv >= 1.5) {
    // pv 2 tail (colorAdjust.mjs finishPv2)
    if (u_contrastS != 0.0) {
      c = clamp(c, 0.0, 1.0);
      c -= u_contrastS * sin(6.283185307 * c);
    }
    if (useGuide && (u_textureAmt != 0.0 || u_clarityAmt != 0.0)) c += detailShift(c, gd, lumIn);
    if (u_whitesAmt != 0.0 || u_blacksAmt != 0.0) {
      float lum = dot(c, LUMA);
      c += u_whitesAmt * smoothstepClamp(0.45, 1.0, lum) + u_blacksAmt * (1.0 - smoothstepClamp(0.0, 0.55, lum));
    }
    gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
    return;
  }

  c = (c - 0.5) * (1.0 + u_contrastAmt) + 0.5;

  float lumH = dot(c, LUMA);
  float hd = smoothstepClamp(0.35, 1.0, lumH) * u_highlightGain;
  c += hd >= 0.0 ? hd * (1.0 - c) : hd * c;
  float lumS = dot(c, LUMA);
  float sd = (1.0 - smoothstepClamp(0.0, 0.5, lumS)) * u_shadowGain;
  c += sd >= 0.0 ? sd * (1.0 - c) : sd * c;

  if (useGuide && (u_textureAmt != 0.0 || u_clarityAmt != 0.0)) c += detailShift(c, gd, lumIn);

  float lo2 = u_blackShift, hi2 = 1.0 - u_whiteShift;
  c = (c - lo2) / max(0.05, hi2 - lo2);

  gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

/* compileAdjust()'s white balance as the shader's mat3 (GLSL matrices are
   column-major): pv 2's matrix transposed, or pv 1's gains on the diagonal. */
function wbMatColumnMajor(c) {
  const m = c.wbMat;
  if (m) return new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
  return new Float32Array([c.wbGainR, 0, 0, 0, c.wbGainG, 0, 0, 0, c.wbGainB]);
}

function compileGLShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const msg = `GLImage shader compile error: ${gl.getShaderInfoLog(sh)}`;
    console.error(msg);
    if (typeof window !== "undefined" && window.meridian?.logError) window.meridian.logError(msg);
  }
  return sh;
}

/* Converts stamp strokes (original-image absolute px) into the pixel space
   of WHATEVER texture is currently uploaded, given texRegion = {sx, sy, sw}:
   the original-image-px sub-rectangle that texture depicts (sx=sy=0, sw=iw
   for the full proxy; tile.sx/sy/sw for a deep-zoom tile). This is what lets
   the same stamp state render correctly on both the base proxy image and a
   higher-res tile without Editor needing to know which texture size a given
   GLImage instance actually uploaded. */
function mapStampToTexSpace(stamp, texRegion, texW) {
  if (!stamp || !stamp.length || !texRegion || !texRegion.sw || !texW) return [];
  const s0 = texW / texRegion.sw;
  return stamp.map((s) => ({
    radius: s.radius * s0,
    dx: s.dx * s0,
    dy: s.dy * s0,
    feather: s.feather,
    opacity: s.opacity,
    mode: s.mode,
    points: s.points.map((p) => ({
      x: (p.x - texRegion.sx) * s0, y: (p.y - texRegion.sy) * s0,
      // per-dab offset override (see stampTool.mjs header) — a relative
      // vector, so only scale applies, same as the stroke-level dx/dy above
      ...(p.dx !== undefined ? { dx: p.dx * s0, dy: p.dy * s0 } : null),
    })),
  }));
}

/* Same as mapStampToTexSpace, for red-eye entries: the uploaded texture is
   the UNROTATED original (CSS rotates it), so only translate + scale. */
function mapRedEyeToTexSpace(redeye, texRegion, texW) {
  if (isEmptyRedEye(redeye) || !texRegion || !texRegion.sw || !texW) return [];
  const s0 = texW / texRegion.sw;
  return mapRedEye(redeye, (x, y) => [(x - texRegion.sx) * s0, (y - texRegion.sy) * s0], 0, s0);
}

/* How long the composited texture has to sit unchanged before GLImage
   recomputes its guide planes (colorAdjust.mjs computeGuides — hundreds
   of ms at proxy size). Painting a stamp stroke re-uploads every frame;
   recomputing on each of those would stall the brush, so the guides lag
   slightly behind retouching instead (still correct for the image as a
   whole, just not yet aware of the newest stroke for a moment). */
const GUIDE_RECOMPUTE_DELAY = 350;

/* Remove mode's local neural fill (MI-GAN, main process — see
   electron/neuralFill.mjs and removeTool.mjs's hybrid). A GLImage
   composite fills Remove strokes with PatchMatch right away and queues
   neural jobs; this sends them over IPC and, as results land, asks EVERY
   GLImage to recomposite (a job queued by the base image may be taken by
   the deep-zoom tile's drain, or vice versa — the module queue is shared). */
const neuralRedraws = new Set();
function drainNeuralJobs() {
  if (!NATIVE || !window.meridian?.neuralInpaint) return;
  const jobs = takeNeuralJobs();
  for (const j of jobs) {
    window.meridian.neuralInpaint({ w: j.w, h: j.h, rgb: j.rgb, hole: j.hole, engine: j.engine })
      .then((r) => putNeuralResult(j.key, r ? new Uint8Array(r) : null))
      .catch(() => putNeuralResult(j.key, null))
      .then(() => { for (const redraw of neuralRedraws) redraw(); });
  }
}

function GLImage({ className, style, src, draggable, adjust, stamp, draft, redeye, texRegion, origDiag }) {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const texRef = useRef(null);
  const rafRef = useRef(0);
  const adjustRef = useRef(adjust);
  adjustRef.current = adjust;
  const stampRef = useRef(stamp);
  stampRef.current = stamp;
  const redeyeRef = useRef(redeye);
  redeyeRef.current = redeye;
  const origDiagRef = useRef(origDiag);
  origDiagRef.current = origDiag;
  // guide planes for the CURRENT composite: { version, airlight } once
  // uploaded to guideTexRef; `compositeRef` is what they get computed from
  const guideTexRef = useRef(null);
  const guideRef = useRef(null);
  const compositeRef = useRef(null); // { version, data, width, height, scale }
  const guideTimerRef = useRef(0);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const texRegionRef = useRef(texRegion);
  texRegionRef.current = texRegion;
  const srcRef = useRef(src);
  srcRef.current = src;
  const pristineRef = useRef(null); // { src, width, height, data, img } — decoded once per unique `src`
  const uploadTokenRef = useRef(0);
  const uploadRafRef = useRef(0);

  /* Guide planes for the latest composite (see GUIDE_RECOMPUTE_DELAY).
     Only ever runs when a guide-reading slider is actually in use — draw()
     is what asks for it — so photos edited with per-pixel sliders only
     never pay for it. */
  const scheduleGuides = useCallback((delay) => {
    clearTimeout(guideTimerRef.current);
    guideTimerRef.current = setTimeout(() => {
      const comp = compositeRef.current;
      const g = glRef.current;
      if (!comp || !g) return;
      const guides = computeGuides(comp.data, comp.width, comp.height, 4, { scale: comp.scale, origDiag: origDiagRef.current });
      const { gl } = g;
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, guideTexRef.current);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, guides.width, guides.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, guides.data);
      gl.activeTexture(gl.TEXTURE0);
      guideRef.current = { src: comp.src, version: comp.version, airlight: guides.airlight };
      // via ref: scheduleDraw is defined below (and itself depends on draw,
      // which depends on this)
      if (redrawRef.current) redrawRef.current();
    }, delay);
  }, []);
  const redrawRef = useRef(null);

  const draw = useCallback(() => {
    const g = glRef.current;
    const canvas = canvasRef.current;
    if (!g || !texRef.current || !canvas) return;
    const { gl } = g;
    const c = compileAdjust(adjustRef.current);
    // guides: usable if they belong to this image (possibly one composite
    // behind — see GUIDE_RECOMPUTE_DELAY); (re)computed on demand
    const comp = compositeRef.current;
    const gr = guideRef.current;
    const wantGuides = needsGuides(c);
    const useGuide = wantGuides && gr && comp && gr.src === comp.src;
    if (wantGuides && comp && (!gr || gr.version !== comp.version)) {
      scheduleGuides(gr && gr.src === comp.src ? GUIDE_RECOMPUTE_DELAY : 0);
    }
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.useProgram(g.prog);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, guideTexRef.current);
    gl.uniform1i(g.uGuide, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texRef.current);
    gl.uniform1i(g.uTex, 0);
    gl.uniform1f(g.uUseGuide, useGuide ? 1 : 0);
    gl.uniformMatrix3fv(g.uWbMat, false, wbMatColumnMajor(c));
    gl.uniform1f(g.uSat, c.satGain);
    gl.uniform1f(g.uExp, c.exposureGain);
    gl.uniform1f(g.uContrast, c.contrastAmt);
    gl.uniform1f(g.uHi, c.highlightGain);
    gl.uniform1f(g.uSh, c.shadowGain);
    gl.uniform1f(g.uWhite, c.whiteShift);
    gl.uniform1f(g.uBlack, c.blackShift);
    gl.uniform1f(g.uPv, c.pv);
    gl.uniform1f(g.uVibrance, c.vibrance);
    gl.uniform1f(g.uTextureAmt, c.textureAmt);
    gl.uniform1f(g.uClarityAmt, c.clarityAmt);
    gl.uniform1f(g.uDehazeAmt, c.dehazeAmt);
    gl.uniform1f(g.uAirlight, useGuide ? gr.airlight : 1);
    gl.uniform1f(g.uShoulderKnee, c.shoulderKnee);
    gl.uniform1f(g.uHiStops, c.hiStops);
    gl.uniform1f(g.uShStops, c.shStops);
    gl.uniform1f(g.uContrastS, c.contrastS);
    gl.uniform1f(g.uWhitesAmt, c.whitesAmt);
    gl.uniform1f(g.uBlacksAmt, c.blacksAmt);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [scheduleGuides]);

  const scheduleDraw = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0;
      draw();
    });
  }, [draw]);
  redrawRef.current = scheduleDraw;

  /* mount-once: WebGL context + program + fullscreen-quad + texture object */
  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext("webgl", { premultipliedAlpha: false });
    if (!gl) return;
    const vs = compileGLShader(gl, gl.VERTEX_SHADER, GL_VERT);
    const fs = compileGLShader(gl, gl.FRAGMENT_SHADER, GL_FRAG);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const msg = `GLImage program link error: ${gl.getProgramInfoLog(prog)}`;
      console.error(msg);
      if (typeof window !== "undefined" && window.meridian?.logError) window.meridian.logError(msg);
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(prog, "a_pos");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // NPOT textures (our proxy/tile images aren't power-of-two): CLAMP_TO_EDGE
    // + LINEAR with no mipmaps, or WebGL1 renders black.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    texRef.current = tex;

    // guide planes texture (unit 1) — starts as a 1×1 placeholder so the
    // sampler is never unbound; u_useGuide gates every read of it
    const guideTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, guideTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
    gl.activeTexture(gl.TEXTURE0);
    guideTexRef.current = guideTex;

    glRef.current = {
      gl, prog,
      uTex: gl.getUniformLocation(prog, "u_tex"),
      uGuide: gl.getUniformLocation(prog, "u_guide"),
      uUseGuide: gl.getUniformLocation(prog, "u_useGuide"),
      uPv: gl.getUniformLocation(prog, "u_pv"),
      uVibrance: gl.getUniformLocation(prog, "u_vibrance"),
      uTextureAmt: gl.getUniformLocation(prog, "u_textureAmt"),
      uClarityAmt: gl.getUniformLocation(prog, "u_clarityAmt"),
      uDehazeAmt: gl.getUniformLocation(prog, "u_dehazeAmt"),
      uAirlight: gl.getUniformLocation(prog, "u_airlight"),
      uShoulderKnee: gl.getUniformLocation(prog, "u_shoulderKnee"),
      uHiStops: gl.getUniformLocation(prog, "u_hiStops"),
      uShStops: gl.getUniformLocation(prog, "u_shStops"),
      uContrastS: gl.getUniformLocation(prog, "u_contrastS"),
      uWhitesAmt: gl.getUniformLocation(prog, "u_whitesAmt"),
      uBlacksAmt: gl.getUniformLocation(prog, "u_blacksAmt"),
      uWbMat: gl.getUniformLocation(prog, "u_wbMat"),
      uSat: gl.getUniformLocation(prog, "u_satGain"),
      uExp: gl.getUniformLocation(prog, "u_exposureGain"),
      uContrast: gl.getUniformLocation(prog, "u_contrastAmt"),
      uHi: gl.getUniformLocation(prog, "u_highlightGain"),
      uSh: gl.getUniformLocation(prog, "u_shadowGain"),
      uWhite: gl.getUniformLocation(prog, "u_whiteShift"),
      uBlack: gl.getUniformLocation(prog, "u_blackShift"),
    };

    return () => {
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
      glRef.current = null;
      texRef.current = null;
      guideTexRef.current = null;
      clearTimeout(guideTimerRef.current);
    };
  }, []);

  /* Recomposite (if there are any stamp strokes) + upload the texture,
     reading the LATEST stamp/draft/texRegion from refs rather than whatever
     was current when this got scheduled (see scheduleUpload). Stamps are
     composited on the CPU — via the exact same electron/stampTool.mjs
     function the export bake paths use — against a pristine pixel buffer
     decoded once per `src`. This is deliberately NOT a shader-side per-dab
     loop sampling a single static texture: that would make every stroke's
     source always read the UNEDITED original, so a second stroke sourcing
     from an area an earlier stroke had already cloned over would show the
     old, pre-edit content instead of what's actually on screen.

     `draft` (Editor's in-progress stroke, while a drag is still down — see
     StampOverlay) gets folded into the SAME composite call, not rendered
     any other way. It used to drive only an SVG outline while the drag was
     live, showing literally the untouched original underneath — which
     read as "looks great while editing" for no reason other than nothing
     had actually been cloned yet, then looked different the instant the
     real algorithm ran for the first time at commit. Running the identical
     function on the identical merged stroke list here means there is no
     longer a second, different code path to diverge from: what's on
     screen during the drag IS what gets committed, pixel for pixel. */
  const uploadTexture = useCallback(() => {
    const p = pristineRef.current;
    const g = glRef.current;
    const canvas = canvasRef.current;
    // p.src !== srcRef.current: `src` and `texRegion` typically change
    // together (a deep-zoom tile swapping in brings both a new crop window
    // AND a new image URL at once), but the new pristine buffer only exists
    // once its own decode effect finishes — an async step. Without this
    // check, a texRegion-change here would fire first, this recomposite
    // would run using the OLD pristine buffer's pixel data, and resize the
    // canvas to the OLD image's dimensions — while the CSS box driving this
    // canvas's on-screen position/size has already jumped to the NEW tile's
    // (different) bounds, since that's driven directly by React state, not
    // by this decode. The result is one visible frame of the previous
    // tile's content stretched/repositioned into the new tile's box (looks
    // like the same photo "blinking" with a shift) before the real decode
    // lands and the src-decode effect below calls scheduleUpload() again
    // with the correct buffer.
    if (!p || !g || !canvas || p.src !== srcRef.current) return;
    canvas.width = p.width;
    canvas.height = p.height;
    g.gl.bindTexture(g.gl.TEXTURE_2D, texRef.current);
    // fold the in-progress draft stroke into the same list `stamp` itself
    // will hold once the drag commits — an appended new stroke, or the
    // relevant existing stroke's own points extended, exactly mirroring
    // the two branches beginWindowDrag's onUp commits (see StampOverlay).
    const d = draftRef.current;
    let strokesForComposite = stampRef.current;
    // a Remove stroke in progress is outline-only: its content-aware fill
    // (electron/removeTool.mjs) runs once, when the stroke is committed —
    // never per pointermove
    const draftIsRemove = d && (d.isNew ? d.mode === "remove" : stampRef.current[d.index]?.mode === "remove");
    if (d && d.points && d.points.length && !draftIsRemove) {
      strokesForComposite = d.isNew
        ? [...strokesForComposite, { points: d.points, radius: d.radius, dx: d.dx, dy: d.dy, feather: d.feather, opacity: d.opacity, mode: d.mode }]
        : strokesForComposite.map((s, i) => (i === d.index ? { ...s, points: [...s.points, ...d.points] } : s));
    }
    const texStamp = mapStampToTexSpace(strokesForComposite, texRegionRef.current, p.width);
    const texRedEye = mapRedEyeToTexSpace(redeyeRef.current, texRegionRef.current, p.width);
    // same order as every bake path: stamp/heal → red eye → (color, in the shader)
    let pixels = p.data;
    if (texStamp.length || texRedEye.length) {
      pixels = new Uint8ClampedArray(p.data);
      if (texStamp.length) {
        // queue neural jobs only for THIS on-screen composite (not for the
        // histogram's or thumbnails' own small bakes) — see drainNeuralJobs
        setNeuralEnabled(NATIVE && !!window.meridian?.neuralInpaint);
        applyStampToRGBBuffer(pixels, p.width, p.height, 4, texStamp);
        setNeuralEnabled(false);
        drainNeuralJobs();
      }
      if (texRedEye.length) applyRedEyeToRGBBuffer(pixels, p.width, p.height, 4, texRedEye);
      g.gl.texImage2D(g.gl.TEXTURE_2D, 0, g.gl.RGBA, p.width, p.height, 0, g.gl.RGBA, g.gl.UNSIGNED_BYTE, pixels);
    } else {
      g.gl.texImage2D(g.gl.TEXTURE_2D, 0, g.gl.RGBA, g.gl.RGBA, g.gl.UNSIGNED_BYTE, p.img);
    }
    // what the guide planes get computed from (see scheduleGuides); version
    // bumps on every upload so draw() knows the current guides are stale
    const tr = texRegionRef.current;
    compositeRef.current = {
      src: p.src,
      version: (compositeRef.current ? compositeRef.current.version : 0) + 1,
      data: pixels, width: p.width, height: p.height,
      scale: tr && tr.sw ? p.width / tr.sw : 1,
    };
    scheduleDraw();
  }, [scheduleDraw]);

  /* Coalesced via requestAnimationFrame, same idea as scheduleDraw — a fast
     drag can emit a new dab well under 16ms apart, and recompositing (a
     typed-array clone + a handful of feathered-circle blends) plus a GPU
     upload is real CPU work, not a cheap uniform tweak; doing it once per
     dab rather than once per frame was measured to occasionally run long
     enough on a large brush to eat into the next pointermove's turn, which
     is a plausible cause of a drag feeling like it "drops" the gesture
     partway through. Only the latest stamp state as of the next frame is
     ever applied — nothing is lost, intermediate dabs just don't each get
     their own upload. */
  const scheduleUpload = useCallback(() => {
    if (uploadRafRef.current) return;
    uploadRafRef.current = requestAnimationFrame(() => {
      uploadRafRef.current = 0;
      uploadTexture();
    });
  }, [uploadTexture]);

  useEffect(() => {
    neuralRedraws.add(scheduleUpload);
    return () => { neuralRedraws.delete(scheduleUpload); };
  }, [scheduleUpload]);

  /* Decode `src` into a pristine pixel buffer once per unique URL. */
  useEffect(() => {
    let alive = true;
    if (!src) return undefined;
    const token = ++uploadTokenRef.current;
    loadImage(src).then((img) => {
      if (!alive || token !== uploadTokenRef.current) return;
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      pristineRef.current = { src, width: c.width, height: c.height, data: ctx.getImageData(0, 0, c.width, c.height).data, img };
      scheduleUpload();
    }).catch(() => {});
    return () => { alive = false; };
  }, [src, scheduleUpload]);

  /* Recomposite + reupload whenever the stamp strokes, the in-progress
     draft stroke, or texRegion change. `draft` changes on every processed
     pointermove during a drag (see StampOverlay's beginWindowDrag handler)
     — scheduleUpload's own requestAnimationFrame coalescing (see its own
     comment above) is what keeps that from recompositing/re-uploading
     more than once per displayed frame, same as it already does for
     rapid stamp commits. */
  useEffect(() => { scheduleUpload(); }, [stamp, draft, redeye, texRegion, scheduleUpload]);

  /* redraw (GPU-cheap: uniform update + one quad) on every adjust change */
  useEffect(() => { scheduleDraw(); }, [adjust, scheduleDraw]);

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (uploadRafRef.current) cancelAnimationFrame(uploadRafRef.current);
  }, []);

  return <canvas ref={canvasRef} className={className} style={style} draggable={draggable} />;
}

const CLONE_OUTLINE = "rgba(130,215,255,.95)"; // unselected Clone stroke outline
const HEAL_OUTLINE = "rgba(120,230,140,.95)";  // unselected Heal stroke outline
const REMOVE_OUTLINE = "rgba(255,160,90,.95)";  // unselected Remove stroke outline
const STAMP_MODES = ["clone", "heal", "remove"];
const STAMP_MODE_LABEL = { clone: "Clone", heal: "Heal", remove: "Remove" };
const svgTool = (children) => <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{children}</svg>;
const TOOL_ICON = {
  crop: svgTool(<><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></>),
  // "echo copy": the paint, with a fainter dashed duplicate behind it (the source)
  clone: svgTool(<><circle cx="15" cy="9" r="5.3" strokeDasharray="2 2" opacity=".55"/><circle cx="10" cy="14" r="5.3"/></>),
  heal: svgTool(<><rect x="2.6" y="8.4" width="18.8" height="7.2" rx="3.6" transform="rotate(-45 12 12)"/><path d="M10.6 10.6h.01M13.4 13.4h.01M10.6 13.4h.01M13.4 10.6h.01" strokeWidth="2.4"/></>),
  remove: svgTool(<><path d="M8 20h12"/><path d="M5.3 14.7 13.6 6.4a2 2 0 0 1 2.8 0l3.2 3.2a2 2 0 0 1 0 2.8L12.4 19.6a1.4 1.4 0 0 1-1 .4H8.8a1.4 1.4 0 0 1-1-.4l-2.5-2.5a1.7 1.7 0 0 1 0-2.4z"/><path d="M9.5 10.5l4.9 4.9"/></>),
  redeye: svgTool(<><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3" fill="currentColor"/></>),
  wb: svgTool(<><path d="M14.5 4.5l5 5"/><path d="M17 7l2.6-2.6a1.9 1.9 0 0 0-2.7-2.7L14.3 4.3"/><path d="M15.5 8.5L6 18l-3 1 1-3 9.5-9.5"/></>),
  trash: svgTool(<><path d="M4 7h16"/><path d="M9 7V4.8A.8.8 0 0 1 9.8 4h4.4a.8.8 0 0 1 .8.8V7"/><path d="M6.5 7l.9 12a1.2 1.2 0 0 0 1.2 1.1h6.8a1.2 1.2 0 0 0 1.2-1.1l.9-12"/></>),
};
const outlineFor = (mode) => (mode === "heal" ? HEAL_OUTLINE : mode === "remove" ? REMOVE_OUTLINE : CLONE_OUTLINE);
const STAMP_SOURCE_OFFSET_MULT = 2.5; // default source distance, as a multiple of radius
const STAMP_MIN_RADIUS = 3;           // image-px floor so a stroke never collapses to zero
const STAMP_DAB_SPACING = 0.5;        // new dab emitted once the pointer has moved radius*this
const STAMP_SRC_HIT_MIN = 22;         // screen-px floor for the draggable "move source" hit target — a small brush's own visual circle is otherwise too tiny to reliably grab

/* Self-overlap avoidance + auto-feather for the clone stamp — both driven
   off the same cheap 16-point ring statistic (mean color + variance)
   sampled from Editor's cached proxy-resolution buffer (editPixelsRef).
   Deliberately a local search, not a full nearest-neighbor-field search
   over the whole photo (PatchMatch/content-aware-fill territory) — a
   different, much bigger algorithm, not a drop-in replacement for
   "drag a source handle" precision clone-stamping. (Splicer briefly had
   an automatic Heal brush built on exactly that algorithm; it was
   removed — Clone Stamp's manual source point proved more reliable in
   practice, see this project's own memory if reviving it.) */
const STAMP_OVERLAP_MARGIN = 1.6;     // × radius — how close counts as "overlap" (a dab's feather reaches past its own radius too)
const STAMP_OFFSET_SEARCH_DEGREES = [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 105, -105, 120, -120, 135, -135, 150, -150, 165, -165, 180];
const STAMP_DIST_MULTS = [1, 1.4, 1.8, 2.2, 2.8, 3.5]; // candidate source distances, as a multiple of the stroke's own nominal distance
const STAMP_DIST_EASE_STEP = 0.15;    // max distance-multiplier change/dab easing back toward 1 (nominal)
const STAMP_ANGLE_EASE_STEP_DEG = 6;  // max degrees/dab a deviated offset eases back toward the stroke's own nominal angle
const STAMP_FEATHER_MIN = 0.15;
const STAMP_FEATHER_MAX = 0.6;
const STAMP_FEATHER_VARIANCE_NORM = 300; // local color variance at/above this maps to the tightest feather
const STAMP_INITIAL_ANGLES = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5];

function angleDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

/* Mean RGB + color variance over a 16-point ring at `radius` around (x,y)
   (image-space), read from a cached { data, width, height, scale } proxy
   buffer. Nearest-neighbor sampling (not bilinear) — this only informs a
   heuristic decision, not the actual blended pixels. Returns null off-buffer. */
function sampleAreaStats(pixels, x, y, radius) {
  if (!pixels) return null;
  const { data, width, height, scale } = pixels;
  const bx = x * scale, by = y * scale, br = Math.max(1, radius * scale);
  const N = 16;
  let sr = 0, sg = 0, sb = 0, sr2 = 0, sg2 = 0, sb2 = 0, n = 0;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const px = Math.round(bx + Math.cos(a) * br), py = Math.round(by + Math.sin(a) * br);
    if (px < 0 || py < 0 || px >= width || py >= height) continue;
    const o = (py * width + px) * 4;
    const r = data[o], g = data[o + 1], b = data[o + 2];
    sr += r; sg += g; sb += b; sr2 += r * r; sg2 += g * g; sb2 += b * b; n++;
  }
  if (!n) return null;
  const mr = sr / n, mg = sg / n, mb = sb / n;
  const variance = (sr2 / n - mr * mr + (sg2 / n - mg * mg) + (sb2 / n - mb * mb)) / 3;
  return { mean: [mr, mg, mb], variance };
}

/* Chooses a NEW stroke's nominal source-offset DIRECTION (distance stays
   fixed at STAMP_SOURCE_OFFSET_MULT × radius). Always defaulting to due
   horizontal is fine when the thing being traced runs roughly vertically
   (the offset lands beside it, in clean surroundings) but actively wrong
   when it runs roughly horizontally — a wire, say — since the offset then
   points ALONG the wire's own length, so the "source" is just more of the
   same wire (or the pole it runs into), not the clean sky beside it.
   Candidates are scored by local variance measured AT the candidate
   itself: a clean patch of background reads as low variance, a patch
   still sitting on a thin linear feature reads as high variance no matter
   which color that feature is — so this works without any real edge or
   orientation detection. Falls back to the old fixed-horizontal default
   when the proxy buffer isn't ready yet or no candidate clearly wins. */
function pickInitialOffset(pixels, point, dist, obstacles, radius, iw, ih) {
  const fallback = { dx: dist, dy: 0 };
  if (!pixels) return fallback;
  const overlapsAny = (sx, sy) =>
    obstacles.some((q) => Math.hypot(sx - q.x, sy - q.y) < Math.max(radius, q.radius) * STAMP_OVERLAP_MARGIN);
  let best = null, bestScore = Infinity;
  for (const deg of STAMP_INITIAL_ANGLES) {
    const angle = (deg * Math.PI) / 180;
    const cx = point.x + Math.cos(angle) * dist, cy = point.y + Math.sin(angle) * dist;
    if (cx < 0 || cy < 0 || cx > iw || cy > ih || overlapsAny(cx, cy)) continue;
    const stats = sampleAreaStats(pixels, cx, cy, radius * 0.6);
    const score = stats ? stats.variance : 0;
    if (score < bestScore) { bestScore = score; best = { dx: cx - point.x, dy: cy - point.y }; }
  }
  return best || fallback;
}

/* Picks this dab's source offset, and the effective {angle, distMult} to
   hand to the NEXT dab's call as `preferred` (both start at the stroke's
   own nominal angle/distMult=1). `obstacles` is every {x,y,radius} dab
   that already exists ANYWHERE on the photo — every OTHER stroke's own
   points (a later stroke's rigid offset can just as easily sample from an
   earlier, nearby stroke's already-stamped destination as from its own —
   two wires stamped close together is exactly this) plus this stroke's
   own points added so far in the SAME drag gesture.

   Three things this deliberately does NOT do, each because an earlier
   version did and it read as a new, invented artifact rather than a clean
   removal:
   1. Score candidates only against the stroke's ORIGINAL nominal angle.
      That let two adjacent dabs independently jump to two very different
      candidate angles whenever their own local overlap situations
      differed slightly — each individually "avoids overlap", but
      consecutive dabs sampling unrelated patches of the photo reads as
      banding/ghosting along the stroke. Scoring against `preferred` (what
      the PREVIOUS dab actually used) instead keeps any deviation as small
      and as continuous as possible dab-to-dab.
   2. Snap an already-deviated stroke straight back to nominal the instant
      it's clear to do so. Instead it eases back gradually — at most
      STAMP_ANGLE_EASE_STEP_DEG / STAMP_DIST_EASE_STEP per dab — and only
      if the eased position is itself still clear.
   3. Treat a ROTATED angle as an equally good fix as a LONGER distance
      along the SAME angle. Rotating samples a genuinely different part of
      the photo — risky even once technically non-overlapping (a nearby
      cloud edge, different lighting). The single most common way this
      whole thing gets triggered is tracing both edges of a linear defect
      (a wire) right next to each other to fully cover its width — the
      return pass is essentially "the same clean patch, just needs to
      reach a little further" for continuity with the first pass, not a
      different patch. distMult candidates are scored far cheaper than
      angle-rotation candidates so that's preferred whenever it's enough. */
function pickDabOffset(stroke, obstacles, point, preferred, pixels, iw, ih) {
  const nominalAngle = Math.atan2(stroke.dy, stroke.dx);
  const nominalDist = Math.hypot(stroke.dx, stroke.dy);
  if (!(nominalDist > 0)) return { angle: nominalAngle, distMult: 1, dx: stroke.dx, dy: stroke.dy };

  const overlapsAny = (sx, sy) =>
    obstacles.some((q) => Math.hypot(sx - q.x, sy - q.y) < Math.max(stroke.radius, q.radius) * STAMP_OVERLAP_MARGIN);
  const at = (angle, distMult) => ({ x: point.x + Math.cos(angle) * nominalDist * distMult, y: point.y + Math.sin(angle) * nominalDist * distMult });

  let { angle, distMult } = preferred;
  let p = at(angle, distMult);
  const stillClear = !overlapsAny(p.x, p.y);
  // only ease toward nominal while the CURRENT, un-eased vector is itself
  // still comfortably clear — easing regardless of that let it wander
  // into a boundary it couldn't yet see coming (a wire running
  // near-parallel to another for a while has a roughly constant safe
  // region, not a shrinking one), which then needed a fresh search a dab
  // or two later — a small jump, but an avoidable one.
  if (stillClear) {
    const angleGap = angleDiff(nominalAngle, angle), distGap = 1 - distMult;
    if (Math.abs(angleGap) > 1e-6 || Math.abs(distGap) > 1e-6) {
      const easedAngle = angle + Math.sign(angleGap) * Math.min(Math.abs(angleGap), (STAMP_ANGLE_EASE_STEP_DEG * Math.PI) / 180);
      const easedDist = distMult + Math.sign(distGap) * Math.min(Math.abs(distGap), STAMP_DIST_EASE_STEP);
      const pe = at(easedAngle, easedDist);
      if (!overlapsAny(pe.x, pe.y)) { angle = easedAngle; distMult = easedDist; p = pe; }
    }
    return { angle, distMult, dx: p.x - point.x, dy: p.y - point.y };
  }

  // the preferred vector itself no longer clears — search near IT, not the
  // stroke's original nominal angle/distance, so the fix stays local.
  const destStats = sampleAreaStats(pixels, point.x, point.y, stroke.radius);
  let best = null, bestScore = Infinity;
  for (const dm of STAMP_DIST_MULTS) {
    for (const deg of STAMP_OFFSET_SEARCH_DEGREES) {
      const cand = preferred.angle + (deg * Math.PI) / 180;
      const cp = at(cand, dm);
      if (cp.x < 0 || cp.y < 0 || cp.x > iw || cp.y > ih || overlapsAny(cp.x, cp.y)) continue;
      let score = Math.abs(deg) + (dm - 1) * 10;
      const srcStats = destStats && sampleAreaStats(pixels, cp.x, cp.y, stroke.radius);
      if (destStats && srcStats) {
        const [dr, dg, db] = destStats.mean, [sr, sg, sb] = srcStats.mean;
        score += Math.hypot(dr - sr, dg - sg, db - sb) * 0.6;
      }
      if (score < bestScore) { bestScore = score; best = { angle: cand, distMult: dm, dx: cp.x - point.x, dy: cp.y - point.y }; }
    }
  }
  // every candidate overlaps SOMETHING (a dense junction where several
  // strokes converge close together, e.g. wires meeting at a pole) — pick
  // whichever in-bounds candidate overlaps the LEAST (smallest penetration
  // past the overlap threshold) rather than silently keep using `preferred`
  // as-is, which can be the worst possible pick (squarely on top of
  // another stroke, or on structure that was never a reasonable source).
  if (!best) {
    let leastBadPenetration = Infinity;
    for (const dm of STAMP_DIST_MULTS) {
      for (const deg of STAMP_OFFSET_SEARCH_DEGREES) {
        const cand = preferred.angle + (deg * Math.PI) / 180;
        const cp = at(cand, dm);
        if (cp.x < 0 || cp.y < 0 || cp.x > iw || cp.y > ih) continue;
        const penetration = obstacles.reduce((worst, q) => {
          const threshold = Math.max(stroke.radius, q.radius) * STAMP_OVERLAP_MARGIN;
          return Math.max(worst, threshold - Math.hypot(cp.x - q.x, cp.y - q.y));
        }, 0);
        if (penetration < leastBadPenetration) { leastBadPenetration = penetration; best = { angle: cand, distMult: dm, dx: cp.x - point.x, dy: cp.y - point.y }; }
      }
    }
  }
  return best || { angle, distMult, dx: p.x - point.x, dy: p.y - point.y };
}

/* New-stroke feather default: wider on smooth/low-variance surroundings
   (sky, out-of-focus background — a soft edge disappears easily there),
   tighter on detailed/high-variance ones (foliage, brick — a wide feather
   would blur real detail into the blend instead of hiding a seam). Only
   ever runs once, for a stroke's first dab — later dabs of the same
   stroke keep whatever this decided, same as a real brush doesn't change
   softness mid-stroke. */
function autoFeather(pixels, point, radius, fallback) {
  const stats = sampleAreaStats(pixels, point.x, point.y, radius);
  if (!stats) return fallback;
  const t = Math.min(1, stats.variance / STAMP_FEATHER_VARIANCE_NORM);
  return STAMP_FEATHER_MAX - t * (STAMP_FEATHER_MAX - STAMP_FEATHER_MIN);
}

/* Resamples a stroke's own recorded points to a fixed, dense arc-length
   spacing — purely for feeding strokeRing's disk-union circle centers, never
   touching stroke.points itself. The PAINTED dab spacing (STAMP_DAB_SPACING)
   is tuned for the clone algorithm's own cost, not for how smooth a disk
   union of that many circles looks once zoomed in close — a fast drag can
   also leave the recorded points coarser than that nominal spacing (the
   interpolation fills gaps by DISTANCE, but a quick flick can still land
   points further apart than intended). A disk union needs roughly even,
   closely-spaced centers to read as one smooth capsule rather than a chain
   of visibly separate circles — resampling first guarantees that regardless
   of how the actual paint-time points happened to land. Capped to a max
   point count so a very long stroke doesn't force rendering thousands of
   SVG circles. */
function resampleForOutline(points, radius) {
  if (points.length < 2) return points;
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  }
  const MAX_OUTLINE_POINTS = 400;
  const spacing = Math.max(radius * 0.25, total / MAX_OUTLINE_POINTS);
  const out = [points[0]];
  let prev = points[0];
  let dist = 0;
  for (let i = 1; i < points.length; i++) {
    let cur = points[i];
    let segLen = Math.hypot(cur.x - prev.x, cur.y - prev.y);
    while (dist + segLen >= spacing) {
      const t = (spacing - dist) / segLen;
      prev = { x: prev.x + (cur.x - prev.x) * t, y: prev.y + (cur.y - prev.y) * t };
      out.push(prev);
      segLen = Math.hypot(cur.x - prev.x, cur.y - prev.y);
      dist = 0;
    }
    dist += segLen;
    prev = cur;
  }
  out.push(points[points.length - 1]);
  return out;
}

/* Clone-stamp brush painting overlay — rendered only in mode === "stamp",
   as a sibling of the (pointer-events:none) .imgspace/GLImage layers inside
   the same .viewclip box view mode already uses. `stamp` is an array of
   brush strokes (electron/stampTool.mjs): { points:[{x,y}], radius, dx, dy }
   — a path of small circular dabs sharing one radius and one rigid source
   offset (drag the source handle once, it applies to the whole stroke,
   same as Photoshop's Clone Stamp / Lightroom's Heal brush).
   screenToImage/imageToLocal reuse the exact same rot/irot + view.k/theta
   conversion the crop tool's own drag math and deep-zoom tile fetch use. */
/* Closest distance from (px,py) to the segment a->b — same projection-and-
   clamp technique as electron/stampTool.mjs's own closestOnSegment, used
   here for StampOverlay's stroke hit-testing (see nearestPointDist) so
   "can I grab this part of the stroke" matches "is this part of the
   stroke actually painted" exactly, rather than a coarser approximation. */
function distToSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax, aby = by - ay;
  const lenSq = abx * abx + aby * aby;
  let t = lenSq > 0 ? ((px - ax) * abx + (py - ay) * aby) / lenSq : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(px - (ax + abx * t), py - (ay + aby * t));
}

function StampOverlay({ stamp, setStamp, draft, setDraft, selectedStroke, setSelectedStroke, snapshot, view, rect, iw, ih, stageRef, toast, brushRadius, feather, opacity, gesturing, editPixelsRef, spaceRef, stampMode }) {
  const screenToImage = useCallback((clientX, clientY) => {
    const box = stageRef.current.getBoundingClientRect();
    const sx = clientX - box.left, sy = clientY - box.top;
    const [ux, uy] = irot(sx - view.icx, sy - view.icy, rad(view.theta));
    return { x: ux / view.k + iw / 2, y: uy / view.k + ih / 2 };
  }, [view, iw, stageRef]);

  const imageToLocal = useCallback((x, y) => {
    const [ox, oy] = rot(x - iw / 2, y - ih / 2, rad(view.theta));
    return { left: view.icx + ox * view.k - rect.x, top: view.icy + oy * view.k - rect.y };
  }, [view, iw, rect]);

  const clampToImage = (x, y) => [Math.min(Math.max(x, 0), iw), Math.min(Math.max(y, 0), ih)];

  /* Space bar = temporary hand tool, same convention as Photoshop/Lightroom:
     hold it to pan (and double-click to zoom-to-fit) without switching out
     of the stamp brush. The canonical down/up state (`spaceRef`, a prop —
     see Editor) is now tracked in Editor, not here, since crop mode needs
     to read it too and has no StampOverlay mounted at all; this effect only
     handles the cursor-style side effects that ARE local to this overlay. */
  const overlayRef = useRef(null);
  useEffect(() => {
    const down = (ev) => {
      if (ev.code !== "Space" || ev.repeat) return;
      if (overlayRef.current) overlayRef.current.style.cursor = "grab";
      hideCursor();
    };
    const up = (ev) => {
      if (ev.code !== "Space") return;
      if (overlayRef.current) overlayRef.current.style.cursor = "none";
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Brush-size cursor preview: a circle matching the current brush radius
     that follows the pointer, so sizing is visible before the first dab
     lands. Position is written straight to the DOM on every move (bypasses
     React state — the same pattern the codebase already uses for drag.current
     elsewhere) so hovering doesn't cause a re-render per pixel of movement;
     only the circle's diameter is a normal React-driven style, since that
     only changes when the brush-size slider or the zoom level changes.
     Hidden while space is held (hand tool, not painting) or while a wheel/
     trackpad zoom gesture is in progress — trackpad pinch-zoom can report
     tiny incidental pointer movement, which otherwise reads as the brush
     cursor jittering around while the user is only trying to zoom. */
  const cursorRef = useRef(null);
  const onOverlayMove = (e) => {
    const el = cursorRef.current;
    if (!el) return;
    if (spaceRef.current || gesturing) { el.style.opacity = "0"; return; }
    const box = stageRef.current.getBoundingClientRect();
    const p = screenToImage(e.clientX, e.clientY);
    // hovering an existing stroke would grab+move it on click, not paint —
    // show a grab cursor and hide the brush-size circle there instead
    const overStroke = stamp.some((s) => nearestPointDist(s, p.x, p.y) <= s.radius);
    if (overlayRef.current) overlayRef.current.style.cursor = overStroke ? "grab" : "none";
    if (overStroke) { el.style.opacity = "0"; return; }
    el.style.left = `${e.clientX - box.left - rect.x}px`;
    el.style.top = `${e.clientY - box.top - rect.y}px`;
    el.style.opacity = "1";
  };
  const showCursor = () => { if (cursorRef.current && !spaceRef.current && !gesturing) cursorRef.current.style.opacity = "1"; };
  const hideCursor = () => { if (cursorRef.current) cursorRef.current.style.opacity = "0"; };

  // The stroke currently being drawn/extended — kept OUT of `stamp` itself
  // until pointerup (undo/redo and persistence only ever see a finished
  // stroke), but lifted up to Editor (not local state here) so the
  // sibling GLImage instances can fold it into their own live composite —
  // see GLImage's uploadTexture. Drives the destination-outline preview
  // below AND, via that lift, the actual live cloned-pixel preview; no
  // source ring/handles until the stroke commits (those still only make
  // sense for a finished, selectable stroke).
  const totalDabs = stamp.reduce((n, s) => n + s.points.length, 0);
  // Distance to the stroke's own CAPSULE (nearest point on any segment of
  // its polyline, matching applyStrokeToBuffer's own closest-segment
  // footprint test in electron/stampTool.mjs) — not just distance to its
  // stored points. A stroke's visible/painted coverage is the whole swept
  // capsule, including the area BETWEEN two points, not a chain of small
  // circles centered exactly on each stored point; checking only stored
  // points meant part of what's visibly stamped (anywhere more than one
  // point's own radius from the nearest single point, but still within the
  // capsule between two points) couldn't be clicked to grab/move the
  // stroke, even though the exact same spot could have been clicked to
  // start painting a stamp there in the first place.
  const nearestPointDist = (stroke, x, y) => {
    const pts = stroke.points;
    if (pts.length < 2) return Math.hypot(pts[0].x - x, pts[0].y - y);
    let min = Infinity;
    for (let i = 0; i < pts.length - 1; i++) {
      const d = distToSegment(x, y, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y);
      if (d < min) min = d;
    }
    return min;
  };

  const updateStroke = (index, patch) => {
    setStamp((prev) => prev.map((s, i) => (i === index ? { ...s, ...patch } : s)));
  };

  const beginWindowDrag = (onMove, onUp) => {
    const move = (ev) => onMove(ev);
    const up = (ev) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (onUp) onUp(ev);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
  };

  const removeStroke = (i) => {
    snapshot();
    setStamp((prev) => prev.filter((_, idx) => idx !== i));
    setSelectedStroke(null);
  };

  const onHandleDown = (e, index, part) => {
    e.preventDefault(); e.stopPropagation();
    setSelectedStroke(index);
    snapshot();
    const startStroke = stamp[index];
    const anchor = startStroke.points[0];
    const startPt = screenToImage(e.clientX, e.clientY);
    if (part === "move") {
      beginWindowDrag((ev) => {
        const cur = screenToImage(ev.clientX, ev.clientY);
        const ddx = cur.x - startPt.x, ddy = cur.y - startPt.y;
        updateStroke(index, { points: startStroke.points.map((p) => ({ x: p.x + ddx, y: p.y + ddy })) });
      });
    } else if (part === "src") {
      beginWindowDrag((ev) => {
        const cur = screenToImage(ev.clientX, ev.clientY);
        updateStroke(index, { dx: startStroke.dx + (cur.x - startPt.x), dy: startStroke.dy + (cur.y - startPt.y) });
      });
    } else if (part === "resize") {
      beginWindowDrag((ev) => {
        const cur = screenToImage(ev.clientX, ev.clientY);
        const radius = Math.max(STAMP_MIN_RADIUS, Math.hypot(cur.x - anchor.x, cur.y - anchor.y));
        updateStroke(index, { radius });
      });
    }
  };

  /* A single, deliberate drag along a diagonal line can arrive as several
     separate pointerdown/pointerup pairs instead of one continuous
     press-move-release (observed with trackpad click-drag, where a brief
     pressure dip mid-swipe reads to the OS as a release+re-press) — without
     this, one intended brush stroke fractures into several disconnected
     strokes, each getting its own (wrong, default-offset) source. If a new
     pointerdown lands soon after and close to where the previous stroke's
     last dab was, continue that stroke instead of starting a new one. */
  const lastStrokeEndRef = useRef(null); // { index, point: {x,y}, at: timestamp }
  const STROKE_CONTINUE_MS = 400;
  const STROKE_CONTINUE_DIST_MULT = 3; // × brushRadius

  const onOverlayDown = (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    if (spaceRef.current) return; // hand tool: let it bubble to .stage's own pan-drag

    e.preventDefault();
    e.stopPropagation();

    // clicking an existing stroke's path selects + drags it, rather than
    // starting a new stroke on top of it
    const p0 = screenToImage(e.clientX, e.clientY);
    const hitIndex = stamp.findIndex((s) => nearestPointDist(s, p0.x, p0.y) <= s.radius);
    if (hitIndex !== -1) {
      onHandleDown(e, hitIndex, "move");
      return;
    }

    const cont = lastStrokeEndRef.current;
    const canContinue =
      cont &&
      cont.index < stamp.length &&
      performance.now() - cont.at < STROKE_CONTINUE_MS &&
      Math.hypot(p0.x - cont.point.x, p0.y - cont.point.y) < brushRadius * STROKE_CONTINUE_DIST_MULT;

    // every OTHER stroke's dabs count as obstacles too, not just this
    // stroke's own — two strokes stamped near each other (two wires a
    // few pixels apart) can sample from one another just as easily as a
    // single curved stroke can sample from itself.
    const otherStrokesObstacles = (excludeIndex) =>
      stamp.flatMap((s, i) => (i === excludeIndex ? [] : s.points.map((p) => ({ x: p.x, y: p.y, radius: s.radius }))));
    // Heal has no user-chosen source: electron/stampTool.mjs findHealSource
    // picks it from the photo (on the cached proxy buffer, then scaled back
    // to original-image px). null if the buffer isn't decoded yet or nothing
    // fits — callers keep whatever offset they already had.
    const autoHealOffset = (points, radius, strokeFeatherVal, excludeIndex) => {
      const px = editPixelsRef.current;
      if (!px || !points.length) return null;
      const s = px.scale;
      const found = findHealSource(
        px.data, px.width, px.height, 4,
        points.map((p) => ({ x: p.x * s, y: p.y * s })), radius * s, strokeFeatherVal,
        otherStrokesObstacles(excludeIndex).map((o) => ({ x: o.x * s, y: o.y * s, radius: o.radius * s })),
      );
      return found ? { dx: found.dx / s, dy: found.dy / s } : null;
    };

    // strokeMeta (dx/dy/radius), obstacles (a plain mutable list of every
    // existing dab, own-stroke and other-strokes alike, updated in
    // lockstep with `last` below rather than read back from React state),
    // and preferred (this stroke's current effective {angle, distMult}, so
    // it can ease smoothly rather than jump per dab — see pickDabOffset)
    // are local for the whole gesture regardless of when it commits.
    // `newPoints` accumulates ONLY this gesture's own new dabs — never
    // written to `stamp` until pointerup (see beginWindowDrag's onUp
    // below); `setDraft` drives the destination-only outline preview in
    // the meantime, so no cloned pixels appear and no source ring/handles
    // show until the stroke is actually finished.
    let index, isNew, last, strokeMeta, obstacles, preferred, newPoints, strokeFeather;
    // fixed for the whole gesture (a mid-drag H press must not split a
    // stroke into two modes); undefined = Clone, the stored default
    const strokeMode = stampMode === "clone" ? undefined : stampMode;
    // a NEW stroke follows the brush mode; continuing one keeps its own
    let healStroke = strokeMode === "heal";
    let removeStroke = strokeMode === "remove";
    // heal: one automatic offset for the whole stroke, re-searched as it
    // grows; remove: no source at all. Neither uses per-dab overrides —
    // those exist for Clone's manual source.
    const placeDab = (np) => {
      if (healStroke || removeStroke) return np;
      const result = pickDabOffset(strokeMeta, obstacles, np, preferred, editPixelsRef.current, iw, ih);
      preferred = result;
      const isNominal = Math.hypot(result.dx - strokeMeta.dx, result.dy - strokeMeta.dy) < 0.01;
      return isNominal ? np : { ...np, dx: result.dx, dy: result.dy };
    };
    const allPoints = () => (isNew ? newPoints : [...stamp[index].points.map((p) => ({ x: p.x, y: p.y })), ...newPoints]);
    const reHeal = () => {
      if (!healStroke) return;
      const o = autoHealOffset(allPoints(), strokeMeta.radius, strokeFeather, index);
      if (o) strokeMeta = { ...strokeMeta, dx: o.dx, dy: o.dy };
    };
    if (canContinue) {
      index = cont.index;
      isNew = false;
      last = cont.point;
      const existing = stamp[index];
      healStroke = existing.mode === "heal";
      removeStroke = existing.mode === "remove";
      strokeFeather = existing.feather;
      strokeMeta = { dx: existing.dx, dy: existing.dy, radius: existing.radius };
      preferred = { angle: Math.atan2(strokeMeta.dy, strokeMeta.dx), distMult: 1 };
      obstacles = [
        ...otherStrokesObstacles(index),
        ...existing.points.map((p) => ({ x: p.x, y: p.y, radius: strokeMeta.radius })),
      ];
      newPoints = [];
      setSelectedStroke(index);
      const [cx, cy] = clampToImage(p0.x, p0.y);
      if (Math.hypot(cx - last.x, cy - last.y) >= brushRadius * STAMP_DAB_SPACING) {
        const finalPoint = placeDab({ x: cx, y: cy });
        last = { x: cx, y: cy };
        obstacles.push({ x: finalPoint.x, y: finalPoint.y, radius: strokeMeta.radius });
        newPoints = [finalPoint];
      }
      reHeal();
      setDraft({ index, isNew: false, points: newPoints, radius: strokeMeta.radius, dx: strokeMeta.dx, dy: strokeMeta.dy });
    } else {
      setSelectedStroke(null);
      if (totalDabs >= MAX_STAMP_DABS) {
        if (toast) toast(`Clone stamp is limited to ${MAX_STAMP_DABS} dabs per photo`);
        return;
      }
      const [sx, sy] = clampToImage(p0.x, p0.y);
      index = stamp.length;
      isNew = true;
      const initialObstacles = otherStrokesObstacles(index);
      // direction picked contextually (see pickInitialOffset) instead of
      // always defaulting to due horizontal — that default only happens to
      // work when the thing being traced runs roughly vertically.
      const initOffset = pickInitialOffset(editPixelsRef.current, { x: sx, y: sy }, brushRadius * STAMP_SOURCE_OFFSET_MULT, initialObstacles, brushRadius, iw, ih);
      strokeMeta = { dx: initOffset.dx, dy: initOffset.dy, radius: brushRadius };
      preferred = { angle: Math.atan2(strokeMeta.dy, strokeMeta.dx), distMult: 1 };
      const firstPoint = { x: sx, y: sy };
      obstacles = [...initialObstacles, { x: sx, y: sy, radius: brushRadius }];
      // auto-feather: decided once, from what's actually around the first
      // dab, instead of the flat session default — see autoFeather's own
      // comment. Falls back to the session slider value if the proxy
      // buffer hasn't finished decoding yet.
      strokeFeather = autoFeather(editPixelsRef.current, firstPoint, brushRadius, feather);
      newPoints = [firstPoint];
      last = firstPoint;
      if (removeStroke) strokeMeta = { ...strokeMeta, dx: 0, dy: 0 };
      reHeal();
      setDraft({ index, isNew: true, points: newPoints, radius: brushRadius, dx: strokeMeta.dx, dy: strokeMeta.dy, feather: strokeFeather, opacity, mode: strokeMode });
    }

    beginWindowDrag((ev) => {
      const cur = screenToImage(ev.clientX, ev.clientY);
      const [cx, cy] = clampToImage(cur.x, cur.y);
      const spacing = brushRadius * STAMP_DAB_SPACING;
      const dist = Math.hypot(cx - last.x, cy - last.y);
      // a fast drag can jump much further than one dab-spacing between two
      // consecutive pointermove events (the browser simply samples less
      // often than the mouse moves) — placing only one dab at the new
      // position, as this used to do, left a gap the eye reads as separate
      // circles instead of one continuous stroke. Resampling the straight
      // segment from the last dab to here at fixed spacing fills that gap
      // regardless of how far a single tick's jump was.
      const steps = Math.floor(dist / spacing);
      if (steps < 1) return;
      const interpolated = [];
      for (let i = 1; i <= steps; i++) {
        const t = (i * spacing) / dist;
        const np = { x: last.x + (cx - last.x) * t, y: last.y + (cy - last.y) * t };
        const finalPoint = placeDab(np);
        interpolated.push(finalPoint);
        obstacles.push({ x: finalPoint.x, y: finalPoint.y, radius: strokeMeta.radius });
      }
      last = { x: interpolated[interpolated.length - 1].x, y: interpolated[interpolated.length - 1].y };
      newPoints = [...newPoints, ...interpolated];
      reHeal();
      setDraft((d) => (d ? { ...d, points: newPoints, dx: strokeMeta.dx, dy: strokeMeta.dy } : d));
    }, () => {
      lastStrokeEndRef.current = { index, point: last, at: performance.now() };
      if (newPoints.length === 0) { setDraft(null); return; }
      if (isNew) {
        snapshot();
        const dabCount = stamp.reduce((n, s) => n + s.points.length, 0);
        const room = MAX_STAMP_DABS - dabCount;
        const toAdd = newPoints.slice(0, Math.max(0, room));
        if (toAdd.length === 0) { setDraft(null); return; }
        setStamp((prev) => [...prev, { points: toAdd, radius: strokeMeta.radius, dx: strokeMeta.dx, dy: strokeMeta.dy, feather: strokeFeather, opacity, ...(strokeMode ? { mode: strokeMode } : null) }]);
      } else {
        setStamp((prev) => {
          const dabCount = prev.reduce((n, s) => n + s.points.length, 0);
          const room = MAX_STAMP_DABS - dabCount;
          if (room <= 0) return prev;
          const toAdd = newPoints.slice(0, room);
          // heal: the whole stroke's automatic source was re-searched (reHeal)
          return prev.map((s, i) => (i === index ? { ...s, points: [...s.points, ...toAdd], ...(healStroke ? { dx: strokeMeta.dx, dy: strokeMeta.dy } : null) } : s));
        });
      }
      setSelectedStroke(index);
      setDraft(null);
    });
  };

  const onOverlayDoubleClick = (e) => {
    if (spaceRef.current) return; // hand tool: let it bubble to .stage's own zoom-to-fit
    e.preventDefault(); e.stopPropagation();
    const p0 = screenToImage(e.clientX, e.clientY);
    const hitIndex = stamp.findIndex((s) => nearestPointDist(s, p0.x, p0.y) <= s.radius);
    if (hitIndex !== -1) removeStroke(hitIndex);
  };

  /* Hollow outline of a stroke's silhouette, drawn as the boundary of the
     UNION of a filled circle at every dab — one continuous "capsule" shape,
     not a separate circle at each end plus a ribbon between them (a real
     brush stroke's ends are just its first/last dab, already circles).
     Built via a <mask> the same way the very first version of this overlay
     was, but with one crucial difference: the mask's shapes are plain
     filled CIRCLES per dab, never a stroked centerline. Overlapping filled
     circles always union into one clean blob with no interior seam, no
     matter how a path loops back over itself (scrubbing the brush, or
     re-entering near its own start) — that's what a self-crossing STROKED
     line failed at (it produced broken/garbled artifacts, since a stroke's
     own outline is a more complex shape that a self-intersecting centerline
     can render inconsistently). A disk union has no such failure mode, and
     it automatically gives just the one true OUTER boundary even when the
     path loops back on itself — no inner-loop seams to filter out.

     The band is a thin two-tone contour — a slightly wider dark casing
     directly behind a thinner bright line, both centered tightly on the
     TRUE silhouette radius (rPx) — not two independently-sized bands. An
     earlier version scaled the dark casing's OUTER edge outward by up to
     8% of rPx with no matching cap, so on a large/zoomed-in stroke the
     "outline" visibly bulged past the stroke's real edge and ate a good
     chunk of its interior — reported directly ("аутлайн гигантский на
     контуре"). Both bands now derive from the same small `ringW`, so the
     whole contour stays a few px thin and hugs the real boundary
     regardless of stroke size or zoom. */
  const strokeRing = (idPrefix, stroke, dx, dy, rPx, ringW, color) => {
    const centers = resampleForOutline(stroke.points, stroke.radius).map((p) => imageToLocal(p.x + dx, p.y + dy));
    // the hole punched out of the middle must NEVER reach (let alone clamp
    // to) 0 — that would leave the outer circle fully unmasked, rendering
    // as a SOLID filled disc instead of a ring. `Math.max(0.1, …)` below
    // guarantees a hole always exists even for a tiny rPx.
    const lineHalf = ringW * 0.5;
    const casingHalf = lineHalf + 1;
    const casingOuter = rPx + casingHalf;
    const casingInner = Math.max(0.1, rPx - casingHalf);
    const lineOuter = rPx + lineHalf;
    const lineInner = Math.max(0.1, rPx - lineHalf);
    const haloId = `${idPrefix}-halo`, ringId = `${idPrefix}-ring`;
    return (
      <g>
        <mask id={haloId}>
          {centers.map((c, k) => <circle key={`ho${k}`} cx={c.left} cy={c.top} r={casingOuter} fill="#fff" />)}
          {centers.map((c, k) => <circle key={`hi${k}`} cx={c.left} cy={c.top} r={casingInner} fill="#000" />)}
        </mask>
        <g mask={`url(#${haloId})`}>
          {centers.map((c, k) => <circle key={`hc${k}`} cx={c.left} cy={c.top} r={casingOuter} fill="rgba(0,0,0,.6)" />)}
        </g>
        <mask id={ringId}>
          {centers.map((c, k) => <circle key={`ro${k}`} cx={c.left} cy={c.top} r={lineOuter} fill="#fff" />)}
          {centers.map((c, k) => <circle key={`ri${k}`} cx={c.left} cy={c.top} r={lineInner} fill="#000" />)}
        </mask>
        <g mask={`url(#${ringId})`}>
          {centers.map((c, k) => <circle key={`rc${k}`} cx={c.left} cy={c.top} r={lineOuter} fill={color} />)}
        </g>
      </g>
    );
  };

  const cursorPx = brushRadius * view.k * 2;
  return (
    <div
      ref={overlayRef}
      className="stamp-overlay"
      onPointerDown={onOverlayDown}
      onPointerMove={onOverlayMove}
      onPointerEnter={showCursor}
      onPointerLeave={hideCursor}
      onDoubleClick={onOverlayDoubleClick}
    >
      <div
        ref={cursorRef}
        className="stamp-cursor"
        style={{ width: cursorPx, height: cursorPx, opacity: 0 }}
      />
      <svg className="stamp-svg">
        {stamp.map((stroke, i) => {
          // a stroke actively being EXTENDED (continuation drag in
          // progress — see `draft`) shows just its destination outline,
          // same as everything else while the mouse is held; the source
          // ring/line/handles reappear once that drag commits.
          const draftActiveHere = draft && !draft.isNew && draft.index === i;
          const isSel = i === selectedStroke && !draftActiveHere;
          const rPx = stroke.radius * view.k;
          // ring/halo thickness scale down for a small brush radius, capped
          // for a large one — a fixed pixel width made a thin brush (e.g.
          // tracing a pole/wire) look almost solid rather than hollow.
          const ringW = Math.max(1.25, Math.min(2.25, rPx * 0.05));
          // a plain white outline all but vanishes against a bright sky —
          // a light cyan reads clearly against both light and dark content;
          // Heal strokes get green instead, so Clone vs Heal is visible at a glance
          const dstColor = isSel ? "#f3c34f" : outlineFor(stroke.mode);
          const anchorLocal = imageToLocal(stroke.points[0].x, stroke.points[0].y);
          const srcAnchorLocal = isSel ? imageToLocal(stroke.points[0].x + stroke.dx, stroke.points[0].y + stroke.dy) : null;
          return (
            <g key={i}>
              {strokeRing(`stamp-dst-${i}`, stroke, 0, 0, rPx, ringW, dstColor)}
              {isSel && stroke.mode !== "heal" && stroke.mode !== "remove" && (
                <>
                  {strokeRing(`stamp-src-${i}`, stroke, stroke.dx, stroke.dy, rPx, ringW, "#f3c34f")}
                  <line
                    x1={anchorLocal.left} y1={anchorLocal.top}
                    x2={srcAnchorLocal.left} y2={srcAnchorLocal.top}
                    stroke="rgba(243,195,79,.85)" strokeWidth={1.5}
                  />
                </>
              )}
            </g>
          );
        })}
        {draft && draft.points.length > 0 && (() => {
          const rPx = draft.radius * view.k;
          const ringW = Math.max(1.25, Math.min(2.25, rPx * 0.05));
          return <g key="stamp-draft">{strokeRing("stamp-draft", draft, 0, 0, rPx, ringW, outlineFor(draft.mode))}</g>;
        })()}
      </svg>
      {stamp.map((stroke, i) => {
        if (i !== selectedStroke || (draft && draft.index === i)) return null;
        const rPx = stroke.radius * view.k;
        const anchor = stroke.points[0];
        const anchorLocal = imageToLocal(anchor.x, anchor.y);
        const src = imageToLocal(anchor.x + stroke.dx, anchor.y + stroke.dy);
        // grabbable regardless of how small the brush's own visual circle
        // is on screen — the hit target is only ever the SAME size as or
        // BIGGER than the visual ring, never smaller, so it stays honest
        // about what it does without needing its own separate outline.
        const hitR = Math.max(rPx, STAMP_SRC_HIT_MIN);
        return (
          <div key={i}>
            {/* Heal's source is automatic and Remove has none — nothing to grab */}
            {stroke.mode !== "heal" && stroke.mode !== "remove" && (
              <div
                className="stamp-hit src"
                style={{ left: src.left - hitR, top: src.top - hitR, width: hitR * 2, height: hitR * 2 }}
                onPointerDown={(e) => onHandleDown(e, i, "src")}
              />
            )}
            <div
              className="stamp-handle"
              style={{ left: anchorLocal.left + rPx * Math.SQRT1_2 - 5, top: anchorLocal.top - rPx * Math.SQRT1_2 - 5 }}
              onPointerDown={(e) => onHandleDown(e, i, "resize")}
            />
          </div>
        );
      })}
    </div>
  );
}

/* ============================================================
   White Balance Selector overlay (Lightroom's "W" eyedropper) — shown
   over the photo while picking. A click samples the ORIGINAL pixels (the
   cached proxy, editPixelsRef — the sliders are absolute, so the sample
   must be the input, not the already-balanced preview) averaged over a
   small square in linear light, and hands it to onPick. Space = hand
   tool, Esc (Editor's key handler) cancels.
   ============================================================ */
const WB_SAMPLE_RADIUS = 2; // proxy px each side → a 5×5 average, steadier than one noisy pixel

function WBPickOverlay({ view, iw, ih, stageRef, editPixelsRef, spaceRef, onPick, toast }) {
  const onDown = (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    if (spaceRef.current) return;
    e.preventDefault();
    e.stopPropagation();
    const b = stageRef.current.getBoundingClientRect();
    const [ux, uy] = irot(e.clientX - b.left - view.icx, e.clientY - b.top - view.icy, rad(view.theta));
    const x = ux / view.k + iw / 2, y = uy / view.k + ih / 2;
    if (x < 0 || y < 0 || x >= iw || y >= ih) return;
    const px = editPixelsRef.current;
    if (!px) { toast("Still loading the photo — try again in a moment"); return; }
    const cx = Math.round(x * px.scale), cy = Math.round(y * px.scale);
    const lin = [0, 0, 0];
    let n = 0;
    for (let j = -WB_SAMPLE_RADIUS; j <= WB_SAMPLE_RADIUS; j++) {
      for (let i = -WB_SAMPLE_RADIUS; i <= WB_SAMPLE_RADIUS; i++) {
        const sx = cx + i, sy = cy + j;
        if (sx < 0 || sy < 0 || sx >= px.width || sy >= px.height) continue;
        const o = (sy * px.width + sx) * 4;
        for (let c = 0; c < 3; c++) {
          const v = px.data[o + c] / 255;
          lin[c] += v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        }
        n++;
      }
    }
    if (!n) return;
    // back to an sRGB byte triple (what neutralizeWhiteBalance expects)
    const rgb = lin.map((v) => {
      const l = v / n;
      return 255 * (l <= 0.0031308 ? l * 12.92 : 1.055 * Math.pow(l, 1 / 2.4) - 0.055);
    });
    onPick(rgb);
  };
  return <div className="wbpick-overlay" onPointerDown={onDown} />;
}

/* ============================================================
   Red-eye overlay — rendered only in mode === "redeye". Drag a box
   around an eye (or just click on it: a default-sized box is used) and
   electron/redEyeTool.mjs's detectRedEye() finds the pupil inside it on
   Editor's cached proxy buffer (editPixelsRef — same one StampOverlay
   reads); the fitted ellipse is stored in original-image px. Click an
   existing correction to select it (Size/Darken then edit it, Delete
   removes it), double-click to remove. Space = hand tool, same as Stamp.
   ============================================================ */
const REDEYE_CLICK_BOX = 0.05;   // click without a drag: box side as a fraction of the image's short edge
const REDEYE_DRAG_MIN_PX = 4;    // screen px of movement before a press counts as a box drag
const REDEYE_OUTLINE_STEPS = 48;

function RedEyeOverlay({ redeye, setRedEye, selectedEye, setSelectedEye, snapshot, view, rect, iw, ih, stageRef, toast, editPixelsRef, spaceRef, newSize, newDarken }) {
  const [box, setBox] = useState(null); // in-progress drag, image px { x0, y0, x1, y1 }

  const screenToImage = useCallback((clientX, clientY) => {
    const b = stageRef.current.getBoundingClientRect();
    const [ux, uy] = irot(clientX - b.left - view.icx, clientY - b.top - view.icy, rad(view.theta));
    return { x: ux / view.k + iw / 2, y: uy / view.k + ih / 2 };
  }, [view, iw, ih, stageRef]);
  const imageToLocal = useCallback((x, y) => {
    const [ox, oy] = rot(x - iw / 2, y - ih / 2, rad(view.theta));
    return { left: view.icx + ox * view.k - rect.x, top: view.icy + oy * view.k - rect.y };
  }, [view, iw, ih, rect]);

  // which correction (if any) is under image point p — inside its drawn
  // outline (pupil edge × size, plus a little slack for a thin pupil)
  const hitEye = (p) => {
    for (let i = redeye.length - 1; i >= 0; i--) {
      const e = redeye[i];
      const sx = e.sx * e.size, sy = e.sy * e.size;
      const dx = (p.x - e.x) / sx, dy = (p.y - e.y) / sy;
      const u = Math.sqrt((dx * dx - 2 * e.rho * dx * dy + dy * dy) / (1 - e.rho * e.rho)) / 2;
      if (u <= 1.3) return i;
    }
    return -1;
  };

  const addFromBox = (b) => {
    const px = editPixelsRef.current;
    if (!px) { toast("Still loading the photo — try again in a moment"); return; }
    const s = px.scale;
    const found = detectRedEye(px.data, px.width, px.height, 4, { x0: b.x0 * s, y0: b.y0 * s, x1: b.x1 * s, y1: b.y1 * s });
    if (!found) { toast("No red eye found there — drag a box around the whole eye"); return; }
    snapshot();
    setSelectedEye(redeye.length);
    setRedEye((prev) => [...prev, { x: found.x / s, y: found.y / s, sx: found.sx / s, sy: found.sy / s, rho: found.rho, size: newSize, darken: newDarken }]);
  };

  const onDown = (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    if (spaceRef.current) return; // hand tool: let it bubble to .stage's own pan-drag
    e.preventDefault();
    e.stopPropagation();
    const p0 = screenToImage(e.clientX, e.clientY);
    const hit = hitEye(p0);
    if (hit !== -1) { setSelectedEye(hit); return; }
    setSelectedEye(null);
    const sx0 = e.clientX, sy0 = e.clientY;
    let dragging = false;
    const move = (ev) => {
      if (!dragging && Math.hypot(ev.clientX - sx0, ev.clientY - sy0) < REDEYE_DRAG_MIN_PX) return;
      dragging = true;
      const p = screenToImage(ev.clientX, ev.clientY);
      setBox({ x0: p0.x, y0: p0.y, x1: p.x, y1: p.y });
    };
    const up = (ev) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setBox(null);
      if (dragging) {
        const p = screenToImage(ev.clientX, ev.clientY);
        addFromBox({ x0: Math.min(p0.x, p.x), y0: Math.min(p0.y, p.y), x1: Math.max(p0.x, p.x), y1: Math.max(p0.y, p.y) });
      } else {
        const h = (Math.min(iw, ih) * REDEYE_CLICK_BOX) / 2;
        addFromBox({ x0: p0.x - h, y0: p0.y - h, x1: p0.x + h, y1: p0.y + h });
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const onDouble = (e) => {
    const hit = hitEye(screenToImage(e.clientX, e.clientY));
    if (hit === -1) return;
    e.stopPropagation(); // not a zoom-to-fit
    snapshot();
    setRedEye((prev) => prev.filter((_, i) => i !== hit));
    setSelectedEye(null);
  };

  // ellipse outline: center + 2·size·L·(cos t, sin t), L = Cholesky factor
  // of the stored covariance — so a tilted pupil draws tilted, and the
  // image→screen mapping (incl. view rotation) is applied per point
  const outline = (e) => {
    const r = 2 * e.size;
    const l11 = e.sx, l21 = e.rho * e.sy, l22 = e.sy * Math.sqrt(1 - e.rho * e.rho);
    const pts = [];
    for (let k = 0; k < REDEYE_OUTLINE_STEPS; k++) {
      const t = (k / REDEYE_OUTLINE_STEPS) * Math.PI * 2;
      const c = Math.cos(t), s = Math.sin(t);
      const q = imageToLocal(e.x + r * l11 * c, e.y + r * (l21 * c + l22 * s));
      pts.push(`${q.left},${q.top}`);
    }
    return pts.join(" ");
  };

  let boxEl = null;
  if (box) {
    const corners = [[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]].map(([x, y]) => imageToLocal(x, y));
    boxEl = <polygon points={corners.map((q) => `${q.left},${q.top}`).join(" ")} fill="none" stroke="#fff" strokeWidth="1.2" strokeDasharray="4 3" />;
  }

  return (
    <div className="redeye-overlay" onPointerDown={onDown} onDoubleClick={onDouble}>
      <svg className="stamp-svg">
        {redeye.map((e, i) => (
          <g key={i}>
            <polygon points={outline(e)} fill="none" stroke="rgba(0,0,0,.55)" strokeWidth="3" />
            <polygon points={outline(e)} fill="none" stroke={i === selectedEye ? "var(--accent)" : "#fff"} strokeWidth="1.4" />
          </g>
        ))}
        {boxEl}
      </svg>
    </div>
  );
}

function Editor({ photo, siblings, onClose, onApply, onSwitch, onExport, onExportSelected, onRate, selected, onToggleSel, onRangeSel, onContextExport, suppressKeys, theme, onToggleTheme, toast, busyExport, filmstripHeight, onFilmstripResize }) {
  const stageRef = useRef(null);
  const [stage, setStage] = useState({ w: 0, h: 0 });
  const [rect, setRect] = useState(null);
  const [view, setView] = useState(null); // { icx, icy, k, theta }
  const [aspect, setAspect] = useState("free");
  const [flipped, setFlipped] = useState(false);
  const [mode, setMode] = useState("view"); // 'view' = clean cropped photo, 'crop' = editing UI, 'stamp' = clone stamp / heal, 'redeye' = red-eye
  // normalizeAdjust, not a bare spread over defaultAdjust(): an adjust saved
  // before process versions must keep pv 1 (see colorAdjust.mjs)
  const [adjust, setAdjust] = useState(() => normalizeAdjust(photo.edits?.adjust));
  const [stamp, setStamp] = useState(() => normalizeStamp(photo.edits?.stamp ?? photo.edits?.heal));
  const [selectedStroke, setSelectedStroke] = useState(null); // index into stamp, local-only (not persisted)
  // red-eye corrections (electron/redEyeTool.mjs), original-image px
  const [redeye, setRedEye] = useState(() => normalizeRedEye(photo.edits?.redeye));
  const [selectedEye, setSelectedEye] = useState(null); // index into redeye, local-only
  // new stamp strokes are Clone (raw copy), Heal (automatic source, re-
  // toned) or Remove (content-aware fill, no source) — session-local brush
  // setting like radius/feather; each stroke keeps its own mode
  const [stampMode, setStampMode] = useState("clone");
  /* Clone/Heal/Remove picker (toolbar chips and H). Always sets the BRUSH — what
     every following stroke will be — and, when a stroke is selected, that
     stroke too. Not "selected stroke only": painting auto-selects the
     stroke just made, so a selected-only rule meant picking Heal after
     the first dab silently converted that one stroke and every later
     stroke stayed Clone (found in real use: not a single heal stroke had
     been saved). */
  const pickStampMode = (next) => {
    setStampMode(next);
    if (selectedStroke == null || !stamp[selectedStroke]) return;
    if ((stamp[selectedStroke].mode || "clone") === next) return;
    const heal = next === "heal";
    snapshot();
    // switching TO Heal also replaces the hand-picked source with an
    // automatic one (Heal has no user source); back to Clone keeps it, so
    // it's then draggable from where Heal had put it
    let auto = null;
    const px = editPixelsRef.current;
    if (heal && px) {
      const st = stamp[selectedStroke], k = px.scale;
      const obstacles = stamp.flatMap((s, i) => (i === selectedStroke ? [] : s.points.map((p) => ({ x: p.x * k, y: p.y * k, radius: s.radius * k }))));
      const f = findHealSource(px.data, px.width, px.height, 4, st.points.map((p) => ({ x: p.x * k, y: p.y * k })), st.radius * k, st.feather, obstacles);
      if (f) auto = { dx: f.dx / k, dy: f.dy / k };
    }
    // Remove has no source at all; Clone keeps whatever offset the stroke had
    const noSource = next === "remove" ? { dx: 0, dy: 0, points: stamp[selectedStroke].points.map((p) => ({ x: p.x, y: p.y })) } : null;
    setStamp((prev) => prev.map((s, i) => (i === selectedStroke ? { ...s, mode: next === "clone" ? undefined : next, ...auto, ...noSource } : s)));
  };
  // Size/Darken for the NEXT red-eye correction (a selected one is edited
  // directly instead — same split as the stamp's Feather/Opacity)
  const [newEyeSize, setNewEyeSize] = useState(REDEYE_DEFAULT_SIZE);
  // White Balance Selector (W): picking a neutral point on the photo
  const [wbPicking, setWbPicking] = useState(false);
  // right panel: Develop sliders or file Info
  const [sideTab, setSideTab] = useState("develop");
  const [newEyeDarken, setNewEyeDarken] = useState(REDEYE_DEFAULT_DARKEN);
  // The clone-stamp stroke currently being drawn/extended, while a drag is
  // still down — lifted up here (not local to StampOverlay) so GLImage can
  // fold it into its live composite too; see GLImage's uploadTexture and
  // StampOverlay's own comment above its pointer handlers. Never part of
  // undo/redo history or `currentEdits()` — only a finished, committed
  // stroke in `stamp` itself is either of those things.
  const [draft, setDraft] = useState(null);
  // Before/After: hold to preview the photo with adjust/stamp reverted to
  // their untouched state, crop/rotation left exactly as-is (comparing
  // color+retouch work, not re-flowing the whole framing). State, not a
  // ref like spaceRef — this needs to actually trigger a re-render (GLImage
  // redraws off an `adjust` reference change, see its own [adjust,
  // scheduleDraw] effect), not just change how a pointer handler routes.
  const [compareBefore, setCompareBefore] = useState(false);
  const [gesturing, setGesturing] = useState(false); // GPU-compositing hint during zoom/pan
  const [tile, setTile] = useState(null);            // deep-zoom sharpness overlay
  const [exif, setExif] = useState(null);
  const tileTimer = useRef(null);
  const tileUrl = useRef(null);
  const tileToken = useRef(0);
  const worldRef = useRef(null);
  const flipSeq = useRef(0);
  const flipBusyUntil = useRef(0);
  const [flipReq, setFlipReq] = useState(null);
  const gestTimer = useRef(null);
  const gestureOn = useCallback(() => {
    setGesturing(true);
    clearTimeout(gestTimer.current);
    gestTimer.current = setTimeout(() => setGesturing(false), 240);
  }, []);

  /* Retiring the deep-zoom tile used to be an instant setTile(null): the
     sharp tile vanished the instant a zoom-out fired, popping the view back
     to the (now hugely upscaled) blurry proxy for the first frame of the
     flip animation -- most visible on a double-click zoom-out from a tight
     zoom, where the proxy/tile quality gap is largest. Cross-fading it out
     instead hides that pop behind a quick opacity transition, and the
     motion of the flip itself masks the rest. */
  const fadeOutTile = useCallback(() => {
    clearTimeout(tileTimer.current);
    tileToken.current++;
    setTile((t) => (t && !t.fading ? { ...t, fading: true } : t));
    const url = tileUrl.current;
    tileUrl.current = null;
    if (url) {
      setTimeout(() => {
        setTile((cur) => (cur && cur.fading ? null : cur));
        URL.revokeObjectURL(url);
      }, 180);
    }
  }, []);

  /* refit()'s "FLIP" trick puts a raw, un-React-managed transform on
     .world that animates via a CSS transition running outside any render.
     Any OTHER path that jumps rect/view straight to a new value (wheel
     zoom, drag, rotate, aspect change, undo) races that transition: the
     imgspace inside recomputes its own transform for the new state
     immediately, while .world is still mid-flight from the OLD one, so the
     two compose into a bogus combined transform for a frame or more —
     visible as a jitter, or the image swinging off-screen entirely. Every
     entry point that can change rect/view outside of refit's own animation
     must cancel any in-flight flip first. */
  const cancelFlip = useCallback(() => {
    flipSeq.current++;
    flipBusyUntil.current = 0;
    const el = worldRef.current;
    if (el) {
      el.style.transition = "none";
      el.style.transform = "none";
      el.style.willChange = "auto";
    }
  }, []);

  /* --- crop history (Ctrl+Z / Ctrl+Shift+Z) --- */
  const hist = useRef({ undo: [], redo: [] });
  const snapshot = useCallback(() => {
    if (!rect || !view) return;
    hist.current.undo.push({ rect: { ...rect }, view: { ...view }, aspect, flipped, adjust: { ...adjust }, stamp: [...stamp], redeye: [...redeye] });
    if (hist.current.undo.length > 50) hist.current.undo.shift();
    hist.current.redo = [];
  }, [rect, view, aspect, flipped, adjust, stamp, redeye]);

  const restoreSnap = useCallback((s) => {
    cancelFlip();
    zoomBase.current = s.view.k;
    setRect(s.rect);
    setView(s.view);
    setAspect(s.aspect);
    setFlipped(s.flipped);
    if (s.adjust) setAdjust(s.adjust);
    if (s.stamp) setStamp(s.stamp);
    if (s.redeye) setRedEye(s.redeye);
    setSelectedStroke(null);
    setSelectedEye(null);
  }, [cancelFlip]);

  const undo = useCallback(() => {
    const s = hist.current.undo.pop();
    if (!s || !rect || !view) return;
    hist.current.redo.push({ rect: { ...rect }, view: { ...view }, aspect, flipped, adjust: { ...adjust }, stamp: [...stamp], redeye: [...redeye] });
    restoreSnap(s);
  }, [rect, view, aspect, flipped, adjust, stamp, redeye, restoreSnap]);

  const redo = useCallback(() => {
    const s = hist.current.redo.pop();
    if (!s || !rect || !view) return;
    hist.current.undo.push({ rect: { ...rect }, view: { ...view }, aspect, flipped, adjust: { ...adjust }, stamp: [...stamp], redeye: [...redeye] });
    restoreSnap(s);
  }, [rect, view, aspect, flipped, adjust, stamp, redeye, restoreSnap]);
  const zoomBase = useRef(1);
  const drag = useRef(null);
  const refitTimer = useRef(null);
  const iw = photo.w, ih = photo.h;

  /* Space bar = temporary hand tool (Photoshop/Lightroom convention), read
     by StampOverlay's own pointer handlers AND by the crop-frame/handle
     drag starters below (dragOrPan) so holding it pans the whole workspace
     instead of painting/dragging-the-photo-under-a-fixed-frame/resizing —
     needs to live here, not inside StampOverlay, since crop mode has no
     StampOverlay mounted at all. A ref, not state: changes on every keydown/
     keyup and doesn't need a re-render, only imperative reads. */
  const spaceRef = useRef(false);
  useEffect(() => {
    const down = (ev) => {
      if (ev.code !== "Space" || ev.repeat || suppressKeys) return;
      const tag = ev.target?.tagName;
      if ((tag === "INPUT" && ev.target.type !== "range") || tag === "TEXTAREA") return;
      spaceRef.current = true;
    };
    const up = (ev) => {
      if (ev.code !== "Space") return;
      spaceRef.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [suppressKeys]);

  /* Before/After: hold Backslash (same key Lightroom uses for this exact
     purpose) to compare against the untouched photo — mirrors the Space
     effect just above (down/up pair, same suppressKeys/text-field guard),
     but sets real STATE rather than a ref, since this needs to trigger a
     redraw, not just change how a pointer handler routes. */
  useEffect(() => {
    const down = (ev) => {
      if (ev.code !== "Backslash" || ev.repeat || suppressKeys) return;
      const tag = ev.target?.tagName;
      if ((tag === "INPUT" && ev.target.type !== "range") || tag === "TEXTAREA") return;
      ev.preventDefault();
      setCompareBefore(true);
    };
    const up = (ev) => {
      if (ev.code !== "Backslash") return;
      setCompareBefore(false);
    };
    // an OS-level app switch (Cmd+Tab etc.) mid-hold never delivers the
    // matching keyup — without this, releasing the key after refocusing
    // wouldn't be seen either, leaving the compare stuck on indefinitely.
    const onBlur = () => setCompareBefore(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", onBlur);
    };
  }, [suppressKeys]);

  // clone stamp brush settings — session-local (not persisted; each
  // already-painted stroke keeps its own radius/feather/opacity regardless
  // of later slider changes, same as real brush tools). feather is the
  // falloff-zone width as a fraction of radius (0 = hard edge, more = softer);
  // opacity caps how fully the clone replaces the original at full strength
  // (less than 1 blends toward the original — softening rather than erasing).
  const [brushRadius, setBrushRadius] = useState(() => Math.round(Math.min(iw, ih) * 0.03));
  const [feather, setFeather] = useState(0.3);
  const [stampOpacity, setStampOpacity] = useState(1);

  /* Read-only proxy-resolution pixel buffer for StampOverlay's own use —
     picking a source offset that avoids a stroke overlapping its own
     already-painted area, and auto-sizing a new stroke's feather to the
     local texture (see StampOverlay's pickDabOffset/autoFeather). Decoded
     once per photo, from the PRISTINE proxy — deliberately not the live,
     progressively-stamped buffer GLImage recomposites: this only needs to
     be "roughly what's around here", the same way a person eyeballing the
     photo would decide where to drag the source handle, not a precise
     up-to-the-last-dab preview (that stays GLImage's job for the actual
     pixels that get blended). */
  const editPixelsRef = useRef(null); // { data, width, height, scale } — scale = buffer px per original-image px
  useEffect(() => {
    editPixelsRef.current = null; // always clear on photo change — never stamp against a stale buffer
  }, [photo.id, photo.path]);
  // Decoding this is a full proxy-resolution canvas draw + getImageData (up
  // to 3200x3200, ~40MB) — real CPU/memory cost that's wasted on every
  // single photo view if the user never opens Stamp. Gated on the retouch
  // modes (stamp, and red-eye — whose pupil detection reads it too) so it
  // only pays for itself when actually needed; low-end-hardware win since
  // most viewing is crop/develop, not retouching.
  useEffect(() => {
    if (mode !== "stamp" && mode !== "redeye" && !wbPicking) return;
    let alive = true;
    loadImage(proxyURL(photo)).then((img) => {
      if (!alive) return;
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      editPixelsRef.current = {
        data: ctx.getImageData(0, 0, c.width, c.height).data,
        width: c.width, height: c.height,
        scale: c.width / iw,
      };
    }).catch(() => {});
    return () => { alive = false; };
  }, [photo.id, photo.path, iw, mode, wbPicking]);

  // stable object identities (GLImage's redraw effect depends on texRegion)
  const baseTexRegion = useMemo(() => ({ sx: 0, sy: 0, sw: iw }), [iw]);
  const tileTexRegion = useMemo(
    () => (tile ? { sx: tile.sx, sy: tile.sy, sw: tile.sw } : undefined),
    [tile],
  );

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const b = el.getBoundingClientRect();
      setStage({ w: b.width, h: b.height });
    });
    ro.observe(el);
    const b = el.getBoundingClientRect();
    setStage({ w: b.width, h: b.height });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!stage.w || !stage.h || rect) return;
    const availW = stage.w - 2 * PAD, availH = stage.h - 2 * PAD;
    const e = photo.edits;
    const cw = e ? e.cw : iw, ch = e ? e.ch : ih, theta = e ? e.theta : 0;
    const k = Math.min(availW / cw, availH / ch);
    const rw = cw * k, rh = ch * k;
    const r = { x: (stage.w - rw) / 2, y: (stage.h - rh) / 2, w: rw, h: rh };
    const c = { x: r.x + rw / 2, y: r.y + rh / 2 };
    let icx = c.x, icy = c.y;
    if (e) {
      const [ox, oy] = rot(k * e.qcx, k * e.qcy, rad(theta));
      icx = c.x - ox; icy = c.y - oy;
    }
    zoomBase.current = k;
    setRect(r);
    setView({ icx, icy, k, theta });
  }, [stage, rect, photo, iw, ih]);

  // margin defaults to a small POSITIVE tolerance (checks pass a hair
  // outside the true edge) — right for one-shot checks like a resize
  // candidate or a wheel-zoom step, which are used immediately and never
  // rescaled afterward. maximizedRect (below) passes a small NEGATIVE
  // margin instead: its search result is always fed straight into refit(),
  // which can rescale by an arbitrary factor (>1 as often as <1 — a narrow
  // aspect on a wide stage genuinely needs to zoom IN to fill it), and that
  // rescale multiplies whatever margin the search settled for by the same
  // factor. A result found riding the positive-tolerance edge can land
  // outside true bounds after a >1x rescale (confirmed empirically —
  // scratchpad/verify_aspect_switch_freeze.mjs — up to ~0.9px over on
  // ordinary photo/window sizes), which then fails EVERY resize-handle
  // candidate's own coverageOK(r0) check, including the no-op one, so
  // dragging looked completely dead after switching aspect ratios. A
  // strictly-negative margin guarantees genuine geometric slack that
  // stays positive under any positive rescale factor, however large.
  const coverageOK = useCallback((r, v, margin = 0.5) => {
    const t = rad(v.theta);
    const hw = (iw * v.k) / 2 + margin, hh = (ih * v.k) / 2 + margin;
    const corners = [
      [r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h],
    ];
    for (const [cx, cy] of corners) {
      const [vx, vy] = irot(cx - v.icx, cy - v.icy, t);
      if (Math.abs(vx) > hw || Math.abs(vy) > hh) return false;
    }
    return true;
  }, [iw, ih]);

  const clampCenter = useCallback((icx, icy, r, k, thetaDeg) => {
    const t = rad(thetaDeg);
    const hw = (iw * k) / 2, hh = (ih * k) / 2;
    const corners = [
      [r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h],
    ];
    let minUx = Infinity, maxUx = -Infinity, minUy = Infinity, maxUy = -Infinity;
    for (const [cx, cy] of corners) {
      const [ux, uy] = irot(cx, cy, t);
      minUx = Math.min(minUx, ux); maxUx = Math.max(maxUx, ux);
      minUy = Math.min(minUy, uy); maxUy = Math.max(maxUy, uy);
    }
    let [tx, ty] = irot(icx, icy, t);
    const loX = maxUx - hw, hiX = minUx + hw;
    const loY = maxUy - hh, hiY = minUy + hh;
    tx = loX > hiX ? (loX + hiX) / 2 : clamp(tx, loX, hiX);
    ty = loY > hiY ? (loY + hiY) / 2 : clamp(ty, loY, hiY);
    const [nx, ny] = rot(tx, ty, t);
    return [nx, ny];
  }, [iw, ih]);

  const minK = useCallback((r, thetaDeg) => {
    const t = rad(thetaDeg);
    const co = Math.abs(Math.cos(t)), si = Math.abs(Math.sin(t));
    return Math.max((r.w * co + r.h * si) / iw, (r.w * si + r.h * co) / ih);
  }, [iw, ih]);

  const applyRotation = useCallback((deg) => {
    cancelFlip();
    setView((v) => {
      if (!v || !rect) return v;
      const c = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
      const [qx, qy] = irot(c.x - v.icx, c.y - v.icy, rad(v.theta));
      const q = [qx / v.k, qy / v.k];
      const k = Math.max(zoomBase.current, minK(rect, deg) * 1.0005);
      const [ox, oy] = rot(k * q[0], k * q[1], rad(deg));
      let icx = c.x - ox, icy = c.y - oy;
      [icx, icy] = clampCenter(icx, icy, rect, k, deg);
      return { icx, icy, k, theta: deg };
    });
  }, [rect, minK, clampCenter, cancelFlip]);

  /* Re-fit uses the FLIP technique: state jumps straight to the final
     layout, and ONE GPU transform on the .world wrapper animates from the
     old geometry to identity. Zero layout work per frame, and the huge
     dim shadow is rasterized once and just composited — no more judder. */
  const refit = useCallback((r, v, animate = true) => {
    if (animate && performance.now() < flipBusyUntil.current) return;
    cancelFlip();
    const availW = stage.w - 2 * PAD, availH = stage.h - 2 * PAD;
    if (availW <= 0 || availH <= 0) return;
    const f = Math.min(availW / r.w, availH / r.h);
    const nw = r.w * f, nh = r.h * f;
    const nr = { x: (stage.w - nw) / 2, y: (stage.h - nh) / 2, w: nw, h: nh };
    const oc = { x: r.x + r.w / 2, y: r.y + r.h / 2 };
    const nc = { x: nr.x + nw / 2, y: nr.y + nh / 2 };
    const nv = {
      icx: nc.x + (v.icx - oc.x) * f,
      icy: nc.y + (v.icy - oc.y) * f,
      k: v.k * f,
      theta: v.theta,
    };
    zoomBase.current = nv.k;
    if (animate && f < 1) {
      // zooming out: retire the deep-zoom tile now, in the same render as
      // the rect/view jump, so it never draws a frame with stale
      // coordinates against the new geometry — fade it rather than pop it
      // so the resolution drop isn't a jolt in itself.
      fadeOutTile();
    }
    setRect(nr);
    setView(nv);
    if (animate && Math.abs(f - 1) > 1e-4) {
      const jump = Math.max(f, 1 / f); // magnitude of the scale change, >=1
      // Past FLIP_MAX_JUMP, skip the animated transition entirely and just
      // snap (state already jumped above either way) — a CSS transform
      // animating such a large scale change in a fixed ~460ms window is
      // what caused visible compositing glitches before (see kHigh's own
      // comment on the wheel-zoom ceiling this call can now reach). This
      // decouples "how close you're allowed to zoom" from "how big an
      // animated fit-to-screen jump ever has to render" — instead of
      // capping the former to avoid the latter, the latter just stops
      // animating past a safe magnitude.
      if (jump <= FLIP_MAX_JUMP) {
        const dur = clamp(260 + Math.log2(jump) * 55, 260, 460);
        flipBusyUntil.current = performance.now() + dur + 60;
        setFlipReq({ tx: oc.x - nc.x / f, ty: oc.y - nc.y / f, s: 1 / f, n: ++flipSeq.current, dur });
      }
    }
  }, [stage, fadeOutTile, cancelFlip]);

  /* Double-clicking empty space around the crop selection (crop mode) zooms
     out to show the whole original photo WITHOUT touching the crop
     selection itself. refit() above is built to make whatever rect it's
     given become the new crop (exactly right for "fit the current crop
     selection to the screen") — feeding it a whole-photo rect instead
     overwrites the actual crop selection with the whole photo's bounds,
     which is indistinguishable from the crop having been reset. This
     instead reads the crop's own image-space coordinates first, computes a
     view that fits the whole photo, then re-projects that SAME crop
     through the new view so it lands in the right place — now surrounded
     by more visible photo instead of filling the screen. */
  const zoomToWholePhoto = useCallback(() => {
    if (!rect || !view || !stage.w || !stage.h) return;
    cancelFlip();
    const t = rad(view.theta);
    const c = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
    const [qx, qy] = irot(c.x - view.icx, c.y - view.icy, t);
    const qcx = qx / view.k, qcy = qy / view.k;
    const cw = rect.w / view.k, ch = rect.h / view.k;
    const availW = stage.w - 2 * PAD, availH = stage.h - 2 * PAD;
    if (availW <= 0 || availH <= 0) return;
    const k = Math.min(availW / iw, availH / ih);
    const nv = { icx: stage.w / 2, icy: stage.h / 2, k, theta: view.theta };
    const [ox, oy] = rot(qcx * k, qcy * k, t);
    const nr = { x: nv.icx + ox - (cw * k) / 2, y: nv.icy + oy - (ch * k) / 2, w: cw * k, h: ch * k };
    zoomBase.current = k;
    setRect(nr);
    setView(nv);
  }, [rect, view, stage, iw, ih, cancelFlip]);

  useLayoutEffect(() => {
    if (!flipReq) return;
    const el = worldRef.current;
    if (!el) return;
    const n = flipReq.n;
    el.style.transition = "none";
    el.style.willChange = "transform";
    el.style.transform = `translate(${flipReq.tx}px, ${flipReq.ty}px) scale(${flipReq.s})`;
    const dur = flipReq.dur || 340;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (flipSeq.current !== n || !worldRef.current) return;
      el.style.transition = `transform ${dur}ms cubic-bezier(.22,.85,.32,1)`;
      el.style.transform = "none";
      setTimeout(() => {
        if (flipSeq.current !== n || !worldRef.current) return;
        el.style.transition = "none";
        el.style.willChange = "auto";
      }, dur + 60);
    }));
  }, [flipReq]);

  /* Switching mode toggles the bottom toolbelt (crop/stamp controls), which
     changes .stage's own available height/width as a side effect — that
     resize would otherwise trip the auto-refit below and rescale whatever
     zoom the user was deliberately at, purely because a panel appeared.
     This flag, set synchronously right as the mode-driven layout change
     commits, tells the very next stage-resize to skip the refit once.

     EXCEPT entering crop mode. Crop renders the FULL original photo (not
     just the current rect) through the same view transform that was tuned
     to fit the smaller, toolbelt-less view-mode stage — without refitting
     here, the photo hangs off the bottom behind the crop toolbelt that
     just appeared, and stays that way (nothing else re-triggers a refit)
     until an unrelated later resize happens to fire one, e.g. releasing a
     dragged crop handle 180ms after pointerup. So don't arm the
     suppression for that specific transition: let the resize-triggered
     refit below run immediately and correct it before the crop tools are
     even usable. */
  const suppressRefitRef = useRef(false);
  useLayoutEffect(() => {
    suppressRefitRef.current = mode !== "crop";
  }, [mode]);

  useEffect(() => {
    if (suppressRefitRef.current) {
      suppressRefitRef.current = false;
      return;
    }
    if (rect && view && stage.w && stage.h) {
      const availW = stage.w - 2 * PAD, availH = stage.h - 2 * PAD;
      if (rect.w > availW || rect.h > availH || (rect.w < availW - 1 && rect.h < availH - 1)) {
        refit(rect, view, false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage.w, stage.h]);

  const startDrag = (mode) => (e) => {
    if (e.button !== undefined && e.button !== 0) return; // left button only
    e.preventDefault(); e.stopPropagation();
    if (!rect || !view) return;
    cancelFlip();
    drag.current = {
      mode,
      sx: e.clientX, sy: e.clientY,
      rect: { ...rect }, view: { ...view },
    };
    if (mode !== "pan") snapshot();
    clearTimeout(refitTimer.current);
    setGesturing(true);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
  };

  /* Crop mode's hand tool: while Space is held, dragging .cropframe or a
     handle pans the photo under the screen-fixed crop selection instead of
     doing whatever that element normally does (reposition-photo-under-frame
     is ALSO exactly what this is, or resize) — reusing "image" mode's own
     existing math and clampCenter coverage guarantee (the photo can never
     be dragged so far that part of the crop rect would show outside it) —
     deliberately NOT an unclamped free-look mode (tried, reverted: let the
     crop selection extend past the photo's own edges, not what was asked
     for). This just gives that same clamped pan a second entry point,
     available from Space anywhere, not only from a plain drag starting
     exactly inside the frame. */
  const dragOrPan = (mode) => (e) => {
    if (spaceRef.current) { startDrag("image")(e); return; }
    startDrag(mode)(e);
  };

  /* "Original" aspect means "this photo's own aspect ratio, as currently
     oriented" — after a quarter-turn (rotate90, portrait<->landscape), that's
     ih/iw, not iw/ih. A small fine-leveling angle doesn't flip it (only a
     90/270 turn does), same distinction coverageOK already draws elsewhere
     between "rotate the frame" and "level a few degrees". */
  const origAspect = useCallback(
    (theta) => (((Math.round(theta / 90) % 2) + 2) % 2 === 1 ? ih / iw : iw / ih),
    [iw, ih],
  );

  const aspectValue = useMemo(() => {
    const a = ASPECTS.find((x) => x.key === aspect);
    let v = null;
    if (a?.key === "orig") v = origAspect(view?.theta || 0);
    else if (a?.v) v = a.v;
    if (v && flipped && aspect !== "1:1") v = 1 / v;
    return v;
  }, [aspect, flipped, view, origAspect]);

  const onMove = useCallback((e) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.sx, dy = e.clientY - d.sy;

    if (d.mode === "image") {
      setView((v) => {
        let icx = d.view.icx + dx, icy = d.view.icy + dy;
        [icx, icy] = clampCenter(icx, icy, d.rect, d.view.k, d.view.theta);
        return { ...v, icx, icy };
      });
      return;
    }

    if (d.mode === "pan") {
      // move the whole workspace (frame + image together) — repositions the
      // viewport within the app window, but since rect and icx/icy shift by
      // the identical amount, it can never reveal different photo content.
      // Used for .stage's own background drag in view/stamp mode; crop
      // mode's hand tool goes through "image" instead (see dragOrPan).
      setRect({ x: d.rect.x + dx, y: d.rect.y + dy, w: d.rect.w, h: d.rect.h });
      setView((v) => ({ ...v, icx: d.view.icx + dx, icy: d.view.icy + dy }));
      return;
    }

    const r0 = d.rect;
    const a = aspectValue;
    let cand = { ...r0 };
    const right = r0.x + r0.w, bottom = r0.y + r0.h;
    const cx0 = r0.x + r0.w / 2, cy0 = r0.y + r0.h / 2;
    const m = d.mode;

    const sized = (w, h) => ({ w: Math.max(MIN_CROP, w), h: Math.max(MIN_CROP, h) });

    if (m === "se" || m === "nw" || m === "ne" || m === "sw") {
      let w1 = m === "se" || m === "ne" ? r0.w + dx : r0.w - dx;
      let h1 = m === "se" || m === "sw" ? r0.h + dy : r0.h - dy;
      if (a) {
        if (w1 / a >= h1) h1 = w1 / a; else w1 = h1 * a;
      }
      const { w, h } = a ? (() => {
        let W = Math.max(w1, MIN_CROP), H = W / a;
        if (H < MIN_CROP) { H = MIN_CROP; W = H * a; }
        return { w: W, h: H };
      })() : sized(w1, h1);
      cand.w = w; cand.h = h;
      cand.x = m === "se" || m === "ne" ? r0.x : right - w;
      cand.y = m === "se" || m === "sw" ? r0.y : bottom - h;
    } else if (m === "e" || m === "w") {
      let w1 = m === "e" ? r0.w + dx : r0.w - dx;
      w1 = Math.max(MIN_CROP, w1);
      let h1 = a ? w1 / a : r0.h;
      if (a && h1 < MIN_CROP) { h1 = MIN_CROP; w1 = h1 * a; }
      cand.w = w1; cand.h = h1;
      cand.x = m === "e" ? r0.x : right - w1;
      cand.y = a ? cy0 - h1 / 2 : r0.y;
    } else if (m === "n" || m === "s") {
      let h1 = m === "s" ? r0.h + dy : r0.h - dy;
      h1 = Math.max(MIN_CROP, h1);
      let w1 = a ? h1 * a : r0.w;
      if (a && w1 < MIN_CROP) { w1 = MIN_CROP; h1 = w1 / a; }
      cand.w = w1; cand.h = h1;
      cand.y = m === "s" ? r0.y : bottom - h1;
      cand.x = a ? cx0 - w1 / 2 : r0.x;
    }

    // Only the photo's own actual bounds (coverageOK) are a real constraint
    // on a resize candidate — requiring the WHOLE rect to also stay within
    // the visible .stage viewport (an earlier `inStage` check) is wrong
    // once the user is zoomed in past "fit": the committed rect can then
    // already extend beyond the stage on its own (wheel-zoom scales rect
    // freely, with no stage-size ceiling), so `inStage` was false even for
    // the UNCHANGED starting rect — every resize candidate the binary
    // search below tried was rejected, including ones a hair's breadth
    // from the start, so it always converged back to ~no visible change at
    // all. Dropping it lets resize work at any zoom; panning/the
    // release-triggered re-fit-to-100% (see the wheel handler's own
    // comment) is what brings an off-screen edge back into view.
    const valid = (r) => coverageOK(r, d.view);
    const lerp = (t) => ({
      x: r0.x + (cand.x - r0.x) * t,
      y: r0.y + (cand.y - r0.y) * t,
      w: r0.w + (cand.w - r0.w) * t,
      h: r0.h + (cand.h - r0.h) * t,
    });
    let applied = cand;
    if (!valid(cand)) {
      let lo = 0, hi = 1;
      for (let i = 0; i < 8; i++) {
        const mid = (lo + hi) / 2;
        if (valid(lerp(mid))) lo = mid; else hi = mid;
      }
      applied = lerp(lo);
    }
    setRect(applied);
  }, [aspectValue, clampCenter, coverageOK]);

  const onUp = useCallback(() => {
    window.removeEventListener("pointermove", onMove);
    const wasResize = drag.current && drag.current.mode !== "image" && drag.current.mode !== "pan";
    drag.current = null;
    setGesturing(false);
    if (wasResize) {
      refitTimer.current = setTimeout(() => {
        // animate=false — see the rotate90/pickAspect/toggleFlip refit calls
        // for why: a quick second resize-drag started before this delayed
        // refit's own animation finishes calls cancelFlip() and freezes
        // .world's transform mid-transition, visually detaching the crop
        // frame (and its handles) from where React actually has it — which
        // reads as "dragging the handles stopped working".
        setRect((r) => { setView((v) => { r && v && refit(r, v, false); return v; }); return r; });
      }, 180);
    }
  }, [onMove, refit]);

  /* --- largest rect of a given aspect that the image can cover,
         centered on the image itself (like Lightroom's aspect switch) --- */
  const maximizedRect = useCallback((v, vw) => {
    const c = { x: vw.icx, y: vw.icy };
    const rectOf = (sz) => {
      const rw = v >= 1 ? sz : sz * v;
      const rh = v >= 1 ? sz / v : sz;
      return { x: c.x - rw / 2, y: c.y - rh / 2, w: rw, h: rh };
    };
    // -1: search strictly inside the true image edge (see coverageOK's own
    // comment above for why this can't just reuse its default +0.5 margin).
    let lo = MIN_CROP, hi = (iw + ih) * vw.k;
    if (!coverageOK(rectOf(lo), vw, -1)) return rectOf(lo);
    for (let i = 0; i < 28; i++) {
      const mid = (lo + hi) / 2;
      if (coverageOK(rectOf(mid), vw, -1)) lo = mid; else hi = mid;
    }
    return rectOf(lo);
  }, [iw, ih, coverageOK]);

  /* --- rotate the crop frame itself by a quarter turn (portrait <-> landscape),
         e.g. for photos shot with the camera on its side. Separate from the
         fine leveling slider below: it swaps the frame's own w/h and carries
         the current fine angle straight through, so a 90 deg turn plus a
         small horizon tweak compose cleanly (baked angle = rot90 + fine,
         wrapped to -180..180). --- */
  const rotate90 = useCallback((dir = 1) => {
    if (!rect || !view) return;
    snapshot();
    cancelFlip();
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
    const nr = { x: cx - rect.h / 2, y: cy - rect.w / 2, w: rect.h, h: rect.w };
    let deg = view.theta + dir * 90;
    if (deg > 180) deg -= 360;
    if (deg < -180) deg += 360;
    const k = Math.max(zoomBase.current, minK(nr, deg) * 1.0005);
    let icx = cx, icy = cy;
    [icx, icy] = clampCenter(icx, icy, nr, k, deg);
    // animate=false: an animated refit here can be interrupted mid-flight by
    // cancelFlip() from another quick action (rotate/aspect/resize again
    // before this one's ~260-460ms CSS transition finishes), which freezes
    // .world's transform on a half-finished frame instead of the correct
    // final one — same hazard as the crop-area double-click bug, fixed the
    // same way (see .cropframe's onDoubleClick comment for the full story).
    //
    // ONE synchronous call, not the old setRect+setView now / refit 20ms
    // later via setTimeout: staggering "rotate in place" and "fit to the
    // visible stage" across two separate render ticks meant the frame
    // visibly double-snapped — an instant jump to the rotated-but-not-yet-
    // fitted size, then a second instant jump to the final fitted size a
    // frame later — which read as a stutter rather than one clean motion.
    // refit() here still doesn't animate (same reason as always), but at
    // least it's a single snap instead of two.
    refit(nr, { icx, icy, k, theta: deg }, false);
  }, [rect, view, snapshot, minK, clampCenter, refit, cancelFlip]);

  /* --- aspect selection: expand to the largest crop the whole image
         allows (no more progressive zoom-in), then animate the re-fit --- */
  const pickAspect = (key) => {
    snapshot();
    cancelFlip();
    setAspect(key);
    const noFlip = key === "free" || key === "orig" || key === "1:1";
    if (noFlip && flipped) setFlipped(false);
    if (!rect || !view) return;
    const def = ASPECTS.find((x) => x.key === key);
    let v = def?.key === "orig" ? origAspect(view.theta) : def?.v || null;
    if (v && flipped && !noFlip) v = 1 / v;
    if (!v) return;
    const nr = maximizedRect(v, view);
    setRect(nr);
    // animate=false: an animated refit here can be interrupted mid-flight by
    // cancelFlip() from another quick action (rotate/aspect/resize again
    // before this one's ~260-460ms CSS transition finishes), which freezes
    // .world's transform on a half-finished frame instead of the correct
    // final one — same hazard as the crop-area double-click bug, fixed the
    // same way (see .cropframe's onDoubleClick comment for the full story).
    setTimeout(() => setView((vv) => { refit(nr, vv, false); return vv; }), 20);
  };

  const toggleFlip = () => {
    if (!rect || !view || aspect === "1:1" || aspect === "free" || aspect === "orig") return;
    snapshot();
    cancelFlip();
    const nf = !flipped;
    setFlipped(nf);
    const def = ASPECTS.find((x) => x.key === aspect);
    let v = def?.key === "orig" ? origAspect(view.theta) : def?.v || null;
    if (!v) {
      // free aspect: flip the current frame's own proportions
      v = rect.h / rect.w;
    } else if (nf) {
      v = 1 / v;
    }
    const nr = maximizedRect(v, view);
    setRect(nr);
    // animate=false: an animated refit here can be interrupted mid-flight by
    // cancelFlip() from another quick action (rotate/aspect/resize again
    // before this one's ~260-460ms CSS transition finishes), which freezes
    // .world's transform on a half-finished frame instead of the correct
    // final one — same hazard as the crop-area double-click bug, fixed the
    // same way (see .cropframe's onDoubleClick comment for the full story).
    setTimeout(() => setView((vv) => { refit(nr, vv, false); return vv; }), 20);
  };

  /* --- mouse-wheel: zoom the whole canvas (frame + image together).
         Zooming out shrinks the working area so the entire photo fits
         on screen; zooming in magnifies for precision. Uniform scaling
         keeps the crop-coverage invariant, so no clamping is needed.
         Releasing a handle drag still re-fits to 100%. --- */
  useEffect(() => {
    const el = stageRef.current;
    if (!el || !rect || !view) return;
    const onWheel = (e) => {
      e.preventDefault();
      const b = el.getBoundingClientRect();
      const m = { x: e.clientX - b.left, y: e.clientY - b.top };
      // trackpad pinch (and literal Ctrl+wheel) synthesize wheel events
      // with ctrlKey:true regardless of whether Ctrl is actually held —
      // that's the standard signal a pinch gesture drove this event rather
      // than linear two-finger/mouse-wheel scrolling, so it gets its own,
      // much stronger multiplier instead of sharing the scroll one.
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.0096 : 0.0024));
      const availW = stage.w - 2 * PAD, availH = stage.h - 2 * PAD;
      // Same "fit" target refit()/double-click computes (Math.min(availW/
      // rect.w, availH/rect.h) applied to the CURRENT k) — NOT a separately
      // -derived "fit the raw iw/ih" number, which silently diverged from
      // the real fit target whenever a crop or a 90/270 rotation was active
      // (rect's own w/h already account for both; recomputing straight from
      // iw/ih did not), leaving wheel-zoom-out unable to reach the same
      // fully-zoomed-out state a double-click reaches.
      const kFit = view.k * Math.min(availW / rect.w, availH / rect.h);
      const kLow = Math.min(view.k, kFit);
      // Raised from an original 6x to 20x on request, now that full-
      // resolution originals (up to ~30MB) are the norm rather than a
      // downsampled proxy — there's real detail to zoom into past the old
      // ceiling. This used to be capped much lower specifically because an
      // ANIMATED fit-to-screen double-click from way out here had to jump
      // such a large scale factor that the FLIP transition's compositing
      // visibly glitched — that's now handled independently by FLIP_MAX_JUMP
      // (refit skips the animation and snaps instead, past a safe jump
      // magnitude), so this ceiling no longer needs to stay low to avoid it.
      // Wheel-zoom itself (this handler) was never animated either way.
      const kHigh = Math.max(view.k, kFit * 20);
      const k2 = clamp(view.k * factor, kLow, kHigh);
      const f = k2 / view.k;
      if (Math.abs(f - 1) < 1e-6) return;
      cancelFlip();
      gestureOn();
      // Hitting the zoom-out floor this tick: snap straight to refit()'s own
      // centered result (exactly what a double-click produces) instead of
      // the cursor-anchored math below. That math only matches the fit
      // SCALE at the floor, not necessarily the same on-screen POSITION —
      // if the cursor wasn't dead-center when the floor was reached, the
      // frame would land at the right zoom but off-center, which doesn't
      // read as "the same fully-zoomed-out state" a double-click gives.
      if (view.k > kFit && k2 <= kFit + 1e-9) {
        refit(rect, view, false);
        return;
      }
      const nr = {
        x: m.x + (rect.x - m.x) * f,
        y: m.y + (rect.y - m.y) * f,
        w: rect.w * f,
        h: rect.h * f,
      };
      zoomBase.current = k2;
      setRect(nr);
      setView({
        icx: m.x + (view.icx - m.x) * f,
        icy: m.y + (view.icy - m.y) * f,
        k: k2,
        theta: view.theta,
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [rect, view, stage, iw, ih, gestureOn, cancelFlip, refit]);

  /* --- deep-zoom sharpness: when the gesture settles and the view is
     zoomed past the proxy's resolution, fetch a tile of the visible
     region from the full-resolution original --- */
  useEffect(() => {
    // already editing the full original (see editSrcURL/isFullResEligible)
    // — no lower-resolution proxy ceiling exists for a tile to patch in.
    if (isFullResEligible(photo)) return;
    if (!NATIVE || !photo.path || !view || !rect || !stage.w) return;
    const kProxy = PROXY_LONG / Math.max(iw, ih);
    if (view.k <= kProxy * 1.08 || gesturing) {
      if (view.k <= kProxy * 1.08 && tile && !tile.fading) fadeOutTile();
      return;
    }
    clearTimeout(tileTimer.current);
    const token = ++tileToken.current;
    tileTimer.current = setTimeout(async () => {
      const t = rad(view.theta);
      const corners = [[0, 0], [stage.w, 0], [0, stage.h], [stage.w, stage.h]];
      let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
      for (const [cx, cy] of corners) {
        const [ux, uy] = irot(cx - view.icx, cy - view.icy, t);
        const x = ux / view.k + iw / 2, y = uy / view.k + ih / 2;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
      const padX = (maxX - minX) * 0.08, padY = (maxY - minY) * 0.08;
      const sx = clamp(Math.floor(minX - padX), 0, iw - 4);
      const sy = clamp(Math.floor(minY - padY), 0, ih - 4);
      const sw = clamp(Math.ceil(maxX + padX) - sx, 4, iw - sx);
      const sh = clamp(Math.ceil(maxY + padY) - sy, 4, ih - sy);
      const dw = Math.min(4096, Math.round(sw * view.k * (window.devicePixelRatio || 1)));
      if (dw <= 8) return;
      const data = await window.meridian.renderTile(photo.path, { sx, sy, sw, sh, dw });
      if (!data || token !== tileToken.current) return;
      const url = URL.createObjectURL(new Blob([data], { type: "image/jpeg" }));
      // Pre-decode before swapping: setTile() below flips .tileimg's CSS
      // position/size to the new tile's bounds immediately, but GLImage's
      // own texture upload only starts once its `src` prop changes and
      // loadImage() resolves — a visible frame or two of the OLD texture
      // stretched into the NEW bounds otherwise (the "stutter" on rapid
      // zoom). loadImage() is cache-backed, so warming it here means
      // GLImage's own call resolves instantly once `tile` actually flips.
      await loadImage(url);
      if (token !== tileToken.current) { URL.revokeObjectURL(url); return; }
      if (tileUrl.current) URL.revokeObjectURL(tileUrl.current);
      tileUrl.current = url;
      setTile({ url, sx, sy, sw, sh });
    }, 160);
    return () => clearTimeout(tileTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, rect, gesturing, stage, iw, ih, photo.path]);

  useEffect(() => () => {
    if (tileUrl.current) URL.revokeObjectURL(tileUrl.current);
  }, []);

  const reset = () => {
    snapshot();
    cancelFlip();
    setAspect("free"); setFlipped(false);
    setAdjust(defaultAdjust());
    setStamp([]);
    setSelectedStroke(null);
    setRedEye([]);
    setSelectedEye(null);
    setDraft(null);
    const availW = stage.w - 2 * PAD, availH = stage.h - 2 * PAD;
    const k = Math.min(availW / iw, availH / ih);
    const rw = iw * k, rh = ih * k;
    const r = { x: (stage.w - rw) / 2, y: (stage.h - rh) / 2, w: rw, h: rh };
    zoomBase.current = k;
    setRect(r);
    setView({ icx: stage.w / 2, icy: stage.h / 2, k, theta: 0 });
    onApply(null, null); // reverting is applied immediately
    cachePreviewToDisk(photo.path, null);
    toast("Restored to original");
  };

  const currentEdits = useCallback(() => {
    if (!rect || !view) return null;
    const c = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
    const [qx, qy] = irot(c.x - view.icx, c.y - view.icy, rad(view.theta));
    const e = {
      theta: view.theta,
      qcx: qx / view.k,
      qcy: qy / view.k,
      cw: Math.min(iw, rect.w / view.k),
      ch: Math.min(ih, rect.h / view.k),
    };
    const isFull =
      Math.abs(e.theta) < 0.05 &&
      Math.abs(e.qcx) < 1 && Math.abs(e.qcy) < 1 &&
      e.cw > iw - 2 && e.ch > ih - 2;
    if (isFull && isDefaultAdjust(adjust) && stamp.length === 0 && redeye.length === 0) return null;
    if (!isDefaultAdjust(adjust)) e.adjust = { ...adjust };
    if (stamp.length > 0) e.stamp = stamp;
    if (redeye.length > 0) e.redeye = redeye;
    return e;
  }, [rect, view, iw, ih, adjust, stamp, redeye]);

  // Same crop/rotate math as currentEdits() above, kept as its own memo
  // (deliberately NOT folding `adjust` in, unlike currentEdits) so the
  // histogram's crop-region decode below only re-triggers on an actual
  // crop/rotate/stamp change — not on every slider tick, which needs to
  // stay cheap and synchronous (see Histogram's own draw()).
  const histCrop = useMemo(() => {
    if (!rect || !view) return null;
    const c = { x: rect.x + rect.w / 2, y: rect.y + rect.h / 2 };
    const [qx, qy] = irot(c.x - view.icx, c.y - view.icy, rad(view.theta));
    return {
      theta: view.theta,
      qcx: qx / view.k,
      qcy: qy / view.k,
      cw: Math.min(iw, rect.w / view.k),
      ch: Math.min(ih, rect.h / view.k),
      stamp,
      redeye,
    };
  }, [rect, view, iw, ih, stamp, redeye]);

  const buildEdits = async () => {
    const e = currentEdits();
    let preview = null;
    if (e) {
      preview = await bakePreview(proxyURL(photo), iw, ih, e, 560);
    }
    cachePreviewToDisk(photo.path, preview);
    return [e, preview];
  };

  /* Unlike crop (an explicit mode with its own Done button), color sliders
     and clone-stamp strokes are always-live with no discrete "commit"
     moment — so auto-commit them into shared state shortly after the user
     stops adjusting. Without this, any export trigger that isn't this
     Editor's own Export/Done/Escape path (the library toolbar, a right-click
     on the grid or on this photo's own filmstrip thumbnail while still
     mid-edit) reads the shared `photos` array directly and would see stale,
     pre-adjustment data. */
  const autoCommitMounted = useRef(false);
  const autoCommitTimer = useRef(null);
  useEffect(() => {
    if (!autoCommitMounted.current) { autoCommitMounted.current = true; return undefined; }
    clearTimeout(autoCommitTimer.current);
    autoCommitTimer.current = setTimeout(() => {
      buildEdits().then(([e, preview]) => onApply(e, preview));
    }, 500);
    return () => clearTimeout(autoCommitTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adjust, stamp, redeye]);

  /* Done / Enter: apply the crop, hide the crop UI, show the clean result */
  const apply = async () => {
    const [e, preview] = await buildEdits();
    onApply(e, preview);
    // Only re-fit after committing a CROP: the frame itself just changed, so
    // normalizing the view to the new crop is expected. Applying from stamp
    // mode (or from view mode, e.g. just color-adjusting) must leave zoom/pan
    // exactly as-is — the user is often applying specifically to check the
    // result of a stamp edit at the zoom they were just working at.
    // animate=false: this was the actual source of a nasty bug — its ~260-
    // 460ms flip animation was still mid-flight when the user quickly
    // re-entered crop mode and picked an aspect, whose own cancelFlip() froze
    // .world's transform on a half-finished frame, visually detaching
    // .cropframe's handles from where they really were (see
    // feedback_flip_animation_interruption.md for the full mechanism).
    if (mode === "crop" && rect && view) refit(rect, view, false);
    setMode("view");
    setDraft(null);
    setSelectedEye(null);
    if (!e) toast("Restored to original");
  };

  /* Back / Escape: apply and return to the library — nothing is lost */
  const exit = async () => {
    const [e, preview] = await buildEdits();
    onApply(e, preview);
    onClose();
  };

  /* filmstrip navigation: edits are applied automatically, Lightroom-style */
  const switchTo = async (nextId) => {
    if (nextId === photo.id) return;
    const [e, preview] = await buildEdits();
    onSwitch(e, preview, nextId);
  };

  /* EXIF for the metadata panel (cached per path) */
  useEffect(() => {
    let alive = true;
    setExif(null);
    if (NATIVE && photo.path && window.meridian.readExif) {
      if (exifCache.has(photo.path)) {
        setExif(exifCache.get(photo.path));
      } else {
        window.meridian.readExif(photo.path).then((d) => {
          exifCache.set(photo.path, d);
          if (alive) setExif(d);
        });
      }
    }
    return () => { alive = false; };
  }, [photo.path]);

  useEffect(() => {
    const onKey = (ev) => {
      if (suppressKeys) return;
      const tag = ev.target?.tagName;
      // a slider keeps keyboard focus after a drag — Enter/Escape must still
      // act as confirm/cancel then, so only text-entry fields block them
      const inTextField = (tag === "INPUT" && ev.target.type !== "range") || tag === "TEXTAREA";
      // Esc while picking white balance only cancels the picker
      if (ev.key === "Escape" && wbPicking) { ev.preventDefault(); setWbPicking(false); return; }
      if (ev.key === "Enter" && !inTextField) { ev.preventDefault(); apply(); return; }
      if (ev.key === "Escape" && !inTextField) { ev.preventDefault(); exit(); return; }
      if (inTextField) return;
      // a slider keeps focus after a drag (see inTextField above), so without
      // this a stray Delete/[/]/etc. right after adjusting Brush size/Feather/
      // Opacity would silently do nothing — only arrow keys still need to be
      // left alone here, since those are the native way to nudge a focused
      // range input and must not be hijacked for sibling-photo navigation.
      if (tag === "INPUT" && (ev.key === "ArrowLeft" || ev.key === "ArrowRight")) return;
      // `ev.code` (the physical key, e.g. "BracketLeft") is used for every
      // shortcut below that's normally a printable character — `ev.key`
      // reflects whatever character the ACTIVE layout maps that key to
      // (e.g. Cyrillic "х" for the same physical key as "[" on a Russian
      // layout), which silently broke every one of these on a non-US
      // layout. Pure control keys (Enter/Escape/Arrow*/Delete/Backspace)
      // aren't remapped by layout, so `ev.key` stays correct for those.
      if ((ev.ctrlKey || ev.metaKey) && ev.code === "KeyZ") {
        ev.preventDefault();
        ev.shiftKey ? redo() : undo();
        return;
      }
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
        if (!siblings || siblings.length < 2) return;
        ev.preventDefault();
        const idx = siblings.findIndex((s) => s.id === photo.id);
        const next = siblings[(idx + (ev.key === "ArrowRight" ? 1 : -1) + siblings.length) % siblings.length];
        if (next) switchTo(next.id);
        return;
      }
      if ((ev.code === "BracketLeft" || ev.code === "BracketRight") && mode === "stamp") {
        ev.preventDefault();
        const min = Math.max(2, Math.round(Math.min(iw, ih) * 0.0012));
        const max = Math.round(Math.min(iw, ih) * 0.15);
        setBrushRadius((r) => Math.round(Math.min(max, Math.max(min, r * (ev.code === "BracketRight" ? 1.15 : 1 / 1.15)))));
      }
      else if ((ev.code === "Equal" || ev.code === "Minus") && mode === "stamp") {
        ev.preventDefault();
        const min = Math.max(2, Math.round(Math.min(iw, ih) * 0.0012));
        const max = Math.round(Math.min(iw, ih) * 0.15);
        const grow = ev.code === "Equal";
        setBrushRadius((r) => Math.round(Math.min(max, Math.max(min, r * (grow ? 1.15 : 1 / 1.15)))));
      }
      else if (ev.code === "BracketLeft" || ev.code === "BracketRight") {
        // available in view mode too, not just crop — rotate90 only touches
        // rect/view (the same state view mode already renders from), so it
        // works identically without needing to open the crop UI first; stamp
        // mode still claims [ / ] for brush size (handled above, returns
        // before reaching here).
        if (mode !== "crop" && mode !== "view") return;
        ev.preventDefault();
        rotate90(ev.code === "BracketRight" ? 1 : -1);
      }
      else if (ev.code === "KeyS" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        if (mode !== "view") return;
        ev.preventDefault();
        setMode("stamp");
      }
      else if (ev.code === "KeyW" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        // White Balance Selector — over the finished view, like Lightroom's
        // Develop W; from a retouch/crop mode it switches to view first
        ev.preventDefault();
        if (mode !== "view") setMode("view");
        setWbPicking((p) => !p);
      }
      else if (ev.code === "KeyE" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        if (mode !== "view") return;
        ev.preventDefault();
        setMode("redeye");
      }
      else if (ev.code === "KeyH" && !ev.ctrlKey && !ev.metaKey && !ev.altKey && mode === "stamp") {
        ev.preventDefault();
        // H cycles Clone → Heal → Remove
        pickStampMode(STAMP_MODES[(STAMP_MODES.indexOf(stampMode) + 1) % STAMP_MODES.length]);
      }
      else if ((ev.key === "Delete" || ev.key === "Backspace") && mode === "stamp" && selectedStroke != null) {
        ev.preventDefault();
        snapshot();
        setStamp((prev) => prev.filter((_, i) => i !== selectedStroke));
        setSelectedStroke(null);
      }
      else if ((ev.key === "Delete" || ev.key === "Backspace") && mode === "redeye" && selectedEye != null) {
        ev.preventDefault();
        snapshot();
        setRedEye((prev) => prev.filter((_, i) => i !== selectedEye));
        setSelectedEye(null);
      }
      else if (/^Digit[0-5]$/.test(ev.code)) {
        const n = parseInt(ev.code.slice(5), 10);
        onRate(photo.id, n);
        toast(n ? `Rated ${"★".repeat(n)}` : "Rating cleared");
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onRate, photo.id, suppressKeys, toast, rect, view, mode, siblings, undo, redo, selectedStroke, selectedEye, snapshot, stampMode, stamp, wbPicking]);

  useEffect(() => { if (mode !== "view") setWbPicking(false); }, [mode]);

  const pickWhiteBalance = (rgb) => {
    const wb = neutralizeWhiteBalance(rgb, adjust.pv);
    snapshot();
    setAdjust({ ...adjust, temp: Math.round(wb.temp), tint: Math.round(wb.tint) });
    setWbPicking(false);
    toast(`White balance: Temperature ${Math.round(wb.temp) > 0 ? "+" : ""}${Math.round(wb.temp)}, Tint ${Math.round(wb.tint) > 0 ? "+" : ""}${Math.round(wb.tint)}`);
  };

  const theta = view ? view.theta : 0;
  // Before/After: what GLImage actually renders — untouched adjust/stamp
  // while comparing, the real ones otherwise. Crop/rotation (rect/view)
  // stay exactly as-is either way, so this compares color+retouch work
  // against the original within the SAME framing, not the literal
  // as-shot file. `adjust`/`stamp` unchanged (new object/array only
  // allocated while actually comparing) so this never affects anything
  // else that reads Editor's own `adjust`/`stamp` state directly (undo,
  // currentEdits, the DevelopPanel sliders, …).
  const displayAdjust = compareBefore ? defaultAdjust() : adjust;
  const displayStamp = compareBefore ? EMPTY_STAMP : stamp;
  const displayRedEye = compareBefore ? EMPTY_STAMP : redeye;
  const origDiag = Math.hypot(iw, ih);
  const rot90Steps = Math.round(theta / 90);
  const fineTheta = Math.round((theta - rot90Steps * 90) * 10) / 10;
  const outW = rect && view ? Math.round(Math.min(iw, rect.w / view.k)) : 0;
  const outH = rect && view ? Math.round(Math.min(ih, rect.h / view.k)) : 0;

  /* ---- tool rail (left of the canvas) ----
     Same entry points as before: a tool starts only from view mode (what
     the old topbar buttons / S / E did), clicking the active tool applies
     it (= Done / Enter), Clone/Heal/Remove inside stamp mode switch the
     brush mode exactly like the old chips (pickStampMode). Tools of
     another mode are disabled until the current one is applied. */
  const toggleWB = () => { if (mode !== "view") setMode("view"); setWbPicking((p) => !p); };
  const railTool = (m, stampM) => {
    if (mode === "view") {
      if (stampM) setStampMode(stampM); // entering: just the brush mode for new strokes
      setMode(m);
      return;
    }
    if (mode !== m) return;
    if (stampM && stampM !== stampMode) pickStampMode(stampM);
    else apply();
  };
  const railBtn = (key, tip, pressed, onClick, disabled) => (
    <Tip text={tip} side="right" key={key}>
      <button className="tool" aria-pressed={pressed} aria-label={tip.split(" — ")[0]} onClick={onClick} disabled={disabled}>
        {TOOL_ICON[key]}
      </button>
    </Tip>
  );
  const railUI = (
    <div className="rail-dock">
      <div className="rail glass">
        {railBtn("crop", mode === "crop" ? "Apply crop (Enter)" : "Crop & straighten", mode === "crop", () => railTool("crop"), mode !== "view" && mode !== "crop")}
        <div className="rail-sep" />
        <div className={`rail-group ${mode === "stamp" ? "engaged" : ""}`}>
          {railBtn("clone", "Clone (S) — exact copy of a source you place · H cycles modes", mode === "stamp" && stampMode === "clone", () => railTool("stamp", "clone"), mode !== "view" && mode !== "stamp")}
          {railBtn("heal", "Heal — automatic source, blended into the surrounding tone · H cycles modes", mode === "stamp" && stampMode === "heal", () => railTool("stamp", "heal"), mode !== "view" && mode !== "stamp")}
          {railBtn("remove", "Remove — content-aware fill, rebuilds the area from around it · H cycles modes", mode === "stamp" && stampMode === "remove", () => railTool("stamp", "remove"), mode !== "view" && mode !== "stamp")}
        </div>
        <div className="rail-sep" />
        {railBtn("redeye", "Red eye (E)", mode === "redeye", () => railTool("redeye"), mode !== "view" && mode !== "redeye")}
        {railBtn("wb", "White balance selector (W) — click something neutral gray or white", wbPicking, toggleWB, false)}
      </div>
    </div>
  );

  /* ---- HUD: the options of the active tool, docked under the photo ----
     Every control is the same native input with the same handlers as the
     old toolbelt rows; only the layout and drawing changed. */
  const doneBtn = (
    <button className="hud-done" onClick={apply}>Done <span className="kbd">⏎</span></button>
  );
  const brushMin = Math.max(2, Math.round(Math.min(iw, ih) * 0.0012));
  const brushMax = Math.round(Math.min(iw, ih) * 0.15);
  const selStroke = mode === "stamp" && selectedStroke != null ? stamp[selectedStroke] : null;
  const curFeather = selStroke ? selStroke.feather : feather;
  const curOpacity = selStroke ? selStroke.opacity : stampOpacity;
  const prevSize = 8 + 22 * Math.sqrt(clamp((brushRadius - brushMin) / Math.max(1, brushMax - brushMin), 0, 1));
  const hudField = (label, value, unit, sel, track, wide) => (
    <div className={`hfield ${wide ? "wide" : ""}`}>
      <div className="hf-head">
        <span className="lab">{sel && <i className="sdot" />}{label}</span>
        <span className="val">{value}{unit && <em>{unit}</em>}</span>
      </div>
      {track}
    </div>
  );
  let hudUI = null;
  if (mode === "stamp") {
    const next = STAMP_MODES[(STAMP_MODES.indexOf(stampMode) + 1) % STAMP_MODES.length];
    hudUI = (
      <div className="hud glass">
        <Tip text={`Switch to ${STAMP_MODE_LABEL[next]} (H)`}>
          <button className="hud-chip" onClick={() => pickStampMode(next)}>
            <span className="chip-ico">{TOOL_ICON[stampMode]}</span>
            <span className="chip-text">
              <span className="chip-name">{STAMP_MODE_LABEL[stampMode]}</span>
              <span className="dots">{STAMP_MODES.map((m) => <i key={m} className={m === stampMode ? "on" : ""} />)}</span>
            </span>
          </button>
        </Tip>
        <div className="hsep" />
        <div className="brush-prev">
          <i style={{
            width: prevSize, height: prevSize, opacity: 0.35 + 0.65 * curOpacity,
            background: `radial-gradient(circle closest-side, var(--text) ${Math.round((1 - curFeather) * 100)}%, transparent 100%)`,
          }} />
        </div>
        {hudField("Size", Math.round(brushRadius), "px", false,
          <RangeTrack
            min={brushMin} max={brushMax} step={1} value={brushRadius}
            onChange={(e) => setBrushRadius(parseFloat(e.target.value))}
          />)}
        {hudField("Feather", Math.round(curFeather * 100), "%", !!selStroke,
          <RangeTrack
            min={0} max={80} step={1} value={Math.round(curFeather * 100)}
            onPointerDown={snapshot}
            onChange={(e) => {
              const v = parseFloat(e.target.value) / 100;
              if (selectedStroke != null) {
                setStamp((prev) => prev.map((s, i) => (i === selectedStroke ? { ...s, feather: v } : s)));
              } else {
                setFeather(v);
              }
            }}
          />)}
        {hudField("Opacity", Math.round(curOpacity * 100), "%", !!selStroke,
          <RangeTrack
            min={10} max={100} step={1} value={Math.round(curOpacity * 100)}
            onPointerDown={snapshot}
            onChange={(e) => {
              const v = parseFloat(e.target.value) / 100;
              if (selectedStroke != null) {
                setStamp((prev) => prev.map((s, i) => (i === selectedStroke ? { ...s, opacity: v } : s)));
              } else {
                setStampOpacity(v);
              }
            }}
          />)}
        {selStroke && (
          <div className="selchip">
            <i className="sdot" />{STAMP_MODE_LABEL[selStroke.mode || "clone"]} stroke
            <Tip text="Delete stroke (Delete)">
              <button
                className="ib"
                onClick={() => {
                  snapshot();
                  setStamp((prev) => prev.filter((_, i) => i !== selectedStroke));
                  setSelectedStroke(null);
                }}
                aria-label="Delete stroke"
              >
                {TOOL_ICON.trash}
              </button>
            </Tip>
            <Tip text="Deselect — Feather/Opacity go back to new strokes">
              <button className="ib" onClick={() => setSelectedStroke(null)} aria-label="Deselect stroke">{I.close}</button>
            </Tip>
          </div>
        )}
        <div className="hsep" />
        {doneBtn}
      </div>
    );
  } else if (mode === "crop") {
    hudUI = (
      <div className="hud glass">
        <div className="hud-chip">
          <span className="chip-ico">{TOOL_ICON.crop}</span>
          <span className="chip-name">Crop</span>
        </div>
        <div className="hsep" />
        <div className="hud-seg">
          {ASPECTS.map((a) => (
            <button key={a.key} aria-pressed={aspect === a.key} onClick={() => pickAspect(a.key)}>
              {flipped && a.v && a.v !== 1 ? swapLabel(a.label) : a.label}
            </button>
          ))}
        </div>
        <Tip text="Swap portrait / landscape">
          <button
            className="ib"
            onClick={toggleFlip}
            disabled={aspect === "1:1" || aspect === "free" || aspect === "orig"}
            aria-pressed={flipped}
            aria-label="Swap portrait / landscape"
          >
            {I.flip}
          </button>
        </Tip>
        <Tip text="Rotate 90° left">
          <button className="ib" onClick={() => rotate90(-1)} aria-label="Rotate 90° left">{I.rotateLeft}</button>
        </Tip>
        <Tip text="Rotate 90° right">
          <button className="ib" onClick={() => rotate90(1)} aria-label="Rotate 90° right">{I.rotateRight}</button>
        </Tip>
        <div className="hsep" />
        <div className="hfield wide">
          <div className="hf-head">
            <span className="lab">Straighten</span>
            <span className={`val ${Math.abs(fineTheta) < 0.05 ? "zero" : ""}`}>{fineTheta > 0 ? "+" : ""}{fineTheta.toFixed(1)}°</span>
          </div>
          <RangeTrack
            min={-45} max={45} step={0.1} origin={0}
            value={fineTheta}
            onPointerDown={snapshot}
            onChange={(e) => applyRotation(rot90Steps * 90 + parseFloat(e.target.value))}
            onDoubleClick={() => { snapshot(); applyRotation(rot90Steps * 90); }}
          />
        </div>
        <div className="hsep" />
        {doneBtn}
      </div>
    );
  } else if (mode === "redeye") {
    const sel = selectedEye != null ? redeye[selectedEye] : null;
    hudUI = (
      <div className="hud glass">
        <div className="hud-chip">
          <span className="chip-ico">{TOOL_ICON.redeye}</span>
          <span className="chip-name">Red eye</span>
        </div>
        <div className="hsep" />
        <div className="hud-hint">
          {redeye.length === 0
            ? "Drag a box around a red eye, or click on it"
            : "Drag around another eye · click one to select · Delete removes it"}
        </div>
        {[
          { key: "size", label: "Pupil size", min: 50, max: 200, value: newEyeSize, set: setNewEyeSize },
          { key: "darken", label: "Darken", min: 0, max: 100, value: newEyeDarken, set: setNewEyeDarken },
        ].map((f) => {
          const v = sel ? sel[f.key] : f.value;
          return (
            <div key={f.key} style={{ display: "contents" }}>
              {hudField(f.label, Math.round(v * 100), "%", !!sel,
                <RangeTrack
                  min={f.min} max={f.max} step={1} value={Math.round(v * 100)}
                  onPointerDown={sel ? snapshot : undefined}
                  onChange={(e) => {
                    const nv = parseFloat(e.target.value) / 100;
                    if (sel) setRedEye((prev) => prev.map((x, i) => (i === selectedEye ? { ...x, [f.key]: nv } : x)));
                    else f.set(nv);
                  }}
                />)}
            </div>
          );
        })}
        <div className="hsep" />
        {doneBtn}
      </div>
    );
  }
  // White balance picking happens over the finished view: its HUD floats
  // over the canvas instead of docking, so the stage never resizes (a
  // resize there would re-fit the zoom the user is picking at)
  const wbHud = mode === "view" && wbPicking && (
    <div className="hud glass hud-float">
      <div className="hud-chip">
        <span className="chip-ico">{TOOL_ICON.wb}</span>
        <span className="chip-name">White balance</span>
      </div>
      <div className="hsep" />
      <div className="hud-hint">Click something that should be neutral gray or white</div>
      <button className="hud-btn" onClick={() => setWbPicking(false)}>Cancel <span className="kbd">Esc</span></button>
    </div>
  );
  const folderName = (photo.folder || "").split(/[\\/]/).filter(Boolean).pop();

  return (
    <div className="editor">
      <div className="topbar">
        <Tip text="Back to Library">
          <button className="btn ghost icon" onClick={exit} aria-label="Back to Library">{I.back}</button>
        </Tip>
        <div className="crumbs">
          {folderName && <><span className="roll">{folderName}</span><span className="sep">/</span></>}
          <span className="file">{photo.name}</span>
        </div>
        <div className="spacer" />
        {busyExport && <div className="topbar-progress">{busyExport}</div>}
        {(!isDefaultAdjust(adjust) || stamp.length > 0 || redeye.length > 0) && (
          <Tip text="Hold to compare with the original (\)">
            <button
              className={`btn ghost icon ${compareBefore ? "active" : ""}`}
              onPointerDown={(e) => { e.preventDefault(); setCompareBefore(true); }}
              onPointerUp={() => setCompareBefore(false)}
              onPointerLeave={() => setCompareBefore(false)}
              onPointerCancel={() => setCompareBefore(false)}
              aria-label="Hold to compare with the original"
            >
              {I.compare}
            </button>
          </Tip>
        )}
        <Tip text="Reset all edits">
          <button className="btn ghost icon" onClick={reset} aria-label="Reset all edits">{I.reset}</button>
        </Tip>
        <Tip text={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
          <button className="btn ghost icon" onClick={onToggleTheme} aria-label="Toggle theme">
            {theme === "dark" ? I.sun : I.moon}
          </button>
        </Tip>
        <Tip text={selected && selected.size > 0 ? `Export ${selected.size} selected photos` : "Export this photo"}>
          <button
            className="btn-export"
            onClick={async () => {
              // commit live edits (crop + adjust + stamp) first — otherwise a
              // tweak made without hitting Done/Enter first gets left behind.
              // onApply() queues an async setPhotos that isn't flushed yet, so
              // the batch-selection branch can't just re-read shared state for
              // THIS photo — hand it the fresh edits directly instead.
              const [e, preview] = await buildEdits();
              onApply(e, preview);
              if (selected && selected.size > 0) onExportSelected(photo.id, e);
              else onExport(e);
            }}
            aria-label="Export"
          >
            {I.export}Export
          </button>
        </Tip>
      </div>

      <div className="ed-body">
      {railUI}
      <div className="workspace">
      <div
        className="stage"
        ref={stageRef}
        onPointerDown={mode === "crop" ? startDrag("image") : startDrag("pan")}
        onDoubleClick={() => {
          if (!view) return;
          // in crop mode, .cropframe's own onDoubleClick (fit-to-crop-area)
          // stops propagation before this fires — reaching here means the
          // double-click landed outside the crop selection, so zoom out to
          // the whole original photo instead (zoomToWholePhoto leaves the
          // crop selection itself untouched — see its own comment above).
          if (mode === "crop") {
            zoomToWholePhoto();
            return;
          }
          if (rect) refit(rect, view, true);
        }}
        onContextMenu={(e) => {
          if (!onContextExport) return;
          e.preventDefault();
          onContextExport(photo.id, e.clientX, e.clientY);
        }}
      >
        <div className="world" ref={worldRef}>
        {view && mode === "crop" && (
          <div
            className={`imgspace ${gesturing ? "gpu" : ""}`}
            style={{
              width: iw, height: ih,
              transform: `translate(${view.icx - iw / 2}px, ${view.icy - ih / 2}px) rotate(${theta}deg) scale(${view.k})`,
            }}
          >
            <GLImage className="base" src={editSrcURL(photo)} adjust={displayAdjust} stamp={displayStamp} redeye={displayRedEye} texRegion={baseTexRegion} origDiag={origDiag} draggable={false} />
            {tile && (
              <GLImage
                className="tileimg"
                src={tile.url}
                adjust={displayAdjust}
                stamp={displayStamp}
                redeye={displayRedEye}
                texRegion={tileTexRegion}
                origDiag={origDiag}
                draggable={false}
                style={{ left: tile.sx, top: tile.sy, width: tile.sw, height: tile.sh, opacity: tile.fading ? 0 : 1 }}
              />
            )}
          </div>
        )}
        {rect && view && (mode === "view" || mode === "stamp" || mode === "redeye") && (
          <div
            className="viewclip"
            style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
          >
            <div
              className={`imgspace ${gesturing ? "gpu" : ""}`}
              style={{
                width: iw, height: ih,
                transform: `translate(${view.icx - rect.x - iw / 2}px, ${view.icy - rect.y - ih / 2}px) rotate(${theta}deg) scale(${view.k})`,
              }}
            >
              <GLImage
                className="base"
                src={editSrcURL(photo)}
                adjust={displayAdjust}
                stamp={displayStamp}
                draft={mode === "stamp" && !compareBefore ? draft : null}
                redeye={displayRedEye}
                texRegion={baseTexRegion}
                origDiag={origDiag}
                draggable={false}
              />
              {tile && (
                <GLImage
                  className="tileimg"
                  src={tile.url}
                  adjust={displayAdjust}
                  stamp={displayStamp}
                  draft={mode === "stamp" && !compareBefore ? draft : null}
                  redeye={displayRedEye}
                  texRegion={tileTexRegion}
                  origDiag={origDiag}
                  draggable={false}
                  style={{ left: tile.sx, top: tile.sy, width: tile.sw, height: tile.sh, opacity: tile.fading ? 0 : 1 }}
                />
              )}
            </div>
            {mode === "stamp" && (
              <StampOverlay
                stamp={stamp}
                setStamp={setStamp}
                draft={draft}
                setDraft={setDraft}
                selectedStroke={selectedStroke}
                setSelectedStroke={setSelectedStroke}
                snapshot={snapshot}
                view={view}
                rect={rect}
                iw={iw}
                ih={ih}
                stageRef={stageRef}
                toast={toast}
                brushRadius={brushRadius}
                feather={feather}
                opacity={stampOpacity}
                gesturing={gesturing}
                editPixelsRef={editPixelsRef}
                spaceRef={spaceRef}
                stampMode={stampMode}
              />
            )}
            {mode === "view" && wbPicking && (
              <WBPickOverlay
                view={view}
                iw={iw}
                ih={ih}
                stageRef={stageRef}
                editPixelsRef={editPixelsRef}
                spaceRef={spaceRef}
                onPick={pickWhiteBalance}
                toast={toast}
              />
            )}
            {mode === "redeye" && (
              <RedEyeOverlay
                redeye={redeye}
                setRedEye={setRedEye}
                selectedEye={selectedEye}
                setSelectedEye={setSelectedEye}
                snapshot={snapshot}
                view={view}
                rect={rect}
                iw={iw}
                ih={ih}
                stageRef={stageRef}
                toast={toast}
                editPixelsRef={editPixelsRef}
                spaceRef={spaceRef}
                newSize={newEyeSize}
                newDarken={newEyeDarken}
              />
            )}
          </div>
        )}
        {rect && mode === "crop" && (
          <div
            className={`cropframe ${drag.current?.mode === "image" ? "grabbing" : ""}`}
            style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}
            onPointerDown={dragOrPan("image")}
            onDoubleClick={(e) => {
              e.stopPropagation(); // otherwise bubbles to .stage's own onDoubleClick below, which fits the whole photo instead
              // animate=false: an animated refit here is a big, frequent
              // enough scale jump that a second quick double-click (e.g.
              // trying the crop-area vs. empty-area behaviors back to back)
              // fires cancelFlip() while the first flip's CSS transition is
              // still mid-flight, which snaps .world's transform straight to
              // "none" on top of a half-finished transition — freezing the
              // crop frame at a wrong intermediate size/position instead of
              // its correct final one, which reads as "the crop reset".
              // Snapping instantly (matching how wheel-zoom already always
              // works, with no animation at all) removes that window
              // entirely.
              if (view) refit(rect, view, false);
            }}
          >
            <div className="thirds">
              <div className="v" style={{ left: "33.333%" }} />
              <div className="v" style={{ left: "66.666%" }} />
              <div className="h" style={{ top: "33.333%" }} />
              <div className="h" style={{ top: "66.666%" }} />
            </div>
            <div className="handle corner h-nw" onPointerDown={dragOrPan("nw")} />
            <div className="handle corner h-ne" onPointerDown={dragOrPan("ne")} />
            <div className="handle corner h-sw" onPointerDown={dragOrPan("sw")} />
            <div className="handle corner h-se" onPointerDown={dragOrPan("se")} />
            <div className="handle edge h-n" onPointerDown={dragOrPan("n")} />
            <div className="handle edge h-s" onPointerDown={dragOrPan("s")} />
            <div className="handle edge h-e" onPointerDown={dragOrPan("e")} />
            <div className="handle edge h-w" onPointerDown={dragOrPan("w")} />
          </div>
        )}
        </div>
      </div>
      {hudUI && <div className="hud-dock">{hudUI}</div>}
      {wbHud}
      </div>
      <div className="meta">
        <div className="side-tabs" role="tablist">
          <button role="tab" aria-selected={sideTab === "develop"} onClick={() => setSideTab("develop")}>Develop</button>
          <button role="tab" aria-selected={sideTab === "info"} onClick={() => setSideTab("info")}>Info</button>
        </div>
        {sideTab === "develop" ? (
          <DevelopPanel
            photo={photo} iw={iw} ih={ih} crop={histCrop} adjust={adjust} onAdjustChange={setAdjust} onSliderStart={snapshot}
            wbPicking={wbPicking}
            onToggleWBPick={toggleWB}
          />
        ) : (
          <MetaPanel photo={photo} outW={outW} outH={outH} theta={theta} exif={exif} />
        )}
      </div>
      </div>

      {siblings && siblings.length >= 2 && (
        <>
          <div
            className="filmstrip-resize"
            onPointerDown={(e) => {
              e.preventDefault();
              const startY = e.clientY;
              const startH = filmstripHeight;
              const move = (ev) => {
                // dragging UP (toward smaller clientY) grows the panel — the
                // handle sits on the panel's TOP edge, same convention as
                // dragging a window's own top edge to make it taller.
                const dy = startY - ev.clientY;
                onFilmstripResize(clamp(startH + dy, FILMSTRIP_MIN_H, FILMSTRIP_MAX_H));
              };
              const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
              };
              window.addEventListener("pointermove", move);
              window.addEventListener("pointerup", up, { once: true });
            }}
          />
          <Filmstrip
            items={siblings}
            currentId={photo.id}
            onSelect={switchTo}
            selectedIds={selected}
            onToggleSel={onToggleSel}
            onRangeSel={onRangeSel}
            onContext={onContextExport}
            height={filmstripHeight}
          />
        </>
      )}
    </div>
  );
}

/* ============================================================
   EXPORT DIALOG
   ============================================================ */
function ExportDialog({ items, onClose, onSavePreset, presets, onRunPreset, onExportNative, toast }) {
  const [format, setFormat] = useState("jpeg");
  const [targetMB, setTargetMB] = useState("2");
  const [scale, setScale] = useState(100);
  const [estimate, setEstimate] = useState(null);
  const [calculating, setCalculating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [progress, setProgress] = useState(null);
  const [presetName, setPresetName] = useState("");
  const [template, setTemplate] = useState("{name}");
  const [keepExif, setKeepExif] = useState(false);
  const veilDownOnSelf = useRef(false);
  const closeOnVeil = useCallback((e) => {
    if (e.target === e.currentTarget && veilDownOnSelf.current && !exporting) onClose();
    veilDownOnSelf.current = false;
  }, [exporting, onClose]);
  const canvasCache = useRef({}); // scale -> canvas (browser fallback)
  const solverCache = useRef({}); // `${scale}|${mime}` -> solve() (browser fallback)
  const single = items.length === 1;
  const mime = format === "jpeg" ? "image/jpeg" : "image/webp";
  const targetBytes = Math.max(0.05, parseFloat(targetMB) || 0.05) * 1024 * 1024;
  const targetMBn = Math.max(0.05, parseFloat(targetMB) || 0.05);

  const getSolver = useCallback(async (item, sc, mm) => {
    const cKey = `${sc}`;
    let canvas = canvasCache.current[cKey];
    if (!canvas) {
      canvas = await bakeCanvas(item.photo, item.edits, sc);
      canvasCache.current[cKey] = canvas;
    }
    const sKey = `${sc}|${mm}`;
    if (!solverCache.current[sKey]) {
      solverCache.current[sKey] = { solve: makeSizeSolver(canvas, mm), canvas };
    }
    return solverCache.current[sKey];
  }, []);

  useEffect(() => {
    if (!single) return;
    let alive = true;
    setCalculating(true);
    const delay = estimate ? 400 : 0; // no delay on first open — nothing to show yet
    const t = setTimeout(async () => {
      try {
        if (NATIVE && items[0].photo.path) {
          const res = await window.meridian.estimatePhoto(
            { path: items[0].photo.path, name: items[0].photo.name, edits: items[0].edits },
            { format, targetMB: targetMBn, scale, keepExif }
          );
          if (alive) {
            setEstimate(res && !res.error
              ? { size: res.size, quality: res.quality / 100, over: res.over, maxed: res.maxed, w: res.w, h: res.h }
              : null);
            setCalculating(false);
          }
        } else {
          const { solve, canvas } = await getSolver(items[0], scale, mime);
          const res = await solve(targetBytes);
          if (alive) {
            setEstimate(res ? { size: res.blob.size, blob: res.blob, quality: res.quality, over: res.over, w: canvas.width, h: canvas.height } : null);
            setCalculating(false);
          }
        }
      } catch {
        if (alive) { setEstimate(null); setCalculating(false); }
      }
    }, delay);
    return () => { alive = false; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [single, items, mime, format, targetBytes, targetMBn, scale, keepExif, getSolver]);

  const browserDownload = (blob, name) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  };

  const doExport = async () => {
    if (NATIVE && items.every((it) => it.photo.path)) {
      // hand off to the app-level runner and close immediately — the actual
      // export continues in the background, progress shown in the
      // persistent top-bar indicator instead of blocking this dialog
      const serial = items.map((it) => ({ path: it.photo.path, name: it.photo.name, edits: it.edits }));
      onClose();
      onExportNative(serial, { format, targetMB: targetMBn, scale, template, keepExif });
      return;
    }

    setExporting(true);
    try {
      /* browser fallback: canvas pipeline + downloads */
      const ext = format === "jpeg" ? "jpg" : "webp";
      const prepared = [];
      if (single && estimate && estimate.blob && !calculating) {
        const { photo, edits } = items[0];
        const base = photo.name.replace(/\.[^.]+$/, "");
        prepared.push({ name: `${base}${edits ? "-edited" : ""}.${ext}`, blob: estimate.blob });
      } else {
        for (let i = 0; i < items.length; i++) {
          setProgress(items.length > 1 ? `Compressing ${i + 1} / ${items.length}…` : "Compressing…");
          const { photo, edits } = items[i];
          const canvas = await bakeCanvas(photo, edits, scale);
          const solve = makeSizeSolver(canvas, mime);
          const res = await solve(targetBytes);
          if (!res) continue;
          const base = photo.name.replace(/\.[^.]+$/, "");
          prepared.push({ name: `${base}${edits ? "-edited" : ""}.${ext}`, blob: res.blob });
        }
      }
      for (const p of prepared) {
        browserDownload(p.blob, p.name);
        if (prepared.length > 1) await new Promise((r) => setTimeout(r, 350));
      }
      toast(prepared.length > 1 ? `Exported ${prepared.length} photos` : "Photo exported");
      onClose();
    } finally {
      setExporting(false);
      setProgress(null);
    }
  };

  return (
    <div
      className="veil"
      onPointerDown={(e) => { veilDownOnSelf.current = e.target === e.currentTarget; }}
      onPointerUp={closeOnVeil}
    >
      <div className="dialog">
        <h3>Export</h3>
        <div className="sub">{single ? items[0].photo.name : `${items.length} photos selected`}</div>

        {presets && presets.length > 0 && (
          <div className="field">
            <label>Presets — one click exports straight to their folder</label>
            <div className="preset-list">
              {presets.map((pr) => (
                <button key={pr.id} className="preset-row" disabled={exporting} onClick={() => onRunPreset(pr)}>
                  <span className="cm-name">{pr.name}</span>
                  <span className="cm-sub">≤{pr.targetMB} MB · {pr.format.toUpperCase()} · {pr.scale}% → {baseName(pr.dir)}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="field">
          <label>Format</label>
          <div className="controls">
            <div className="seg">
              <button className={format === "jpeg" ? "on" : ""} onClick={() => setFormat("jpeg")}>JPEG</button>
              <button className={format === "webp" ? "on" : ""} onClick={() => setFormat("webp")}>WebP</button>
            </div>
          </div>
        </div>

        <div className="field">
          <label>Max file size</label>
          <div className="controls">
            <div className="numinput">
              <input
                type="number" min="0.1" step="0.1" value={targetMB}
                onChange={(e) => setTargetMB(e.target.value)}
              />
              <span className="unit">MB{single ? "" : " each"}</span>
            </div>
          </div>
        </div>

        <div className="field">
          <label>Resolution</label>
          <div className="controls">
            <div className="seg">
              {[100, 75, 50, 25].map((p) => (
                <button key={p} className={scale === p ? "on" : ""} onClick={() => setScale(p)}>{p}%</button>
              ))}
            </div>
          </div>
        </div>

        {NATIVE && (
          <div className="field">
            <Tip text="Adds camera, lens and date to the file">
              <label className="chk">
                <input type="checkbox" checked={keepExif} onChange={(e) => setKeepExif(e.target.checked)} />
                Keep EXIF
              </label>
            </Tip>
          </div>
        )}

        {single && (
          <div className={`estimate ${estimate && estimate.over && !calculating ? "warn" : ""}`}>
            {calculating && <span className="spin" />}
            {estimate ? (
              <span style={calculating ? { opacity: 0.55 } : undefined}>
                {estimate.over ? (
                  <>Smallest possible at this resolution is <strong>{fmtMB(estimate.size)}</strong> — lower the resolution to hit the target.</>
                ) : estimate.maxed ? (
                  <>Maximum quality reached: <strong>{fmtMB(estimate.size)}</strong> · quality 100 · {estimate.w}×{estimate.h}px — already under your limit, extra bytes wouldn't add detail.</>
                ) : (
                  <>Estimated file: <strong>{fmtMB(estimate.size)}</strong> · quality {Math.round(estimate.quality * 100)} · {estimate.w}×{estimate.h}px{NATIVE ? " · MozJPEG" : ""}</>
                )}
              </span>
            ) : calculating ? (
              <span>Calculating size…</span>
            ) : (
              <span>Estimate unavailable</span>
            )}
          </div>
        )}
        {!single && (
          <div className="estimate">
            <span>Each photo will be compressed to stay under <strong>{parseFloat(targetMB) || 0} MB</strong>{NATIVE ? " (MozJPEG / WebP)" : ""}.</span>
          </div>
        )}

        {onSavePreset && (
          <div className="field">
            <Tip text="Remembers folder, format and size"><label>Save as preset</label></Tip>
            <div className="controls">
              <div className="numinput" style={{ flex: 1 }}>
                <input
                  type="text"
                  placeholder="Preset name"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  style={{ fontFamily: "var(--sans)" }}
                />
              </div>
              <Tip text="Pick a folder and save">
                <button
                  className="btn"
                  disabled={!presetName.trim() || exporting}
                  onClick={async () => {
                    const ok = await onSavePreset({
                      name: presetName.trim(), format, targetMB: targetMBn, scale, template, keepExif,
                    });
                    if (ok) setPresetName("");
                  }}
                >
                  Save…
                </button>
              </Tip>
            </div>
          </div>
        )}

        <div className="actions">
          <button className="btn ghost" onClick={onClose} disabled={exporting}>Cancel</button>
          <button className="btn primary" onClick={doExport} disabled={exporting}>
            {exporting ? (progress || "Exporting…") : `Export${single ? "" : ` ${items.length}`}`}
          </button>
        </div>
      </div>
    </div>
  );
}

/* context menu that never falls off-screen and never "jumps":
   it is remounted per position (key), measured synchronously before
   the first paint, and only then made visible */
function CtxMenuInner({ x, y, children }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ left: x, top: y, visibility: "hidden" });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    let left = x, top = y;
    if (left + r.width > window.innerWidth - 8) left = Math.max(8, window.innerWidth - r.width - 8);
    if (top + r.height > window.innerHeight - 8) top = Math.max(8, y - r.height);
    setPos({ left, top, visibility: "visible" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div className="ctxmenu" ref={ref} style={pos} onClick={(e) => e.stopPropagation()}>
      {children}
    </div>
  );
}
function CtxMenu(props) {
  return <CtxMenuInner key={`${props.x}:${props.y}`} {...props} />;
}

/* image that fades in over the skeleton shimmer; no flash for cached files */
function Thumb({ src, alt }) {
  const r = useRef(null);
  useLayoutEffect(() => {
    if (r.current?.complete) r.current.style.opacity = 1;
  });
  return (
    <img
      ref={r}
      src={src}
      alt={alt || ""}
      loading="lazy"
      draggable={false}
      style={{ opacity: 0, transition: "opacity .18s" }}
      onLoad={(e) => { e.currentTarget.style.opacity = 1; }}
    />
  );
}

const FILMSTRIP_MIN_H = 92;   // = the previous fixed height, kept as the floor
const FILMSTRIP_MAX_H = 184;  // 2× the floor — "adequate limits", not half the screen
const SIDEBAR_MIN_W = 208;    // = the previous fixed width, kept as the floor — long folder names still ellipsize below this, same as before
const SIDEBAR_MAX_W = 416;    // 2× the floor, same "adequate limits" reasoning as the filmstrip

/* ============================================================
   APP
   ============================================================ */
export default function App() {
  const [photos, setPhotos] = useState([]);
  const [libLoaded, setLibLoaded] = useState(!NATIVE);
  const [selected, setSelected] = useState(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [activeFolder, setActiveFolder] = useState("__all__");
  const [thumb, setThumb] = useState(164);
  const [sortDesc, setSortDesc] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [exportItems, setExportItems] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [toastMsg, setToastMsg] = useState(null);
  const [watchedRoots, setWatchedRoots] = useState([]);
  const [theme, setTheme] = useState("dark");
  const [presets, setPresets] = useState([]);      // export presets {id,name,format,targetMB,scale,dir}
  const [filmstripHeight, setFilmstripHeight] = useState(FILMSTRIP_MIN_H); // persisted like theme — see the settings load/save effects below
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_MIN_W);         // persisted the same way — see the settings load/save effects below
  const [ctxMenu, setCtxMenu] = useState(null);    // {x, y, ids}
  const [minRating, setMinRating] = useState(0);   // sidebar rating filter
  const [busyExport, setBusyExport] = useState(null);
  const fileInput = useRef(null);
  const dragCount = useRef(0);
  const toastTimer = useRef(null);
  const busyExportTimer = useRef(null);
  const loadedOnce = useRef(false);
  const photosRef = useRef([]);
  useEffect(() => { photosRef.current = photos; }, [photos]);

  /* A <button> clicked with a mouse otherwise keeps DOM focus afterward —
     visually a distracting ring around whatever was last clicked (Reset,
     a library sidebar folder, ...), and functionally worse: a focused
     button intercepts Space/Enter and re-fires its own click, so an
     unrelated later keypress (e.g. Space, used elsewhere as a shortcut)
     can silently re-trigger it. preventDefault on mousedown stops a
     button from ever taking focus on a mouse click in the first place —
     the click itself still fires normally, and Tab-based keyboard
     navigation is untouched (this only intercepts mouse-originated
     mousedown). One listener here covers every button in the app, rather
     than needing this on each one individually. */
  useEffect(() => {
    const onMouseDown = (e) => {
      if (e.target.closest("button")) e.preventDefault();
    };
    window.addEventListener("mousedown", onMouseDown);
    return () => window.removeEventListener("mousedown", onMouseDown);
  }, []);

  const toast = useCallback((msg, action = null, ttl = 2600) => {
    setToastMsg(msg ? { text: msg, action } : null);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToastMsg(null), action ? Math.max(ttl, 6000) : ttl);
  }, []);

  /* shows a result message (e.g. "Photo exported") in the same top-bar slot
     export progress already uses, instead of the bottom toast — so export
     feedback stays where the user was already looking, then clears itself */
  const flashBusy = useCallback((msg, ttl = 2600) => {
    clearTimeout(busyExportTimer.current);
    setBusyExport(msg);
    busyExportTimer.current = setTimeout(() => setBusyExport(null), ttl);
  }, []);

  const undoRemoval = useCallback(async (removedPhotos, roots) => {
    setPhotos((prev) => {
      const have = new Set(prev.map((p) => p.id));
      return [...prev, ...removedPhotos.filter((p) => !have.has(p.id))];
    });
    if (NATIVE && roots.length) {
      const updated = await window.meridian.watchRoots(roots);
      setWatchedRoots(updated);
    }
    toast("Restored");
  }, [toast]);


  /* ---- persistent library (native) ---- */
  useEffect(() => {
    if (!NATIVE || loadedOnce.current) return;
    loadedOnce.current = true;
    (async () => {
      const { items, missing, watchedRoots: roots, settings } = await window.meridian.loadLibrary();
      setPhotos(items.map((rec) => ({
        ...rec,
        sizeBytes: rec.sizeBytes ?? rec.size ?? 0,
        mtime: rec.mtime ?? 0,
        rating: rec.rating ?? 0,
        url: photoURL(rec.path),
      })));
      if (settings?.theme === "light") setTheme("light");
      if (Array.isArray(settings?.exportPresets)) setPresets(settings.exportPresets);
      if (typeof settings?.filmstripHeight === "number") setFilmstripHeight(clamp(settings.filmstripHeight, FILMSTRIP_MIN_H, FILMSTRIP_MAX_H));
      if (typeof settings?.sidebarWidth === "number") setSidebarWidth(clamp(settings.sidebarWidth, SIDEBAR_MIN_W, SIDEBAR_MAX_W));
      setWatchedRoots(roots || []);
      setLibLoaded(true);
      if (missing > 0) toast(`${missing} photo${missing > 1 ? "s" : ""} removed — files no longer on disk`);
    })();
  }, [toast]);

  /* ---- helper: records [{path,name,folder}] -> photo objects with dims ---- */
  const loadRecords = useCallback(async (records) => {
    const base = Date.now();
    const loaded = [];
    for (let i = 0; i < records.length; i++) {
      const r = records[i];
      const url = photoURL(r.path);
      try {
        const img = await loadImage(url);
        loaded.push({
          id: uid(), path: r.path, name: r.name, folder: r.folder || "",
          url, w: img.naturalWidth, h: img.naturalHeight,
          sizeBytes: r.size || 0, mtime: r.mtime || 0, rating: 0,
          addedAt: base + i, edits: null, previewUrl: null,
        });
      } catch {
        imgCache.delete(url);
      }
    }
    return loaded;
  }, []);

  /* ---- live folder sync: watched folders push changes from disk ---- */
  useEffect(() => {
    if (!NATIVE) return;
    const off = window.meridian.onFolderScan(async ({ root, records }) => {
      const present = new Set(records.map((r) => r.path));
      const currentPaths = new Set(photosRef.current.map((p) => p.path));
      const fresh = records.filter((r) => !currentPaths.has(r.path));
      const loaded = await loadRecords(fresh);

      let removed = 0;
      const recByPath = new Map(records.map((r) => [r.path, r]));
      setPhotos((prev) => {
        const kept = prev
          .filter((p) => !p.path || !isUnder(p.path, root) || present.has(p.path))
          .map((p) => {
            const r = p.path && recByPath.get(p.path);
            if (r && (r.size !== p.sizeBytes || r.mtime !== p.mtime)) {
              return { ...p, sizeBytes: r.size, mtime: r.mtime };
            }
            return p;
          });
        removed = prev.length - kept.length;
        const keptPaths = new Set(kept.map((p) => p.path));
        const add = loaded.filter((p) => !keptPaths.has(p.path));
        return add.length ? [...kept, ...add] : kept;
      });

      if (loaded.length || removed) {
        const parts = [];
        if (loaded.length) parts.push(`+${loaded.length}`);
        if (removed) parts.push(`−${removed}`);
        toast(`"${baseName(root)}" synced (${parts.join(", ")})`);
      }
    });
    return off;
  }, [loadRecords, toast]);

  useEffect(() => {
    if (!NATIVE || !libLoaded) return;
    const t = setTimeout(() => {
      // previewUrl is a blob: URL scoped to this session — dead on restart
      const serializable = photos.map(({ url, previewUrl, ...rest }) => rest);
      window.meridian.saveLibrary(serializable, { theme, exportPresets: presets, filmstripHeight, sidebarWidth });
    }, 400);
    return () => clearTimeout(t);
  }, [photos, libLoaded, theme, presets, filmstripHeight, sidebarWidth]);

  /* ---- import: native records [{path, name, folder}] ---- */
  const addNativeRecords = useCallback(async (records) => {
    if (!records || !records.length) { toast("No supported images found"); return; }
    const existing = new Set(photosRef.current.map((p) => p.path));
    const fresh = records.filter((r) => !existing.has(r.path));
    if (!fresh.length) { toast("These photos are already in the library"); return; }
    const loaded = await loadRecords(fresh);
    if (!loaded.length) { toast("Couldn't decode those files"); return; }
    setPhotos((prev) => [...prev, ...loaded]);
    toast(`Imported ${loaded.length} photo${loaded.length > 1 ? "s" : ""}`);
  }, [loadRecords, toast]);

  /* ---- import: browser fallback [{file, folder}] ---- */
  const addBrowserFiles = useCallback(async (entries) => {
    if (!entries.length) { toast("No supported images found"); return; }
    const base = Date.now();
    const loaded = [];
    for (let i = 0; i < entries.length; i++) {
      const { file, folder } = entries[i];
      const url = URL.createObjectURL(file);
      try {
        const img = await loadImage(url);
        loaded.push({
          id: uid(), name: file.name, folder: folder || "",
          url, w: img.naturalWidth, h: img.naturalHeight,
          sizeBytes: file.size || 0, mtime: file.lastModified || 0, rating: 0,
          addedAt: base + i, edits: null, previewUrl: null,
        });
      } catch {
        URL.revokeObjectURL(url);
        imgCache.delete(url);
      }
    }
    if (!loaded.length) { toast("Couldn't decode those files"); return; }
    setPhotos((prev) => [...prev, ...loaded]);
    toast(`Imported ${loaded.length} photo${loaded.length > 1 ? "s" : ""}`);
  }, [toast]);

  const importPhotosOrFolder = async () => {
    if (NATIVE) {
      const paths = await window.meridian.pickFilesOrFolder();
      if (!paths?.length) return;
      const { records, roots } = await window.meridian.importPaths(paths);
      if (roots?.length) {
        setWatchedRoots((prev) => [...prev, ...roots.filter((r) => !prev.includes(r))]);
      }
      await addNativeRecords(records);
    } else {
      fileInput.current?.click();
    }
  };

  const onPickFiles = (e) => {
    const list = Array.from(e.target.files || [])
      .filter(isImageFile)
      .map((file) => {
        const rel = file.webkitRelativePath || "";
        const folder = rel.includes("/") ? rel.split("/").slice(0, -1).join("/") : "";
        return { file, folder };
      });
    addBrowserFiles(list);
    e.target.value = "";
  };

  const onDrop = async (e) => {
    e.preventDefault();
    dragCount.current = 0;
    setDragOver(false);
    if (NATIVE) {
      const paths = Array.from(e.dataTransfer.files || [])
        .map((f) => {
          try { return window.meridian.getPathForFile(f); } catch { return null; }
        })
        .filter(Boolean);
      const { records, roots } = await window.meridian.importPaths(paths);
      if (roots?.length) {
        setWatchedRoots((prev) => [...prev, ...roots.filter((r) => !prev.includes(r))]);
      }
      addNativeRecords(records);
    } else {
      const entries = await filesFromDataTransfer(e.dataTransfer);
      addBrowserFiles(entries);
    }
  };

  /* ---- library derived state: hierarchical folder tree ---- */
  const folderTree = useMemo(() => {
    const counts = new Map(); // label -> photos in the whole subtree
    const labels = new Set();
    for (const p of photos) {
      if (!p.folder) continue;
      const parts = p.folder.split("/");
      let acc = "";
      for (let i = 0; i < parts.length; i++) {
        acc = i ? `${acc}/${parts[i]}` : parts[i];
        labels.add(acc);
        counts.set(acc, (counts.get(acc) || 0) + 1);
      }
    }
    return [...labels].sort((a, b) => a.localeCompare(b)).map((label) => ({
      label,
      depth: label.split("/").length - 1,
      name: label.split("/").pop(),
      count: counts.get(label) || 0,
    }));
  }, [photos]);

  const visible = useMemo(() => {
    let list = photos;
    if (activeFolder === "__edited__") list = list.filter((p) => p.edits);
    else if (activeFolder !== "__all__") {
      // a folder node shows its whole subtree
      list = list.filter(
        (p) => p.folder === activeFolder || (p.folder && p.folder.startsWith(activeFolder + "/"))
      );
    }
    if (minRating > 0) list = list.filter((p) => (p.rating || 0) >= minRating);
    return [...list].sort((a, b) => (sortDesc ? b.addedAt - a.addedAt : a.addedAt - b.addedAt));
  }, [photos, activeFolder, sortDesc, minRating]);

  useEffect(() => {
    if (activeFolder !== "__all__" && activeFolder !== "__edited__" &&
        !folderTree.some((n) => n.label === activeFolder)) {
      setActiveFolder("__all__");
    }
  }, [folderTree, activeFolder]);

  /* ---- selection / delete ---- */
  const toggleSelect = (id) => {
    setSelected((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };
  const clearSelection = () => { setSelected(new Set()); setSelectMode(false); };

  /* Esc cancels a library selection — but not while the Editor, a context
     menu, or the export dialog is open, since each of those already owns
     Esc for its own close/cancel behavior. */
  useEffect(() => {
    if (editingId || ctxMenu || exportItems) return;
    const onKey = (e) => {
      if (e.key !== "Escape" || selected.size === 0) return;
      const tag = e.target?.tagName;
      const inTextField = (tag === "INPUT" && e.target.type !== "range") || tag === "TEXTAREA";
      if (!inTextField) clearSelection();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editingId, ctxMenu, exportItems, selected]);

  const undoResetSelected = (prevState) => {
    setPhotos((prev) => prev.map((p) => {
      const found = prevState.find((x) => x.id === p.id);
      return found ? { ...p, edits: found.edits, previewUrl: found.previewUrl } : p;
    }));
    for (const p of prevState) cachePreviewToDisk(p.path, p.previewUrl ?? null);
    toast("Restored edits");
  };

  /* Bulk equivalent of the Editor's own per-photo "Reset" button (onApply
     (null, null) there) — clears crop/adjust/stamp for every selected
     photo at once. Captures each photo's previous edits/previewUrl first
     so the toast's Undo action can restore them exactly. */
  const resetSelected = () => {
    const targets = photosRef.current.filter((p) => selected.has(p.id));
    if (!targets.length) return;
    const prevState = targets.map((p) => ({ id: p.id, path: p.path, edits: p.edits, previewUrl: p.previewUrl }));
    setPhotos((prev) => prev.map((p) => (selected.has(p.id) ? { ...p, edits: null, previewUrl: null } : p)));
    for (const p of targets) cachePreviewToDisk(p.path, null);
    toast(
      `Reset ${targets.length} photo${targets.length === 1 ? "" : "s"}`,
      { label: "Undo", fn: () => undoResetSelected(prevState) }
    );
  };

  /* ---- migrate: folders imported before watching existed start syncing
         automatically, without re-adding them ---- */
  const migrated = useRef(false);
  useEffect(() => {
    if (!NATIVE || !libLoaded || migrated.current) return;
    migrated.current = true;
    (async () => {
      const have = new Set(watchedRoots.map((r) => r.replace(/\\/g, "/")));
      const roots = new Set();
      for (const p of photosRef.current) {
        const r = deriveRoot(p);
        if (r && !have.has(r)) roots.add(r);
      }
      if (roots.size) {
        const updated = await window.meridian.watchRoots([...roots]);
        setWatchedRoots(updated);
      }
    })();
  }, [libLoaded, watchedRoots]);

  /* ---- per-folder sync toggle (sidebar) ---- */
  const rootForGroup = useCallback((group) =>
    watchedRoots.find((r) => baseName(r) === group) || null,
  [watchedRoots]);

  /* ---- remove a whole folder from the library ---- */
  const removeFolder = useCallback(async (folderLabel) => {
    const group = folderLabel.split("/")[0];
    const inFolder = (p) =>
      p.folder === folderLabel || (p.folder && p.folder.startsWith(folderLabel + "/"));
    const targets = photosRef.current.filter(inFolder);
    if (!targets.length) return;
    const unwatched = [];
    if (NATIVE) {
      const watched = rootForGroup(group);
      if (watched) {
        await window.meridian.unwatchRoot(watched);
        setWatchedRoots((prev) => prev.filter((r) => r !== watched));
        unwatched.push(watched);
      }
    }
    setPhotos((prev) => prev.filter((p) => !inFolder(p)));
    toast(
      `Removed "${folderLabel.split("/").pop()}" (${targets.length} photo${targets.length > 1 ? "s" : ""})${NATIVE ? " — files stay on disk" : ""}`,
      { label: "Undo", fn: () => undoRemoval(targets, unwatched) }
    );
  }, [rootForGroup, toast, undoRemoval]);

  /* ---- ratings ---- */
  const setRating = useCallback((id, rating) => {
    setPhotos((prev) => prev.map((p) => (p.id === id ? { ...p, rating } : p)));
  }, []);

  /* ---- export presets & context menu ---- */
  useEffect(() => {
    if (!NATIVE || !window.meridian.onExportProgress) return;
    const off = window.meridian.onExportProgress(({ done, total, name }) => {
      if (done < total) setBusyExport(`Exporting ${done + 1} / ${total}${name ? ` — ${name}` : ""}…`);
      else setBusyExport(null);
    });
    return off;
  }, []);

  const openContextMenu = useCallback((photoId, x, y) => {
    const ids = selected.has(photoId) && selected.size > 0 ? [...selected] : [photoId];
    setCtxMenu({ x, y, type: "export", ids });
  }, [selected]);

  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    const onKey = (e) => { if (e.key === "Escape") close(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [ctxMenu]);

  const serializeItems = useCallback((ids) =>
    photosRef.current
      .filter((p) => ids.includes(p.id) && p.path)
      .map((p) => ({ path: p.path, name: p.name, edits: p.edits })),
  []);

  const runPreset = useCallback(async (preset, ids) => {
    setCtxMenu(null);
    const items = serializeItems(ids);
    if (!items.length) return;
    setBusyExport(`Exporting ${items.length}…`);
    try {
      const res = await window.meridian.exportPhotos(items, {
        format: preset.format, targetMB: preset.targetMB, scale: preset.scale, dir: preset.dir,
        template: preset.template || "{name}", keepExif: !!preset.keepExif,
      });
      if (res.ok) {
        flashBusy(`"${preset.name}": exported ${res.exported}${res.failed ? `, ${res.failed} failed` : ""} → ${baseName(res.dir)}`);
      } else {
        setBusyExport(null);
      }
    } catch {
      setBusyExport(null);
    }
  }, [serializeItems, flashBusy]);

  /* fires from ExportDialog's Export button, which closes itself immediately
     — this runs the actual native export in the background, independent of
     the (by-then-unmounted) dialog, with progress surfaced through the same
     persistent busyExport indicator runPreset above already uses */
  const runExport = useCallback(async (serial, opts) => {
    setBusyExport(serial.length > 1 ? `Exporting ${serial.length}…` : "Exporting…");
    try {
      const res = await window.meridian.exportPhotos(serial, {
        format: opts.format, targetMB: opts.targetMB, scale: opts.scale, dir: null,
        template: opts.template, keepExif: opts.keepExif,
      });
      if (res.canceled) { setBusyExport(null); return; }
      flashBusy(serial.length > 1
        ? `Exported ${res.exported}${res.failed ? `, ${res.failed} failed` : ""}`
        : "Photo exported");
    } catch {
      setBusyExport(null);
    }
  }, [flashBusy]);

  const savePreset = useCallback(async (draft) => {
    const dir = await window.meridian.pickDir();
    if (!dir) return false;
    setPresets((prev) => [...prev, { ...draft, dir, id: uid() }]);
    toast(`Preset "${draft.name}" saved → ${baseName(dir)}`);
    return true;
  }, [toast]);

  const deletePreset = useCallback((id) => {
    setPresets((prev) => prev.filter((p) => p.id !== id));
  }, []);

  /* ---- editor plumbing ---- */
  const editingPhoto = photos.find((p) => p.id === editingId) || null;

  const editorSiblings = useMemo(() => {
    if (!editingPhoto) return [];
    return photos
      .filter((p) => p.folder === editingPhoto.folder)
      .sort((a, b) => (sortDesc ? b.addedAt - a.addedAt : a.addedAt - b.addedAt));
  }, [photos, editingPhoto, sortDesc]);

  const applyEdits = (edits, previewUrl) => {
    setPhotos((prev) => prev.map((p) => (p.id === editingId ? { ...p, edits, previewUrl } : p)));
  };

  const switchEditing = (edits, previewUrl, nextId) => {
    setPhotos((prev) => prev.map((p) => (p.id === editingId ? { ...p, edits, previewUrl } : p)));
    setEditingId(nextId);
  };

  const openExportFromEditor = (liveEdits) => {
    if (!editingPhoto) return;
    setExportItems([{ photo: editingPhoto, edits: liveEdits }]);
  };

  /* overrideId/overrideEdits: the Editor's Export button calls onApply()
     (an async setPhotos) immediately before this — since that update isn't
     flushed yet, `photos` here would still hold the pre-edit version of
     whichever photo is currently open. The override lets that one call site
     hand over the edits it just computed directly, instead of this reading
     stale state a render behind. */
  const exportSelected = (overrideId, overrideEdits) => {
    const items = photos
      .filter((p) => selected.has(p.id))
      .map((p) => ({ photo: p, edits: p.id === overrideId ? overrideEdits : p.edits }));
    if (items.length) setExportItems(items);
  };

  const anchorIdx = useRef(null);

  /* ---- virtualized grid: only visible cells are rendered ---- */
  const gridRef = useRef(null);
  const [gridScroll, setGridScroll] = useState(0);
  const [gridSize, setGridSize] = useState({ w: 0, h: 0 });
  const scrollRaf = useRef(false);
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setGridSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setGridSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, [photos.length > 0]);
  const onGridScroll = () => {
    if (scrollRaf.current) return;
    scrollRaf.current = true;
    requestAnimationFrame(() => {
      scrollRaf.current = false;
      if (gridRef.current) setGridScroll(gridRef.current.scrollTop);
    });
  };
  const GAP = 6, PADG = 18;
  const cols = Math.max(1, Math.floor((gridSize.w - PADG * 2 + GAP) / (thumb + GAP)));
  const cellW = cols > 0 ? (gridSize.w - PADG * 2 - GAP * (cols - 1)) / cols : thumb;
  const rowH = cellW + GAP;
  const totalRows = Math.ceil(visible.length / cols);
  const totalH = Math.max(0, totalRows * rowH - GAP) + PADG * 2;
  const startRow = Math.max(0, Math.floor((gridScroll - PADG) / rowH) - 2);
  const endRow = Math.min(totalRows, Math.ceil((gridScroll + gridSize.h - PADG) / rowH) + 2);
  const winStart = startRow * cols;
  const winEnd = Math.min(visible.length, endRow * cols);

  const rangeSelect = useCallback((ids) => {
    setSelected((s) => new Set([...s, ...ids]));
  }, []);

  const cellClick = (p, idx, e) => {
    if (e.shiftKey && anchorIdx.current !== null) {
      const [a, b] = [Math.min(anchorIdx.current, idx), Math.max(anchorIdx.current, idx)];
      rangeSelect(visible.slice(a, b + 1).map((x) => x.id));
      return;
    }
    if (e.ctrlKey || e.metaKey) {
      toggleSelect(p.id);
      anchorIdx.current = idx;
      return;
    }
    if (selectMode || selected.size > 0) {
      toggleSelect(p.id);
      anchorIdx.current = idx;
    } else {
      setEditingId(p.id);
    }
  };

  return (
    <div
      className={`app ${theme === "light" ? "light" : ""}`}
      onDragEnter={(e) => { e.preventDefault(); dragCount.current++; setDragOver(true); }}
      onDragLeave={() => { dragCount.current--; if (dragCount.current <= 0) { dragCount.current = 0; setDragOver(false); } }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
    >
      <style>{CSS}</style>

      <div className="topbar">
        <div className="brand"><span className="dot" /><span className="name">SPLICER LAB</span></div>
        <Tip text="Import photos or a folder">
          <button className="btn ghost icon" onClick={importPhotosOrFolder} aria-label="Import photos or a folder">{I.upload}</button>
        </Tip>
        {!NATIVE && (
          <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={onPickFiles} />
        )}
        <div className="spacer" />
        {busyExport && <div className="topbar-progress">{busyExport}</div>}
        {photos.length > 0 && (
          <>
            <Tip text={sortDesc ? "Sort: Newest first" : "Sort: Oldest first"}>
              <button className="btn ghost icon" onClick={() => setSortDesc((s) => !s)} aria-label="Toggle sort order">
                {I.sort}
              </button>
            </Tip>
            <div className="thumbslider">
              <input
                type="range" className="rot" style={{ width: 110 }}
                min={96} max={280} value={thumb}
                onChange={(e) => setThumb(+e.target.value)}
              />
            </div>
            {selected.size > 0 ? (
              <>
                {/* only the left side needs extra room: the slider has no
                    inset of its own, while the icon buttons on the right
                    already carry ~10px of inset around their glyph — adding
                    padding on both sides double-counted that and pushed the
                    right gap too wide. */}
                <span style={{ color: "var(--muted)", fontSize: 13, paddingLeft: 10 }}>{selected.size}</span>
                <Tip text="Cancel selection">
                  <button className="btn ghost icon" onClick={clearSelection} aria-label="Cancel selection">{I.close}</button>
                </Tip>
                <Tip text="Reset all edits on the selected photos">
                  <button className="btn ghost icon" onClick={resetSelected} aria-label="Reset selected">{I.reset}</button>
                </Tip>
                <Tip text={`Export ${selected.size} selected photos`}>
                  <button className="btn ghost icon" onClick={() => exportSelected()} aria-label="Export selected">{I.export}</button>
                </Tip>
              </>
            ) : (
              <Tip text="Select photos">
                <button className={`btn ghost icon ${selectMode ? "active" : ""}`} onClick={() => setSelectMode((s) => !s)} aria-label="Select photos">
                  {I.select}
                </button>
              </Tip>
            )}
          </>
        )}
        <Tip text={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
          <button
            className="btn ghost icon"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? I.sun : I.moon}
          </button>
        </Tip>
      </div>

      <div className="body">
        <div className="sidebar" style={{ width: sidebarWidth }}>
          <button className={`side-item ${activeFolder === "__all__" ? "active" : ""}`} onClick={() => setActiveFolder("__all__")}>
            {I.photos} All Photos <span className="count">{photos.length}</span>
          </button>
          <button className={`side-item ${activeFolder === "__edited__" ? "active" : ""}`} onClick={() => setActiveFolder("__edited__")}>
            {I.level} Edited <span className="count">{photos.filter((p) => p.edits).length}</span>
          </button>
          <div className="side-label">Sort by rating</div>
          <div className="side-rating">
            {[1, 2, 3, 4, 5].map((n) => (
              <Tip key={n} text={`Show ★${n} and up`}>
                <button
                  className={n <= minRating ? "lit" : ""}
                  onClick={() => setMinRating((m) => (m === n ? 0 : n))}
                >
                  ★
                </button>
              </Tip>
            ))}
            {minRating > 0 && (
              <Tip text="Clear filter">
                <button className="clear" onClick={() => setMinRating(0)} aria-label="Clear filter">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
              </Tip>
            )}
          </div>
          {folderTree.length > 0 && <div className="side-label">Folders</div>}
          {folderTree.map((node) => {
            const { label: f, depth, name, count } = node;
            const isRoot = depth === 0;
            return (
              <Tip key={f} text={`${f} · right-click to remove`}>
              <button
                className={`side-item ${activeFolder === f ? "active" : ""}`}
                style={{ paddingLeft: 10 + depth * 16 }}
                onClick={() => setActiveFolder(f)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  setCtxMenu({ x: e.clientX, y: e.clientY, type: "folder", label: f, name });
                }}
              >
                {isRoot ? I.folder : (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" style={{ opacity: .6 }}><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/></svg>
                )}
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                <span className="count">{count}</span>
              </button>
              </Tip>
            );
          })}
        </div>

        <div
          className="sidebar-resize"
          onPointerDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startW = sidebarWidth;
            const move = (ev) => {
              const dx = ev.clientX - startX; // dragging right grows the sidebar
              setSidebarWidth(clamp(startW + dx, SIDEBAR_MIN_W, SIDEBAR_MAX_W));
            };
            const up = () => {
              window.removeEventListener("pointermove", move);
              window.removeEventListener("pointerup", up);
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up, { once: true });
          }}
        />

        <div className="main">
          {photos.length === 0 ? (
            <div className="empty">
              <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="var(--faint)" strokeWidth="1.4"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="8.5" cy="10" r="1.6"/><path d="M21 15l-5-5-9 9"/></svg>
              <h2>{libLoaded ? "Your library is empty" : "Loading library…"}</h2>
              {libLoaded && (
                <>
                  <p>Drop photos or folders anywhere in this window, or use the import button. Originals on disk are never modified — every edit can be reverted.</p>
                  <div className="row">
                    <button className="btn primary" onClick={importPhotosOrFolder}>{I.upload} Import photos or a folder</button>
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="gridwrap" ref={gridRef} onScroll={onGridScroll} style={{ padding: 0 }}>
              <div style={{ position: "relative", height: totalH }}>
                {visible.slice(winStart, winEnd).map((p, i) => {
                  const idx = winStart + i;
                  const row = Math.floor(idx / cols), col = idx % cols;
                  const cellStyle = {
                    position: "absolute",
                    left: PADG + col * (cellW + GAP),
                    top: PADG + row * rowH,
                    width: cellW,
                    height: cellW,
                  };
                  const isSel = selected.has(p.id);
                  return (
                    <Tip key={p.id} text={p.name}>
                      <div
                        style={cellStyle}
                        className={`cell ${isSel ? "selected" : ""} ${selectMode || selected.size > 0 ? "selectmode" : ""}`}
                        onClick={(e) => cellClick(p, idx, e)}
                        onContextMenu={(e) => {
                          if (!NATIVE) return;
                          e.preventDefault();
                          openContextMenu(p.id, e.clientX, e.clientY);
                        }}
                        onMouseDown={(e) => { if (e.shiftKey) e.preventDefault(); }}
                      >
                        <Thumb src={p.previewUrl || thumbURL(p)} alt={p.name} />
                        <div
                          className="check"
                          onClick={(e) => { e.stopPropagation(); toggleSelect(p.id); anchorIdx.current = idx; }}
                        >
                          {isSel && I.check}
                        </div>
                        {p.edits && <div className="badge">Edited</div>}
                        {p.rating > 0 && <div className="rbadge">{"★".repeat(p.rating)}</div>}
                      </div>
                    </Tip>
                  );
                })}
              </div>
            </div>
          )}
          {dragOver && <div className="dropveil">Drop photos or folders to import</div>}
        </div>
      </div>

      {editingPhoto && (
        <Editor
          key={editingPhoto.id}
          photo={editingPhoto}
          siblings={editorSiblings}
          suppressKeys={!!exportItems}
          busyExport={busyExport}
          theme={theme}
          onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          onClose={() => setEditingId(null)}
          onApply={applyEdits}
          onSwitch={switchEditing}
          onExport={openExportFromEditor}
          onExportSelected={exportSelected}
          selected={selected}
          onToggleSel={toggleSelect}
          onRangeSel={rangeSelect}
          onContextExport={NATIVE ? openContextMenu : null}
          onRate={setRating}
          toast={toast}
          filmstripHeight={filmstripHeight}
          onFilmstripResize={setFilmstripHeight}
        />
      )}

      {exportItems && (
        <ExportDialog
          items={exportItems}
          onClose={() => setExportItems(null)}
          onSavePreset={NATIVE ? savePreset : null}
          presets={NATIVE ? presets : []}
          onRunPreset={(pr) => {
            const ids = exportItems.map((i) => i.photo.id);
            setExportItems(null);
            runPreset(pr, ids);
          }}
          onExportNative={runExport}
          toast={toast}
        />
      )}

      {ctxMenu && ctxMenu.type === "folder" && (
        <CtxMenu x={ctxMenu.x} y={ctxMenu.y}>
          <div className="cm-title">{ctxMenu.name}</div>
          <div
            className="cm-item"
            onClick={() => { const f = ctxMenu.label; setCtxMenu(null); removeFolder(f); }}
          >
            <div className="cm-main">
              <span className="cm-name" style={{ color: "var(--danger)" }}>Remove folder from library</span>
              <span className="cm-sub">files stay on disk</span>
            </div>
          </div>
        </CtxMenu>
      )}

      {ctxMenu && ctxMenu.type === "export" && (
        <CtxMenu x={ctxMenu.x} y={ctxMenu.y}>
          <div className="cm-title">Export {ctxMenu.ids.length > 1 ? `${ctxMenu.ids.length} photos` : "photo"}</div>
          {presets.map((pr) => (
            <div key={pr.id} className="cm-item" onClick={() => runPreset(pr, ctxMenu.ids)}>
              <div className="cm-main">
                <span className="cm-name">{pr.name}</span>
                <span className="cm-sub">≤{pr.targetMB} MB · {pr.format.toUpperCase()} · {pr.scale}% → {baseName(pr.dir)}</span>
              </div>
              <Tip text="Delete preset">
                <button
                  className="cm-del"
                  onClick={(e) => { e.stopPropagation(); deletePreset(pr.id); }}
                  aria-label="Delete preset"
                >
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
              </Tip>
            </div>
          ))}
          {presets.length === 0 && (
            <div className="cm-empty">No presets yet — create one in the export dialog</div>
          )}
          <div className="cm-sep" />
          <div
            className="cm-item"
            onClick={() => {
              const items = photosRef.current
                .filter((p) => ctxMenu.ids.includes(p.id))
                .map((p) => ({ photo: p, edits: p.edits }));
              setCtxMenu(null);
              if (items.length) setExportItems(items);
            }}
          >
            <div className="cm-main"><span className="cm-name">Export with settings…</span></div>
          </div>
        </CtxMenu>
      )}

      {toastMsg && (
        <div className="toast">
          {toastMsg.text}
          {toastMsg.action && (
            <button
              className="toast-act"
              onClick={() => { const fn = toastMsg.action.fn; setToastMsg(null); fn(); }}
            >
              {toastMsg.action.label}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
