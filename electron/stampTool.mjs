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
   ============================================================ */

const DEFAULT_FEATHER = 0.3;      // default feather zone width, as a fraction of radius (Photoshop's brush "hardness", inverted)
const DEFAULT_OPACITY = 1;        // default strength — 1 = fully replace, less = blend with the original (Photoshop's Opacity/Flow)

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
  const dxs = pts.map((p) => (p.dx ?? stroke.dx));
  const dys = pts.map((p) => (p.dy ?? stroke.dy));

  const readMargin = outer + 1;
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
  // whole read footprint falls off-buffer (e.g. a stroke that's off-frame
  // in a cropped preview, or outside a deep-zoom tile's crop window) —
  // bail before allocating the snapshot below.
  if (rx1 < rx0 || ry1 < ry0) return;
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
  const best = new Map();
  for (let s = 0; s < segCount; s++) {
    const i0 = s, i1 = pts.length > 1 ? s + 1 : 0;
    const ax = pts[i0].x, ay = pts[i0].y, bx = pts[i1].x, by = pts[i1].y;
    const sx0 = Math.max(0, Math.floor(Math.min(ax, bx) - outer));
    const sx1 = Math.min(width - 1, Math.ceil(Math.max(ax, bx) + outer));
    const sy0 = Math.max(0, Math.floor(Math.min(ay, by) - outer));
    const sy1 = Math.min(height - 1, Math.ceil(Math.max(ay, by) + outer));
    if (sx1 < sx0 || sy1 < sy0) continue;
    for (let y = sy0; y <= sy1; y++) {
      for (let x = sx0; x <= sx1; x++) {
        const c = closestOnSegment(x, y, ax, ay, bx, by);
        const dist = Math.hypot(x - c.x, y - c.y);
        if (dist > outer) continue;
        const key = y * width + x;
        const prev = best.get(key);
        if (prev && prev.dist <= dist) continue;
        const t = c.t;
        best.set(key, {
          dist,
          dx: dxs[i0] + (dxs[i1] - dxs[i0]) * t,
          dy: dys[i0] + (dys[i1] - dys[i0]) * t,
        });
      }
    }
  }

  for (const [key, v] of best) {
    const alphaGeom = 1 - smoothstep(radius, outer, v.dist);
    if (alphaGeom <= 0) continue;
    const x = key % width, y = (key - x) / width;
    // DITHERED feather — see the module header's own section on this.
    // Not color correction: this never reads the destination's own
    // surrounding content, only the brush's geometric falloff.
    const dithered = alphaGeom >= 1 ? 1 : ditherThreshold(x, y) < alphaGeom ? 1 : 0;
    const alpha = dithered * opacity;
    if (alpha <= 0) continue;
    const [sr, sg, sb] = sampleNearest(snap, rw, rh, 3, x + v.dx - rx0, y + v.dy - ry0);
    const o = key * channels;
    // Raw 1:1 copy — no correction added to sr/sg/sb, matching real
    // Photoshop's Clone Stamp (see module header).
    buf[o] = buf[o] * (1 - alpha) + sr * alpha;
    buf[o + 1] = buf[o + 1] * (1 - alpha) + sg * alpha;
    buf[o + 2] = buf[o + 2] * (1 - alpha) + sb * alpha;
  }
}

/* Mutates buf in place (a Node Buffer or a Uint8ClampedArray alike) and
   returns it. `stamp` is an array of strokes (see header); strokes are
   composited in array order, so a later stroke's source/blend can
   legitimately read pixels an earlier stroke in the same call already
   modified (matches stacking clone-stamp edits in Photoshop). No-ops
   when isEmptyStamp(stamp). */
export function applyStampToRGBBuffer(buf, width, height, channels, stamp) {
  if (isEmptyStamp(stamp)) return buf;
  for (const stroke of normalizeStamp(stamp)) {
    applyStrokeToBuffer(buf, width, height, channels, stroke);
  }
  return buf;
}
