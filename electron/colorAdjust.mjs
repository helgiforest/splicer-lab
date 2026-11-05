/* ============================================================
   Shared tone/color adjustment math — used by:
     - the live WebGL preview shader in src/App.jsx (GLSL twin,
       marked "KEEP IN SYNC WITH electron/colorAdjust.mjs")
     - the renderer's CPU bake paths (bakePreview/bakeCanvas in
       src/App.jsx, thumbnails + browser-fallback export)
     - the native export pipeline (bakeRaw in electron/main.cjs)

   .mjs on purpose: Node always treats this as ESM regardless of
   package.json's missing "type": "module", so electron/main.cjs
   (CommonJS) can `await import()` it, while Vite bundles it for
   the renderer like any other ES module. Lives under electron/
   rather than src/ because electron-builder's `files` allowlist
   only packages "electron/**", not "src/**" — a src/ path would
   404 in a built app even though it works fine in dev.

   All strength constants below are eyeballed, not derived from
   any Adobe/Apple spec — expect a tuning pass. Known simplifications:
   white balance is a per-channel gain model (not a true Kelvin/CCT
   chromatic-adaptation transform), contrast is a linear pivot (not
   a parametric spline), and Highlights/Shadows/White Point operate on
   an already-clipped 8-bit sRGB buffer so they can lift/darken existing
   tones but can't reconstruct genuinely blown detail the way
   RAW-domain recovery can.

   LINEAR-LIGHT white balance/exposure/saturation, not straight sRGB:
   every serious raw/photo pipeline (darktable's exposure & white-balance
   modules, RawTherapee, Lightroom's Camera Raw engine) does these three
   specifically in LINEAR light, not on the gamma-encoded display values —
   because that's what they physically are: white balance is a per-channel
   gain on how much of each color of LIGHT hit the sensor, exposure is a
   gain on how much light there was, period, and Rec.709 luma weights
   (used here for saturation's own "distance from gray") are themselves
   only defined for linear RGB. Multiplying the GAMMA-encoded (sRGB, ~x^
   (1/2.4)-shaped) value by a stops factor instead — which is what this
   file used to do — means the same physical gain lands very differently
   depending on where a pixel already sits on that curve: shadows/
   midtones swing much harder than a true exposure change would (gamma
   compresses a lot of real dynamic range into a small band of code
   values there), while highlights clip almost immediately. That's
   exactly the "touchy/imprecise, highlights clip too fast" character
   reported firsthand ("не очень точные... изменения не плавные").
   srgbToLinear/linearToSrgb below are the standard IEC 61966-2-1 sRGB
   transfer function (the real piecewise curve, not a naive x^2.2), not
   darktable's own source (GPL, and not something to fold into this
   codebase's license) — this is the same publicly-documented transfer
   function every color-managed tool implements from the spec.
   Contrast and Highlights/Shadows deliberately STAY in gamma/display
   space: they're each already a "how should this look on screen"
   judgment (a simple contrast slider's usual convention pivots on the
   displayed midpoint; the highlight/shadow luma mask is explicitly a
   perceived-brightness classifier), not a physical light quantity, so
   there's no equivalent "this is measurably wrong" case for moving them
   the way there was for exposure/white-balance/saturation.

   PROCESS VERSIONS (`adjust.pv`). Everything above describes process
   version 1, and it stays exactly as it was — byte for byte — for every
   adjust saved before pv existed (no `pv` field ⇒ 1), so a photo edited
   earlier never changes its look. New edits get pv 2 (defaultAdjust()),
   which changes four things — same idea as Lightroom's "Process
   Version", re-implemented here from the published techniques, not from
   anyone's code:
   - Exposure: same linear-light gain, plus a soft highlight SHOULDER that
     grows with positive exposure (identity at 0 EV), so pushing exposure
     rolls highlights off instead of slamming into a hard clip.
   - Highlights/Shadows: LOCAL, not a per-pixel luma mask. A smooth,
     edge-preserving "base" layer of the image's luminance (a guided
     filter — He, Sun, Tang, "Guided Image Filtering", 2010) decides which
     REGIONS are highlights or shadows; the whole pixel is then scaled by
     a gain in stops computed from that base (the base/detail tone-mapping
     idea of Durand & Dorsey 2002). Local contrast inside a region (its
     detail relative to the base) is untouched, which is why a big
     Shadows move no longer flattens texture the way a per-pixel curve
     does.
   - Contrast: a smooth S-curve with fixed endpoints instead of a linear
     pivot that clips both ends.
   - Whites/Blacks (still stored as whitePoint/blackPoint): soft-kneed
     shifts of the top/bottom of the tonal range instead of hard levels.
   - Temperature/Tint: Lightroom's incremental white balance for rendered
     images (Temperature K / Tint → CIE xy → Bradford adaptation to D50),
     see electron/whiteBalance.mjs, instead of per-channel gains.

   New sliders, available in BOTH process versions (all identity at 0, so
   adding them never changes an older photo's look on its own):
   - Texture: amplifies (or smooths, negative) FINE detail — luminance
     minus a small-radius guided-filter base.
   - Clarity: the same at a larger radius, weighted toward midtones.
   - Dehaze: the dark channel prior (He, Sun, Tang, "Single Image Haze
     Removal Using Dark Channel Prior", 2009): a smoothed per-region
     minimum-of-RGB estimates how much haze sits over each area; the haze
     model I = J·t + A·(1−t) is inverted toward J (positive) or pushed
     toward A (negative).
   - Vibrance: saturation that mostly lifts the LESS saturated colors and
     goes easier on skin tones.

   The three local effects (Highlights/Shadows in pv 2, Texture/Clarity,
   Dehaze) need neighborhood information a per-pixel shader can't compute
   on its own, so computeGuides() precomputes it ONCE per image into four
   8-bit planes — fine base, mid base, coarse base, smoothed dark channel —
   which depend only on the image, never on slider values. Sliders stay
   GPU-cheap uniform tweaks; the guide planes are a second texture the
   GLSL twin samples, and the CPU paths sample the same planes bilinearly.
   Planes are computed on a downsampled working copy (≤ GUIDE_MAX_DIM long
   edge — all three effects are smooth at that scale, and this keeps a
   200-MP export from allocating gigabytes), and every filter radius is
   defined as a fraction of the ORIGINAL image's diagonal, converted
   through the caller's buffer scale — so the proxy preview, the
   480-px histogram sample and the full-resolution export all filter the
   same physical image area.
   ============================================================ */

import { wbMatrix, neutralizingIncrementals } from "./whiteBalance.mjs";

const EV_RANGE = 2.5;              // exposure ±100% -> ±2.5 stops
const TEMP_STRENGTH = 0.45;        // temperature ±100% -> R/B gain swing
const TINT_STRENGTH = 0.35;        // tint ±100% -> G gain swing (+ light R/B trim)
const CONTRAST_STRENGTH = 1.0;     // contrast ±100% -> pivot factor 0..2
const HIGHLIGHT_STRENGTH = 0.9;
const SHADOW_STRENGTH = 0.9;
const BLACKPOINT_STRENGTH = 0.3;   // black point ±100% -> levels shift (floor)
const WHITEPOINT_STRENGTH = 0.3;   // white point ±100% -> levels shift (ceiling)
const SATURATION_STRENGTH = 1.0;   // saturation ±100% -> uniform gain 0..2 on distance from gray

// ---- process version 2 + new sliders (see header) ----
export const CURRENT_PV = 2;
const PV2_SHOULDER_MAX = 0.35;     // highlight shoulder at +2 EV and beyond: knee drops from 1.0 to 1 − this (linear luminance)
const PV2_HIGHLIGHT_STOPS = 1.5;   // highlights ±100% -> ±stops applied to highlight regions
const PV2_SHADOW_STOPS = 1.75;     // shadows ±100% -> ±stops applied to shadow regions
const PV2_CONTRAST_AMP = 0.12;     // S-curve amplitude at ±100% (< 1/2π keeps the curve monotonic)
const PV2_WHITES_STRENGTH = 0.25;  // whites ±100% -> max shift of the top of the range
const PV2_BLACKS_STRENGTH = 0.2;   // blacks ±100% -> max shift of the bottom of the range
const TEXTURE_STRENGTH = 1.2;      // texture ±100% -> fine-detail gain −1.2..+1.2
const CLARITY_STRENGTH = 0.8;      // clarity ±100% -> mid-detail gain
const VIBRANCE_STRENGTH = 1.0;     // vibrance ±100% -> max saturation gain on fully-unsaturated colors
export const DEHAZE_T_MIN = 0.3;          // floor on estimated transmission: bright objects (a white wall) have a high dark channel
                                   // too and would otherwise get their grain amplified up to 1/t
export const DEHAZE_OMEGA = 0.9;          // fraction of the dark channel attributed to haze (He et al. use 0.95; a bit gentler here)

export function defaultAdjust() {
  return {
    pv: CURRENT_PV,
    temp: 0, tint: 0, saturation: 0, vibrance: 0,
    exposure: 0, contrast: 0, highlights: 0, shadows: 0,
    whitePoint: 0, blackPoint: 0,
    texture: 0, clarity: 0, dehaze: 0,
  };
}

/* The ONE place a stored adjust becomes an editable one: a stored object
   with no `pv` predates process versions and must keep rendering with
   pv 1 (see header), while "nothing stored yet" starts on the current
   version. Every read site that turns persisted edits into editor state
   goes through this, never a bare `{ ...defaultAdjust(), ...stored }` —
   that spread would silently stamp CURRENT_PV onto an old edit. */
export function normalizeAdjust(stored) {
  if (!stored) return defaultAdjust();
  return { ...defaultAdjust(), ...stored, pv: stored.pv ?? 1 };
}

// pv deliberately ignored: a pv-2 adjust with every slider at 0 is still a no-op
export function isDefaultAdjust(adjust) {
  if (!adjust) return true;
  return (
    (adjust.temp || 0) === 0 &&
    (adjust.tint || 0) === 0 &&
    (adjust.saturation || 0) === 0 &&
    (adjust.vibrance || 0) === 0 &&
    (adjust.exposure || 0) === 0 &&
    (adjust.contrast || 0) === 0 &&
    (adjust.highlights || 0) === 0 &&
    (adjust.shadows || 0) === 0 &&
    (adjust.whitePoint || 0) === 0 &&
    (adjust.blackPoint || 0) === 0 &&
    (adjust.texture || 0) === 0 &&
    (adjust.clarity || 0) === 0 &&
    (adjust.dehaze || 0) === 0
  );
}

/* Maps -100..100 slider values to the derived numbers the per-pixel
   apply step actually uses. This is the single source of truth for
   what each slider % means — both the CPU path (adjustPixel below)
   and the GLSL shader consume exactly these derived values (as
   uniforms), so only the small final apply formula needs a
   hand-mirrored GLSL twin, not this mapping. */
export function compileAdjust(adjust) {
  const a = { ...defaultAdjust(), ...(adjust || {}) };
  // NOT a.pv: the spread above would hand an old (pv-less) stored adjust
  // CURRENT_PV — see normalizeAdjust
  const pv = adjust && adjust.pv ? adjust.pv : 1;
  const temp = a.temp / 100;
  const tint = a.tint / 100;
  const ev = (a.exposure / 100) * EV_RANGE;
  const c = {
    pv,
    wbGainR: 1 + temp * TEMP_STRENGTH + tint * TINT_STRENGTH * 0.3,
    wbGainG: 1 - tint * TINT_STRENGTH,
    wbGainB: 1 - temp * TEMP_STRENGTH + tint * TINT_STRENGTH * 0.3,
    satGain: 1 + (a.saturation / 100) * SATURATION_STRENGTH,
    exposureGain: Math.pow(2, ev),
    contrastAmt: (a.contrast / 100) * CONTRAST_STRENGTH,
    highlightGain: (a.highlights / 100) * HIGHLIGHT_STRENGTH,
    shadowGain: (a.shadows / 100) * SHADOW_STRENGTH,
    whiteShift: (a.whitePoint / 100) * WHITEPOINT_STRENGTH,
    blackShift: -(a.blackPoint / 100) * BLACKPOINT_STRENGTH,
    // both process versions
    vibrance: (a.vibrance / 100) * VIBRANCE_STRENGTH,
    textureAmt: (a.texture / 100) * TEXTURE_STRENGTH,
    clarityAmt: (a.clarity / 100) * CLARITY_STRENGTH,
    dehazeAmt: a.dehaze / 100,
    airlight: 1, // replaced by the image's own estimate from computeGuides()
    // pv 2 only (zeroed for pv 1 so nothing downstream has to re-check pv)
    wbMat: null, // pv 2 white balance (3×3, row-major) — set below
    shoulderKnee: 1,
    hiStops: 0, shStops: 0, contrastS: 0, whitesAmt: 0, blacksAmt: 0,
  };
  if (pv >= 2) {
    // Lightroom-style incremental white balance (electron/whiteBalance.mjs):
    // one 3×3 on linear sRGB replaces pv 1's per-channel gains. Row-major.
    c.wbGainR = c.wbGainG = c.wbGainB = 1;
    if (a.temp !== 0 || a.tint !== 0) c.wbMat = wbMatrix(a.temp, a.tint).flat();
    c.shoulderKnee = 1 - PV2_SHOULDER_MAX * Math.min(1, Math.max(0, ev / 2));
    c.hiStops = (a.highlights / 100) * PV2_HIGHLIGHT_STOPS;
    c.shStops = (a.shadows / 100) * PV2_SHADOW_STOPS;
    c.contrastS = (a.contrast / 100) * PV2_CONTRAST_AMP;
    c.whitesAmt = (a.whitePoint / 100) * PV2_WHITES_STRENGTH;
    c.blacksAmt = (a.blackPoint / 100) * PV2_BLACKS_STRENGTH;
  }
  return c;
}

/* White Balance Selector (eyedropper): the temp/tint slider values (−100..
   +100) that make the sampled color neutral, in whichever white-balance
   model this adjust's process version uses. `rgb` is the sampled INPUT
   color (0-255, before any adjustment). pv 2: Lightroom's model, solved
   exactly (electron/whiteBalance.mjs). pv 1: its per-channel gains are
   linear in temp/tint, so "R·gR = G·gG = B·gB" is a 2×2 linear system. */
export function neutralizeWhiteBalance(rgb, pv) {
  const lr = SRGB_TO_LINEAR_LUT[Math.round(rgb[0])], lg = SRGB_TO_LINEAR_LUT[Math.round(rgb[1])], lb = SRGB_TO_LINEAR_LUT[Math.round(rgb[2])];
  const clamp100 = (v) => Math.max(-100, Math.min(100, v));
  if ((pv || 1) >= 2) {
    const r = neutralizingIncrementals([lr, lg, lb]);
    return { temp: clamp100(r.temp), tint: clamp100(r.tint) };
  }
  // gR = 1 + T·t + 0.3·N·n, gG = 1 − N·n, gB = 1 − T·t + 0.3·N·n
  // (t, n = temp, tint as fractions; T, N = TEMP/TINT_STRENGTH)
  const T = TEMP_STRENGTH, N = TINT_STRENGTH;
  const a11 = T * lr, a12 = 0.3 * N * lr + N * lg, b1 = lg - lr;
  const a21 = -T * lb, a22 = 0.3 * N * lb + N * lg, b2 = lg - lb;
  const det = a11 * a22 - a12 * a21;
  if (!(Math.abs(det) > 1e-12)) return { temp: 0, tint: 0 };
  return {
    temp: clamp100(((b1 * a22 - a12 * b2) / det) * 100),
    tint: clamp100(((a11 * b2 - b1 * a21) / det) * 100),
  };
}

/* Does this compiled adjust read the guide planes at all? Lets callers
   skip computeGuides() (the only expensive step) whenever only the
   per-pixel sliders are in use. */
export function needsGuides(c) {
  return c.textureAmt !== 0 || c.clarityAmt !== 0 || c.dehazeAmt !== 0 || c.hiStops !== 0 || c.shStops !== 0;
}

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/* Standard sRGB (IEC 61966-2-1) transfer function, not a naive x^2.2 —
   the real curve every color-managed tool uses to go between the
   gamma-encoded display values a file stores and the physically-linear
   light those values represent. See this module's header for why white
   balance/exposure/saturation specifically need to happen on the LINEAR
   side of this conversion. Negative inputs are clamped before the pow()
   (can otherwise happen transiently from float math) since a fractional
   power of a negative base is NaN. */
function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow(Math.max(0, (c + 0.055) / 1.055), 2.4);
}
function linearToSrgb(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(0, c), 1 / 2.4) - 0.055;
}

/* srgbToLinear precomputed for every one of the 256 possible input BYTE
   values — the buffer this module reads is always 8-bit, so there's no
   need to call Math.pow twice per channel per pixel when the input only
   ever takes 256 distinct values. (linearToSrgb, the OUTPUT side, stays
   a direct call: by then the value is a continuous float, not one of a
   fixed 256 levels, so a lookup table doesn't apply the same way.) A
   module-level constant, not part of compileAdjust()'s result — it
   doesn't depend on any adjust setting, only on the fixed sRGB curve
   itself, so it's built once total, not once per adjust change. */
const SRGB_TO_LINEAR_LUT = (() => {
  const t = new Float64Array(256);
  for (let i = 0; i < 256; i++) t[i] = srgbToLinear(i / 255);
  return t;
})();

/* r,g,b: 0-255 in, 0-255 out. `c` is a compileAdjust() result. `gd` is
   this pixel's [fine, mid, coarse, darkChannel] guide sample (0..1, see
   computeGuides/sampleGuide) — only read when needsGuides(c); may be
   omitted otherwise.
   KEEP IN SYNC WITH the GLSL fragment shader in src/App.jsx (GLImage). */
export function adjustPixel(r, g, b, c, gd) {
  // white balance, exposure, saturation: all physically multiplicative
  // gains on LIGHT, done here in LINEAR space — see this module's header.
  let lr = SRGB_TO_LINEAR_LUT[r], lg = SRGB_TO_LINEAR_LUT[g], lb = SRGB_TO_LINEAR_LUT[b];
  // the INPUT pixel's perceptual luminance — what the guide bases were
  // computed from, so detail = this − base is measured like-for-like
  // (only when guides are in play — it costs a pow() per pixel)
  const lumIn = gd ? linearToSrgb(0.2126 * lr + 0.7152 * lg + 0.0722 * lb) : 0;

  // dehaze: invert (or apply) the haze model I = J·t + A·(1−t) in linear
  // light, before white balance (the haze is part of the captured light)
  if (c.dehazeAmt !== 0 && gd) {
    const A = c.airlight;
    const t = Math.max(DEHAZE_T_MIN, 1 - (DEHAZE_OMEGA * srgbToLinear(gd[3])) / A);
    if (c.dehazeAmt > 0) {
      const k = c.dehazeAmt;
      lr += ((lr - A) / t + A - lr) * k;
      lg += ((lg - A) / t + A - lg) * k;
      lb += ((lb - A) / t + A - lb) * k;
    } else {
      const k = -c.dehazeAmt * 0.5 * (0.4 + 0.6 * (1 - t));
      lr += (A - lr) * k; lg += (A - lg) * k; lb += (A - lb) * k;
    }
    lr = Math.max(0, lr); lg = Math.max(0, lg); lb = Math.max(0, lb);
  }

  if (c.wbMat) {
    const m = c.wbMat;
    const r2 = m[0] * lr + m[1] * lg + m[2] * lb;
    const g2 = m[3] * lr + m[4] * lg + m[5] * lb;
    const b2 = m[6] * lr + m[7] * lg + m[8] * lb;
    lr = Math.max(0, r2); lg = Math.max(0, g2); lb = Math.max(0, b2);
  } else {
    lr *= c.wbGainR; lg *= c.wbGainG; lb *= c.wbGainB;
  }
  lr *= c.exposureGain; lg *= c.exposureGain; lb *= c.exposureGain;

  // vibrance: saturation gain weighted toward LOW-saturation colors, eased
  // off on skin-like (R ≥ G ≥ B) hues
  if (c.vibrance !== 0) {
    const mx = Math.max(lr, lg, lb), mn = Math.min(lr, lg, lb);
    const sat = mx > 0 ? (mx - mn) / mx : 0;
    const skin = lr >= lg && lg >= lb ? 0.5 : 1;
    const gain = 1 + c.vibrance * (1 - sat) * (1 - sat) * skin;
    const gray = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    lr = gray + (lr - gray) * gain;
    lg = gray + (lg - gray) * gain;
    lb = gray + (lb - gray) * gain;
  }

  if (c.satGain !== 1) {
    const gray = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    lr = gray + (lr - gray) * c.satGain;
    lg = gray + (lg - gray) * c.satGain;
    lb = gray + (lb - gray) * c.satGain;
  }

  // pv 2 local highlights/shadows: gain in stops from the COARSE base's
  // brightness (post-exposure), applied to the whole pixel — see header
  if ((c.hiStops !== 0 || c.shStops !== 0) && gd) {
    const base = Math.max(1e-4, srgbToLinear(gd[2]) * c.exposureGain);
    const lb2 = Math.log2(base);
    const stops = c.hiStops * smoothstep(-2.5, 0, lb2) + c.shStops * (1 - smoothstep(-6.5, -2, lb2));
    const k = Math.pow(2, stops);
    lr *= k; lg *= k; lb *= k;
  }
  // pv 2 exposure shoulder (identity at ≤ 0 EV: knee = 1)
  if (c.shoulderKnee < 1) {
    const y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    const kn = c.shoulderKnee;
    if (y > kn) {
      const s = (kn + (1 - kn) * (1 - Math.exp(-(y - kn) / (1 - kn)))) / y;
      lr *= s; lg *= s; lb *= s;
    }
  }

  // back to gamma-encoded/display-referred for everything below: each of
  // these is already a "how should this look on screen" judgment, not a
  // physical light quantity — see this module's header for why they stay
  // here rather than also moving into the linear block above.
  let cr = linearToSrgb(Math.max(0, lr)), cg = linearToSrgb(Math.max(0, lg)), cb = linearToSrgb(Math.max(0, lb));

  if (c.pv >= 2) {
    return finishPv2(cr, cg, cb, c, gd, lumIn);
  }

  // contrast: linear pivot around mid-gray
  cr = (cr - 0.5) * (1 + c.contrastAmt) + 0.5;
  cg = (cg - 0.5) * (1 + c.contrastAmt) + 0.5;
  cb = (cb - 0.5) * (1 + c.contrastAmt) + 0.5;

  // highlights / shadows: luma-masked lift (positive) or recovery
  // (negative), one shared mask across a pixel's own R/G/B channels (not
  // a per-channel independent mask) so a highlight/shadow move can't
  // shift hue the way adjusting each channel by its own separate amount
  // would. Shadows recomputes luminance AFTER highlights has already run
  // (rather than reusing highlights' own `lum`) — the two operate on
  // genuinely different tonal ranges (0.35-1.0 vs 0.0-0.5, minimal
  // overlap) so this rarely matters in practice, but it's the actually-
  // correct order: each step should see the pixel as the PREVIOUS step
  // actually left it, not a stale reading from before that step ran.
  if (c.highlightGain !== 0) {
    const lum = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
    const d = smoothstep(0.35, 1.0, lum) * c.highlightGain;
    cr += d >= 0 ? d * (1 - cr) : d * cr;
    cg += d >= 0 ? d * (1 - cg) : d * cg;
    cb += d >= 0 ? d * (1 - cb) : d * cb;
  }
  if (c.shadowGain !== 0) {
    const lum = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
    const d = (1 - smoothstep(0.0, 0.5, lum)) * c.shadowGain;
    cr += d >= 0 ? d * (1 - cr) : d * cr;
    cg += d >= 0 ? d * (1 - cg) : d * cg;
    cb += d >= 0 ? d * (1 - cb) : d * cb;
  }

  if ((c.textureAmt !== 0 || c.clarityAmt !== 0) && gd) {
    const d = detailShift(cr, cg, cb, c, gd, lumIn);
    cr += d; cg += d; cb += d;
  }

  // black point / white point: levels-style remap of the [lo, hi] range to
  // [0, 1]. Positive blackPoint lifts/lightens shadows (raises the floor,
  // reduces contrast); negative crushes blacks. Positive whitePoint pulls
  // the ceiling down (clips/brightens highlights, more contrast); negative
  // pushes it up (recovers/dims highlights, less contrast).
  if (c.blackShift !== 0 || c.whiteShift !== 0) {
    const lo = c.blackShift, hi = 1 - c.whiteShift;
    const denom = Math.max(0.05, hi - lo);
    cr = (cr - lo) / denom;
    cg = (cg - lo) / denom;
    cb = (cb - lo) / denom;
  }

  return [
    Math.round(Math.min(1, Math.max(0, cr)) * 255),
    Math.round(Math.min(1, Math.max(0, cg)) * 255),
    Math.round(Math.min(1, Math.max(0, cb)) * 255),
  ];
}

/* Texture + Clarity, as one luminance shift added equally to R/G/B (so it
   changes local contrast without shifting hue). Detail is the INPUT
   pixel's perceptual luminance minus the fine/mid guide base; clarity is
   weighted toward midtones (current luminance), where it reads as "punch"
   rather than as crushed shadows or halos in the highlights. */
function detailShift(cr, cg, cb, c, gd, lumIn) {
  const lum = Math.min(1, Math.max(0, 0.2126 * cr + 0.7152 * cg + 0.0722 * cb));
  const mid = 1 - (2 * lum - 1) * (2 * lum - 1);
  return c.textureAmt * (lumIn - gd[0]) + c.clarityAmt * (lumIn - gd[1]) * (0.3 + 0.7 * mid);
}

/* pv 2's display-referred tail: S-curve contrast, texture/clarity, soft
   whites/blacks. KEEP IN SYNC WITH the GLSL twin's pv-2 branch. */
function finishPv2(cr, cg, cb, c, gd, lumIn) {
  if (c.contrastS !== 0) {
    // endpoints fixed (sin(0) = sin(2π) = 0), steeper middle for +, flatter
    // for −; PV2_CONTRAST_AMP < 1/2π keeps it monotonic. Clamped first: the
    // sine is periodic, so an out-of-range value would fold back.
    const k = c.contrastS, tau = 2 * Math.PI;
    cr = Math.min(1, Math.max(0, cr)); cg = Math.min(1, Math.max(0, cg)); cb = Math.min(1, Math.max(0, cb));
    cr -= k * Math.sin(tau * cr); cg -= k * Math.sin(tau * cg); cb -= k * Math.sin(tau * cb);
  }
  if ((c.textureAmt !== 0 || c.clarityAmt !== 0) && gd) {
    const d = detailShift(cr, cg, cb, c, gd, lumIn);
    cr += d; cg += d; cb += d;
  }
  if (c.whitesAmt !== 0 || c.blacksAmt !== 0) {
    const lum = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
    const d = c.whitesAmt * smoothstep(0.45, 1.0, lum) + c.blacksAmt * (1 - smoothstep(0.0, 0.55, lum));
    cr += d; cg += d; cb += d;
  }
  return [
    Math.round(Math.min(1, Math.max(0, cr)) * 255),
    Math.round(Math.min(1, Math.max(0, cg)) * 255),
    Math.round(Math.min(1, Math.max(0, cb)) * 255),
  ];
}

/* ============================================================
   Guide planes (see header): computed once per image, independent of
   slider values.
   ============================================================ */
const GUIDE_MAX_DIM = 2048;        // working-copy long edge cap
// filter radii as fractions of the ORIGINAL image diagonal
const GUIDE_R_FINE = 0.0015;       // texture
const GUIDE_R_MID = 0.008;         // clarity
const GUIDE_R_COARSE = 0.03;       // pv-2 highlights/shadows regions
const GUIDE_R_DARK_PATCH = 0.0008; // dark-channel minimum patch — deliberately tiny: a wider min-window drags a dark
                                   // neighbor's low dark channel across an edge into a bright object, a visible
                                   // rim after dehaze (measured: ~6.5 → ~1.4 levels at a face/ground edge)
const GUIDE_R_DARK_SMOOTH = 0.004; // dark-channel edge-aware smoothing
// guided-filter regularization (perceptual-luminance units²): small eps
// keeps edges in the base (so detail = lum − base has no halos at edges)
const GUIDE_EPS_FINE = 0.0004;
const GUIDE_EPS_MID = 0.0025;
const GUIDE_EPS_COARSE = 0.005;
const GUIDE_EPS_DARK = 0.0001;

/* Separable mean filter with a (2r+1)² window, edge-normalized (border
   pixels average only the in-bounds part of the window). O(n) per pass via
   running sums, independent of r. */
function boxMean(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) s += src[row + x];
    for (let x = 0; x < w; x++) {
      const lo = Math.max(0, x - r), hi = Math.min(w - 1, x + r);
      tmp[row + x] = s / (hi - lo + 1);
      if (x + r + 1 < w) s += src[row + x + r + 1];
      if (x - r >= 0) s -= src[row + x - r];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = 0; y <= Math.min(r, h - 1); y++) s += tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      const lo = Math.max(0, y - r), hi = Math.min(h - 1, y + r);
      out[y * w + x] = s / (hi - lo + 1);
      if (y + r + 1 < h) s += tmp[(y + r + 1) * w + x];
      if (y - r >= 0) s -= tmp[(y - r) * w + x];
    }
  }
  return out;
}

/* Guided filter (He, Sun, Tang 2010) of `p` with guide `I`. */
function guidedFilter(I, p, w, h, r, eps) {
  const n = w * h;
  const meanI = boxMean(I, w, h, r), meanP = I === p ? meanI : boxMean(p, w, h, r);
  const ip = new Float32Array(n);
  for (let i = 0; i < n; i++) ip[i] = I[i] * p[i];
  const corrIP = boxMean(ip, w, h, r);
  let corrII = corrIP;
  if (I !== p) {
    for (let i = 0; i < n; i++) ip[i] = I[i] * I[i];
    corrII = boxMean(ip, w, h, r);
  }
  const a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = corrII[i] - meanI[i] * meanI[i];
    const cov = corrIP[i] - meanI[i] * meanP[i];
    a[i] = cov / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  const ma = boxMean(a, w, h, r), mb = boxMean(b, w, h, r);
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = ma[i] * I[i] + mb[i];
  return q;
}

/* Separable (2r+1)² minimum filter. r is small (a few px), so the plain
   O(n·r) scan is fine. */
function minFilter(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = Infinity;
    for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) m = Math.min(m, src[y * w + k]);
    tmp[y * w + x] = m;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = Infinity;
    for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++) m = Math.min(m, tmp[k * w + x]);
    out[y * w + x] = m;
  }
  return out;
}

/* Computes the four guide planes for an RGB(A) byte buffer.
   opts.scale    — this buffer's px per ORIGINAL-image px (1 for a full-res
                   original; s for a proxy/crop bake at scale s)
   opts.origDiag — the original image's diagonal in its own px
   Returns { width, height, data: Uint8ClampedArray RGBA (fine, mid, coarse,
   dark channel), airlight } — width/height are the working copy's, which
   callers sample with sampleGuide (CPU) or as a LINEAR-filtered texture
   covering the same extent (GPU). */
export function computeGuides(buf, width, height, channels, opts = {}) {
  const scale = opts.scale > 0 ? opts.scale : 1;
  const origDiag = opts.origDiag > 0 ? opts.origDiag : Math.hypot(width, height) / scale;
  const gs = Math.min(1, GUIDE_MAX_DIM / Math.max(width, height));
  const gw = Math.max(1, Math.round(width * gs)), gh = Math.max(1, Math.round(height * gs));
  const n = gw * gh;
  // area-average into the working copy: perceptual luminance + per-pixel
  // min(R,G,B) (the dark channel's per-pixel part)
  const L = new Float32Array(n), D = new Float32Array(n), cnt = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    const gy = Math.min(gh - 1, Math.floor((y * gh) / height));
    for (let x = 0; x < width; x++) {
      const gx = Math.min(gw - 1, Math.floor((x * gw) / width));
      const o = (y * width + x) * channels, q = gy * gw + gx;
      const r = buf[o], g = buf[o + 1], b = buf[o + 2];
      L[q] += 0.2126 * SRGB_TO_LINEAR_LUT[r] + 0.7152 * SRGB_TO_LINEAR_LUT[g] + 0.0722 * SRGB_TO_LINEAR_LUT[b];
      // dark channel in LINEAR light — the haze model it feeds is linear
      D[q] += SRGB_TO_LINEAR_LUT[Math.min(r, g, b)];
      cnt[q]++;
    }
  }
  for (let q = 0; q < n; q++) {
    const k = cnt[q] || 1;
    L[q] = linearToSrgb(L[q] / k);
    // stored sRGB-encoded (decoded again at use): 8 bits of a LINEAR value
    // would leave the shadows, where haze matters most, with a handful of levels
    D[q] = linearToSrgb(D[q] / k);
  }
  const px = (frac) => Math.max(1, Math.round(frac * origDiag * scale * gs));
  const fine = guidedFilter(L, L, gw, gh, px(GUIDE_R_FINE), GUIDE_EPS_FINE);
  const mid = guidedFilter(L, L, gw, gh, px(GUIDE_R_MID), GUIDE_EPS_MID);
  const coarse = guidedFilter(L, L, gw, gh, px(GUIDE_R_COARSE), GUIDE_EPS_COARSE);
  const dark = guidedFilter(L, minFilter(D, gw, gh, px(GUIDE_R_DARK_PATCH)), gw, gh, px(GUIDE_R_DARK_SMOOTH), GUIDE_EPS_DARK);

  // airlight: mean luminance of the brightest-0.1% dark-channel cells (the
  // haziest, most-distant areas), as a gray linear value
  // (threshold found via a 1024-bin histogram rather than a full sort —
  // the working copy can hold ~4M cells)
  const hist = new Uint32Array(1024);
  for (let q = 0; q < n; q++) hist[Math.min(1023, Math.max(0, Math.floor(dark[q] * 1024)))]++;
  const want = Math.max(1, Math.round(n * 0.001));
  let bin = 1023, acc = 0;
  while (bin > 0 && acc + hist[bin] < want) acc += hist[bin--];
  const thr = bin / 1024;
  let a = 0, an = 0;
  for (let q = 0; q < n; q++) if (dark[q] >= thr) { a += L[q]; an++; }
  const airlight = srgbToLinear(Math.min(1, Math.max(0.6, a / Math.max(1, an))));

  const data = new Uint8ClampedArray(n * 4);
  for (let q = 0; q < n; q++) {
    data[q * 4] = fine[q] * 255 + 0.5;
    data[q * 4 + 1] = mid[q] * 255 + 0.5;
    data[q * 4 + 2] = coarse[q] * 255 + 0.5;
    data[q * 4 + 3] = dark[q] * 255 + 0.5;
  }
  return { width: gw, height: gh, data, airlight };
}

/* Bilinear guide sample for buffer pixel (x, y) of a width×height buffer —
   the CPU equivalent of the GPU's LINEAR texture filtering over the same
   extent (texel centers at (i + 0.5)/gw). Writes 0..1 values into `out`. */
export function sampleGuide(guides, x, y, width, height, out) {
  const { width: gw, height: gh, data } = guides;
  const fx = Math.min(Math.max(((x + 0.5) * gw) / width - 0.5, 0), gw - 1);
  const fy = Math.min(Math.max(((y + 0.5) * gh) / height - 0.5, 0), gh - 1);
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, gw - 1), y1 = Math.min(y0 + 1, gh - 1);
  const tx = fx - x0, ty = fy - y0;
  const q00 = (y0 * gw + x0) * 4, q10 = (y0 * gw + x1) * 4, q01 = (y1 * gw + x0) * 4, q11 = (y1 * gw + x1) * 4;
  for (let k = 0; k < 4; k++) {
    const top = data[q00 + k] + (data[q10 + k] - data[q00 + k]) * tx;
    const bot = data[q01 + k] + (data[q11 + k] - data[q01 + k]) * tx;
    out[k] = (top + (bot - top) * ty) / 255;
  }
  return out;
}

/* Mutates buf in place (works on a Node Buffer or a Uint8ClampedArray
   alike) and returns it. Only touches the first 3 bytes/pixel — alpha,
   if present, is left untouched. No-ops (and skips the compile step)
   when adjust is at defaults, so crop-only edits pay zero extra cost.
   `opts` = { scale, origDiag } (see computeGuides) and optionally a
   precomputed `guides` for this exact buffer content — only consulted
   when a guide-reading slider is in use. */
export function applyAdjustToRGBBuffer(buf, width, height, channels, adjust, opts = {}) {
  if (isDefaultAdjust(adjust)) return buf;
  const c = compileAdjust(adjust);
  const guides = needsGuides(c) ? opts.guides || computeGuides(buf, width, height, channels, opts) : null;
  if (guides) c.airlight = guides.airlight;
  const gd = guides ? [0, 0, 0, 0] : undefined;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * channels;
      if (gd) sampleGuide(guides, x, y, width, height, gd);
      const [r, g, b] = adjustPixel(buf[o], buf[o + 1], buf[o + 2], c, gd);
      buf[o] = r; buf[o + 1] = g; buf[o + 2] = b;
    }
  }
  return buf;
}
