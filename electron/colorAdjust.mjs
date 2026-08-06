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
   ============================================================ */

const EV_RANGE = 2.5;              // exposure ±100% -> ±2.5 stops
const TEMP_STRENGTH = 0.45;        // temperature ±100% -> R/B gain swing
const TINT_STRENGTH = 0.35;        // tint ±100% -> G gain swing (+ light R/B trim)
const CONTRAST_STRENGTH = 1.0;     // contrast ±100% -> pivot factor 0..2
const HIGHLIGHT_STRENGTH = 0.9;
const SHADOW_STRENGTH = 0.9;
const BLACKPOINT_STRENGTH = 0.3;   // black point ±100% -> levels shift (floor)
const WHITEPOINT_STRENGTH = 0.3;   // white point ±100% -> levels shift (ceiling)
const SATURATION_STRENGTH = 1.0;   // saturation ±100% -> uniform gain 0..2 on distance from gray

export function defaultAdjust() {
  return {
    temp: 0, tint: 0, saturation: 0,
    exposure: 0, contrast: 0, highlights: 0, shadows: 0,
    whitePoint: 0, blackPoint: 0,
  };
}

export function isDefaultAdjust(adjust) {
  if (!adjust) return true;
  return (
    (adjust.temp || 0) === 0 &&
    (adjust.tint || 0) === 0 &&
    (adjust.saturation || 0) === 0 &&
    (adjust.exposure || 0) === 0 &&
    (adjust.contrast || 0) === 0 &&
    (adjust.highlights || 0) === 0 &&
    (adjust.shadows || 0) === 0 &&
    (adjust.whitePoint || 0) === 0 &&
    (adjust.blackPoint || 0) === 0
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
  const temp = a.temp / 100;
  const tint = a.tint / 100;
  return {
    wbGainR: 1 + temp * TEMP_STRENGTH + tint * TINT_STRENGTH * 0.3,
    wbGainG: 1 - tint * TINT_STRENGTH,
    wbGainB: 1 - temp * TEMP_STRENGTH + tint * TINT_STRENGTH * 0.3,
    satGain: 1 + (a.saturation / 100) * SATURATION_STRENGTH,
    exposureGain: Math.pow(2, (a.exposure / 100) * EV_RANGE),
    contrastAmt: (a.contrast / 100) * CONTRAST_STRENGTH,
    highlightGain: (a.highlights / 100) * HIGHLIGHT_STRENGTH,
    shadowGain: (a.shadows / 100) * SHADOW_STRENGTH,
    whiteShift: (a.whitePoint / 100) * WHITEPOINT_STRENGTH,
    blackShift: -(a.blackPoint / 100) * BLACKPOINT_STRENGTH,
  };
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

/* r,g,b: 0-255 in, 0-255 out. `c` is a compileAdjust() result.
   KEEP IN SYNC WITH the GLSL fragment shader in src/App.jsx (GLImage). */
export function adjustPixel(r, g, b, c) {
  // white balance, exposure, saturation: all physically multiplicative
  // gains on LIGHT, done here in LINEAR space — see this module's header.
  let lr = SRGB_TO_LINEAR_LUT[r], lg = SRGB_TO_LINEAR_LUT[g], lb = SRGB_TO_LINEAR_LUT[b];

  lr *= c.wbGainR; lg *= c.wbGainG; lb *= c.wbGainB;
  lr *= c.exposureGain; lg *= c.exposureGain; lb *= c.exposureGain;

  if (c.satGain !== 1) {
    const gray = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    lr = gray + (lr - gray) * c.satGain;
    lg = gray + (lg - gray) * c.satGain;
    lb = gray + (lb - gray) * c.satGain;
  }

  // back to gamma-encoded/display-referred for everything below: each of
  // these is already a "how should this look on screen" judgment, not a
  // physical light quantity — see this module's header for why they stay
  // here rather than also moving into the linear block above.
  let cr = linearToSrgb(Math.max(0, lr)), cg = linearToSrgb(Math.max(0, lg)), cb = linearToSrgb(Math.max(0, lb));

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

/* Mutates buf in place (works on a Node Buffer or a Uint8ClampedArray
   alike) and returns it. Only touches the first 3 bytes/pixel — alpha,
   if present, is left untouched. No-ops (and skips the compile step)
   when adjust is at defaults, so crop-only edits pay zero extra cost. */
export function applyAdjustToRGBBuffer(buf, width, height, channels, adjust) {
  if (isDefaultAdjust(adjust)) return buf;
  const c = compileAdjust(adjust);
  const n = width * height;
  for (let i = 0; i < n; i++) {
    const o = i * channels;
    const [r, g, b] = adjustPixel(buf[o], buf[o + 1], buf[o + 2], c);
    buf[o] = r; buf[o + 1] = g; buf[o + 2] = b;
  }
  return buf;
}
