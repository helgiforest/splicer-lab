/* ============================================================
   White balance — Lightroom-style Temperature/Tint for rendered images
   (JPEG/TIFF/HEIC, and RAWs, which reach this app already rendered).
   Used by electron/colorAdjust.mjs (process version 2 only; pv 1 keeps
   its original per-channel gain model, see that module's header).

   How it works — this is how Adobe's Camera Raw engine treats a NON-raw
   image (its "incremental" Temperature/Tint, the −100..+100 sliders
   Lightroom shows for a JPEG), reconstructed from the published DNG SDK
   algorithm plus the slider mapping observed in Lightroom Classic 15.5;
   the code below is written from those formulas, not copied:

   1. The image's own white is taken to be D50 (the ICC connection-space
      white), and the sliders describe a CHANGE from it. The incremental
      slider pair maps to an absolute (Temperature K, Tint) pair:
        temperature:  x = slider/100 ∈ [−1, 1],  T = 50000 / (3x² − 12x + 10)
                      → 0 ↦ 5000 K, +100 ↦ 50000 K, −100 ↦ 2000 K
        tint:         u = ½·x·(|x| + 1),  tint = c + 150u − c·u²,  c = 9.562965
                      → 0 ↦ +9.563, ±100 ↦ ±150
      (5000 K, +9.563) is exactly D50 under step 2, which is why slider
      zero is a no-op.
   2. (T, tint) → CIE xy via Robertson's isotemperature lines (the
      standard 31-row table, A. R. Robertson 1968): interpolate the
      Planckian locus point and isotherm direction at 10⁶/T in CIE 1960
      uv, then offset along the isotherm by tint/−3000.
   3. Pixels: "the light was really this xy" → chromatic adaptation from
      that white to D50 with the Bradford transform, done on linear-light
      XYZ (sRGB primaries, D50-adapted). So a higher Temperature tells the
      engine the light was bluer, and compensating for it warms the
      image — the same direction as Lightroom's slider.
   The whole thing collapses to one 3×3 matrix on linear sRGB per slider
   setting (wbMatrix), so it costs the same per pixel as the old gains.
   ============================================================ */

// Robertson isotemperature table: [10⁶/T (mired), u, v, isotherm slope t]
const ROBERTSON = [
  [0, 0.18006, 0.26352, -0.24341], [10, 0.18066, 0.26589, -0.25479], [20, 0.18133, 0.26846, -0.26876],
  [30, 0.18208, 0.27119, -0.28539], [40, 0.18293, 0.27407, -0.30470], [50, 0.18388, 0.27709, -0.32675],
  [60, 0.18494, 0.28021, -0.35156], [70, 0.18611, 0.28342, -0.37915], [80, 0.18740, 0.28668, -0.40955],
  [90, 0.18880, 0.28997, -0.44278], [100, 0.19032, 0.29326, -0.47888], [125, 0.19462, 0.30141, -0.58204],
  [150, 0.19962, 0.30921, -0.70471], [175, 0.20525, 0.31647, -0.84901], [200, 0.21142, 0.32312, -1.0182],
  [225, 0.21807, 0.32909, -1.2168], [250, 0.22511, 0.33439, -1.4512], [275, 0.23247, 0.33904, -1.7298],
  [300, 0.24010, 0.34308, -2.0637], [325, 0.24702, 0.34655, -2.4681], [350, 0.25591, 0.34951, -2.9641],
  [375, 0.26400, 0.35200, -3.5814], [400, 0.27218, 0.35407, -4.3633], [425, 0.28039, 0.35577, -5.3762],
  [450, 0.28863, 0.35714, -6.7262], [475, 0.29685, 0.35823, -8.5955], [500, 0.30505, 0.35907, -11.324],
  [525, 0.31320, 0.35968, -15.628], [550, 0.32129, 0.36011, -23.325], [575, 0.32931, 0.36038, -40.770],
  [600, 0.33724, 0.36051, -116.45],
];
const TINT_SCALE = -3000;
const TINT_REF = 9.562965;           // incremental tint 0 ↔ this absolute tint
export const WB_TEMP_MIN = 1500;     // incremental −150 (the slider itself stops at −100 = 2000 K)
export const WB_TEMP_MAX = 50000;
export const WB_TINT_LIMIT = 150;

/* ---- incremental sliders ↔ absolute temperature/tint ---- */
export function incrementalToTemperature(inc) {
  const v = Math.max(-150, Math.min(100, inc));
  if (v >= -100) {
    const x = v / 100;
    return 50000 / (x * (3 * x - 12) + 10);
  }
  // below −100: cubic Hermite in 1/T-space down to 1500 K at −150, matching
  // value (25 → 2000 K) and slope (−0.18 per unit) of the quadratic at −100
  const t = (v + 150) / 50;
  const h00 = 2 * t ** 3 - 3 * t ** 2 + 1, h10 = t ** 3 - 2 * t ** 2 + t;
  const h01 = -2 * t ** 3 + 3 * t ** 2, h11 = t ** 3 - t ** 2;
  return 50000 / (h00 * (100 / 3) + h10 * 50 * (-1 / 6) + h01 * 25 + h11 * 50 * -0.18);
}
export function temperatureToIncremental(temp) {
  const T = Math.max(WB_TEMP_MIN, Math.min(WB_TEMP_MAX, temp));
  if (T >= 2000) {
    const p = Math.min(25, 50000 / T);
    const x = (12 - Math.sqrt(144 - 12 * (p < 1 ? 9 : 10 - p))) / 6;
    return Math.max(-100, Math.min(100, x * 100));
  }
  // invert the Hermite segment by bisection (monotonic in inc)
  let lo = -150, hi = -100;
  for (let i = 0; i < 64; i++) {
    const mid = 0.5 * (lo + hi);
    if (incrementalToTemperature(mid) < T) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}
export function incrementalToTint(inc) {
  const x = Math.max(-1, Math.min(1, inc / 100));
  const u = 0.5 * x * (Math.abs(x) + 1);
  return TINT_REF + u * (150 - TINT_REF * u);
}
export function tintToIncremental(tint) {
  const t = Math.max(-WB_TINT_LIMIT, Math.min(WB_TINT_LIMIT, tint));
  const u = (150 - Math.sqrt(22500 + 4 * TINT_REF * (TINT_REF - t))) / (2 * TINT_REF);
  return u >= 0 ? 50 * (Math.sqrt(1 + 8 * u) - 1) : -50 * (Math.sqrt(1 - 8 * u) - 1);
}

/* ---- absolute temperature/tint ↔ CIE xy (Robertson) ---- */
export function temperatureTintToXY(temp, tint) {
  const r = 1e6 / temp;
  const off = tint / TINT_SCALE;
  for (let k = 0; k < 30; k++) {
    if (r < ROBERTSON[k + 1][0] || k === 29) {
      const a = ROBERTSON[k], b = ROBERTSON[k + 1];
      const f = (b[0] - r) / (b[0] - a[0]);
      let u = a[1] * f + b[1] * (1 - f);
      let v = a[2] * f + b[2] * (1 - f);
      const l1 = Math.hypot(1, a[3]), l2 = Math.hypot(1, b[3]);
      let du = f / l1 + (1 - f) / l2;
      let dv = (a[3] / l1) * f + (b[3] / l2) * (1 - f);
      const l3 = Math.hypot(du, dv);
      du /= l3; dv /= l3;
      u += du * off; v += dv * off;
      const d = u - 4 * v + 2;
      return [(1.5 * u) / d, v / d];
    }
  }
  return [0.3457, 0.3585];
}
/* Inverse — what a white-balance eyedropper needs (xy of the clicked
   neutral → temperature/tint). */
export function xyToTemperatureTint(x, y) {
  const d0 = 1.5 - x + 6 * y;
  const u = (2 * x) / d0, v = (3 * y) / d0;
  let lastDt = 0, lastDu = 0, lastDv = 0;
  for (let k = 1; k <= 30; k++) {
    const row = ROBERTSON[k], prev = ROBERTSON[k - 1];
    const len = Math.hypot(1, row[3]);
    let du = 1 / len, dv = row[3] / len;
    let dt = -(u - row[1]) * dv + (v - row[2]) * du;
    if (dt <= 0 || k === 30) {
      if (dt > 0) dt = 0;
      dt = -dt;
      const f = k === 1 ? 0 : dt / (lastDt + dt);
      const temperature = 1e6 / (prev[0] * f + row[0] * (1 - f));
      const uu = u - (prev[1] * f + row[1] * (1 - f));
      const vv = v - (prev[2] * f + row[2] * (1 - f));
      du = du * (1 - f) + lastDu * f;
      dv = dv * (1 - f) + lastDv * f;
      const l = Math.hypot(du, dv);
      return { temperature, tint: ((uu * du + vv * dv) / l) * TINT_SCALE };
    }
    lastDt = dt; lastDu = du; lastDv = dv;
  }
  return { temperature: 5000, tint: TINT_REF };
}

/* ---- the pixel transform ---- */
const mul3 = (A, B) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]));
const apply3 = (A, v) => [0, 1, 2].map((i) => A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2]);
function inv3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}
const xyToXYZ = ([x, y]) => [x / y, 1, (1 - x - y) / y];
const BRADFORD = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
const BRADFORD_INV = inv3(BRADFORD);
function bradford(srcXY, dstXY) {
  const s = apply3(BRADFORD, xyToXYZ(srcXY)), d = apply3(BRADFORD, xyToXYZ(dstXY));
  const D = [[d[0] / s[0], 0, 0], [0, d[1] / s[1], 0], [0, 0, d[2] / s[2]]];
  return mul3(BRADFORD_INV, mul3(D, BRADFORD));
}
// linear sRGB → XYZ with the image white at D50 (the standard D65 sRGB
// matrix, Bradford-adapted to D50 — how a color-managed engine places a
// rendered sRGB image in a D50 connection space)
const D50 = temperatureTintToXY(5000, TINT_REF);
const D65 = [0.3127, 0.3290];
const SRGB_TO_XYZ_D65 = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];
const SRGB_TO_XYZ_D50 = mul3(bradford(D65, D50), SRGB_TO_XYZ_D65);
const XYZ_D50_TO_SRGB = inv3(SRGB_TO_XYZ_D50);

/* 3×3 (row-major, nested) on LINEAR sRGB for incremental slider values
   (−100..+100 each). Identity (to float precision) at 0/0. */
export function wbMatrix(incTemp, incTint) {
  const xy = temperatureTintToXY(incrementalToTemperature(incTemp), incrementalToTint(incTint));
  return mul3(XYZ_D50_TO_SRGB, mul3(bradford(xy, D50), SRGB_TO_XYZ_D50));
}

/* Eyedropper helper: the incremental slider values that make linear-sRGB
   color `rgb` neutral (what Lightroom's White Balance Selector does). */
export function neutralizingIncrementals(rgb) {
  const [X, Y, Z] = apply3(SRGB_TO_XYZ_D50, rgb);
  const s = X + Y + Z;
  if (!(s > 0)) return { temp: 0, tint: 0 };
  const { temperature, tint } = xyToTemperatureTint(X / s, Y / s);
  return { temp: temperatureToIncremental(temperature), tint: tintToIncremental(tint) };
}
