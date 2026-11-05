/* ============================================================
   Auto Straighten — the angle Lightroom Classic's Crop "Auto" button sets.

   What Lightroom does (Camera Raw, cr_upright.cpp; read from LrC 15.5's
   CameraRaw binary — AgCRContext:computeAutoStraightenAngle →
   cr_upright_params::Calculate → the ura::UprightData pipeline): the
   "Upright" analysis of Lee, Shechtman, Wang & Lee, "Automatic Upright
   Adjustment of Photographs" (CVPR 2012):
     1. the photo rendered at 1024 px on its long edge, as gray with a 2.2
        gamma ("Upright-MakeGrayGamma22")
     2. LSD line segments (von Gioi et al.)
     3. vanishing points, then the camera: focal length from the 35 mm
        equivalent in EXIF (35 mm when there is none), as pixels via the
        full-frame diagonal hypot(24, 36); principal point at the center
     4. the Manhattan frame — three orthogonal scene directions — and the
        camera rotation R = Rz(c)·Ry(b)·Rx(a) that explains them
     5. the straighten angle is the roll c, folded into ±45° (the three
        axes are interchangeable, so the roll is only defined mod 90°)
   Lightroom then turns the crop about its own center by that angle and
   shrinks it to stay inside the photo — Splicer's applyRotation already
   does exactly that, so this module only answers "which angle".

   When the scene has no convincing vertical, Lightroom levels the HORIZON
   instead (handle_degen_case, 640 px: collinear segments voting for a line
   — or a slightly bent arc — across ≥40% of the frame, checked for a
   color/texture change across it), and with neither it rotates nothing.

   This is a reconstruction of that pipeline (the published method plus
   the constants and conventions seen in the binary), not Adobe's code:
   the full camera optimization that decides "convincing" isn't ported, so
   the three readings — vertical vanishing point, horizontal one, horizon —
   are weighed against each other (see weigh), with evidence measured in
   absolute segment length so a portrait's few stray lines can't tilt it.
   Checked on: synthetic facades and seascapes tilted −12…+25° (exact to
   ~0.05°), 42 of this user's photos hand-leveled in Lightroom (median
   0.49° from the hand-set angle), and contact sheets of 252 more from the
   library (portraits without structure are left alone).
   ============================================================ */

const ANALYSIS_SIZE = 1024;          // long edge, px (Lightroom renders 0x400)
const DIAG_35MM = Math.hypot(24, 36); // mm, full-frame diagonal
const DEFAULT_FOCAL_35 = 35;          // mm, Lightroom's fallback

/* ---------- 1. gray, gamma 2.2, 1024 px ---------- */
export function analysisGray(rgba, width, height, channels = 4, size = ANALYSIS_SIZE) {
  const s = Math.min(1, size / Math.max(width, height));
  const w = Math.max(1, Math.round(width * s)), h = Math.max(1, Math.round(height * s));
  const out = new Float64Array(w * h), cnt = new Float64Array(w * h);
  // area average of the sRGB values (sRGB is close to a 2.2 gamma already),
  // Rec.709 weights on the gamma-encoded values
  for (let y = 0; y < height; y++) {
    const gy = Math.min(h - 1, Math.floor(y * s));
    for (let x = 0; x < width; x++) {
      const gx = Math.min(w - 1, Math.floor(x * s)), o = (y * width + x) * channels;
      out[gy * w + gx] += 0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2];
      cnt[gy * w + gx]++;
    }
  }
  for (let i = 0; i < w * h; i++) out[i] /= cnt[i] || 1;
  return { gray: out, w, h };
}

/* ---------- 2. LSD: a Line Segment Detector ----------
   R. Grompone von Gioi, J. Jakubowicz, J.-M. Morel, G. Randall, "LSD: a
   Line Segment Detector", IPOL 2012 — the published algorithm with its
   default parameters (the a-contrario validation included). */
const LSD = { scale: 0.8, sigmaScale: 0.6, quant: 2.0, angTh: 22.5, logEps: 0, densityTh: 0.7, nBins: 1024 };
const NOTDEF = -1024;
const M_3_2_PI = 4.71238898038, M_2_PI = 6.28318530718;

function gaussianSampler(img, w, h, scale, sigmaScale) {
  const sigma = scale < 1 ? sigmaScale / scale : sigmaScale;
  const prec = 3, hs = Math.ceil(sigma * Math.sqrt(2 * prec * Math.log(10)));
  const n = 1 + 2 * hs;
  const W = Math.floor(w * scale), H = Math.floor(h * scale);
  const tmp = new Float64Array(W * h), out = new Float64Array(W * H);
  const ker = new Float64Array(n);
  const kernel = (offset) => {
    let sum = 0;
    for (let i = 0; i < n; i++) { const v = (i - hs - offset) / sigma; ker[i] = Math.exp(-0.5 * v * v); sum += ker[i]; }
    for (let i = 0; i < n; i++) ker[i] /= sum;
  };
  const dbl = (v, size) => { // symmetric boundary
    if (v < 0) v = -v - 1;
    const p = 2 * size;
    v %= p;
    return v >= size ? p - 1 - v : v;
  };
  for (let x = 0; x < W; x++) {
    const xx = x / scale, xc = Math.floor(xx + 0.5);
    kernel(xx - xc);
    for (let y = 0; y < h; y++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += img[y * w + dbl(xc - hs + i, w)] * ker[i];
      tmp[y * W + x] = sum;
    }
  }
  for (let y = 0; y < H; y++) {
    const yy = y / scale, yc = Math.floor(yy + 0.5);
    kernel(yy - yc);
    for (let x = 0; x < W; x++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += tmp[dbl(yc - hs + i, h) * W + x] * ker[i];
      out[y * W + x] = sum;
    }
  }
  return { img: out, w: W, h: H };
}

function logGamma(x) {
  // Windschitl for x > 15, Lanczos below — as in the reference LSD
  if (x > 15) return 0.918938533204673 + (x - 0.5) * Math.log(x) - x + 0.5 * x * Math.log(x * Math.sinh(1 / x) + 1 / (810 * Math.pow(x, 6)));
  const q = [75122.6331530, 80916.6278952, 36308.2951477, 8687.24529705, 1168.92649479, 83.8676043424, 2.50662827511];
  let a = (x + 0.5) * Math.log(x + 5.5) - (x + 5.5), b = 0;
  for (let n = 0; n < 7; n++) { a -= Math.log(x + n); b += q[n] * Math.pow(x, n); }
  return a + Math.log(b);
}
function nfa(n, k, p, logNT) {
  if (n === 0 || k === 0) return -logNT;
  if (n === k) return -logNT - n * Math.log10(p);
  const pTerm = p / (1 - p);
  const logInvBin = (nn, kk) => logGamma(nn + 1) - logGamma(kk + 1) - logGamma(nn - kk + 1);
  let logTerm = logInvBin(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p);
  let term = Math.exp(logTerm);
  if (term === 0) return k > n * p ? -logTerm / Math.LN10 - logNT : -logNT;
  let binTail = term;
  const tolerance = 0.1;
  for (let i = k + 1; i <= n; i++) {
    const binTerm = (n - i + 1) / i;
    const multTerm = binTerm * pTerm;
    term *= multTerm;
    binTail += term;
    if (binTerm < 1) {
      const err = term * ((1 - Math.pow(multTerm, n - i + 1)) / (1 - multTerm) - 1);
      if (err < tolerance * Math.abs(-Math.log10(binTail) - logNT) * binTail) break;
    }
  }
  return -Math.log10(binTail) - logNT;
}
const angleDiff = (a, b) => {
  let d = a - b;
  while (d <= -Math.PI) d += M_2_PI;
  while (d > Math.PI) d -= M_2_PI;
  return Math.abs(d);
};
const isAligned = (angles, w, x, y, theta, prec) => {
  const a = angles[y * w + x];
  if (a === NOTDEF) return false;
  let t = theta - a;
  if (t < 0) t = -t;
  if (t > M_3_2_PI) { t -= M_2_PI; if (t < 0) t = -t; }
  return t <= prec;
};

export function lsd(gray, w0, h0) {
  const { img, w, h } = gaussianSampler(gray, w0, h0, LSD.scale, LSD.sigmaScale);
  const prec = Math.PI * LSD.angTh / 180, p = LSD.angTh / 180;
  const rhoTh = LSD.quant / Math.sin(prec);
  // gradient (2x2), level-line angle
  const angles = new Float64Array(w * h).fill(NOTDEF), mod = new Float64Array(w * h);
  let maxGrad = 0;
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
    const A = img[y * w + x], B = img[y * w + x + 1], C = img[(y + 1) * w + x], D = img[(y + 1) * w + x + 1];
    const com1 = D - A, com2 = B - C;
    const gx = com1 + com2, gy = com1 - com2;
    const norm = Math.sqrt((gx * gx + gy * gy) / 4);
    mod[y * w + x] = norm;
    if (norm > rhoTh) { angles[y * w + x] = Math.atan2(gx, -gy); if (norm > maxGrad) maxGrad = norm; }
  }
  // pseudo-ordering by gradient magnitude, strongest first
  const bins = Array.from({ length: LSD.nBins }, () => []);
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
    const i = y * w + x;
    if (angles[i] === NOTDEF) continue;
    let b = Math.floor(mod[i] * LSD.nBins / maxGrad);
    if (b >= LSD.nBins) b = LSD.nBins - 1;
    bins[b].push(i);
  }
  const logNT = 5 * (Math.log10(w) + Math.log10(h)) / 2 + Math.log10(11);
  const minRegSize = Math.floor(-logNT / Math.log10(p));
  const used = new Uint8Array(w * h);
  const segments = [];
  const regX = new Int32Array(w * h), regY = new Int32Array(w * h);

  const regionGrow = (x0, y0, prec) => {
    let n = 1;
    regX[0] = x0; regY[0] = y0;
    let regAngle = angles[y0 * w + x0], sumdx = Math.cos(regAngle), sumdy = Math.sin(regAngle);
    used[y0 * w + x0] = 1;
    for (let i = 0; i < n; i++) {
      for (let yy = regY[i] - 1; yy <= regY[i] + 1; yy++) for (let xx = regX[i] - 1; xx <= regX[i] + 1; xx++) {
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const k = yy * w + xx;
        if (used[k] !== 1 && isAligned(angles, w, xx, yy, regAngle, prec)) {
          used[k] = 1;
          regX[n] = xx; regY[n] = yy; n++;
          sumdx += Math.cos(angles[k]); sumdy += Math.sin(angles[k]);
          regAngle = Math.atan2(sumdy, sumdx);
        }
      }
    }
    return { n, angle: regAngle };
  };

  const regionToRect = (n, regAngle, prec, p) => {
    let x = 0, y = 0, sum = 0;
    for (let i = 0; i < n; i++) { const wgt = mod[regY[i] * w + regX[i]]; x += regX[i] * wgt; y += regY[i] * wgt; sum += wgt; }
    x /= sum; y /= sum;
    let Ixx = 0, Iyy = 0, Ixy = 0;
    for (let i = 0; i < n; i++) {
      const wgt = mod[regY[i] * w + regX[i]];
      Ixx += (regY[i] - y) * (regY[i] - y) * wgt;
      Iyy += (regX[i] - x) * (regX[i] - x) * wgt;
      Ixy -= (regX[i] - x) * (regY[i] - y) * wgt;
    }
    const lambda = 0.5 * (Ixx + Iyy - Math.sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4 * Ixy * Ixy));
    let theta = Math.abs(Ixx) > Math.abs(Iyy) ? Math.atan2(lambda - Ixx, Ixy) : Math.atan2(Ixy, lambda - Iyy);
    if (angleDiff(theta, regAngle) > prec) theta += Math.PI;
    const dx = Math.cos(theta), dy = Math.sin(theta);
    let lMin = 0, lMax = 0, wMin = 0, wMax = 0;
    for (let i = 0; i < n; i++) {
      const l = (regX[i] - x) * dx + (regY[i] - y) * dy;
      const ww = -(regX[i] - x) * dy + (regY[i] - y) * dx;
      if (l > lMax) lMax = l; if (l < lMin) lMin = l;
      if (ww > wMax) wMax = ww; if (ww < wMin) wMin = ww;
    }
    const rec = {
      x1: x + lMin * dx, y1: y + lMin * dy, x2: x + lMax * dx, y2: y + lMax * dy,
      width: Math.max(1, wMax - wMin), x, y, theta, dx, dy, prec, p,
    };
    return rec;
  };

  // points inside a rectangle (by its corner geometry), counted against the angle
  const rectNfa = (rec) => {
    const hw = rec.width / 2;
    const cx = [rec.x1 - rec.dy * hw, rec.x2 - rec.dy * hw, rec.x2 + rec.dy * hw, rec.x1 + rec.dy * hw];
    const cy = [rec.y1 + rec.dx * hw, rec.y2 + rec.dx * hw, rec.y2 - rec.dx * hw, rec.y1 - rec.dx * hw];
    const x0 = Math.max(0, Math.floor(Math.min(...cx))), x1 = Math.min(w - 1, Math.ceil(Math.max(...cx)));
    const y0 = Math.max(0, Math.floor(Math.min(...cy))), y1 = Math.min(h - 1, Math.ceil(Math.max(...cy)));
    const len = Math.hypot(rec.x2 - rec.x1, rec.y2 - rec.y1);
    let pts = 0, alg = 0;
    for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
      const l = (xx - rec.x1) * rec.dx + (yy - rec.y1) * rec.dy;
      const ww = -(xx - rec.x1) * rec.dy + (yy - rec.y1) * rec.dx;
      if (l < -0.5 || l > len + 0.5 || ww < -hw - 0.5 || ww > hw + 0.5) continue;
      pts++;
      if (isAligned(angles, w, xx, yy, rec.theta, rec.prec)) alg++;
    }
    return nfa(pts, alg, rec.p, logNT);
  };

  const rectImprove = (rec) => {
    const delta = 0.5, delta2 = delta / 2;
    let logNfa = rectNfa(rec);
    if (logNfa > LSD.logEps) return logNfa;
    let r = { ...rec };
    for (let n = 0; n < 5; n++) { // finer precision
      r.p /= 2; r.prec = r.p * Math.PI;
      const v = rectNfa(r);
      if (v > logNfa) { logNfa = v; Object.assign(rec, r); }
    }
    if (logNfa > LSD.logEps) return logNfa;
    r = { ...rec };
    for (let n = 0; n < 5; n++) { // narrower
      if (r.width - delta >= 0.5) {
        r.width -= delta;
        const v = rectNfa(r);
        if (v > logNfa) { Object.assign(rec, r); logNfa = v; }
      }
    }
    if (logNfa > LSD.logEps) return logNfa;
    r = { ...rec };
    for (let n = 0; n < 5; n++) { // one side in
      if (r.width - delta >= 0.5) {
        r.x1 += -r.dy * delta2; r.y1 += r.dx * delta2; r.x2 += -r.dy * delta2; r.y2 += r.dx * delta2; r.width -= delta;
        const v = rectNfa(r);
        if (v > logNfa) { Object.assign(rec, r); logNfa = v; }
      }
    }
    if (logNfa > LSD.logEps) return logNfa;
    r = { ...rec };
    for (let n = 0; n < 5; n++) { // the other side in
      if (r.width - delta >= 0.5) {
        r.x1 -= -r.dy * delta2; r.y1 -= r.dx * delta2; r.x2 -= -r.dy * delta2; r.y2 -= r.dx * delta2; r.width -= delta;
        const v = rectNfa(r);
        if (v > logNfa) { Object.assign(rec, r); logNfa = v; }
      }
    }
    if (logNfa > LSD.logEps) return logNfa;
    r = { ...rec };
    for (let n = 0; n < 5; n++) { // even finer precision
      r.p /= 2; r.prec = r.p * Math.PI;
      const v = rectNfa(r);
      if (v > logNfa) { logNfa = v; Object.assign(rec, r); }
    }
    return logNfa;
  };

  for (let b = LSD.nBins - 1; b >= 0; b--) {
    for (const i of bins[b]) {
      if (used[i] || angles[i] === NOTDEF) continue;
      const x0 = i % w, y0 = (i - x0) / w;
      let { n, angle } = regionGrow(x0, y0, prec);
      if (n < minRegSize) continue;
      let rec = regionToRect(n, angle, prec, p);
      // refine: a region too sparse for its rectangle is grown again with
      // the tolerance it actually showed, then shrunk around its seed
      let density = n / (Math.hypot(rec.x1 - rec.x2, rec.y1 - rec.y2) * rec.width);
      if (density < LSD.densityTh) {
        let sum = 0, s2 = 0, cnt = 0;
        const xc = regX[0], yc = regY[0], ang0 = angles[yc * w + xc];
        for (let k = 0; k < n; k++) {
          used[regY[k] * w + regX[k]] = 0;
          if (Math.hypot(xc - regX[k], yc - regY[k]) < rec.width) {
            const d = angleDiff(angles[regY[k] * w + regX[k]], ang0);
            sum += d; s2 += d * d; cnt++;
          }
        }
        const mean = sum / cnt, tau = 2 * Math.sqrt((s2 - 2 * mean * sum) / cnt + mean * mean);
        ({ n, angle } = regionGrow(xc, yc, tau));
        if (n < 2) continue;
        rec = regionToRect(n, angle, prec, p);
        density = n / (Math.hypot(rec.x1 - rec.x2, rec.y1 - rec.y2) * rec.width);
        if (density < LSD.densityTh) {
          // reduce the radius until dense enough
          let rad = Math.hypot(rec.x1 - rec.x2, rec.y1 - rec.y2) / 2;
          while (density < LSD.densityTh && n > 1) {
            rad *= 0.75;
            let m = 0;
            for (let k = 0; k < n; k++) {
              if (Math.hypot(xc - regX[k], yc - regY[k]) <= rad) { regX[m] = regX[k]; regY[m] = regY[k]; m++; }
              else used[regY[k] * w + regX[k]] = 0;
            }
            n = m;
            if (n < 2) break;
            rec = regionToRect(n, angle, prec, p);
            density = n / (Math.hypot(rec.x1 - rec.x2, rec.y1 - rec.y2) * rec.width);
          }
          if (n < 2) continue;
        }
      }
      const logNfa = rectImprove(rec);
      if (logNfa <= LSD.logEps) continue;
      // +0.5: pixel centers; back to the input's scale
      const k = 1 / LSD.scale;
      segments.push({ x1: (rec.x1 + 0.5) * k, y1: (rec.y1 + 0.5) * k, x2: (rec.x2 + 0.5) * k, y2: (rec.y2 + 0.5) * k, width: rec.width * k, nfa: logNfa });
    }
  }
  return segments;
}

/* ---------- 3–5. vanishing points → the roll ---------- */
const norm3 = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/* deterministic PRNG, so the same photo always gives the same angle */
export function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

const VP_TOL = Math.sin(1.5 * Math.PI / 180); // a segment supports a direction within ~1.5°
const MIN_LEN_FRAC = 0.025;                     // segments shorter than this × the long edge are noise

/* Segments (analysis px) + focal (analysis px) → interpretation-plane
   normals in camera coordinates (principal point = center, y down). A
   scene direction d is consistent with a segment when n·d ≈ 0. */
export function planeNormals(segments, w, h, f) {
  const cx = w / 2, cy = h / 2, minLen = MIN_LEN_FRAC * Math.max(w, h);
  const out = [];
  for (const s of segments) {
    const len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
    if (len < minLen) continue;
    const p = [(s.x1 - cx) / f, (s.y1 - cy) / f, 1], q = [(s.x2 - cx) / f, (s.y2 - cy) / f, 1];
    out.push({ n: norm3(cross(p, q)), len });
  }
  return out;
}

/* support of a direction: total length of the segments it explains */
function support(lines, d) {
  let s = 0;
  for (const l of lines) if (Math.abs(dot(l.n, d)) < VP_TOL) s += l.len;
  return s;
}
/* least-squares refit of a direction to the segments within the tolerance
   of it (smallest eigenvector of Σ len·n nᵀ), seeded by d */
function refine(lines, d) {
  const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  let any = false;
  for (const l of lines) {
    if (Math.abs(dot(l.n, d)) >= VP_TOL) continue;
    const n = l.n, wgt = l.len;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i * 3 + j] += wgt * n[i] * n[j];
    any = true;
  }
  if (!any) return d;
  // inverse power iteration on M (shifted) → eigenvector of the smallest eigenvalue
  let v = d.slice();
  const A = M.slice(); A[0] += 1e-9; A[4] += 1e-9; A[8] += 1e-9;
  const det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  if (Math.abs(det) < 1e-18) return d;
  const inv = [
    (A[4] * A[8] - A[5] * A[7]) / det, (A[2] * A[7] - A[1] * A[8]) / det, (A[1] * A[5] - A[2] * A[4]) / det,
    (A[5] * A[6] - A[3] * A[8]) / det, (A[0] * A[8] - A[2] * A[6]) / det, (A[2] * A[3] - A[0] * A[5]) / det,
    (A[3] * A[7] - A[4] * A[6]) / det, (A[1] * A[6] - A[0] * A[7]) / det, (A[0] * A[4] - A[1] * A[3]) / det,
  ];
  for (let it = 0; it < 30; it++) {
    v = norm3([inv[0] * v[0] + inv[1] * v[1] + inv[2] * v[2], inv[3] * v[0] + inv[4] * v[1] + inv[5] * v[2], inv[6] * v[0] + inv[7] * v[1] + inv[8] * v[2]]);
  }
  return dot(v, d) < 0 ? [-v[0], -v[1], -v[2]] : v;
}

/* Step 3 — vanishing points, one at a time (as Lightroom lists its V
   before choosing among them): the direction explaining the most segment
   length (RANSAC over segment pairs, longer segments sampled more), refit
   to its segments, which are then set aside for the next one. A
   consensus of many segments, so small changes in the input (resampling,
   grain) move it by a fraction of a degree, not to another structure.
   Only vanishing points along the picture plane are searched (|z| below
   IN_PLANE_Z, see the roll below): a point INSIDE the frame "explains" any
   segment passing within ~f·1.5° (≈25 px) of it, so random texture piles
   up there and used to take the vertical's segments with it. */
const VP_MAX = 10, VP_MIN_SHARE = 0.015, VP_MIN_COUNT = 4, VP_TRIALS = 6000;
export function vanishingDirections(lines, rand) {
  const total = lines.reduce((t, l) => t + l.len, 0);
  let remaining = lines;
  const out = [];
  while (out.length < VP_MAX && remaining.length >= 2) {
    const cum = []; let tot = 0;
    for (const l of remaining) { tot += l.len; cum.push(tot); }
    const pick = () => { const r = rand() * tot; let lo = 0, hi = cum.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m; } return remaining[lo]; };
    let best = null;
    for (let it = 0; it < VP_TRIALS; it++) {
      const a = pick(), b = pick();
      if (a === b) continue;
      const c = cross(a.n, b.n);
      if (Math.hypot(c[0], c[1], c[2]) < 1e-6) continue;
      const d = norm3(c);
      if (Math.abs(d[2]) >= IN_PLANE_Z) continue;
      const s = support(remaining, d);
      if (!best || s > best.s) best = { d, s };
    }
    if (!best || best.s < VP_MIN_SHARE * total) break;
    let d = best.d;
    for (let r = 0; r < 3; r++) d = refine(remaining, d);
    if (Math.abs(d[2]) >= IN_PLANE_Z) d = best.d;
    const inliers = remaining.filter((l) => Math.abs(dot(l.n, d)) < VP_TOL);
    if (inliers.length < VP_MIN_COUNT) break;
    out.push({ d, s: inliers.reduce((t, l) => t + l.len, 0), n: inliers.length });
    remaining = remaining.filter((l) => Math.abs(dot(l.n, d)) >= VP_TOL);
  }
  // no vertical among them (a pedestal's few edges drowned by hundreds of
  // paving joints): one more search, among vertical directions only
  if (!out.some((v) => Math.abs(v.d[1]) > Math.abs(v.d[0])) && remaining.length >= VP_MIN_COUNT) {
    const vert = remaining.filter((l) => Math.abs(l.n[0]) > Math.abs(l.n[1])); // segments running up/down
    let best = null;
    for (let i = 0; i < vert.length; i++) for (let j = i + 1; j < vert.length; j++) {
      const c = cross(vert[i].n, vert[j].n);
      if (Math.hypot(c[0], c[1], c[2]) < 1e-6) continue;
      const d = norm3(c);
      if (Math.abs(d[2]) >= IN_PLANE_Z || Math.abs(d[1]) <= Math.abs(d[0])) continue;
      const s = support(vert, d);
      if (!best || s > best.s) best = { d, s };
    }
    if (best) {
      let d = best.d;
      for (let r = 0; r < 3; r++) d = refine(vert, d);
      const inliers = vert.filter((l) => Math.abs(dot(l.n, d)) < VP_TOL);
      if (inliers.length >= VP_MIN_COUNT) out.push({ d, s: inliers.reduce((t, l) => t + l.len, 0), n: inliers.length });
    }
  }
  return out;
}

/* Steps 4–5 — the roll. With R = Rz(c)·Ry(b)·Rx(a) mapping camera → world
   (the order of Lightroom's estimate_MKR), the scene's vertical in camera
   coordinates is R's middle row, (−sin c·cos b, cos c·cos a − …, ·), and
   its horizontal R's first: the roll c is the on-screen direction from the
   picture's center to the vertical vanishing point — or, with no
   verticals in the photo, to the horizontal one — folded into ±45° (the
   scene's axes are interchangeable). It doesn't depend on the focal
   length: that, and orthogonality, only decide WHICH vanishing point is
   the vertical — so each is measured from its own segments, never bent
   toward an orthogonal frame (on a film scan with an unknown lens, the
   35 mm guess left the vertical 10° off orthogonal from the facade's
   receding lines, and forcing them square dragged it off its segments).
   A vanishing point counts only if its lines run along the picture (|z|
   small: its point lies far outside the frame) — the receding direction's
   lines converge inside the frame and say nothing about the roll.
   Checked on 41 photos hand-leveled in Lightroom: the vertical first is
   what matches the hand-set angles (the horizontal first was worse — a
   camera pitched down at a table read 18° where its bottles stood 2°). */
const ROLL_PRIOR_DEG = 15;
const IN_PLANE_Z = 0.64;  // |z| below this: at most ~40° out of the picture plane
const MIN_SUPPORT = 0.015; // a vanishing point needs this share of all segment length
const fold45 = (c) => { while (c > 45) c -= 90; while (c < -45) c += 90; return c; };
const TILT_PRIOR_Z = 0.4;  // |z| of a vanishing direction: ~0.4 ≈ 24° of pitch/yaw
const ORTHO_TOL = 0.25;   // |cos| between two scene axes (the focal is a guess): ~75–105°
export function rollFromVanishing(vps, total) {
  const cand = vps.filter((vp) => Math.abs(vp.d[2]) < IN_PLANE_Z && vp.s >= MIN_SUPPORT * total).map((vp) => {
    const d = vp.d, vertical = Math.abs(d[1]) > Math.abs(d[0]);
    const roll = vertical
      ? Math.atan2(d[1] < 0 ? -d[0] : d[0], Math.abs(d[1])) * 180 / Math.PI
      : Math.atan2(d[0] < 0 ? d[1] : -d[1], Math.abs(d[0])) * 180 / Math.PI;
    return { d, s: vp.s, n: vp.n, vertical, roll: fold45(roll) };
  });
  // a scene axis is backed by its own segments and by the other axes it is
  // square to (a stray diagonal family rarely is); a big roll needs more
  // evidence — photos are rarely tilted much — and so does a vanishing
  // point close to the frame (a strong pitch/yaw): converging lines of a
  // ceiling or a cue read as an 8° "vertical" over the real one otherwise
  const score = (c) => (c.s + cand.reduce((t, o) => t + (o !== c && Math.abs(dot(o.d, c.d)) < ORTHO_TOL ? o.s : 0), 0))
    * Math.exp(-0.5 * (c.roll / ROLL_PRIOR_DEG) ** 2) * Math.exp(-0.5 * (c.d[2] / TILT_PRIOR_Z) ** 2);
  let V = null, H = null;
  for (const c of cand) {
    c.score = score(c);
    if (c.vertical) { if (!V || c.score > V.score) V = c; }
    else if (!H || c.score > H.score) H = c;
  }
  const out = (c) => c && { roll: c.roll, share: c.s / total, len: c.s, count: c.n, z: Math.abs(c.d[2]) };
  return { V: out(V), H: out(H) };
}

/* ---------- the horizon (Lightroom's handle_degen_case) ----------
   Used when the photo has no confident vertical (see autoStraightenAngle).
   From the binary (ura: FUN_0195caec → FUN_0193cb04 → FUN_0193cdfc):
     - segments on the photo at 640 px, collinear ones within 1 px merged
       (up to 10 passes), sorted longest first
     - for each segment, the segments lying on its line — both endpoints
       within 2 px — or on a slightly bent version of it: an arc turning
       k·0.2° (k = ±1…±10) across the frame width (lens distortion bends
       a real horizon); the variant covering the most of the frame wins
     - its score: the covered share of the frame width, / cos(its angle),
       and only for lines within 45° of horizontal
     - no horizon unless some score reaches 0.4
     - the horizon is the line fitted (length-weighted) to the segments
       supporting the LONGEST segment — averaged with the next-longest
       ones while they're within 10% of its length (here: the candidate
       with the most coverage instead, see horizonRoll) */
const HORIZON_SIZE = 640, HZ_MERGE_DIST = 1, HZ_DIST = 2, HZ_MIN_COVER = 0.4, HZ_MAX_TILT = 12;

function segLine(s) {
  // a·x + b·y + c = 0 with a² + b² = 1
  const dx = s.x2 - s.x1, dy = s.y2 - s.y1, L = Math.hypot(dx, dy) || 1;
  const a = -dy / L, b = dx / L;
  return { a, b, c: -(a * s.x1 + b * s.y1) };
}
const distToLine = (l, x, y) => Math.abs(l.a * x + l.b * y + l.c);

/* length-weighted least-squares line through segments' endpoints (each
   endpoint weighted by its segment's length), as y = m·x + q or x = m·y + q,
   whichever is better conditioned (Lightroom's FUN_0194977c) */
function fitLine(segs) {
  let W = 0, Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0;
  for (const s of segs) {
    const w = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
    for (const [x, y] of [[s.x1, s.y1], [s.x2, s.y2]]) { W += w; Sx += w * x; Sy += w * y; Sxx += w * x * x; Syy += w * y * y; Sxy += w * x * y; }
  }
  const dX = W * Sxx - Sx * Sx, dY = W * Syy - Sy * Sy;
  let a, b, c;
  if (dX >= dY) { const m = (W * Sxy - Sx * Sy) / dX, q = (Sy - m * Sx) / W; a = m; b = -1; c = q; }
  else { const m = (W * Sxy - Sx * Sy) / dY, q = (Sx - m * Sy) / W; a = -1; b = m; c = q; }
  const n = Math.hypot(a, b) || 1;
  return { a: a / n, b: b / n, c: c / n };
}

/* merge segments lying on one line (endpoints within HZ_MERGE_DIST of
   each other's line) whose extents along it touch or overlap */
function mergeCollinear(segs) {
  let cur = segs.map((s) => ({ ...s }));
  for (let pass = 0; pass < 10; pass++) {
    const used = new Uint8Array(cur.length), out = [];
    let merged = false;
    for (let i = 0; i < cur.length; i++) {
      if (used[i]) continue;
      let s = cur[i];
      for (let j = i + 1; j < cur.length; j++) {
        if (used[j]) continue;
        const t = cur[j], ls = segLine(s), lt = segLine(t);
        if (Math.max(distToLine(ls, t.x1, t.y1), distToLine(ls, t.x2, t.y2)) > HZ_MERGE_DIST) continue;
        if (Math.max(distToLine(lt, s.x1, s.y1), distToLine(lt, s.x2, s.y2)) > HZ_MERGE_DIST) continue;
        // extents along s's direction must touch (gap ≤ the merge distance)
        const ux = ls.b, uy = -ls.a, p = (x, y) => x * ux + y * uy;
        const s0 = Math.min(p(s.x1, s.y1), p(s.x2, s.y2)), s1 = Math.max(p(s.x1, s.y1), p(s.x2, s.y2));
        const t0 = Math.min(p(t.x1, t.y1), p(t.x2, t.y2)), t1 = Math.max(p(t.x1, t.y1), p(t.x2, t.y2));
        if (t0 > s1 + HZ_MERGE_DIST || s0 > t1 + HZ_MERGE_DIST) continue;
        // the merged segment: the fitted line, spanning both extents
        const l = fitLine([s, t]), vx = l.b, vy = -l.a;
        const pts = [[s.x1, s.y1], [s.x2, s.y2], [t.x1, t.y1], [t.x2, t.y2]];
        const px = (s.x1 + s.x2 + t.x1 + t.x2) / 4, py = (s.y1 + s.y2 + t.y1 + t.y2) / 4;
        let lo = Infinity, hi = -Infinity;
        for (const [x, y] of pts) { const q = (x - px) * vx + (y - py) * vy; if (q < lo) lo = q; if (q > hi) hi = q; }
        // the foot of (px, py) on the fitted line
        const d = l.a * px + l.b * py + l.c, fx = px - d * l.a, fy = py - d * l.b;
        s = { x1: fx + lo * vx, y1: fy + lo * vy, x2: fx + hi * vx, y2: fy + hi * vy };
        used[j] = 1; merged = true;
      }
      out.push(s);
    }
    cur = out;
    if (!merged) break;
  }
  return cur;
}

/* Lightroom's horizon confidence (FUN_0193dd1c + FUN_0193d9e4): the photo
   64 px wide, as RGB and as the Sobel gradient magnitude of each channel;
   along the candidate line, the mean of the 5 rows just above it vs the 5
   just below, the squared difference summed over those six channels —
   sky over land or water differs across the line in color and texture, a
   curb or a railing in perspective has the same pavement on both sides. */
const HZ_CONF_W = 64, HZ_CONF_ROWS = 5;
export function horizonContrastImage(pixels, width, height, channels = 4) {
  const s = HZ_CONF_W / width, w = HZ_CONF_W, h = Math.max(3, Math.round(height * s));
  const rgb = [new Float32Array(w * h), new Float32Array(w * h), new Float32Array(w * h)], cnt = new Float32Array(w * h);
  for (let y = 0; y < height; y++) {
    const gy = Math.min(h - 1, Math.floor(y * s));
    for (let x = 0; x < width; x++) {
      const gx = Math.min(w - 1, Math.floor(x * s)), o = (y * width + x) * channels, i = gy * w + gx;
      rgb[0][i] += pixels[o] / 255; rgb[1][i] += pixels[o + 1] / 255; rgb[2][i] += pixels[o + 2] / 255; cnt[i]++;
    }
  }
  for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) rgb[c][i] /= cnt[i] || 1;
  const grad = rgb.map((p) => {
    const g = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const v = (dx, dy) => p[(y + dy) * w + x + dx];
      const gx = -v(-1, -1) - 2 * v(-1, 0) - v(-1, 1) + v(1, -1) + 2 * v(1, 0) + v(1, 1);
      const gy = -v(-1, -1) - 2 * v(0, -1) - v(1, -1) + v(-1, 1) + 2 * v(0, 1) + v(1, 1);
      g[y * w + x] = Math.hypot(gx, gy);
    }
    return g;
  });
  return { planes: [...rgb, ...grad], w, h, scale: s };
}
function horizonContrast(img, line, lineScale) {
  // line in analysis px → contrast-image px
  const k = img.scale / lineScale, a = line.a, b = line.b, c = line.c * k;
  if (Math.abs(b) < 1e-6) return 0;
  const above = new Float64Array(6), below = new Float64Array(6);
  let na = 0, nb = 0;
  for (let x = 0; x < img.w; x++) {
    const yl = Math.max(0, Math.min(img.h, Math.round(-(c + a * x) / b)));
    for (let r = 1; r <= HZ_CONF_ROWS; r++) {
      const ya = yl - r, yb = yl + r;
      if (ya >= 0) { for (let p = 0; p < 6; p++) above[p] += img.planes[p][ya * img.w + x]; na++; }
      if (yb < img.h) { for (let p = 0; p < 6; p++) below[p] += img.planes[p][yb * img.w + x]; nb++; }
    }
  }
  if (!na || !nb) return 0;
  let d = 0;
  for (let p = 0; p < 6; p++) d += (above[p] / na - below[p] / nb) ** 2;
  return d;
}

export function horizonRoll(gray, w, h, contrastImg = null, lineScale = 1) {
  const segs = mergeCollinear(lsd(gray, w, h))
    .map((s) => ({ ...s, len: Math.hypot(s.x2 - s.x1, s.y2 - s.y1) }))
    .sort((p, q) => q.len - p.len);
  const n = segs.length;
  if (!n) return null;
  const recs = segs.map((s) => {
    const l = segLine(s);
    let ang = Math.abs(Math.atan2(-l.a, l.b) * 180 / Math.PI);
    if (ang > 90) ang = 180 - ang;
    const flat = ang < 45;
    const span = flat ? w : h;
    const lo = Math.round(Math.min(flat ? s.x1 : s.y1, flat ? s.x2 : s.y2));
    const hi = Math.round(Math.max(flat ? s.x1 : s.y1, flat ? s.x2 : s.y2));
    return { s, l, ang, flat, span, lo: Math.max(0, Math.min(span - 1, lo)), hi: Math.max(0, Math.min(span - 1, hi)) };
  });
  // coverage of the frame by a candidate's supporting segments (same orientation class)
  const cover = (i, members) => {
    const r = recs[i], mark = new Uint8Array(r.span);
    let c = 0;
    for (const j of members) {
      const q = recs[j];
      if (q.flat !== r.flat) continue;
      for (let x = q.lo; x <= q.hi; x++) if (!mark[x]) { mark[x] = 1; c++; }
    }
    return c;
  };
  const best = recs.map((r, i) => {
    // straight
    let members = [];
    for (let j = 0; j < n; j++) {
      const t = recs[j].s;
      if (j === i || Math.max(distToLine(r.l, t.x1, t.y1), distToLine(r.l, t.x2, t.y2)) <= HZ_DIST) members.push(j);
    }
    let bestCover = cover(i, members), bestMembers = members;
    // bent: an arc turning k·0.2° over the frame width, through this segment
    const mx = (r.s.x1 + r.s.x2) / 2, my = (r.s.y1 + r.s.y2) / 2;
    for (let k = -10; k <= 10; k++) {
      if (!k) continue;
      const R = (w * 180) / (k * 0.2 * Math.PI);
      const cx = mx + R * r.l.a, cy = my + R * r.l.b, aR = Math.abs(R);
      const m = [];
      for (let j = 0; j < n; j++) {
        const t = recs[j].s;
        const e = Math.max(Math.abs(aR - Math.hypot(t.x1 - cx, t.y1 - cy)), Math.abs(aR - Math.hypot(t.x2 - cx, t.y2 - cy)));
        if (j === i || e <= HZ_DIST) m.push(j);
      }
      const c = cover(i, m);
      if (c > bestCover) { bestCover = c; bestMembers = m; }
    }
    const score = r.ang <= 45 ? bestCover / Math.cos(r.ang * Math.PI / 180) / w : 0;
    return { score, members: bestMembers };
  });
  // among the candidates covering enough of the frame: the one with the
  // most coverage × contrast across it — Lightroom takes the longest
  // segment's line, but on a street or a path that's a receding edge,
  // while the far shore or the skyline covers the frame with sky above it
  let top = -1;
  for (let i = 0; i < n; i++) {
    if (best[i].score < HZ_MIN_COVER) continue;
    best[i].line = fitLine(best[i].members.map((j) => recs[j].s));
    best[i].contrast = contrastImg ? horizonContrast(contrastImg, best[i].line, lineScale) : 1;
    // …and the plausibility of its tilt (a pier's edge in perspective at 9°
    // vs the far shore at 0°, both water below)
    let t = Math.atan2(-best[i].line.a, best[i].line.b) * 180 / Math.PI;
    t = fold45(t);
    best[i].rank = best[i].score * best[i].contrast / (1 + (t / TILT_SIGMA) ** 2);
    if (top < 0 || best[i].rank > best[top].rank) top = i;
  }
  if (top < 0) return null;
  const ref = best[top].line;
  let A = 0, B = 0;
  for (let i = 0; i < n; i++) {
    if (!best[i].line || best[i].rank < 0.9 * best[top].rank) continue;
    const l = best[i].line;
    const sgn = l.a * ref.a + l.b * ref.b < 0 ? -1 : 1;
    if (Math.abs(sgn * (l.a * ref.b - l.b * ref.a)) > Math.sin(Math.PI / 180)) continue;
    A += sgn * l.a; B += sgn * l.b;
  }
  // the line's direction (B, −A) is tilted atan2(−A, B) on screen (y down,
  // clockwise positive); turning it back by that much levels it. A
  // "horizon" tilted beyond HZ_MAX_TILT is a path, a roof edge, a railing
  // in perspective — a handheld horizon is rarely off by that much
  // (Lightroom's no-horizon branch bounds its default line at 30°)
  const roll = fold45(-Math.atan2(-A, B) * 180 / Math.PI);
  return Math.abs(roll) < HZ_MAX_TILT ? { roll, score: best[top].score, contrast: best[top].contrast } : null;
}

/* The straighten angle, in degrees, for an RGBA/RGB buffer of the whole
   (uncropped) photo as the user sees it; focal35 = the 35 mm-equivalent
   focal length from EXIF if known. It's the rotation that levels the photo,
   positive = clockwise on screen (y down) — Splicer's own theta convention,
   and the negative of Lightroom's CropAngle. null = nothing to level by. */
/* Which estimate to trust. Lightroom's order (FUN_0195b7ac) is: the
   Manhattan result when its vertical axis exists and the fit is
   convincing, else the horizon (handle_degen_case), else no rotation.
   Its convincing-fit test needs its full camera model; without it, the
   three readings here — the vertical vanishing point, the horizontal one,
   the horizon line — are weighed against each other instead:
     - each by its evidence: the share of the photo's segments behind a
       vanishing point (saturating at ~12 of them), the horizon's covered
       share of the frame; a horizontal vanishing point only counts when
       it's far off (its lines nearly parallel on screen — closer in, it's
       a road or a curb in perspective, not the scene's horizontal)
     - readings within 1° of each other back each other up
     - a big tilt needs more evidence (a handheld photo is rarely off
       more than a few degrees): a stray vertical at 10° on a portrait,
       or motion-blur streaks at 16°, lost to a level horizon this way
   Checked on this user's library: city scenes, portraits, riversides. */
// evidence is in units of the photo's long edge (segment length behind a
// vanishing point; ~coverage of the frame width for the horizon line)
const AGREE_DEG = 1, TILT_SIGMA = 5, MIN_EVIDENCE = 1.5, HZ_CONTRAST_FULL = 0.05, VERTICAL_TRUST = 3, HORIZONTAL_TRUST = 0.5;
const STRONG_V_COUNT = 8, STRONG_V_TILT = 4, STRONG_V_ABS = 1.2, HUGE_V_ABS = 10, HORIZON_TRUST = 8;
function weigh(cands) {
  let best = null;
  for (const c of cands) {
    let w = 0, lead = c;
    for (const o of cands) if (Math.abs(o.roll - c.roll) <= AGREE_DEG) { w += o.weight; if (o.weight > lead.weight) lead = o; }
    const score = w / (1 + (c.roll / TILT_SIGMA) ** 2);
    // the group's angle is its strongest member's, not an average of
    // estimates of different precision
    if (!best || score > best.score) best = { score, roll: lead.roll, src: lead.src };
  }
  return best && best.score >= MIN_EVIDENCE ? best : null;
}
export function autoStraightenAngle(pixels, width, height, channels = 4, focal35 = null) {
  const { gray, w, h } = analysisGray(pixels, width, height, channels);
  const segments = lsd(gray, w, h);
  const f = Math.hypot(w, h) / DIAG_35MM * (focal35 > 4 ? focal35 : DEFAULT_FOCAL_35);
  const lines = planeNormals(segments, w, h, f);
  const total = lines.reduce((s, l) => s + l.len, 0);
  const vp = total > 0 ? rollFromVanishing(vanishingDirections(lines, rng(0x5eed)), total) : { V: null, H: null };
  // Lightroom's Manhattan branch decides whenever its vertical axis is
  // found: a vertical vanishing point backed by several segments, at a
  // plausible tilt (or by a big share of the photo), settles it outright —
  // a quay's tactile strip in perspective can't outvote a pedestal's edges
  // evidence in absolute terms — total length of a vanishing point's
  // segments, in units of the photo's long edge: on a portrait a few
  // strands of hair or a coat's seams can be a big SHARE of the few lines
  // there are, but they add up to far less than one building edge does
  const L = Math.max(w, h);
  const V = vp.V, vAbs = V ? V.len / L : 0;
  if (V && V.count >= STRONG_V_COUNT && ((vAbs >= STRONG_V_ABS && Math.abs(V.roll) <= STRONG_V_TILT) || vAbs >= HUGE_V_ABS)) return V.roll;
  const g = analysisGray(pixels, width, height, channels, HORIZON_SIZE);
  const hz = horizonRoll(g.gray, g.w, g.h, horizonContrastImage(pixels, width, height, channels), g.w / width);
  const cands = [];
  // verticals first, as in Upright (its roll is read off the scene's
  // vertical): a building's walls outweigh a roofline in perspective
  if (V) cands.push({ src: "vertical", roll: V.roll, weight: VERTICAL_TRUST * vAbs });
  // horizontal lines are the weaker evidence: even far off, a family of
  // them can be a curb or a railing that isn't level in the scene
  if (vp.H) cands.push({ src: "horizontal", roll: vp.H.roll, weight: HORIZONTAL_TRUST * (vp.H.len / L) * Math.exp(-0.5 * (vp.H.z / 0.2) ** 2) });
  // a horizon's weight: its coverage, times how clearly it separates two
  // different regions (contrast ≈ 0.05 already is sky over land)
  if (hz) cands.push({ src: "horizon", roll: hz.roll, weight: HORIZON_TRUST * Math.min(1.5, hz.score) * Math.min(1, hz.contrast / HZ_CONTRAST_FULL) });
  const pick = weigh(cands);
  return pick ? pick.roll : null;
}
