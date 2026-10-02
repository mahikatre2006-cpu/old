// Pure maths for gaze calibration: no DOM, no WebGazer, so it can be unit tested.

export interface Point { x: number; y: number }
/** x' = a*x + b*y + c ; y' = d*x + e*y + f, in screen pixels */
export interface Affine { a: number; b: number; c: number; d: number; e: number; f: number }
export const IDENTITY: Affine = { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };

export function applyAffine(m: Affine, p: Point): Point {
  return { x: m.a * p.x + m.b * p.y + m.c, y: m.d * p.x + m.e * p.y + m.f };
}

export function median(v: number[]): number {
  if (!v.length) return NaN;
  const s = [...v].sort((p, q) => p - q);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Solve a small linear system A x = b by Gaussian elimination with partial pivoting. */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = col + 1; r < n; r++) {
      const k = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= k * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return x;
}

/**
 * Least-squares affine map from what the tracker reported (pred) to where the person really looked (target).
 * Fitted in screen-normalised coordinates and shrunk toward the identity, so a few noisy points cannot
 * produce a wild warp: with ridge = 0 it is plain least squares, with a large ridge it stays the identity.
 * Returns null if there are too few (or degenerate) points.
 */
export function fitAffine(pred: Point[], target: Point[], viewport: { w: number; h: number }, ridge = 0.05): Affine | null {
  const n = pred.length;
  if (n < 4 || target.length !== n) return null;
  const nx = (p: Point) => [p.x / viewport.w, p.y / viewport.h, 1];
  const X = pred.map(nx);
  const XtX = [0, 1, 2].map((i) => [0, 1, 2].map((j) => X.reduce((s, r) => s + r[i] * r[j], 0)));
  // shrink the 2x2 linear part toward identity, but leave the offset free
  const R = [ridge * n, ridge * n, 0];
  const A = XtX.map((row, i) => row.map((v, j) => v + (i === j ? R[i] : 0)));
  const solveFor = (axis: "x" | "y", prior: number[]) => {
    const q = target.map((t) => (axis === "x" ? t.x / viewport.w : t.y / viewport.h));
    const rhs = [0, 1, 2].map((i) => X.reduce((s, r, k) => s + r[i] * q[k], 0) + R[i] * prior[i]);
    return solve(A, rhs);
  };
  const tx = solveFor("x", [1, 0, 0]);
  const ty = solveFor("y", [0, 1, 0]);
  if (!tx || !ty) return null;
  // back to pixels: x' = W*(tx0*x/W + tx1*y/H + tx2) = tx0*x + tx1*(W/H)*y + W*tx2
  return {
    a: tx[0], b: tx[1] * (viewport.w / viewport.h), c: tx[2] * viewport.w,
    d: ty[0] * (viewport.h / viewport.w), e: ty[1], f: ty[2] * viewport.h,
  };
}

const MIN_GAIN_PX = 12; // a correction must buy at least this many pixels to be worth applying

export function meanDistance(a: Point[], b: Point[]): number {
  return a.reduce((s, p, i) => s + Math.hypot(p.x - b[i].x, p.y - b[i].y), 0) / a.length;
}

/** Honest accuracy of the correction: fit on all-but-one point, test on the held-out one, repeat. */
export function leaveOneOutError(pred: Point[], target: Point[], viewport: { w: number; h: number }, ridge = 0.05): number | null {
  if (pred.length < 5) return null;
  const errs: number[] = [];
  for (let i = 0; i < pred.length; i++) {
    const rest = pred.map((_, k) => k).filter((k) => k !== i);
    const m = fitAffine(rest.map((k) => pred[k]), rest.map((k) => target[k]), viewport, ridge);
    if (!m) return null;
    const p = applyAffine(m, pred[i]);
    errs.push(Math.hypot(p.x - target[i].x, p.y - target[i].y));
  }
  return median(errs);
}

export interface CorrectionResult {
  affine: Affine | null; // null = leave the tracker's output alone
  rawError: number; // px, median over validation points, before correction
  correctedError: number | null; // px, leave-one-out estimate after correction
}

/**
 * Decide whether the affine correction is worth applying. It must beat the raw tracker by a clear margin in the
 * leave-one-out test; otherwise it would only be fitting noise and we keep the raw output.
 */
export function chooseCorrection(pred: Point[], target: Point[], viewport: { w: number; h: number }): CorrectionResult {
  const rawError = median(pred.map((p, i) => Math.hypot(p.x - target[i].x, p.y - target[i].y)));
  const loo = leaveOneOutError(pred, target, viewport);
  const m = fitAffine(pred, target, viewport);
  // clear relative AND absolute gain, otherwise we would only be fitting noise (or "fixing" 2 px)
  if (!m || loo === null || !(loo < rawError * 0.9) || rawError - loo < MIN_GAIN_PX) return { affine: null, rawError, correctedError: loo };
  return { affine: m, rawError, correctedError: loo };
}

/** One Euro filter (Casiez et al. 2012): heavy smoothing when the eye rests, little lag when it moves. */
export class OneEuro {
  private x: number | null = null;
  private dx = 0;
  private t: number | null = null;
  constructor(private minCutoff = 1.0, private beta = 0.01, private dCutoff = 1.0) {}

  private alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value: number, tMs: number): number {
    if (this.x === null || this.t === null) { this.x = value; this.t = tMs; return value; }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000);
    const rawDx = (value - this.x) / dt;
    this.dx += this.alpha(this.dCutoff, dt) * (rawDx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += this.alpha(cutoff, dt) * (value - this.x);
    this.t = tMs;
    return this.x;
  }

  reset() { this.x = null; this.t = null; this.dx = 0; }
}

/** Beyond this (as a fraction of screen width) a calibration is not remembered; beyond UNUSABLE it is rejected. */
export const SAVE_LIMIT = 0.09;
export const UNUSABLE_LIMIT = 0.15;

export function qualityLabel(px: number, screenWidth: number): { text: string; level: "good" | "fair" | "poor" } {
  const f = px / screenWidth;
  if (f < 0.05) return { text: "good", level: "good" };
  if (f < 0.09) return { text: "fair", level: "fair" };
  return { text: "poor", level: "poor" };
}
