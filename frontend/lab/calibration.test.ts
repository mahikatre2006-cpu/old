import { describe, expect, it } from "vitest";
import {
  IDENTITY, OneEuro, applyAffine, chooseCorrection, fitAffine, leaveOneOutError, median, qualityLabel, type Affine, type Point,
} from "./calibrationMath";
import {
  STORAGE_KEY, ageText, clearCalibration, compatibility, deserializeTraining, loadCalibration, saveCalibration, serializeTraining,
  type SavedCalibration, type TrainingItem,
} from "./calibrationStore";

const VP = { w: 1920, h: 1080 };
const TARGETS: Point[] = [[0.5, 0.5], [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.5, 0.2], [0.5, 0.8]].map(([x, y]) => ({ x: x * VP.w, y: y * VP.h }));

// small deterministic noise
function rng(seed: number) { let s = seed; return () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296 - 0.5) * 2; }

describe("fitAffine", () => {
  it("recovers a known affine distortion when the data is clean", () => {
    const truth: Affine = { a: 1.08, b: 0.04, c: -35, d: -0.03, e: 0.93, f: 28 };
    // the tracker reports what the inverse of the distortion would produce; fitting pred -> target must undo it
    const pred = TARGETS.map((t) => applyAffine(truth, t));
    const m = fitAffine(pred, TARGETS, VP, 1e-9)!;
    for (let i = 0; i < pred.length; i++) {
      const fixed = applyAffine(m, pred[i]);
      expect(Math.hypot(fixed.x - TARGETS[i].x, fixed.y - TARGETS[i].y)).toBeLessThan(0.5);
    }
  });

  it("stays at the identity when the tracker was already right", () => {
    const m = fitAffine(TARGETS, TARGETS, VP)!;
    expect(m.a).toBeCloseTo(1, 6); expect(m.e).toBeCloseTo(1, 6);
    expect(Math.abs(m.b) + Math.abs(m.d) + Math.abs(m.c) + Math.abs(m.f)).toBeLessThan(1e-3);
  });

  it("removes a constant offset (the most common webcam error)", () => {
    const pred = TARGETS.map((t) => ({ x: t.x + 90, y: t.y - 55 }));
    const m = fitAffine(pred, TARGETS, VP)!;
    const fixed = applyAffine(m, { x: 1000, y: 500 });
    expect(Math.hypot(fixed.x - 910, fixed.y - 555)).toBeLessThan(6);
  });

  it("refuses too few or mismatched points", () => {
    expect(fitAffine(TARGETS.slice(0, 3), TARGETS.slice(0, 3), VP)).toBeNull();
    expect(fitAffine(TARGETS, TARGETS.slice(0, 5), VP)).toBeNull();
    expect(fitAffine(TARGETS.map(() => ({ x: 5, y: 5 })), TARGETS, VP, 0)).toBeNull(); // all predictions identical: degenerate
  });
});

describe("chooseCorrection", () => {
  it("applies the correction when there is systematic error, and reports the honest (leave-one-out) result", () => {
    const n = rng(1);
    const pred = TARGETS.map((t) => ({ x: t.x + 110 + n() * 8, y: t.y - 70 + n() * 8 }));
    const r = chooseCorrection(pred, TARGETS, VP);
    expect(r.affine).not.toBeNull();
    expect(r.rawError).toBeGreaterThan(120);
    expect(r.correctedError!).toBeLessThan(25);
  });

  it("leaves the tracker alone when the errors are random jitter (nothing to fix)", () => {
    const n = rng(7);
    const pred = TARGETS.map((t) => ({ x: t.x + n() * 90, y: t.y + n() * 90 }));
    const r = chooseCorrection(pred, TARGETS, VP);
    expect(r.affine).toBeNull();
  });

  it("does not fiddle with an already accurate tracker", () => {
    const pred = TARGETS.map((t) => ({ x: t.x + 2, y: t.y - 1 }));
    expect(chooseCorrection(pred, TARGETS, VP).affine).toBeNull();
  });

  it("leave-one-out needs enough points", () => {
    expect(leaveOneOutError(TARGETS.slice(0, 4), TARGETS.slice(0, 4), VP)).toBeNull();
  });
});

describe("OneEuro", () => {
  it("passes a constant signal through unchanged", () => {
    const f = new OneEuro();
    let v = 0;
    for (let t = 0; t < 1000; t += 33) v = f.filter(500, t);
    expect(v).toBeCloseTo(500, 6);
  });

  it("smooths jitter while the eye rests", () => {
    const n = rng(3), f = new OneEuro(1.0, 0.004);
    const raw: number[] = [], out: number[] = [];
    for (let t = 0; t < 4000; t += 33) { const x = 800 + n() * 60; raw.push(x); out.push(f.filter(x, t)); }
    const sd = (a: number[]) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
    expect(sd(out.slice(20))).toBeLessThan(0.5 * sd(raw.slice(20)));
  });

  it("follows a fast jump (a saccade) within a few frames instead of lagging", () => {
    const f = new OneEuro(1.0, 0.02);
    for (let t = 0; t < 1000; t += 33) f.filter(200, t);
    let v = 0, frames = 0;
    for (let t = 1000; t < 2000; t += 33) { v = f.filter(1400, t); frames++; if (Math.abs(v - 1400) < 120) break; }
    expect(frames).toBeLessThanOrEqual(8);
  });
});

describe("helpers", () => {
  it("median and quality thresholds", () => {
    expect(median([5, 1, 9])).toBe(5); expect(median([1, 2, 3, 4])).toBe(2.5);
    expect(qualityLabel(60, 1920).level).toBe("good");
    expect(qualityLabel(140, 1920).level).toBe("fair");
    expect(qualityLabel(260, 1920).level).toBe("poor");
    expect(IDENTITY.a).toBe(1);
  });
});

// ---------------- storage ----------------

function fakeItems(n: number): TrainingItem[] {
  return Array.from({ length: n }, (_, i) => ({
    eyes: {
      left: { patch: { data: new Uint8ClampedArray(60 * 4).map((_, k) => (i + k) % 256) }, width: 10, height: 6, imagex: 3, imagey: 4 },
      right: { patch: { data: new Uint8ClampedArray(60 * 4).map((_, k) => (i * 2 + k) % 256) }, width: 10, height: 6, imagex: 30, imagey: 4 },
    },
    screenPos: [100 + i, 200 + i] as [number, number],
    type: "click",
  }));
}

function memStorage() {
  const m = new Map<string, string>();
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => { m.set(k, v); }, removeItem: (k: string) => { m.delete(k); }, _m: m };
}

const meta = { savedAt: Date.now(), viewport: VP, errorPx: 88, rawErrorPx: 140, samples: 60 };
const makeCal = (): SavedCalibration => ({ version: 1, meta, affine: { ...IDENTITY, c: 12 }, training: serializeTraining(fakeItems(60)) });

describe("calibration storage", () => {
  it("round-trips WebGazer training samples through JSON without losing pixels", () => {
    const items = fakeItems(30);
    const back = deserializeTraining(JSON.parse(JSON.stringify(serializeTraining(items))));
    expect(back).toHaveLength(30);
    expect(Array.from(back[7].eyes.left.patch.data)).toEqual(Array.from(items[7].eyes.left.patch.data));
    expect(back[7].screenPos).toEqual([107, 207]);
    expect(back[7].eyes.right.width).toBe(10); expect(back[7].type).toBe("click");
  });

  it("saves, loads and clears", () => {
    const st = memStorage();
    expect(loadCalibration(st)).toBeNull();
    expect(saveCalibration(makeCal(), st)).toBe(true);
    const got = loadCalibration(st)!;
    expect(got.meta.errorPx).toBe(88); expect(got.training).toHaveLength(60); expect(got.affine!.c).toBe(12);
    clearCalibration(st);
    expect(loadCalibration(st)).toBeNull();
  });

  it("ignores corrupted, foreign or tampered data instead of crashing", () => {
    const st = memStorage();
    st.setItem(STORAGE_KEY, "{not json");
    expect(loadCalibration(st)).toBeNull();
    st.setItem(STORAGE_KEY, JSON.stringify({ version: 2, meta, affine: null, training: [] }));
    expect(loadCalibration(st)).toBeNull();
    const bad = makeCal(); bad.training[3].l.data = [1, 2, 3]; // patch size no longer matches width x height
    st.setItem(STORAGE_KEY, JSON.stringify(bad));
    expect(loadCalibration(st)).toBeNull();
    const nan = makeCal(); (nan.affine as Affine).a = NaN;
    st.setItem(STORAGE_KEY, JSON.stringify(nan));
    expect(loadCalibration(st)).toBeNull();
    const few = makeCal(); few.training = few.training.slice(0, 5);
    st.setItem(STORAGE_KEY, JSON.stringify(few));
    expect(loadCalibration(st)).toBeNull();
  });

  it("reports failure (not an exception) when storage is full", () => {
    const full = { setItem: () => { throw new DOMException("quota", "QuotaExceededError"); } };
    expect(saveCalibration(makeCal(), full)).toBe(false);
  });

  it("is only reused on the same screen size and while fresh", () => {
    const now = meta.savedAt + 3600_000;
    expect(compatibility(meta, VP, now).ok).toBe(true);
    expect(compatibility(meta, { w: 1920 * 1.02, h: 1080 }, now).ok).toBe(true); // within tolerance
    expect(compatibility(meta, { w: 1280, h: 720 }, now)).toEqual({ ok: false, reason: "screen-changed" });
    expect(compatibility(meta, VP, meta.savedAt + 31 * 24 * 3600_000)).toEqual({ ok: false, reason: "expired" });
  });

  it("describes age in plain words", () => {
    const t = 1_000_000_000_000;
    expect(ageText(t, t + 30_000)).toBe("just now");
    expect(ageText(t, t + 20 * 60_000)).toBe("20 min ago");
    expect(ageText(t, t + 5 * 3600_000)).toBe("5 h ago");
    expect(ageText(t, t + 3 * 24 * 3600_000)).toBe("3 days ago");
  });
});
