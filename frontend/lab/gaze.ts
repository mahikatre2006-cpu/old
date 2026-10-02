// Gaze sources for the eye lab. Webcam frames never leave the browser: only (x, y, t) points are ever sent.
import { applyAffine, median, type Affine, type Point } from "./calibrationMath";
import type { TrainingItem } from "./calibrationStore";

export type { Point } from "./calibrationMath";

export const FRAME_W = 1280;
export const FRAME_H = 720;

export interface Rect { left: number; top: number; width: number; height: number }
export interface FramePoint { x: number; y: number; t?: number; order?: number }

/** Screen (client) pixels -> 1280x720 thumbnail-frame pixels, given where the frame is drawn on screen. */
export function toFrame(p: Point, r: Rect): Point {
  return { x: ((p.x - r.left) * FRAME_W) / r.width, y: ((p.y - r.top) * FRAME_H) / r.height };
}

export interface GazeSource {
  readonly kind: "mouse" | "webgazer" | "sim";
  start(): Promise<void>;
  /** Releases the camera / listeners. Always safe to call, more than once. */
  stop(): void;
  /** Gaze after our correction. */
  subscribe(cb: (p: Point) => void): () => void;
  /** Exactly what the tracker reported, before our correction (used to fit the correction). */
  subscribeRaw(cb: (p: Point) => void): () => void;
  /** Calibration: the viewer is looking at (x, y) right now. Returns true if a sample was actually learned
   *  (the tracker silently skips a sample when it cannot see the eyes). */
  train(x: number, y: number): boolean;
  /** Show or hide WebGazer's small camera preview with the face box (so people can centre their face). */
  showPreview(show: boolean): void;
  /** What the tracker has learned so far, so a calibration can be saved. */
  getTraining(): TrainingItem[];
  /** Replace what the tracker has learned with a saved calibration. */
  setTraining(items: TrainingItem[]): Promise<void>;
  /** Forget everything learned (start a fresh calibration). */
  clearTraining(): Promise<void>;
  setCorrection(m: Affine | null): void;
  /** Is a face being tracked right now? */
  faceFound(): boolean;
  /** ms timestamp of the most recent sample (0 if none yet). */
  readonly lastSampleAt: number;
}

/** The mouse pointer stands in for gaze: lets the whole lab be rehearsed and tested without a webcam. */
export function createMouseGaze(): GazeSource {
  let last: Point | null = null;
  let timer: number | undefined;
  let lastAt = 0;
  const subs = new Set<(p: Point) => void>();
  const onMove = (e: PointerEvent) => { last = { x: e.clientX, y: e.clientY }; };
  return {
    kind: "mouse",
    async start() {
      window.addEventListener("pointermove", onMove);
      timer = window.setInterval(() => { if (last) { lastAt = performance.now(); subs.forEach((cb) => cb(last!)); } }, 33); // ~30 Hz
    },
    stop() {
      window.removeEventListener("pointermove", onMove);
      window.clearInterval(timer);
      subs.clear();
    },
    subscribe(cb) { subs.add(cb); return () => subs.delete(cb); },
    subscribeRaw(cb) { subs.add(cb); return () => subs.delete(cb); },
    train: () => false,
    showPreview() {},
    getTraining: () => [],
    async setTraining() {},
    async clearTraining() {},
    setCorrection() {},
    faceFound: () => last !== null,
    get lastSampleAt() { return lastAt; },
  };
}

/**
 * DEV ONLY. Behaves like the webcam tracker so the calibration flow can be exercised without a camera: the mouse is the
 * "eye", but the reported gaze is deliberately wrong (a constant offset plus jitter) until a correction is fitted,
 * and it needs training, can save/restore what it learned, and loses the "face" when asked.
 */
export function createSimulatedWebcam(bias: Point = { x: 70, y: -45 }, jitter = 14): GazeSource {
  let eye: Point | null = null;
  let timer: number | undefined;
  let lastAt = 0;
  let training: TrainingItem[] = [];
  let affine: Affine | null = null;
  const subs = new Set<(p: Point) => void>();
  const rawSubs = new Set<(p: Point) => void>();
  const onMove = (e: PointerEvent) => { eye = { x: e.clientX, y: e.clientY }; };
  const noise = () => (Math.random() + Math.random() + Math.random() - 1.5) * jitter;
  const fakeEye = (x: number): { patch: { data: number[] }; width: number; height: number } => ({ patch: { data: Array.from({ length: 240 }, (_, k) => (x + k) % 256) }, width: 10, height: 6 });
  return {
    kind: "sim",
    async start() {
      window.addEventListener("pointermove", onMove);
      timer = window.setInterval(() => {
        if (!eye || (window as any).__simFaceLost) return;
        lastAt = performance.now();
        const raw = { x: eye.x + bias.x + noise(), y: eye.y + bias.y + noise() };
        rawSubs.forEach((cb) => cb(raw));
        const fixed = affine ? applyAffine(affine, raw) : raw;
        subs.forEach((cb) => cb(fixed));
      }, 33);
    },
    stop() { window.removeEventListener("pointermove", onMove); window.clearInterval(timer); subs.clear(); rawSubs.clear(); },
    subscribe(cb) { subs.add(cb); return () => subs.delete(cb); },
    subscribeRaw(cb) { rawSubs.add(cb); return () => rawSubs.delete(cb); },
    train(x, y) {
      if ((window as any).__simFaceLost || !eye) return false;
      training.push({ eyes: { left: fakeEye(training.length), right: fakeEye(training.length + 7) }, screenPos: [x, y], type: "click" });
      return true;
    },
    showPreview() {},
    getTraining: () => [...training],
    async setTraining(items) { training = [...items]; },
    async clearTraining() { training = []; },
    setCorrection(m) { affine = m; },
    faceFound: () => performance.now() - lastAt < 600 && !(window as any).__simFaceLost,
    get lastSampleAt() { return lastAt; },
  };
}

/* eslint-disable @typescript-eslint/no-explicit-any */
/** WebGazer's bundle is a classic script that defines a global `webgazer` (it exports nothing as a module),
 *  so it is served from our own /public and loaded with a <script> tag, only when the lab is used. */
function loadWebGazerScript(): Promise<any> {
  const w = window as any;
  if (w.webgazer) return Promise.resolve(w.webgazer);
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/webgazer.js";
    s.async = true;
    s.onload = () => (w.webgazer ? resolve(w.webgazer) : reject(new Error("The eye tracker script loaded but did not initialise.")));
    s.onerror = () => reject(new Error("Could not load /webgazer.js. Run `npm run sync:webgazer` in the frontend folder, then reload."));
    document.head.appendChild(s);
  });
}

const TRAINING_CAP = 690; // WebGazer keeps at most 700 training samples

export async function createWebGazer(): Promise<GazeSource> {
  const wg: any = await loadWebGazerScript();
  const subs = new Set<(p: Point) => void>();
  const rawSubs = new Set<(p: Point) => void>();
  let started = false;
  let affine: Affine | null = null;
  let lastAt = 0;
  let lastFaceAt = 0;

  const releaseCamera = () => {
    // webgazer.end() leaves the camera running (its own stopVideo call is commented out), so stop the tracks ourselves
    const v = document.getElementById(wg.params.videoElementId) as HTMLVideoElement | null;
    (v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
  };

  return {
    kind: "webgazer",
    async start() {
      if (started) return;
      wg.params.faceMeshSolutionPath = "/mediapipe/face_mesh"; // served from our own origin: works offline
      wg.params.saveDataAcrossSessions = false; // WebGazer's own persistence is off; we save a validated copy ourselves
      wg.params.dataTimestep = 33; // ~30 Hz
      wg.params.showGazeDot = false; // WebGazer's red dot is replaced by our own blob where we want one
      wg.params.showFaceOverlay = false;
      wg.params.applyKalmanFilter = true;
      wg.setGazeListener((d: Point | null) => {
        if (!d || !Number.isFinite(d.x) || !Number.isFinite(d.y)) return;
        lastAt = lastFaceAt = performance.now();
        const raw = { x: d.x, y: d.y };
        rawSubs.forEach((cb) => cb(raw));
        const fixed = affine ? applyAffine(affine, raw) : raw;
        subs.forEach((cb) => cb(fixed));
      });
      try {
        await wg.begin();
      } catch (e) {
        releaseCamera();
        throw new Error(
          e instanceof DOMException && (e.name === "NotAllowedError" || e.name === "NotFoundError")
            ? "Camera access was blocked or no camera was found. Allow the camera for this site, or use the mouse stand-in."
            : "Could not start the webcam tracker.",
        );
      }
      wg.removeMouseEventListeners?.(); // mouse movement is not gaze: only train on calibration points we record ourselves
      wg.showPredictionPoints?.(false);
      started = true;
    },
    stop() {
      subs.clear();
      rawSubs.clear();
      if (!started) return;
      started = false;
      releaseCamera();
      try { wg.clearGazeListener(); wg.pause(); wg.end(); } catch { /* already torn down */ }
    },
    subscribe(cb) { subs.add(cb); return () => subs.delete(cb); },
    subscribeRaw(cb) { rawSubs.add(cb); return () => rawSubs.delete(cb); },
    train(x, y) {
      // WebGazer only emits gaze AFTER it has been trained, so training cannot wait for gaze output. It adds a sample
      // only when it can see the eyes; detect that by whether its training set grew (it is capped, hence the >= check).
      const n = () => (wg.getRegression()?.[0]?.getData?.() ?? []).length;
      const before = n();
      wg.recordScreenPosition(x, y, "click");
      const grew = n() > before || before >= TRAINING_CAP;
      if (grew) lastFaceAt = performance.now();
      return grew;
    },
    showPreview(show) { try { wg.showVideoPreview(show); wg.showFaceOverlay(false); wg.showFaceFeedbackBox(show); } catch { /* not ready yet */ } },
    getTraining: () => [...((wg.getRegression()?.[0]?.getData?.() as TrainingItem[]) ?? [])],
    async setTraining(items) {
      await wg.clearData();
      // WebGazer re-wraps each stored eye patch into ImageData itself; hand it fresh copies
      wg.getRegression()[0].setData(items.map((it) => ({ ...it, eyes: { left: { ...it.eyes.left, patch: { data: Array.from(it.eyes.left.patch.data) } }, right: { ...it.eyes.right, patch: { data: Array.from(it.eyes.right.patch.data) } } } })));
    },
    async clearTraining() { await wg.clearData(); },
    setCorrection(m) { affine = m; },
    faceFound: () => performance.now() - lastFaceAt < 600,
    get lastSampleAt() { return lastAt; },
  };
}

/** Median distance in screen px between samples and where the person was asked to look. */
export function medianError(samples: Point[], target: Point): number | null {
  if (samples.length < 5) return null;
  return median(samples.map((s) => Math.hypot(s.x - target.x, s.y - target.y)));
}

export function shuffle<T>(items: T[]): T[] {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** 13 calibration dots in a snaking order (short eye movements between neighbours). */
export const CALIBRATION_POINTS: Point[] = [
  { x: 0.08, y: 0.1 }, { x: 0.5, y: 0.1 }, { x: 0.92, y: 0.1 },
  { x: 0.7, y: 0.3 }, { x: 0.3, y: 0.3 },
  { x: 0.08, y: 0.5 }, { x: 0.5, y: 0.5 }, { x: 0.92, y: 0.5 },
  { x: 0.7, y: 0.7 }, { x: 0.3, y: 0.7 },
  { x: 0.92, y: 0.9 }, { x: 0.5, y: 0.9 }, { x: 0.08, y: 0.9 },
];

/** 5 validation targets, deliberately different from the calibration dots so the check is honest. */
export const VALIDATION_POINTS: Point[] = [
  { x: 0.5, y: 0.5 }, { x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 },
];
