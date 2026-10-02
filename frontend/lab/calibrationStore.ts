// Saves a finished calibration in THIS browser only (localStorage) so it is not asked for again.
// Nothing here is ever sent to the server.
import type { Affine } from "./calibrationMath";

export const STORAGE_KEY = "eyelab.calibration.v1";
export const MAX_AGE_MS = 30 * 24 * 3600 * 1000; // a month: faces, rooms and chairs change
const VIEWPORT_TOLERANCE = 0.03; // saved mapping is in screen pixels, so the screen/window must match

export interface EyeSample { patch: { data: ArrayLike<number> }; width: number; height: number; imagex?: number; imagey?: number }
export interface TrainingItem { eyes: { left: EyeSample; right: EyeSample }; screenPos: [number, number]; type: string }

interface SerialEye { data: number[]; width: number; height: number; imagex?: number; imagey?: number }
interface SerialItem { l: SerialEye; r: SerialEye; p: [number, number]; t: string }

export interface CalibrationMeta {
  savedAt: number;
  viewport: { w: number; h: number };
  errorPx: number; // validation error after correction, px
  rawErrorPx: number; // before correction
  samples: number;
}

export interface SavedCalibration {
  version: 1;
  meta: CalibrationMeta;
  affine: Affine | null;
  training: SerialItem[];
}

const ser = (e: EyeSample): SerialEye => ({ data: Array.from(e.patch.data), width: e.width, height: e.height, imagex: e.imagex, imagey: e.imagey });

export function serializeTraining(items: TrainingItem[]): SerialItem[] {
  return items.map((it) => ({ l: ser(it.eyes.left), r: ser(it.eyes.right), p: [it.screenPos[0], it.screenPos[1]], t: it.type }));
}

/** Back into the shape WebGazer's setData() expects (it re-wraps the patch arrays into ImageData itself). */
export function deserializeTraining(items: SerialItem[]): TrainingItem[] {
  const eye = (e: SerialEye): EyeSample => ({ patch: { data: e.data }, width: e.width, height: e.height, imagex: e.imagex, imagey: e.imagey });
  return items.map((it) => ({ eyes: { left: eye(it.l), right: eye(it.r) }, screenPos: it.p, type: it.t }));
}

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function valid(c: unknown): c is SavedCalibration {
  const s = c as SavedCalibration | null;
  if (!s || s.version !== 1 || !s.meta || !Array.isArray(s.training) || s.training.length < 20) return false;
  const m = s.meta;
  if (!finite(m.savedAt) || !finite(m.errorPx) || !finite(m.viewport?.w) || !finite(m.viewport?.h)) return false;
  if (s.affine !== null && !(["a", "b", "c", "d", "e", "f"] as const).every((k) => finite(s.affine![k]))) return false;
  return s.training.every((t) => Array.isArray(t.l?.data) && Array.isArray(t.r?.data) && Array.isArray(t.p) && finite(t.p[0]) && finite(t.p[1]) && t.l.data.length === t.l.width * t.l.height * 4);
}

export function saveCalibration(cal: SavedCalibration, storage: Pick<Storage, "setItem"> = localStorage): boolean {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(cal));
    return true;
  } catch {
    return false; // storage full or blocked (private window): the session still works, it just won't be remembered
  }
}

export function loadCalibration(storage: Pick<Storage, "getItem"> = localStorage): SavedCalibration | null {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return valid(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function clearCalibration(storage: Pick<Storage, "removeItem"> = localStorage): void {
  try { storage.removeItem(STORAGE_KEY); } catch { /* nothing to clear */ }
}

export type Compat = { ok: true } | { ok: false; reason: "expired" | "screen-changed" };

export function compatibility(meta: CalibrationMeta, viewport: { w: number; h: number }, now = Date.now()): Compat {
  if (now - meta.savedAt > MAX_AGE_MS) return { ok: false, reason: "expired" };
  const dw = Math.abs(viewport.w - meta.viewport.w) / meta.viewport.w;
  const dh = Math.abs(viewport.h - meta.viewport.h) / meta.viewport.h;
  if (dw > VIEWPORT_TOLERANCE || dh > VIEWPORT_TOLERANCE) return { ok: false, reason: "screen-changed" };
  return { ok: true };
}

export function ageText(savedAt: number, now = Date.now()): string {
  const m = Math.max(0, Math.round((now - savedAt) / 60000));
  if (m < 2) return "just now";
  if (m < 90) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 36) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
}
