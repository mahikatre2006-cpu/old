import { useCallback, useEffect, useRef, useState } from "react";
import type { Affine } from "./calibrationMath";
import {
  clearCalibration, compatibility, deserializeTraining, loadCalibration, saveCalibration, serializeTraining, type SavedCalibration,
} from "./calibrationStore";
import { createMouseGaze, createSimulatedWebcam, createWebGazer, type GazeSource } from "./gaze";

export type TrackerMode = "webcam" | "mouse" | "sim";

export type Plan = "ready" | "check" | "calibrate";
export interface CalibrationOutcome { rawErrorPx: number; errorPx: number; affine: Affine | null; ok: boolean }
interface Session { errorPx: number; rawErrorPx: number; checkedAt: number; viewport: { w: number; h: number } }

const RECENT_MS = 15 * 60 * 1000; // within a quarter of an hour of a check, nobody is asked anything
const near = (a: { w: number; h: number }, b: { w: number; h: number }) => Math.abs(a.w - b.w) / a.w < 0.03 && Math.abs(a.h - b.h) / a.h < 0.03;
const viewport = () => ({ w: window.innerWidth, h: window.innerHeight });

/**
 * Owns the gaze source for the whole Eye lab visit: the camera stays on (with a visible indicator) between runs,
 * so nothing is re-calibrated unless the saved calibration no longer fits.
 */
export function useTracker() {
  const [status, setStatus] = useState<"off" | "starting" | "on">("off");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<SavedCalibration | null>(() => loadCalibration());
  const [session, setSession] = useState<Session | null>(null);
  const [kind, setKind] = useState<TrackerMode | null>(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const source = useRef<GazeSource | null>(null);
  const modelLoaded = useRef(false); // has this camera session's model been given a calibration (restored or trained)?

  const stop = useCallback(() => {
    source.current?.stop(); // releases the camera
    source.current = null;
    modelLoaded.current = false;
    setSession(null);
    setKind(null);
    setStatus("off");
  }, []);

  useEffect(() => () => { source.current?.stop(); }, []); // leaving the lab turns the camera off

  /** Starts the camera (or the mouse stand-in) if it is not already running as the requested kind. */
  const start = useCallback(async (want: TrackerMode): Promise<GazeSource> => {
    if (source.current && kind === want) return source.current;
    if (source.current) stop();
    setError(null);
    setStatus("starting");
    try {
      const src = want === "mouse" ? createMouseGaze() : want === "sim" ? createSimulatedWebcam() : await createWebGazer();
      await src.start();
      source.current = src;
      modelLoaded.current = false;
      setKind(want);
      setStatus("on");
      return src;
    } catch (e) {
      setStatus("off");
      const msg = e instanceof Error ? e.message : "Could not start the gaze tracker.";
      setError(msg);
      throw e;
    }
  }, [kind, stop]);

  /** What does this run need? Restores a saved calibration into the tracker when it still fits this screen. */
  const prepare = useCallback(async (): Promise<Plan> => {
    const src = source.current;
    if (!src) throw new Error("Tracker is not running.");
    if (src.kind === "mouse") return "ready";
    const vp = viewport();
    if (session && Date.now() - session.checkedAt < RECENT_MS && near(session.viewport, vp)) return "ready";
    const s = loadCalibration();
    if (s && compatibility(s.meta, vp).ok) {
      if (!modelLoaded.current) { await src.setTraining(deserializeTraining(s.training)); modelLoaded.current = true; }
      src.setCorrection(s.affine);
      return "check";
    }
    await src.clearTraining(); // start the model from nothing so old data cannot pollute the new calibration
    src.setCorrection(null);
    modelLoaded.current = false;
    return "calibrate";
  }, [session]);

  /** A calibration (or a quick check) finished: remember it for the rest of the visit and for next time. */
  const commit = useCallback((o: CalibrationOutcome, persist = true): boolean => {
    const src = source.current;
    if (!src || src.kind === "mouse" || !o.ok) return false;
    const vp = viewport();
    const training = src.getTraining();
    const cal: SavedCalibration = {
      version: 1,
      meta: { savedAt: Date.now(), viewport: vp, errorPx: o.errorPx, rawErrorPx: o.rawErrorPx, samples: training.length },
      affine: o.affine,
      training: serializeTraining(training),
    };
    const persisted = persist && saveCalibration(cal);
    setSaveFailed(persist && !persisted); // only a problem if we wanted to remember it
    if (persist) setSaved(persisted ? cal : null);
    setSession({ errorPx: o.errorPx, rawErrorPx: o.rawErrorPx, checkedAt: Date.now(), viewport: vp });
    modelLoaded.current = true;
    return persisted;
  }, []);

  /** Forget the saved calibration (storage and the running model). */
  const forget = useCallback(async () => {
    clearCalibration();
    setSaved(null);
    setSession(null);
    modelLoaded.current = false;
    await source.current?.clearTraining();
    source.current?.setCorrection(null);
  }, []);

  return { status, kind, error, source, saved, session, saveFailed, start, stop, prepare, commit, forget };
}

export type Tracker = ReturnType<typeof useTracker>;
