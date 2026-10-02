import { useEffect, useRef, useState } from "react";
import Calibration from "./Calibration";
import type { GazeSource } from "./gaze";
import type { Plan, Tracker, TrackerMode } from "./useTracker";

const overlay = "fixed inset-0 z-50 flex flex-col items-center justify-center bg-black text-white";

/**
 * Gets the tracker ready, asking for as little as possible:
 *   camera not running            -> start it (the first start loads the face model)
 *   fine within the last 15 min   -> nothing
 *   saved calibration still fits  -> 5-point quick check (also absorbs small changes in how you sit)
 *   otherwise                     -> full calibration, then saved for next time
 */
export default function TrackerGate({ tracker, mode, force, onReady, onCancel }: {
  tracker: Tracker; mode: TrackerMode; force?: Plan; onReady: (s: GazeSource) => void; onCancel: () => void;
}) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [src, setSrc] = useState<GazeSource | null>(null);
  const ran = useRef(false);

  useEffect(() => {
    if (ran.current) return;
    ran.current = true;
    (async () => {
      try {
        const s = await tracker.start(mode);
        let p = await tracker.prepare();
        if (force && p === "ready") p = force;
        if (force === "calibrate" && p !== "calibrate") { await s.clearTraining(); s.setCorrection(null); p = "calibrate"; }
        setSrc(s);
        if (p === "ready") onReady(s);
        else setPlan(p);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not start the gaze tracker.");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return (
      <div className={overlay} role="alert">
        <p className="max-w-lg px-6 text-center text-lg">{error}</p>
        <button type="button" autoFocus onClick={onCancel} className="mt-6 rounded-lg bg-amber-500 px-6 py-3 font-medium text-stone-950">Back</button>
      </div>
    );
  }
  if (!src || !plan) {
    return (
      <div className={overlay} role="status">
        <p className="text-lg">{mode === "mouse" ? "Starting…" : "Starting the camera…"}</p>
        {mode === "webcam" && <p className="mt-1 text-sm text-stone-400">The first start loads the face model (a few seconds). Allow the camera if the browser asks.</p>}
      </div>
    );
  }
  return (
    <Calibration source={src} mode={plan === "check" ? "check" : "full"} hasSaved={!!tracker.saved}
      commit={tracker.commit} onFinish={() => onReady(src)} onCancel={onCancel} />
  );
}
