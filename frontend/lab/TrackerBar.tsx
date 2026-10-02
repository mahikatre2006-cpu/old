import { qualityLabel } from "./calibrationMath";
import { ageText } from "./calibrationStore";
import type { Tracker, TrackerMode } from "./useTracker";

/** Always-visible status of the camera and the saved calibration, with the controls to stop or forget them. */
export default function TrackerBar({ tracker, mode, setMode, showMode }: {
  tracker: Tracker; mode: TrackerMode; setMode: (m: TrackerMode) => void; showMode: boolean;
}) {
  const { status, kind, saved, session, saveFailed, error } = tracker;
  const on = status === "on";
  const cal = saved?.meta;
  const q = cal ? qualityLabel(cal.errorPx, window.innerWidth) : null;
  const btn = "rounded-md border border-stone-300 px-2.5 py-1 text-sm hover:bg-stone-100 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800";

  return (
    <section aria-label="Eye tracker status" className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-stone-200 bg-white px-4 py-3 text-sm dark:border-stone-800 dark:bg-stone-900">
      <div className="flex items-center gap-2">
        <span aria-hidden="true" className={`inline-block h-2.5 w-2.5 rounded-full ${on ? (kind === "webcam" ? "animate-pulse bg-red-500" : "bg-emerald-500") : "bg-stone-400"}`} />
        <span className="font-medium">
          {status === "starting" ? "Starting…" : on ? (kind === "webcam" ? "Camera on" : kind === "sim" ? "Simulated camera on" : "Mouse stand-in on") : "Camera off"}
        </span>
        {on && <button type="button" onClick={tracker.stop} className={btn}>{kind === "webcam" ? "Turn camera off" : "Stop"}</button>}
        {!on && status !== "starting" && (
          <button type="button" className={btn} onClick={() => { void tracker.start(mode).catch(() => {}); }}>Start {mode === "mouse" ? "mouse stand-in" : mode === "sim" ? "simulated camera" : "camera"}</button>
        )}
      </div>

      {mode !== "mouse" && (
        <div className="flex flex-wrap items-center gap-2 text-stone-600 dark:text-stone-300">
          {cal ? (
            <>
              <span>Saved calibration: ≈ {Math.round(cal.errorPx)} px ({q!.text}), {ageText(cal.savedAt)}{session ? " · checked this visit" : ""}</span>
              <button type="button" className={btn} onClick={() => void tracker.forget()}>Forget it</button>
            </>
          ) : (
            <span>No saved calibration: the first run asks you to calibrate once (≈ 1 min), then remembers it in this browser.</span>
          )}
        </div>
      )}
      {saveFailed && <span role="status" className="text-amber-700 dark:text-amber-300">Could not save the calibration in this browser, so it will be asked again.</span>}
      {error && <span role="alert" className="text-red-700 dark:text-red-300">{error}</span>}

      {showMode && (
        <label className="ml-auto flex items-center gap-2">
          Input
          <select value={mode} onChange={(e) => setMode(e.target.value as TrackerMode)} className="rounded border border-stone-300 bg-transparent px-2 py-1 dark:border-stone-700">
            <option value="webcam">Webcam</option>
            <option value="mouse">Mouse stand-in (rehearsal)</option>
            {import.meta.env.DEV && <option value="sim">Simulated webcam (dev test)</option>}
          </select>
        </label>
      )}
    </section>
  );
}
