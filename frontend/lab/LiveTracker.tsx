import { useEffect, useState } from "react";
import { qualityLabel } from "./calibrationMath";
import GazeBlob from "./GazeBlob";
import { VALIDATION_POINTS, type GazeSource } from "./gaze";
import type { Tracker } from "./useTracker";

/** Full-screen live view: the blob follows your gaze everywhere, so you can see the tracking quality directly. */
export default function LiveTracker({ source, tracker, onQuickCheck, onRecalibrate, onExit }: {
  source: GazeSource; tracker: Tracker; onQuickCheck: () => void; onRecalibrate: () => void; onExit: () => void;
}) {
  const [hz, setHz] = useState(0);
  const [face, setFace] = useState(true);
  const [trail, setTrail] = useState(true);

  useEffect(() => {
    let n = 0;
    const unsub = source.subscribe(() => { n++; });
    const iv = window.setInterval(() => { setHz(n * 2); n = 0; setFace(source.faceFound()); }, 500);
    return () => { unsub(); window.clearInterval(iv); };
  }, [source]);

  const err = tracker.session?.errorPx;
  const q = err !== undefined ? qualityLabel(err, window.innerWidth) : null;
  const btn = "rounded-lg border border-stone-600 bg-stone-900/80 px-4 py-2 text-sm hover:bg-stone-800 focus-visible:outline-2 focus-visible:outline-amber-400";

  return (
    <div className="fixed inset-0 z-50 bg-stone-950 text-white" role="dialog" aria-modal="true" aria-label="Live gaze tracker">
      {/* targets to test the tracker against: look at each and watch where the glow lands */}
      {[{ x: 0.5, y: 0.5 }, ...VALIDATION_POINTS.slice(1), { x: 0.1, y: 0.1 }, { x: 0.9, y: 0.1 }, { x: 0.1, y: 0.9 }, { x: 0.9, y: 0.9 }].map((p, i) => (
        <span key={i} aria-hidden="true" className="absolute h-5 w-5 rounded-full border border-stone-600"
          style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%`, transform: "translate(-50%,-50%)" }}>
          <span className="absolute left-1/2 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-stone-500" />
        </span>
      ))}
      <GazeBlob source={source} trail={trail} radius={42} />

      <div className="absolute left-4 top-4 space-y-1 rounded-lg bg-black/60 px-3 py-2 text-sm" aria-live="polite">
        <div className="flex items-center gap-2"><span className={`inline-block h-2.5 w-2.5 rounded-full ${face ? "bg-emerald-400" : "bg-rose-500"}`} /> {face ? "Tracking your face" : "Face not found"}</div>
        <div className="text-stone-400">{source.kind === "mouse" ? "Mouse stand-in" : `${hz} samples/s${source.kind === "sim" ? " (simulated)" : ""}`}{q ? ` · calibrated ≈ ${Math.round(err!)} px (${q.text})` : ""}</div>
        <div className="text-stone-500">Look at the rings: the glow should land on them.</div>
      </div>

      <div className="absolute bottom-5 left-1/2 flex -translate-x-1/2 flex-wrap justify-center gap-2">
        <label className={`${btn} flex items-center gap-2`}><input type="checkbox" checked={trail} onChange={(e) => setTrail(e.target.checked)} /> Trail</label>
        {source.kind !== "mouse" && <button type="button" className={btn} onClick={onQuickCheck}>Quick check (5 dots)</button>}
        {source.kind !== "mouse" && <button type="button" className={btn} onClick={onRecalibrate}>Recalibrate</button>}
        <button type="button" autoFocus className={`${btn} border-amber-500`} onClick={onExit}>Exit (Esc)</button>
      </div>
    </div>
  );
}
