import { useCallback, useEffect, useRef, useState } from "react";
import { SAVE_LIMIT, UNUSABLE_LIMIT, chooseCorrection, median, qualityLabel, type Point } from "./calibrationMath";
import GazeBlob from "./GazeBlob";
import { CALIBRATION_POINTS, VALIDATION_POINTS, type GazeSource } from "./gaze";
import type { CalibrationOutcome } from "./useTracker";

const SAMPLES_PER_DOT = 16; // continuous training while the eyes rest on the dot: far more data than a few clicks
const SAMPLE_EVERY_MS = 70;
const SETTLE_MS = 600;
const MEASURE_MS = 1000;
const MIN_VALID = 8;

type Phase = "intro" | "dots" | "validate" | "result";
const overlay = "fixed inset-0 z-50 flex flex-col items-center justify-center bg-black text-white";
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * mode "full":  13 training dots (look at each, click once, hold your gaze) then a 5-point accuracy check.
 * mode "check": only the 5-point check, which also re-fits the correction, so small changes in how you sit are absorbed
 *               without repeating the whole calibration.
 */
export default function Calibration({ source, mode, commit, onFinish, onCancel, hasSaved }: {
  source: GazeSource; mode: "full" | "check"; commit: (o: CalibrationOutcome, persist?: boolean) => boolean;
  onFinish: () => void; onCancel: () => void; hasSaved: boolean;
}) {
  const [phase, setPhase] = useState<Phase>(mode === "full" ? "intro" : "validate");
  const [idx, setIdx] = useState(0);
  const [collecting, setCollecting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [valIdx, setValIdx] = useState(0);
  const [valMsg, setValMsg] = useState("");
  const [outcome, setOutcome] = useState<(CalibrationOutcome & { persisted: boolean; failed: number; unusable?: boolean; poor?: boolean }) | null>(null);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const [, refresh] = useState(0); // re-render a few times a second so the face-found hint stays current
  useEffect(() => { const i = window.setInterval(() => refresh((n) => n + 1), 300); return () => window.clearInterval(i); }, []);

  // preview + face box help people centre themselves; hidden again when we leave
  useEffect(() => { source.showPreview(true); return () => source.showPreview(false); }, [source]);

  // ---------- training dots ----------
  const onDot = useCallback(async (el: HTMLButtonElement) => {
    if (collecting) return;
    setCollecting(true);
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2; // train on the dot's centre, not on where the cursor was
    let good = 0;
    const t0 = performance.now();
    while (alive.current && good < SAMPLES_PER_DOT && performance.now() - t0 < 3500) {
      if (source.train(cx, cy)) good++; // true only when the tracker really saw the eyes
      setProgress(good / SAMPLES_PER_DOT);
      await sleep(SAMPLE_EVERY_MS);
    }
    if (!alive.current) return;
    setCollecting(false);
    setProgress(0);
    if (good < 6) return; // face was lost: the same dot stays up so it can be redone
    if (idx + 1 < CALIBRATION_POINTS.length) setIdx(idx + 1);
    else setPhase("validate");
  }, [collecting, idx, source]);

  // ---------- 5-point accuracy check (also fits the correction) ----------
  useEffect(() => {
    if (phase !== "validate") return;
    let cancelled = false;
    (async () => {
      const vp = { w: window.innerWidth, h: window.innerHeight };
      source.setCorrection(null); // measure the tracker itself, then decide about a correction
      const preds: Point[] = [], targets: Point[] = [];
      let failed = 0;
      await sleep(500);
      for (let i = 0; i < VALIDATION_POINTS.length; i++) {
        if (cancelled) return;
        setValIdx(i);
        setValMsg("");
        const target = { x: VALIDATION_POINTS[i].x * vp.w, y: VALIDATION_POINTS[i].y * vp.h };
        await sleep(SETTLE_MS); // let the eyes arrive
        const got: Point[] = [];
        const unsub = source.subscribeRaw((p) => got.push(p));
        await sleep(MEASURE_MS);
        unsub();
        if (got.length < MIN_VALID) { failed++; continue; }
        preds.push({ x: median(got.map((p) => p.x)), y: median(got.map((p) => p.y)) });
        targets.push(target);
      }
      if (cancelled) return;
      if (failed > 0 || preds.length < 5) {
        setOutcome({ ok: false, rawErrorPx: NaN, errorPx: NaN, affine: null, persisted: false, failed });
        setPhase("result");
        return;
      }
      const res = chooseCorrection(preds, targets, vp);
      source.setCorrection(res.affine);
      const o: CalibrationOutcome = { ok: true, rawErrorPx: res.rawError, errorPx: res.affine ? res.correctedError! : res.rawError, affine: res.affine };
      if (o.errorPx > UNUSABLE_LIMIT * vp.w) { // garbage in, garbage out: do not keep it, do not pretend it is fine
        source.setCorrection(null);
        setOutcome({ ...o, ok: false, persisted: false, failed: 0, unusable: true });
        setPhase("result");
        return;
      }
      const poor = o.errorPx > SAVE_LIMIT * vp.w;
      const persisted = commit(o, !poor); // a poor calibration is used for this visit but not remembered
      setOutcome({ ...o, persisted, failed: 0, poor });
      setPhase("result");
    })();
    return () => { cancelled = true; };
  }, [phase, source, commit]);

  // Esc is handled by the parent (it also leaves fullscreen); here only the screens
  if (phase === "intro") {
    return (
      <div className={overlay} role="dialog" aria-modal="true" aria-label="Calibration instructions">
        <div className="max-w-xl space-y-3 px-6 text-center">
          <h2 className="text-2xl font-semibold">Quick calibration</h2>
          <ul className="space-y-1 text-left text-stone-300">
            <li>• Sit 50–70 cm from the screen with your face well lit from the front (not a window behind you).</li>
            <li>• Keep your head fairly still; move your <b>eyes</b>, not your head.</li>
            <li>• 13 dots appear. Look at a dot, <b>click it once</b>, and keep looking at it until it fills.</li>
            <li>• It is saved in this browser, so you will normally only do this once. Esc cancels.</li>
          </ul>
          <p className="text-sm text-stone-400">Check the small camera box (bottom-left): your face should sit inside its frame.</p>
          <div className="flex justify-center gap-3 pt-2">
            <button type="button" autoFocus onClick={() => setPhase("dots")} className="rounded-lg bg-amber-500 px-6 py-3 font-medium text-stone-950">Start</button>
            <button type="button" onClick={onCancel} className="rounded-lg border border-stone-600 px-5 py-3">Cancel</button>
          </div>
        </div>
      </div>
    );
  }

  if (phase === "dots") {
    const p = CALIBRATION_POINTS[idx];
    return (
      <div className={overlay} role="dialog" aria-modal="true" aria-label="Calibration">
        {idx >= 4 && <GazeBlob source={source} trail={false} radius={26} label={false} />}
        <div className="absolute left-1/2 top-[18%] -translate-x-1/2 text-center">
          <p className="text-lg font-medium">Dot {idx + 1} of {CALIBRATION_POINTS.length}</p>
          <p className="text-sm text-stone-400" aria-live="polite">
            {collecting && progress < 0.15 && !source.faceFound() ? "Face not found: move into view" : collecting ? "Keep looking at the dot…" : idx > 0 && !source.faceFound() ? "Face not found: move into view" : "Look at the dot, then click it once"}
          </p>
        </div>
        <button
          key={idx}
          type="button"
          aria-label={`Calibration dot ${idx + 1} of ${CALIBRATION_POINTS.length}`}
          onClick={(e) => void onDot(e.currentTarget)}
          disabled={collecting}
          className="absolute grid h-10 w-10 place-items-center rounded-full border-2 border-white bg-red-500 focus-visible:outline-2 focus-visible:outline-amber-400"
          style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%`, transform: `translate(-50%,-50%) scale(${collecting ? 1 - 0.55 * progress : 1})`, boxShadow: collecting ? `0 0 0 ${6 + 10 * progress}px rgba(251,191,36,.35)` : "0 0 0 6px rgba(255,255,255,.15)" }}
        />
      </div>
    );
  }

  if (phase === "validate") {
    const t = VALIDATION_POINTS[valIdx];
    return (
      <div className={overlay} role="dialog" aria-modal="true" aria-label="Accuracy check">
        <GazeBlob source={source} trail={false} radius={30} label />
        <div className="absolute left-1/2 top-[14%] -translate-x-1/2 text-center">
          <p className="text-lg font-medium">{mode === "check" ? "Quick check" : "Accuracy check"}: point {valIdx + 1} of {VALIDATION_POINTS.length}</p>
          <p className="text-sm text-stone-400">Look at the dot without clicking. The glow shows where the tracker thinks you are looking. {valMsg}</p>
        </div>
        <div className="absolute h-6 w-6 rounded-full border-2 border-white bg-red-500" style={{ left: `${t.x * 100}%`, top: `${t.y * 100}%`, transform: "translate(-50%,-50%)" }} />
      </div>
    );
  }

  // result
  const o = outcome!;
  const q = o.ok ? qualityLabel(o.errorPx, window.innerWidth) : null;
  return (
    <div className={overlay} role="dialog" aria-modal="true" aria-label="Calibration result">
      <GazeBlob source={source} trail label={false} />
      <div className="max-w-xl space-y-3 px-6 text-center">
        {o.ok ? (
          <>
            <h2 className="text-2xl font-semibold">Tracker accuracy: about {Math.round(o.errorPx)} px ({q!.text})</h2>
            <p className="text-stone-300">
              {o.affine ? `The correction brought it from ${Math.round(o.rawErrorPx)} px down to about ${Math.round(o.errorPx)} px.` : "No correction was needed."}{" "}
              Move your eyes around: the glow should follow them.
            </p>
            <p className="text-sm text-stone-400">
              {o.persisted ? "Saved in this browser only: you will not be asked again unless your setup changes."
                : o.poor ? "Not saved because it is not accurate enough to reuse, so you will be asked again next time."
                : "Could not save to this browser (storage blocked), so it will be asked again next time."}
              {o.poor && " For better results: face a light source, sit closer, keep your head still, and recalibrate."}
            </p>
          </>
        ) : (
          <>
            <h2 className="text-2xl font-semibold">
              {o.unusable ? `The tracker is about ${Math.round(o.errorPx)} px off: too much to be useful` : `Could not read your gaze at ${o.failed} of ${VALIDATION_POINTS.length} points`}
            </h2>
            <p className="text-stone-300">
              {o.unusable ? "Look straight at each dot with your head still, keep your face inside the camera box and well lit, then recalibrate. This result was not saved."
                : "Make sure your face is inside the camera box, evenly lit, and not too far away."}
            </p>
          </>
        )}
        <div className="flex flex-wrap justify-center gap-3 pt-2">
          {o.ok && <button type="button" autoFocus onClick={onFinish} className="rounded-lg bg-amber-500 px-6 py-3 font-medium text-stone-950">Continue</button>}
          {!o.ok && <button type="button" autoFocus onClick={() => { setOutcome(null); setValIdx(0); setPhase("validate"); }} className="rounded-lg bg-amber-500 px-6 py-3 font-medium text-stone-950">Try the check again</button>}
          <button type="button" onClick={() => { void source.clearTraining(); source.setCorrection(null); setIdx(0); setOutcome(null); setPhase("intro"); }} className="rounded-lg border border-stone-500 px-5 py-3">Recalibrate fully</button>
          {!hasSaved && !o.ok && <button type="button" onClick={onCancel} className="rounded-lg border border-stone-600 px-5 py-3">Cancel</button>}
        </div>
      </div>
    </div>
  );
}
