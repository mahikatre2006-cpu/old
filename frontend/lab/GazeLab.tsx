import { useCallback, useEffect, useRef, useState } from "react";
import { enterFullscreen, leaveFullscreen, useExitGuard } from "./fullscreen";
import { shuffle, toFrame, type FramePoint, type GazeSource, type Point } from "./gaze";
import { postGazeSession, type LabImage } from "./labApi";
import { SessionTable, SideBySide, type LabRow } from "./Results";
import TrackerGate from "./TrackerGate";
import type { Tracker, TrackerMode } from "./useTracker";

const VIEW_MS = 3000;
const CROSS_MS = 1000;
type Phase = "intro" | "gate" | "ready" | "cross" | "view" | "submitting" | "done" | "error";

const STIMULUS_WIDTH = "min(100vw, calc(100vh * 16 / 9))";
const overlay = "fixed inset-0 z-50 flex flex-col items-center justify-center bg-black text-white";

/**
 * Gaze test (and the live demo, which is the same flow on one thumbnail): per thumbnail a fixation cross, then 3 s of
 * free viewing, then compare with the prediction. The tracker is prepared by TrackerGate, which asks for calibration
 * only when there is no saved one that still fits. The camera stays on afterwards (see the bar above), so running
 * again does not ask for anything.
 */
export default function GazeLab({ images, participant, mode, variant, tracker }: {
  images: LabImage[]; participant: string; mode: TrackerMode; variant: "test" | "demo"; tracker: Tracker;
}) {
  const [phase, setPhase] = useState<Phase>("intro");
  const [error, setError] = useState<string | null>(null);
  const [order, setOrder] = useState<LabImage[]>([]);
  const [idx, setIdx] = useState(0);
  const [rows, setRows] = useState<LabRow[]>([]);
  const [selected, setSelected] = useState(0);

  const source = useRef<GazeSource | null>(null);
  const stimulus = useRef<HTMLDivElement>(null);
  const samples = useRef<(Point & { t: number })[]>([]);
  const current = order[idx];

  const abort = useCallback((msg?: string) => {
    leaveFullscreen(); // the camera stays on; the bar above has a Stop button
    setError(msg ?? null);
    setPhase(msg ? "error" : "intro");
  }, []);

  useExitGuard(["gate", "ready", "cross", "view"].includes(phase), () => abort());
  useEffect(() => () => leaveFullscreen(), []);

  async function begin() {
    setError(null); setRows([]); setIdx(0); setSelected(0);
    setOrder(shuffle(images));
    await enterFullscreen();
    setPhase("gate");
  }

  useEffect(() => {
    if (phase !== "cross") return;
    const t = window.setTimeout(() => setPhase("view"), CROSS_MS);
    return () => window.clearTimeout(t);
  }, [phase, idx]);

  useEffect(() => {
    if (phase !== "view" || !source.current) return;
    samples.current = [];
    const t0 = performance.now();
    const unsub = source.current.subscribe((p) => samples.current.push({ ...p, t: performance.now() - t0 }));
    const t = window.setTimeout(() => { unsub(); void submit(); }, VIEW_MS);
    return () => { unsub(); window.clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, idx]);

  async function submit() {
    const el = stimulus.current;
    if (!el || !current) return;
    setPhase("submitting");
    const rect = el.getBoundingClientRect();
    const points: FramePoint[] = samples.current.map((s) => ({ ...toFrame(s, rect), t: Math.round(s.t) }));
    try {
      const session = await postGazeSession({
        thumbnail_id: current.id, method: "webgazer", points: points.slice(0, 3000),
        screen: { width: window.innerWidth, height: window.innerHeight }, participant: participant || undefined,
      });
      setRows((r) => [...r, { image: current, session }]);
      if (idx + 1 < order.length) { setIdx(idx + 1); setPhase("cross"); }
      else { leaveFullscreen(); setPhase("done"); }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not score this viewing.");
      setPhase("error");
    }
  }

  // ---------- screens ----------
  if (phase === "gate") {
    return <TrackerGate tracker={tracker} mode={mode} onReady={(s) => { source.current = s; setPhase("ready"); }} onCancel={() => abort()} />;
  }

  if (phase === "ready") {
    const err = tracker.session?.errorPx;
    return (
      <div className={overlay} role="dialog" aria-modal="true" aria-label="Ready">
        <p className="max-w-lg text-center text-lg">
          {order.length} thumbnail{order.length === 1 ? "" : "s"} will appear one at a time for <b>3 seconds</b>. Just look at each one naturally, and sit as you did for calibration.
          {mode !== "webcam" && " (Stand-in: move the pointer to where you would look.)"}
        </p>
        {err !== undefined && mode !== "mouse" && <p className="mt-2 text-sm text-stone-400">Tracker accuracy ≈ {Math.round(err)} px</p>}
        <button type="button" autoFocus onClick={() => setPhase("cross")} className="mt-6 rounded-lg bg-amber-500 px-6 py-3 font-medium text-stone-950">Start viewing</button>
      </div>
    );
  }

  if (phase === "cross" || phase === "view" || phase === "submitting") {
    return (
      <div className={overlay} aria-label={phase === "view" ? "Viewing thumbnail" : "Get ready"}>
        <div ref={stimulus} className="relative" style={{ width: STIMULUS_WIDTH, aspectRatio: "16 / 9" }}>
          {phase === "view" && current && <img src={current.url} alt="" draggable={false} className="absolute inset-0 h-full w-full select-none object-fill" />}
          {phase === "cross" && <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-5xl font-light text-white">+</span>}
          {phase === "submitting" && <span className="absolute inset-0 grid place-items-center text-stone-300">Scoring…</span>}
        </div>
        <span className="absolute bottom-3 left-1/2 -translate-x-1/2 text-xs text-stone-500" aria-live="polite">{idx + 1} / {order.length}</span>
      </div>
    );
  }

  if (phase === "done") {
    const row = rows[Math.min(selected, rows.length - 1)];
    return (
      <div className="space-y-5">
        {rows.length > 1 && <SessionTable rows={rows} />}
        {rows.length > 1 && (
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Choose a thumbnail to inspect">
            {rows.map((r, i) => (
              <button key={r.session.session_id} role="tab" aria-selected={i === selected} onClick={() => setSelected(i)}
                className={`rounded-md border px-2 py-1 text-sm ${i === selected ? "border-amber-500 bg-amber-50 dark:bg-amber-950/40" : "border-stone-300 dark:border-stone-700"}`}>{r.image.name}</button>
            ))}
          </div>
        )}
        {row && <SideBySide row={row} />}
        <button type="button" onClick={() => setPhase("intro")} className="rounded-md border border-stone-300 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800">Run again (no recalibration needed)</button>
      </div>
    );
  }

  // intro / error
  return (
    <div className="space-y-3">
      {error && (
        <div role="alert" className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
          {rows.length > 0 && <button type="button" onClick={() => { setError(null); setPhase("done"); }} className="ml-3 underline">Show what was completed</button>}
        </div>
      )}
      <p className="max-w-2xl text-sm">
        {variant === "demo"
          ? "Live demo: look at the thumbnail for 3 seconds. We then show the model's prediction next to where you really looked."
          : "Webcam test: each thumbnail appears for 3 seconds in random order while your gaze is recorded."}{" "}
        {mode === "mouse" ? "You chose the mouse stand-in, so no camera is used. " : "The first time, you calibrate once (about a minute); it is then saved in this browser. "}
        Your camera video never leaves this browser; only gaze coordinates are sent to the local server.
      </p>
      <button type="button" disabled={!images.length} onClick={() => void begin()}
        className="rounded-lg bg-amber-500 px-5 py-2.5 font-medium text-stone-950 hover:bg-amber-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-900 disabled:opacity-50">
        {variant === "demo" ? "Start live demo" : "Start webcam test"}
      </button>
      {!images.length && <p className="text-sm text-stone-500">Add a thumbnail above first.</p>}
    </div>
  );
}
