import { useEffect, useMemo, useRef, useState } from "react";
import { shuffle, toFrame, type FramePoint } from "./gaze";
import { postGazeSession, type LabImage } from "./labApi";
import { SessionTable, SideBySide, type LabRow } from "./Results";

const SHOW_MS = 1500;
type Phase = "intro" | "show" | "click1" | "click2" | "submitting" | "done";

/** Click test: each thumbnail is shown for 1.5 s, then hidden; the viewer clicks where they looked first and second. */
export default function ClickLab({ images, participant }: { images: LabImage[]; participant: string }) {
  const [phase, setPhase] = useState<Phase>("intro");
  const [order, setOrder] = useState<LabImage[]>([]);
  const [idx, setIdx] = useState(0);
  const [clicks, setClicks] = useState<FramePoint[]>([]);
  const [rows, setRows] = useState<LabRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const current = order[idx];

  function start() {
    setOrder(shuffle(images)); // random order, so position in the run does not bias the first look
    setIdx(0); setRows([]); setClicks([]); setError(null);
    setPhase("show");
  }

  useEffect(() => {
    if (phase !== "show") return;
    const t = window.setTimeout(() => setPhase("click1"), SHOW_MS);
    return () => window.clearTimeout(t);
  }, [phase, idx]);

  async function finish(points: FramePoint[]) {
    setPhase("submitting");
    try {
      const session = await postGazeSession({
        thumbnail_id: current.id, method: "click", points,
        screen: { width: window.innerWidth, height: window.innerHeight }, participant: participant || undefined,
      });
      setRows((r) => [...r, { image: current, session }]);
      if (idx + 1 < order.length) { setIdx(idx + 1); setClicks([]); setPhase("show"); } else setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save this answer.");
      setPhase("click1"); setClicks([]);
    }
  }

  function onFrameClick(e: React.MouseEvent) {
    if (!frame.current || (phase !== "click1" && phase !== "click2")) return;
    const p = toFrame({ x: e.clientX, y: e.clientY }, frame.current.getBoundingClientRect());
    const pt: FramePoint = { x: Math.max(0, Math.min(1280, p.x)), y: Math.max(0, Math.min(720, p.y)), order: phase === "click1" ? 1 : 2 };
    const next = [...clicks, pt];
    setClicks(next);
    if (phase === "click1") setPhase("click2"); else void finish(next);
  }

  const last = useMemo(() => rows[rows.length - 1], [rows]);

  if (phase === "intro") {
    return (
      <div className="space-y-3">
        <p className="max-w-2xl text-sm">
          Each thumbnail appears for <b>1.5 seconds</b> and then disappears. Click where you looked <b>first</b>, then where you looked <b>second</b>.
          Go with your gut. {images.length} thumbnail{images.length === 1 ? "" : "s"} in random order. The person who picked the thumbnails should not take the test.
        </p>
        <button type="button" disabled={!images.length} onClick={start} className="rounded-lg bg-amber-500 px-5 py-2.5 font-medium text-stone-950 hover:bg-amber-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-900 disabled:opacity-50">
          Start click test
        </button>
        {!images.length && <p className="text-sm text-stone-500">Add thumbnails above first.</p>}
      </div>
    );
  }

  if (phase === "done") {
    return (
      <div className="space-y-5">
        <SessionTable rows={rows} />
        {last && <details><summary className="cursor-pointer text-sm font-medium">See the last one side by side</summary><div className="mt-3"><SideBySide row={last} /></div></details>}
        <button type="button" onClick={() => setPhase("intro")} className="rounded-md border border-stone-300 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800">Run again</button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-3">
      <div className="flex items-center justify-between text-sm text-stone-500 dark:text-stone-400">
        <span>Thumbnail {idx + 1} of {order.length}</span>
        <span aria-live="polite">{phase === "show" ? "Look…" : phase === "click1" ? "Click where you looked FIRST" : phase === "click2" ? "Now click where you looked SECOND" : "Saving…"}</span>
      </div>
      {error && <div role="alert" className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">{error}</div>}
      <div ref={frame} onClick={onFrameClick} className={`relative aspect-video w-full overflow-hidden rounded-xl bg-stone-900 ${phase === "click1" || phase === "click2" ? "cursor-crosshair ring-2 ring-amber-500" : ""}`}>
        {phase === "show" && current && <img src={current.url} alt="" draggable={false} className="absolute inset-0 h-full w-full select-none object-fill" />}
        {(phase === "click1" || phase === "click2") && (
          <div className="absolute inset-0 grid place-items-center text-stone-500" style={{ backgroundImage: "linear-gradient(#292524 1px, transparent 1px), linear-gradient(90deg, #292524 1px, transparent 1px)", backgroundSize: "10% 11.11%" }}>
            <span>The thumbnail is hidden. Click on this frame where it was.</span>
          </div>
        )}
        {clicks.map((c) => (
          <span key={c.order} className="pointer-events-none absolute grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-white bg-black text-lg font-bold text-white"
            style={{ left: `${(c.x / 1280) * 100}%`, top: `${(c.y / 720) * 100}%` }}>{c.order}</span>
        ))}
      </div>
      {phase === "click2" && (
        <button type="button" onClick={() => void finish(clicks)} className="rounded-md border border-stone-300 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800">
          I only looked in one place
        </button>
      )}
    </div>
  );
}
