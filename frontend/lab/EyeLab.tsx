import { useRef, useState } from "react";
import { analyze } from "../api";
import ClickLab from "./ClickLab";
import { enterFullscreen, leaveFullscreen, useExitGuard } from "./fullscreen";
import GazeLab from "./GazeLab";
import type { GazeSource } from "./gaze";
import type { LabImage } from "./labApi";
import LiveTracker from "./LiveTracker";
import TrackerBar from "./TrackerBar";
import TrackerGate from "./TrackerGate";
import { useTracker, type Plan, type Tracker, type TrackerMode } from "./useTracker";

type Tab = "live" | "demo" | "gaze" | "click";
const TABS: [Tab, string, string][] = [
  ["live", "Live tracker", "A glowing blob follows your eyes across the screen in real time. Use it to see (and tune) the tracking."],
  ["demo", "Live demo", "One thumbnail: look at it for 3 s, then see the prediction next to your real gaze."],
  ["gaze", "Webcam test", "Several thumbnails in random order, 3 s each, with webcam gaze."],
  ["click", "Click test", "No camera: thumbnail flashes for 1.5 s, then click where you looked first and second."],
];

/** The glowing-blob view. Prepares the tracker (asking for calibration only if needed), then shows it. */
function LiveRun({ tracker, mode }: { tracker: Tracker; mode: TrackerMode }) {
  const [phase, setPhase] = useState<"idle" | "gate" | "live">("idle");
  const [force, setForce] = useState<Plan | undefined>();
  const [gateKey, setGateKey] = useState(0);
  const src = useRef<GazeSource | null>(null);

  async function open(f?: Plan) {
    setForce(f); setGateKey((k) => k + 1);
    await enterFullscreen();
    setPhase("gate");
  }
  const exit = () => { leaveFullscreen(); setPhase("idle"); };
  useExitGuard(phase !== "idle", exit);

  if (phase === "gate") {
    return <TrackerGate key={gateKey} tracker={tracker} mode={mode} force={force} onReady={(s) => { src.current = s; setPhase("live"); }} onCancel={exit} />;
  }
  if (phase === "live" && src.current) {
    return <LiveTracker source={src.current} tracker={tracker} onExit={exit} onQuickCheck={() => void open("check")} onRecalibrate={() => void open("calibrate")} />;
  }
  return (
    <div className="space-y-3">
      <p className="max-w-2xl text-sm">
        A glowing blob follows your gaze across the whole screen with a short trail, so you can see how well the tracker is doing: look at each corner and the
        rings in the middle and watch where the glow lands. {mode !== "webcam" ? "(Stand-in: the blob is driven by the mouse pointer.)" : "The first time you calibrate once (≈ 1 minute); it is saved in this browser."}
      </p>
      <button type="button" onClick={() => void open()} className="rounded-lg bg-amber-500 px-5 py-2.5 font-medium text-stone-950 hover:bg-amber-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-900">
        Open live tracker
      </button>
    </div>
  );
}

/** Eye-tracking lab (M11): checks the model's predictions against what real people actually look at. */
export default function EyeLab() {
  const input = useRef<HTMLInputElement>(null);
  const tracker = useTracker();
  const [tab, setTab] = useState<Tab>("live");
  const [images, setImages] = useState<LabImage[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [participant, setParticipant] = useState("");
  const [mode, setMode] = useState<TrackerMode>("webcam");
  const [samples, setSamples] = useState<{ file: string; label: string }[]>([]);
  const [sampleLoaded, setSampleLoaded] = useState(false);

  if (!sampleLoaded && import.meta.env.DEV) {
    setSampleLoaded(true);
    fetch("/samples/index.json").then((r) => (r.ok ? r.json() : [])).then(setSamples).catch(() => {});
  }

  async function add(files: File[]) {
    setError(null);
    for (const f of files) {
      setBusy(`Analysing ${f.name}…`);
      try {
        const a = await analyze(f, f.name);
        if (!a.preview_jpg) throw new Error("The server did not return a preview. Restart the backend.");
        setImages((cur) => (cur.some((i) => i.id === a.id) ? cur : [...cur, { id: a.id, name: f.name, url: `data:image/jpeg;base64,${a.preview_jpg}`, analysis: a }]));
      } catch (e) {
        setError(`${f.name}: ${e instanceof Error ? e.message : "could not be analysed"}`);
      }
    }
    setBusy(null);
  }

  async function addSample(file: string) {
    const blob = await (await fetch(`/samples/${file}`)).blob();
    await add([new File([blob], file, { type: blob.type || "image/jpeg" })]);
  }

  const needsOne = tab === "demo";
  const shown = needsOne ? images.slice(0, 1) : images;

  return (
    <div className="space-y-5">
      <TrackerBar tracker={tracker} mode={mode} setMode={setMode} showMode={tab !== "click"} />

      <nav aria-label="Eye lab mode" className="flex flex-wrap gap-2">
        {TABS.map(([id, label, hint]) => (
          <button key={id} type="button" aria-pressed={tab === id} title={hint} onClick={() => setTab(id)}
            className={`rounded-lg border px-3 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-amber-500 ${tab === id ? "border-amber-500 bg-amber-50 dark:bg-amber-950/40" : "border-stone-300 hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"}`}>
            {label}
          </button>
        ))}
      </nav>
      <p className="text-sm text-stone-500 dark:text-stone-400">{TABS.find((t) => t[0] === tab)![2]}</p>

      {tab !== "live" && (
        <section className="space-y-3 rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
          <div className="flex flex-wrap items-center gap-3">
            <button type="button" disabled={!!busy} onClick={() => input.current?.click()} className="rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-700 disabled:opacity-50 dark:bg-amber-400 dark:text-stone-950">Add thumbnails</button>
            <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden onChange={(e) => { void add(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
            {samples.map((s) => (
              <button key={s.file} type="button" disabled={!!busy} onClick={() => void addSample(s.file)} className="rounded-md border border-stone-300 px-2 py-1 text-sm hover:bg-stone-100 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800">{s.label}</button>
            ))}
            {busy && <span className="text-sm text-stone-500" role="status">{busy}</span>}
          </div>
          {error && <div role="alert" className="text-sm text-red-700 dark:text-red-300">{error}</div>}
          {images.length > 0 && (
            <ul className="flex flex-wrap gap-2">
              {images.map((im, i) => (
                <li key={im.id} className={`relative ${needsOne && i > 0 ? "opacity-40" : ""}`} title={needsOne && i > 0 ? "The live demo uses the first thumbnail" : im.name}>
                  <img src={im.url} alt={im.name} className="h-16 rounded border border-stone-300 dark:border-stone-700" />
                  <button type="button" aria-label={`Remove ${im.name}`} onClick={() => setImages(images.filter((x) => x.id !== im.id))} className="absolute -right-1 -top-1 grid h-5 w-5 place-items-center rounded-full bg-black text-xs text-white">✕</button>
                </li>
              ))}
            </ul>
          )}
          <label className="flex items-center gap-2 text-sm">Participant (optional, anonymous)
            <input value={participant} maxLength={40} onChange={(e) => setParticipant(e.target.value)} className="w-32 rounded border border-stone-300 bg-transparent px-2 py-1 dark:border-stone-700" placeholder="e.g. p1" />
          </label>
        </section>
      )}

      {tab === "live" && <LiveRun tracker={tracker} mode={mode} />}
      {tab === "click" && <ClickLab images={images} participant={participant} />}
      {tab === "gaze" && <GazeLab key="gaze" images={shown} participant={participant} mode={mode} variant="test" tracker={tracker} />}
      {tab === "demo" && <GazeLab key="demo" images={shown} participant={participant} mode={mode} variant="demo" tracker={tracker} />}
    </div>
  );
}
