import { useEffect, useState } from "react";
import { analyze } from "./api";
import HeatmapViewer from "./components/HeatmapViewer";
import { AttentionShare, Card, Details, Footer, Recommendations, ScoreGrid } from "./components/Panels";
import CompareView from "./components/CompareView";
import EyeLab from "./lab/EyeLab";
import Uploader from "./components/Uploader";
import type { AnalysisResult } from "./types";

export default function App() {
  const [mode, setMode] = useState<"analyze" | "compare" | "lab">("analyze");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [localUrl, setLocalUrl] = useState<string | null>(null);

  const [opacity, setOpacity] = useState(0.7);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showPath, setShowPath] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [replayKey, setReplayKey] = useState(0);
  const [highlight, setHighlight] = useState<[number, number, number, number] | null>(null);

  useEffect(() => () => { if (localUrl) URL.revokeObjectURL(localUrl); }, [localUrl]);

  async function run(file: File) {
    setBusy(true);
    setError(null);
    try {
      const res = await analyze(file, file.name);
      setLocalUrl(URL.createObjectURL(file));
      setResult(res);
      setHighlight(null);
      setReplayKey((k) => k + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  // Prefer the backend's preview: it is the exact frame the model analysed, so overlays line up
  // even when the upload was not 16:9.
  const imageUrl = result?.preview_jpg ? `data:image/jpeg;base64,${result.preview_jpg}` : localUrl;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Thumbnail attention heatmap</h1>
          <p className="text-sm text-stone-500 dark:text-stone-400">Where viewers look, in what order, and what to fix. Runs locally.</p>
        </div>
        <nav aria-label="Mode" className="flex rounded-lg border border-stone-300 p-0.5 text-sm dark:border-stone-700">
          {(["analyze", "compare", "lab"] as const).map((m) => (
            <button key={m} type="button" aria-pressed={mode === m} onClick={() => { setMode(m); setError(null); }}
              className={`rounded-md px-3 py-1.5 font-medium focus-visible:outline-2 focus-visible:outline-amber-500 ${mode === m ? "bg-stone-900 text-white dark:bg-amber-400 dark:text-stone-950" : "text-stone-600 hover:bg-stone-100 dark:text-stone-300 dark:hover:bg-stone-800"}`}>
              {m === "analyze" ? "Analyze" : m === "compare" ? "A/B compare" : "Eye lab"}
            </button>
          ))}
        </nav>
      </header>

      {mode === "compare" ? <CompareView /> : mode === "lab" ? <EyeLab /> : <>

      {error && (
        <div role="alert" className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {!result ? (
        <Uploader busy={busy} onFile={run} onError={setError} />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="space-y-4">
            {imageUrl && (
              <HeatmapViewer imageUrl={imageUrl} result={result} opacity={opacity} showHeatmap={showHeatmap} showPath={showPath} showBoxes={showBoxes} replayKey={replayKey} highlight={highlight} />
            )}

            {result.regions.some((r) => r.type === "text" && r.mobile_readable === false) && (
              <p className="text-sm text-rose-600 dark:text-rose-300">
                ⚠ Red text boxes are too small to read at sidebar size (168 px wide).
              </p>
            )}

            <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-stone-200 bg-white px-4 py-3 text-sm dark:border-stone-800 dark:bg-stone-900">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showHeatmap} onChange={(e) => setShowHeatmap(e.target.checked)} /> Heatmap
              </label>
              <label className="flex items-center gap-2">
                Opacity
                <input type="range" min={0} max={100} value={Math.round(opacity * 100)} disabled={!showHeatmap} onChange={(e) => setOpacity(Number(e.target.value) / 100)} aria-label="Heatmap opacity" className="w-32 accent-amber-500" />
                <span className="w-9 tabular-nums text-stone-500">{Math.round(opacity * 100)}%</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} /> Gaze order
              </label>
              <button type="button" disabled={!showPath} onClick={() => setReplayKey((k) => k + 1)} className="rounded-md border border-stone-300 px-2 py-1 hover:bg-stone-100 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800">
                Replay
              </button>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> Regions
              </label>
              <div className="ml-auto flex items-center gap-2 text-xs text-stone-500 dark:text-stone-400" aria-label="Heatmap colour scale">
                Low
                <span className="h-2.5 w-28 rounded-full" style={{ background: "linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)" }} />
                High attention
              </div>
            </div>

            <Uploader busy={busy} onFile={run} onError={setError} compact />
          </div>

          <aside className="space-y-4">
            <Card title="Attention share"><AttentionShare shares={result.metrics.attention_share} /></Card>
            <Card title="Scores"><ScoreGrid scores={result.scores} detail={result.metrics.scores_detail} /></Card>
            <Card title="Fixes"><Recommendations items={result.recommendations} onHover={setHighlight} /></Card>
            <Card title="Details"><Details metrics={result.metrics} /></Card>
            <Footer result={result} />
          </aside>
        </div>
      )}
      </>}
    </div>
  );
}
