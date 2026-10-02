import { useEffect, useMemo, useRef, useState } from "react";
import { compare } from "../api";
import type { CompareResult, Scores } from "../types";
import HeatmapViewer from "./HeatmapViewer";

const MAX = 4;
const MIN = 2;
const OK_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ROWS: [keyof Scores, string][] = [
  ["overall", "Overall"], ["focus", "Focus"], ["text_readability", "Text readability"], ["face_impact", "Face impact"], ["clutter", "Cleanliness"],
];

function Delta({ v }: { v: number | null | undefined }) {
  if (v === null || v === undefined) return <span className="text-stone-400">–</span>;
  if (v === 0) return <span className="text-stone-400">±0</span>;
  // sign and arrow carry the meaning; colour only reinforces it
  return v > 0
    ? <span className="font-medium text-emerald-600 dark:text-emerald-400">▲ +{v}</span>
    : <span className="font-medium text-rose-600 dark:text-rose-400">▼ −{Math.abs(v)}</span>;
}

export default function CompareView() {
  const input = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [urls, setUrls] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [showHeat, setShowHeat] = useState(true);
  const [showDiff, setShowDiff] = useState(false);
  const [opacity, setOpacity] = useState(0.7);
  const [samples, setSamples] = useState<{ file: string; label: string }[]>([]);

  useEffect(() => {
    const u = files.map((f) => URL.createObjectURL(f));
    setUrls(u);
    return () => u.forEach(URL.revokeObjectURL);
  }, [files]);

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    fetch("/samples/index.json").then((r) => (r.ok ? r.json() : [])).then(setSamples).catch(() => {});
  }, []);

  function add(list: File[] | FileList | null | undefined) {
    if (!list) return;
    const next = [...files];
    for (const f of Array.from(list)) {
      if (!OK_TYPES.includes(f.type)) { setError("Please use JPG, PNG or WEBP images."); continue; }
      if (f.size > 10 * 1024 * 1024) { setError("An image is over 10 MB."); continue; }
      if (next.length >= MAX) { setError(`You can compare at most ${MAX} images.`); break; }
      next.push(f);
    }
    setFiles(next);
  }

  async function addSample(file: string) {
    const blob = await (await fetch(`/samples/${file}`)).blob();
    add([new File([blob], file, { type: blob.type || "image/jpeg" })]);
  }

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await compare(files));
      setShowDiff(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  const labels = result?.labels ?? [];
  const winnerIdx = useMemo(() => (result ? result.labels.indexOf(result.winner_label) : -1), [result]);

  if (result) {
    const n = result.variants.length;
    return (
      <div className="space-y-5">
        <section
          role="status"
          className={`rounded-xl border p-4 ${result.close ? "border-amber-300 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/40" : "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/40"}`}
        >
          <h2 className="text-lg font-semibold">
            {result.close ? `Too close to call: ${result.winner_label} edges ahead by ${result.margin} point${result.margin === 1 ? "" : "s"}` : `${result.winner_label} wins by ${result.margin} points`}
          </h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {result.reasons.slice(1).map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </section>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-stone-200 bg-white px-4 py-3 text-sm dark:border-stone-800 dark:bg-stone-900">
          <label className="flex items-center gap-2"><input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Heatmap</label>
          <label className="flex items-center gap-2">
            Opacity
            <input type="range" min={0} max={100} value={Math.round(opacity * 100)} onChange={(e) => setOpacity(Number(e.target.value) / 100)} aria-label="Heatmap opacity" className="w-28 accent-amber-500" />
          </label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} /> Difference vs A</label>
          {showDiff && (
            <span className="flex items-center gap-2 text-xs text-stone-500 dark:text-stone-400">
              <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "rgb(245,140,30)" }} /> draws more attention than A
              <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "rgb(30,110,200)" }} /> draws less
            </span>
          )}
          <button type="button" onClick={() => { setResult(null); setFiles([]); }} className="ml-auto rounded-md border border-stone-300 px-3 py-1 hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800">
            Compare different images
          </button>
        </div>

        <div className={`grid gap-4 ${n === 2 ? "sm:grid-cols-2" : n === 3 ? "sm:grid-cols-2 xl:grid-cols-3" : "sm:grid-cols-2 xl:grid-cols-4"}`}>
          {result.variants.map((v, i) => {
            const diff = result.difference_maps[i];
            const shown = showDiff && i > 0 && diff ? { ...v, heatmap_png: diff, regions: [], scanpath: [] } : { ...v, regions: [], scanpath: [] };
            const img = v.preview_jpg ? `data:image/jpeg;base64,${v.preview_jpg}` : urls[i];
            return (
              <figure key={v.id} className={`space-y-2 rounded-xl p-2 ${i === winnerIdx ? "ring-2 ring-emerald-500" : ""}`}>
                {img && <HeatmapViewer imageUrl={img} result={shown} opacity={opacity} showHeatmap={showHeat && !(showDiff && i === 0)} showPath={false} showBoxes={false} replayKey={0} />}
                <figcaption className="flex items-center justify-between px-1 text-sm">
                  <span className="font-semibold">
                    {labels[i]} {i === winnerIdx && <span className="ml-1 rounded bg-emerald-600 px-1.5 py-0.5 text-xs font-medium text-white">{result.close ? "Slight edge" : "Winner"}</span>}
                  </span>
                  <span className="tabular-nums">
                    <span className="text-lg font-semibold">{v.scores?.overall ?? "–"}</span> <Delta v={i === 0 ? null : result.score_deltas[i].overall} />
                  </span>
                </figcaption>
              </figure>
            );
          })}
        </div>

        <section className="overflow-x-auto rounded-xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
          <table className="w-full text-sm">
            <caption className="sr-only">Scores per variant, with the difference from variant A</caption>
            <thead>
              <tr className="border-b border-stone-200 text-left text-xs uppercase tracking-wide text-stone-500 dark:border-stone-800 dark:text-stone-400">
                <th className="px-4 py-2">Score</th>
                {labels.map((l) => <th key={l} className="px-4 py-2">{l}{l === result.winner_label ? " ★" : ""}</th>)}
              </tr>
            </thead>
            <tbody>
              {ROWS.map(([k, name]) => (
                <tr key={k} className="border-b border-stone-100 last:border-0 dark:border-stone-800">
                  <th scope="row" className="px-4 py-2 text-left font-normal">{name}</th>
                  {result.variants.map((v, i) => (
                    <td key={v.id} className="px-4 py-2 tabular-nums">
                      {v.scores?.[k] ?? <span className="text-stone-400">n/a</span>} {i > 0 && <span className="ml-1 text-xs"><Delta v={result.score_deltas[i][k]} /></span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {result.rules_differ.length > 0 && (
          <section className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">Fixes that differ between variants</h2>
            <ul className="space-y-2 text-sm">
              {result.rules_differ.map((d) => (
                <li key={d.id} className="flex gap-3">
                  <span className="w-20 shrink-0 font-semibold">Needed in {d.present_in.join(", ")}</span>
                  <span>{d.example}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); add(e.dataTransfer.files); }}
        className="rounded-2xl border-2 border-dashed border-stone-300 p-5 dark:border-stone-700"
      >
        <p className="font-medium">Add {MIN} to {MAX} thumbnail variants</p>
        <p className="text-sm text-stone-500 dark:text-stone-400">Drop them here or choose files. They are labelled A, B, C, D in the order you add them.</p>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button type="button" disabled={busy || files.length >= MAX} onClick={() => input.current?.click()}
            className="rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-500 disabled:opacity-50 dark:bg-amber-400 dark:text-stone-950 dark:hover:bg-amber-300">
            Choose files
          </button>
          <input ref={input} type="file" accept={OK_TYPES.join(",")} multiple hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
          {samples.length > 0 && (
            <span className="flex flex-wrap items-center gap-2 text-sm">
              <span className="text-stone-500 dark:text-stone-400">Dev samples:</span>
              {samples.map((s) => (
                <button key={s.file} type="button" disabled={busy || files.length >= MAX} onClick={() => addSample(s.file)}
                  className="rounded-md border border-stone-300 px-2 py-1 hover:bg-stone-100 disabled:opacity-50 dark:border-stone-700 dark:hover:bg-stone-800">{s.label}</button>
              ))}
            </span>
          )}
        </div>
      </div>

      {error && <div role="alert" className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">{error}</div>}

      {files.length > 0 && (
        <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {files.map((f, i) => (
            <li key={f.name + i} className="relative overflow-hidden rounded-xl border border-stone-200 dark:border-stone-800">
              {urls[i] && <img src={urls[i]} alt={`Variant ${String.fromCharCode(65 + i)}: ${f.name}`} className="aspect-video w-full object-cover" />}
              <span className="absolute left-2 top-2 rounded bg-black/70 px-2 py-0.5 text-sm font-bold text-white">{String.fromCharCode(65 + i)}</span>
              <button type="button" aria-label={`Remove variant ${String.fromCharCode(65 + i)}`} disabled={busy}
                onClick={() => setFiles(files.filter((_, j) => j !== i))}
                className="absolute right-2 top-2 rounded bg-black/70 px-2 py-0.5 text-sm text-white hover:bg-black focus-visible:outline-2 focus-visible:outline-amber-400">✕</button>
            </li>
          ))}
        </ul>
      )}

      <button type="button" disabled={busy || files.length < MIN} onClick={run}
        className="rounded-lg bg-amber-500 px-5 py-2.5 font-medium text-stone-950 hover:bg-amber-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-900 disabled:opacity-50">
        {busy ? `Analysing ${files.length} variants…` : files.length < MIN ? `Add ${MIN - files.length} more to compare` : `Compare ${files.length} variants`}
      </button>
      {busy && <p className="text-sm text-stone-500 dark:text-stone-400">Each variant takes a couple of seconds, so this can take up to ~10 s.</p>}
    </div>
  );
}
