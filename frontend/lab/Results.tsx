import HeatmapViewer from "../components/HeatmapViewer";
import type { GazeSessionResult, LabImage } from "./labApi";

export interface LabRow { image: LabImage; session: GazeSessionResult }

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg border border-stone-200 px-3 py-2 dark:border-stone-800" title={hint}>
      <div className="text-xs text-stone-500 dark:text-stone-400">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

const verdictCC = (cc: number) => (cc >= 0.5 ? "strong match" : cc >= 0.3 ? "moderate match" : cc >= 0.1 ? "weak match" : "no clear match");

/** Prediction on the left, what real viewers did on the right. */
export function SideBySide({ row, replayKey = 0 }: { row: LabRow; replayKey?: number }) {
  const { image, session } = row;
  const predicted = image.analysis;
  const real = {
    ...predicted,
    heatmap_png: session.ground_truth_png,
    scanpath: session.fixations.map((f) => ({ x: Math.round(f.x), y: Math.round(f.y), order: f.order })),
    regions: [],
  };
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <figure className="space-y-2">
          <HeatmapViewer imageUrl={image.url} result={{ ...predicted, regions: [] }} opacity={0.75} showHeatmap showPath showBoxes={false} replayKey={replayKey} />
          <figcaption className="text-sm"><b>Predicted</b>: attention map and gaze order from the model</figcaption>
        </figure>
        <figure className="space-y-2">
          <HeatmapViewer imageUrl={image.url} result={real} opacity={0.75} showHeatmap showPath showBoxes={false} replayKey={replayKey} />
          <figcaption className="text-sm">
            <b>Real viewer{session.method === "webgazer" ? " (webcam gaze)" : " (clicks)"}</b>: {session.n_fixations} {session.method === "webgazer" ? "fixation" : "look"}
            {session.n_fixations === 1 ? "" : "s"}, numbered in order
          </figcaption>
        </figure>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="CC (map similarity)" value={session.cc.toFixed(2)} hint="Linear correlation between the predicted and the real attention map. -1 to 1, higher is better." />
        <Metric label="NSS (hit strength)" value={session.nss.toFixed(2)} hint="How strongly the prediction lights up where the viewer actually looked. 0 is chance, above 1 is solid." />
        <Metric label="First look" value={session.first_look_hit === null ? "n/a" : session.first_look_hit ? "Predicted ✓" : session.first_look_hit_top3 ? "In our top 3" : "Missed ✗"}
          hint={`Real first look vs our predicted first fixation${session.first_look_distance_px != null ? ` (${Math.round(session.first_look_distance_px)} px apart)` : ""}`} />
        <Metric label="Verdict" value={verdictCC(session.cc)} hint="Plain-language reading of CC" />
      </div>
      <p className="text-xs text-stone-500 dark:text-stone-400">
        A single viewer is noisy{session.method === "webgazer" ? ", and webcam gaze is typically 100–200 px off" : ", and clicks are a memory of where the eye went"}. Treat one session as a
        sanity check; the numbers firm up when many viewers are pooled.
      </p>
      {session.warnings.map((w) => <p key={w} className="rounded bg-amber-100 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-200">{w}</p>)}
    </div>
  );
}

/** Summary over a whole pilot run: how often did the real first look land where we predicted? */
export function SessionTable({ rows }: { rows: LabRow[] }) {
  const hits = rows.filter((r) => r.session.first_look_hit);
  const top3 = rows.filter((r) => r.session.first_look_hit_top3);
  const mean = (f: (r: LabRow) => number) => (rows.length ? rows.reduce((a, r) => a + f(r), 0) / rows.length : 0);
  return (
    <section className="space-y-3 rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
      <h2 className="text-lg font-semibold">
        First look predicted: {hits.length} of {rows.length}
        <span className="ml-3 text-sm font-normal text-stone-500 dark:text-stone-400">(within our top 3: {top3.length} of {rows.length})</span>
      </h2>
      <p className="text-sm text-stone-500 dark:text-stone-400">Mean CC {mean((r) => r.session.cc).toFixed(2)} · mean NSS {mean((r) => r.session.nss).toFixed(2)}</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-stone-200 text-left text-xs uppercase tracking-wide text-stone-500 dark:border-stone-800 dark:text-stone-400">
              <th className="py-2 pr-3">Thumbnail</th><th className="pr-3">First look</th><th className="pr-3">CC</th><th className="pr-3">NSS</th><th>Looks</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.session.session_id} className="border-b border-stone-100 last:border-0 dark:border-stone-800">
                <td className="py-2 pr-3"><img src={r.image.url} alt="" className="inline h-8 rounded align-middle" /> <span className="ml-2">{r.image.name}</span></td>
                <td className="pr-3">{r.session.first_look_hit ? "✓ predicted" : r.session.first_look_hit_top3 ? "in top 3" : "✗ missed"}</td>
                <td className="pr-3 tabular-nums">{r.session.cc.toFixed(2)}</td>
                <td className="pr-3 tabular-nums">{r.session.nss.toFixed(2)}</td>
                <td className="tabular-nums">{r.session.n_fixations}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
