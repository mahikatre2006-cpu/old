import type { AnalysisResult, Recommendation, Scores } from "../types";

const SHARE_ORDER = ["face", "text", "subject", "background"] as const;
const SHARE_COLOR: Record<string, string> = { face: "bg-cyan-500", text: "bg-yellow-400", subject: "bg-violet-500", background: "bg-stone-400" };

export function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">{title}</h2>
      {children}
    </section>
  );
}

export function AttentionShare({ shares }: { shares: Record<string, unknown> | undefined }) {
  if (!shares) return <p className="text-sm text-stone-500">Not available.</p>;
  return (
    <ul className="space-y-2.5">
      {SHARE_ORDER.map((k) => {
        const v = Number(shares[k] ?? 0);
        return (
          <li key={k}>
            <div className="mb-1 flex justify-between text-sm">
              <span className="capitalize">{k}</span>
              <span className="tabular-nums font-medium">{v.toFixed(1)}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-stone-200 dark:bg-stone-800">
              <div className={`h-full rounded-full ${SHARE_COLOR[k]}`} style={{ width: `${Math.min(100, v)}%` }} />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

const SCORE_LABELS: [keyof Scores, string][] = [
  ["overall", "Overall"], ["focus", "Focus"], ["text_readability", "Text readability"], ["face_impact", "Face impact"], ["clutter", "Cleanliness"],
];

interface Component { id: string; label: string; value: number | null; score: number | null; weight: number }
interface ScoreDetail { note?: string; applicable?: boolean; components?: Component[]; penalties?: { id: string; label: string; points: number }[] }

function Bar({ pct, tone }: { pct: number; tone: string }) {
  return (
    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-stone-200 dark:bg-stone-800">
      <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

const tone = (v: number) => (v >= 70 ? "bg-emerald-500" : v >= 45 ? "bg-amber-500" : "bg-rose-500");

export function ScoreGrid({ scores, detail }: { scores: Scores | null; detail?: Record<string, unknown> }) {
  if (!scores) return <p className="text-sm text-stone-500 dark:text-stone-400">Scores are not available.</p>;
  const det = (k: string) => (detail?.[k] ?? {}) as ScoreDetail;
  return (
    <ul className="space-y-3">
      {SCORE_LABELS.map(([k, label]) => {
        const v = scores[k];
        const d = det(k);
        const big = k === "overall";
        return (
          <li key={k}>
            <details className="group">
              <summary className="flex cursor-pointer list-none items-center gap-3 rounded focus-visible:outline-2 focus-visible:outline-amber-500">
                <span className={`${big ? "w-28 font-semibold" : "w-28 text-sm"}`}>{label}</span>
                {v === null ? <span className="text-sm text-stone-500 dark:text-stone-400">n/a · {d.note || "not applicable"}</span> : (
                  <>
                    <Bar pct={v} tone={tone(v)} />
                    <span className={`${big ? "text-xl font-semibold" : "text-sm font-medium"} w-9 text-right tabular-nums`}>{v}</span>
                  </>
                )}
              </summary>
              {v !== null && (d.components?.length || d.penalties?.length) ? (
                <ul className="mt-2 space-y-1.5 border-l-2 border-stone-200 pl-3 text-xs text-stone-600 dark:border-stone-700 dark:text-stone-400">
                  {d.components!.map((c) => (
                    <li key={c.id} className="flex items-center gap-2">
                      <span className="w-40 shrink-0 truncate" title={c.label}>{c.label}</span>
                      {c.score === null ? <span className="italic">no data</span> : <><Bar pct={c.score * 100} tone={tone(c.score * 100)} /><span className="w-8 text-right tabular-nums">{Math.round(c.score * 100)}</span></>}
                    </li>
                  ))}
                  {d.penalties?.map((p) => (
                    <li key={p.id} className="text-rose-600 dark:text-rose-400">− {Math.round(p.points)} points: {p.label}</li>
                  ))}
                </ul>
              ) : null}
            </details>
          </li>
        );
      })}
    </ul>
  );
}

const SEVERITY: Record<Recommendation["severity"], string> = {
  high: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  medium: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  low: "bg-stone-200 text-stone-700 dark:bg-stone-800 dark:text-stone-300",
};

export function Recommendations({ items, onHover }: { items: Recommendation[]; onHover?: (box: [number, number, number, number] | null) => void }) {
  if (!items.length) return <p className="text-sm text-stone-500 dark:text-stone-400">No fixes yet.</p>;
  return (
    <ol className="space-y-3">
      {items.map((r, i) => (
        <li
          key={r.id}
          tabIndex={r.box ? 0 : undefined}
          onMouseEnter={() => onHover?.(r.box ?? null)}
          onMouseLeave={() => onHover?.(null)}
          onFocus={() => onHover?.(r.box ?? null)}
          onBlur={() => onHover?.(null)}
          title={r.box ? "Hover to see the element this is about" : undefined}
          className={`flex gap-3 rounded text-sm outline-offset-4 focus-visible:outline-2 focus-visible:outline-amber-500 ${r.box ? "cursor-help" : ""}`}
        >
          <span className="mt-0.5 font-semibold tabular-nums">{i + 1}.</span>
          <div>
            <span className={`mr-2 rounded px-1.5 py-0.5 text-xs font-medium ${SEVERITY[r.severity]}`}>{r.severity}</span>
            {r.message}
          </div>
        </li>
      ))}
    </ol>
  );
}

const n = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function Details({ metrics }: { metrics: AnalysisResult["metrics"] }) {
  const face = metrics.face ?? {}, text = metrics.text ?? {}, subject = metrics.subject ?? {}, color = metrics.color ?? {}, comp = metrics.composition ?? {};
  const palette = Array.isArray(color.palette_hex) ? (color.palette_hex as string[]) : [];
  const share = Array.isArray(color.palette_share) ? (color.palette_share as number[]) : [];
  const rows: [string, string | null][] = [
    ["Faces", n(face.count) !== null ? String(n(face.count)) : null],
    ["Text lines", n(text.count) !== null ? `${n(text.count)} (${n(text.unreadable_mobile_count) ?? 0} unreadable on mobile)` : null],
    ["Lowest text contrast", n(text.min_contrast) !== null ? `${n(text.min_contrast)}:1` : null],
    ["Subject size", n(subject.area_pct) !== null ? `${n(subject.area_pct)}% of frame` : null],
    ["Focal points", n(comp.focal_points) !== null ? String(n(comp.focal_points)) : null],
    ["Negative space", n(comp.negative_space_pct) !== null ? `${Math.round(n(comp.negative_space_pct) as number)}% of frame` : null],
    ["Under duration badge", n(comp.ui_overlap_pct) !== null ? `${n(comp.ui_overlap_pct)}% of attention` : null],
    ["Colourfulness", n(color.colorfulness_score) !== null ? `${n(color.colorfulness_score)} / 100` : null],
  ];
  return (
    <div className="space-y-3">
      {palette.length > 0 && (
        <div className="flex h-6 overflow-hidden rounded-md ring-1 ring-black/10" role="img" aria-label={`Colour palette: ${palette.join(", ")}`}>
          {palette.map((hex, i) => (
            <div key={hex + i} title={`${hex} · ${Math.round((share[i] ?? 0) * 100)}%`} style={{ background: hex, flexGrow: Math.max(share[i] ?? 0.05, 0.05) * 100 }} />
          ))}
        </div>
      )}
      <dl className="space-y-1.5 text-sm">
        {rows.filter(([, v]) => v !== null).map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-stone-500 dark:text-stone-400">{k}</dt>
            <dd className="text-right font-medium tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function Footer({ result }: { result: AnalysisResult }) {
  const t = result.timing_ms;
  return (
    <div className="space-y-2 text-xs text-stone-500 dark:text-stone-400">
      <p>
        {result.cached ? "Served from cache" : "Analysed"} in {Math.round(t.total ?? 0)} ms
        {!result.cached && t.saliency ? ` (saliency ${Math.round(t.saliency)} ms)` : ""}.
      </p>
      {result.unavailable.length > 0 && (
        <p className="rounded bg-amber-100 px-2 py-1 text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          Unavailable: {result.unavailable.join(", ")}. The rest of the report is still valid.
        </p>
      )}
    </div>
  );
}
