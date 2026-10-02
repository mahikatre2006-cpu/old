import { useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "../types";

interface Props {
  imageUrl: string;
  result: AnalysisResult;
  opacity: number; // 0..1
  showHeatmap: boolean;
  showPath: boolean;
  showBoxes: boolean;
  replayKey: number; // bump to restart the gaze animation
  highlight?: [number, number, number, number] | null; // element a hovered fix is about
}

const STEP_MS = 650;
const FRAME_W = 1280;
const BOX_COLOR: Record<string, string> = { face: "#22d3ee", text: "#facc15", subject: "#a78bfa" };
const WARN_COLOR = "#fb7185"; // text that will not survive at sidebar size

export default function HeatmapViewer({ imageUrl, result, opacity, showHeatmap, showPath, showBoxes, replayKey, highlight }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(FRAME_W);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The SVG lives in 1280x720 frame units, so on a small display everything shrinks with it.
  // Scale markers up so they never fall below a readable on-screen size.
  const k = Math.max(1, 640 / Math.max(width, 1));

  const path = [...result.scanpath].sort((a, b) => a.order - b.order);
  const boxes = result.regions.filter((r) => r.box && r.type !== "background" && r.attention_pct >= 1); // skip noise-level boxes

  return (
    <div ref={ref} className="relative aspect-video w-full overflow-hidden rounded-xl bg-stone-900 shadow-lg ring-1 ring-black/10">
      <img src={imageUrl} alt="Analysed thumbnail" className="absolute inset-0 h-full w-full object-fill" />

      {showHeatmap && result.heatmap_png && (
        <img
          src={`data:image/png;base64,${result.heatmap_png}`}
          alt="Attention heatmap"
          className="absolute inset-0 h-full w-full object-fill"
          style={{ opacity }}
        />
      )}

      <svg viewBox={`0 0 ${FRAME_W} 720`} className="absolute inset-0 h-full w-full" role="img" aria-label="Predicted gaze order">
        {showBoxes &&
          boxes.map((r, i) => {
            const [x0, y0, x1, y1] = r.box!;
            const tooSmall = r.type === "text" && r.mobile_readable === false;
            const color = tooSmall ? WARN_COLOR : BOX_COLOR[r.type] ?? "#fff";
            const name = r.type === "face" ? r.emotion ?? "face" : r.type === "text" && r.text ? `"${r.text.slice(0, 14)}"` : r.type;
            const label = `${name} ${r.attention_pct.toFixed(0)}%${tooSmall ? " ⚠" : ""}`;
            return (
              <g key={i}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={color} strokeWidth={4 * k} strokeDasharray={`${14 * k} ${8 * k}`} rx={6} />
                <text x={x0 + 8} y={Math.max(26 * k, y0 - 8)} fontSize={26 * k} fontWeight={700} fill={color} stroke="#000" strokeWidth={4 * k} paintOrder="stroke">
                  {label}
                </text>
              </g>
            );
          })}

        {highlight && (
          <g pointerEvents="none">
            <rect x={highlight[0] - 6} y={highlight[1] - 6} width={highlight[2] - highlight[0] + 12} height={highlight[3] - highlight[1] + 12}
              fill="rgba(255,255,255,0.12)" stroke="#000" strokeWidth={9 * k} rx={10} />
            <rect x={highlight[0] - 6} y={highlight[1] - 6} width={highlight[2] - highlight[0] + 12} height={highlight[3] - highlight[1] + 12}
              fill="none" stroke="#fff" strokeWidth={5 * k} rx={10} />
          </g>
        )}

        {showPath && path.length > 0 && (
          <g key={replayKey}>
            {path.slice(1).map((p, i) => {
              const prev = path[i];
              return (
                <line
                  key={`s${p.order}`}
                  x1={prev.x} y1={prev.y} x2={p.x} y2={p.y}
                  pathLength={1}
                  className="gaze-seg"
                  stroke="#fff" strokeWidth={5 * k} strokeLinecap="round"
                  style={{ ["--d" as string]: `${i * STEP_MS + 300}ms`, filter: "drop-shadow(0 0 3px rgba(0,0,0,.9))" }}
                />
              );
            })}
            {path.map((p, i) => (
              <g key={`d${p.order}`} className="gaze-dot" style={{ ["--d" as string]: `${i * STEP_MS}ms` }}>
                <circle cx={p.x} cy={p.y} r={28 * k} fill="#0c0a09" stroke="#fff" strokeWidth={4 * k} />
                <text x={p.x} y={p.y + 10 * k} textAnchor="middle" fontSize={30 * k} fontWeight={800} fill="#fff">
                  {p.order}
                </text>
              </g>
            ))}
          </g>
        )}
      </svg>
    </div>
  );
}
