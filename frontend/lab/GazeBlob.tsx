import { useEffect, useRef } from "react";
import { OneEuro, type Point } from "./calibrationMath";
import type { GazeSource } from "./gaze";

const TRAIL = 30;

/**
 * A soft blob that follows the viewer's gaze across the whole screen, with a fading trail.
 * Drawn on a canvas at display rate; the raw ~30 Hz samples are smoothed (adaptive One Euro filter) and eased,
 * so it glides when the eye drifts and still jumps promptly when the eye jumps.
 */
export default function GazeBlob({ source, trail = true, radius = 36, label = true }: { source: GazeSource; trail?: boolean; radius?: number; label?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const fx = new OneEuro(1.2, 0.012), fy = new OneEuro(1.2, 0.012);
    let target: Point | null = null;
    let pos: Point | null = null;
    const points: Point[] = [];
    let raf = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const unsub = source.subscribe((p) => {
      const t = performance.now();
      target = { x: fx.filter(p.x, t), y: fy.filter(p.y, t) };
    });

    const draw = () => {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      const face = source.faceFound();
      if (target) {
        pos = pos ? { x: pos.x + (target.x - pos.x) * 0.4, y: pos.y + (target.y - pos.y) * 0.4 } : { ...target };
        if (face) { points.push({ ...pos }); if (points.length > TRAIL) points.shift(); }
      }
      if (pos) {
        ctx.globalAlpha = face ? 1 : 0.25;
        if (trail) {
          for (let i = 0; i < points.length; i++) {
            const k = (i + 1) / points.length;
            ctx.fillStyle = `rgba(251,191,36,${0.32 * k * k})`;
            ctx.beginPath();
            ctx.arc(points[i].x, points[i].y, radius * (0.25 + 0.45 * k), 0, Math.PI * 2);
            ctx.fill();
          }
        }
        const g = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius);
        g.addColorStop(0, "rgba(255,255,255,0.95)");
        g.addColorStop(0.25, "rgba(251,191,36,0.85)");
        g.addColorStop(1, "rgba(245,158,11,0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (label && !face) {
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        const w = 300, h = 34, x = window.innerWidth / 2 - w / 2, y = 18;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = "#fbbf24";
        ctx.font = "600 15px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Face not found: move into view", window.innerWidth / 2, y + 22);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      unsub();
      window.removeEventListener("resize", resize);
    };
  }, [source, trail, radius, label]);

  return <canvas ref={ref} aria-hidden="true" className="pointer-events-none fixed inset-0 z-[60] h-full w-full" />;
}
