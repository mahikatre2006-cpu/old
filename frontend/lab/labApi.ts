import type { AnalysisResult } from "../types";
import type { FramePoint } from "./gaze";

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export interface Fixation { x: number; y: number; order: number; start_ms?: number | null; duration_ms?: number | null }

export interface GazeSessionResult {
  session_id: string;
  method: "click" | "webgazer";
  cc: number;
  nss: number;
  sim: number;
  kld: number;
  n_points: number;
  n_fixations: number;
  fixations: Fixation[];
  first_look_hit: boolean | null;
  first_look_distance_px: number | null;
  first_look_hit_top3: boolean | null;
  warnings: string[];
  ground_truth_png: string | null;
}

export interface GazeAggregate {
  thumbnail_id: string;
  n_sessions: number;
  mean_cc: number | null;
  mean_nss: number | null;
  first_look_hit_rate: number | null;
  aggregate_cc: number | null;
  aggregate_nss: number | null;
  aggregate_png: string | null;
}

/** A thumbnail loaded into the lab: analysed once, so we know the prediction to compare real viewers against. */
export interface LabImage {
  id: string;
  name: string;
  url: string; // the exact 16:9 frame the model saw (data URL)
  analysis: AnalysisResult;
}

export async function postGazeSession(body: {
  thumbnail_id: string;
  method: "click" | "webgazer";
  points: FramePoint[];
  screen: { width: number; height: number };
  participant?: string;
}): Promise<GazeSessionResult> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/gaze-session`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new Error("Cannot reach the analysis server. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    const j = await res.json().catch(() => null);
    const d = j?.detail;
    throw new Error(typeof d === "string" ? d : `Could not score the session (HTTP ${res.status}).`);
  }
  return res.json();
}

export async function getGazeAggregate(thumbnailId: string): Promise<GazeAggregate> {
  const res = await fetch(`${BASE}/gaze-sessions/${thumbnailId}`);
  if (!res.ok) throw new Error(`Could not load saved sessions (HTTP ${res.status}).`);
  return res.json();
}
