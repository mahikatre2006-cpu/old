// Mirrors backend/api/schemas.py (the frozen /analyze contract).
export interface ScanpathPoint { x: number; y: number; order: number }

export interface Region {
  type: "face" | "text" | "subject" | "background";
  box: [number, number, number, number] | null;
  attention_pct: number;
  emotion?: string | null;
  intensity?: number | null;
  text?: string | null;
  mobile_readable?: boolean | null;
  contrast?: number | null;
  height_pct?: number | null;
}

export interface Scores { overall: number | null; focus: number | null; text_readability: number | null; face_impact: number | null; clutter: number | null }
export interface Recommendation { id: string; severity: "high" | "medium" | "low"; message: string; impact?: number | null; box?: [number, number, number, number] | null }

export interface AnalysisResult {
  id: string;
  heatmap_png: string | null;
  attention_png?: string | null; // 160x90 grayscale fused attention map
  preview_jpg?: string | null; // normalized 16:9 frame the model analysed; overlays align to this
  scanpath: ScanpathPoint[];
  regions: Region[];
  scores: Scores | null;
  recommendations: Recommendation[];
  metrics: Record<string, Record<string, unknown>>;
  timing_ms: Record<string, number>;
  unavailable: string[];
  cached: boolean;
}

export interface CompareResult {
  id: string;
  labels: string[];
  variants: AnalysisResult[];
  score_deltas: Record<string, number | null>[]; // each variant minus variant A
  winner_id: string;
  winner_label: string;
  margin: number;
  close: boolean; // margin too small to call confidently
  reasons: string[];
  rules_differ: { id: string; present_in: string[]; example: string }[];
  difference_maps: (string | null)[]; // base64 RGBA PNG vs variant A; [0] is null
}
