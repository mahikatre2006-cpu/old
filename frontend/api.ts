import type { AnalysisResult, CompareResult } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export async function analyze(file: Blob, filename = "thumbnail.png"): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("file", file, filename);
  let res: Response;
  try {
    res = await fetch(`${BASE}/analyze`, { method: "POST", body });
  } catch {
    throw new Error("Cannot reach the analysis server. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `Analysis failed (HTTP ${res.status})`);
  }
  return res.json();
}

export async function compare(files: File[]): Promise<CompareResult> {
  const body = new FormData();
  files.forEach((f) => body.append("files", f, f.name));
  let res: Response;
  try {
    res = await fetch(`${BASE}/compare`, { method: "POST", body });
  } catch {
    throw new Error("Cannot reach the analysis server. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `Comparison failed (HTTP ${res.status})`);
  }
  return res.json();
}
