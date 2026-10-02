"""M4e. Composition analyzer (stage 2). Reads the fused attention map and measures how it is arranged:
focal points, clutter (entropy), rule of thirds, centre of mass, negative space, UI overlap.
"""
from __future__ import annotations

import cv2
import numpy as np

from analyzers.base import Analyzer
from pipeline.context import AnalyzerResult, Context

THIRDS = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]


def find_peaks(attention: np.ndarray, nms_frac: float = 0.08, min_ratio: float = 0.35, max_peaks: int = 8, work_w: int = 320) -> list[dict]:
    """Local maxima with non-maximum suppression. Strength is relative to the map's global maximum."""
    H, W = attention.shape
    scale = work_w / W
    small = cv2.resize(attention.astype(np.float32), (work_w, max(1, round(H * scale))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    top = float(small.max())
    if top <= 1e-6:
        return []
    k = max(3, int(nms_frac * work_w) | 1)
    is_max = (small == cv2.dilate(small, np.ones((k, k), np.uint8))) & (small >= min_ratio * top)
    ys, xs = np.where(is_max)
    cand = sorted(zip(small[ys, xs], xs, ys), key=lambda t: -t[0])
    peaks: list[tuple[float, int, int]] = []
    for v, x, y in cand:  # plateaus produce neighbours with equal value; keep the first within the NMS radius
        if all(np.hypot(x - px, y - py) > k / 2 for _, px, py in peaks):
            peaks.append((float(v), int(x), int(y)))
        if len(peaks) >= max_peaks:
            break
    return [{"x": int((x + 0.5) / scale), "y": int((y + 0.5) / scale), "strength": round(v / top, 3)} for v, x, y in peaks]


def thirds_distance(x: float, y: float) -> float:
    """Distance (normalized frame units, so the diagonal is ~1.4) to the nearest rule-of-thirds intersection."""
    return float(min(np.hypot(x - tx, y - ty) for tx, ty in THIRDS))


def normalized_entropy(attention: np.ndarray, work_w: int = 160) -> float:
    """Shannon entropy of the attention distribution divided by its maximum: 0 = one hot spot, 1 = uniform."""
    h = max(1, round(attention.shape[0] * work_w / attention.shape[1]))
    small = cv2.resize(attention.astype(np.float32), (work_w, h), interpolation=cv2.INTER_AREA)
    s = float(small.sum())
    if s <= 0:
        return 1.0
    p = small.ravel() / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(small.size))


class CompositionAnalyzer(Analyzer):
    name = "composition"
    metrics_group = "composition"

    def __init__(self, focal_strength: float = 0.5, negative_attention: float = 0.15, negative_texture: float = 10.0):
        self.focal_strength = focal_strength
        self.negative_attention = negative_attention
        self.negative_texture = negative_texture

    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult:
        fused = ctx.fused
        if fused is None or float(fused.sum()) <= 0:
            return AnalyzerResult(metrics={"focal_points": 0.0, "entropy": 1.0})
        H, W = fused.shape
        total = float(fused.sum())

        peaks = find_peaks(fused)
        focal = [p for p in peaks if p["strength"] >= self.focal_strength]

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = float((fused * xx).sum() / total) / W, float((fused * yy).sum() / total) / H
        var = float((fused * ((xx / W - cx) ** 2 + (yy / H - cy) ** 2)).sum() / total)

        # negative space: calm areas that attract no attention (low attention AND low local texture)
        gray = cv2.cvtColor(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.blur(gray, (9, 9))
        local_std = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mu * mu, 0))
        att_small = cv2.resize(fused, (320, 180), interpolation=cv2.INTER_AREA)
        negative = (att_small < self.negative_attention) & (local_std < self.negative_texture)

        flat = np.sort(fused.ravel())[::-1]
        top10 = float(flat[: max(1, flat.size // 10)].sum() / total)

        metrics = {
            "focal_points": float(len(focal)),
            "peak_count": float(len(peaks)),
            "top_peak_ratio": round(peaks[1]["strength"] / peaks[0]["strength"], 3) if len(peaks) > 1 else 0.0,
            "entropy": round(normalized_entropy(fused), 3),
            "concentration_top10": round(top10, 3),
            "com_x": round(cx, 3),
            "com_y": round(cy, 3),
            "com_offset_center": round(float(np.hypot(cx - 0.5, cy - 0.5)), 3),
            "spread": round(float(np.sqrt(var)), 3),
            "rot_dist_com": round(thirds_distance(cx, cy), 3),
            "negative_space_pct": round(100 * float(negative.mean()), 1),
            "ui_overlap_pct": round(100 * float((fused * ctx.ui_mask).sum() / total), 2),
        }
        if peaks:
            metrics["rot_dist_peak"] = round(thirds_distance(peaks[0]["x"] / W, peaks[0]["y"] / H), 3)
            metrics["peaks"] = peaks
        return AnalyzerResult(metrics=metrics)
