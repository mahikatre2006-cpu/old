"""Saliency evaluation metrics (Bylinskii et al., "What do different evaluation metrics tell us about saliency models?").

All maps are 2D float arrays of the same shape. `pred` is the model's attention map, `gt` the human map built
from real gaze, and fixation points are (x, y) in the same pixel grid as the maps.
AUC-Judd is added in M12 with the rest of the offline evaluation script.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _as_dist(m: np.ndarray) -> np.ndarray:
    m = np.clip(np.asarray(m, np.float64), 0, None)
    s = m.sum()
    return m / s if s > 0 else np.full_like(m, 1.0 / m.size)


def cc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pearson linear correlation between the two maps, -1..1 (higher is better). 0 if either map is flat."""
    a, b = np.asarray(pred, np.float64).ravel(), np.asarray(gt, np.float64).ravel()
    if a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def nss(pred: np.ndarray, points: list[tuple[float, float]], weights: list[float] | None = None) -> float:
    """Normalized Scanpath Saliency: mean of the z-scored prediction at the real fixation points (higher is better;
    0 means no better than chance, ~1+ is solid). Points are (x, y) in map pixels and are clamped to the map."""
    p = np.asarray(pred, np.float64)
    if not points or p.std() < EPS:
        return 0.0
    z = (p - p.mean()) / p.std()
    H, W = z.shape
    vals = [z[min(H - 1, max(0, int(round(y)))), min(W - 1, max(0, int(round(x))))] for x, y in points]
    return float(np.average(vals, weights=weights))


def sim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Histogram intersection of the two maps treated as distributions, 0..1 (higher is better)."""
    return float(np.minimum(_as_dist(pred), _as_dist(gt)).sum())


def kld(pred: np.ndarray, gt: np.ndarray) -> float:
    """KL divergence of the prediction from the human map (lower is better, 0 is identical)."""
    p, g = _as_dist(pred), _as_dist(gt)
    return float((g * np.log(EPS + g / (p + EPS))).sum())
