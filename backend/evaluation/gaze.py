"""M11 analysis: raw gaze or clicks -> fixations -> human ground-truth map -> CC / NSS against the prediction."""
from __future__ import annotations

import math

import cv2
import numpy as np

from evaluation.metrics import cc, kld, nss, sim

FRAME_W, FRAME_H = 1280, 720


def idt(samples: list[tuple[float, float, float]], dispersion_px: float, min_duration_ms: float) -> list[dict]:
    """I-DT fixation detection. samples = [(t_ms, x, y)] sorted by time.
    A fixation is a run of samples lasting at least min_duration_ms whose dispersion
    (x-range + y-range) stays under dispersion_px. Returns [{x, y, start_ms, duration_ms}] in time order."""
    out: list[dict] = []
    n, i = len(samples), 0
    while i < n:
        j = i
        while j < n and samples[j][0] - samples[i][0] < min_duration_ms:
            j += 1
        if j >= n:
            break
        win = samples[i : j + 1]
        if _dispersion(win) <= dispersion_px:
            while j + 1 < n and _dispersion(samples[i : j + 2]) <= dispersion_px:
                j += 1
            win = samples[i : j + 1]
            xs, ys = [s[1] for s in win], [s[2] for s in win]
            out.append({"x": float(np.mean(xs)), "y": float(np.mean(ys)), "start_ms": win[0][0], "duration_ms": win[-1][0] - win[0][0]})
            i = j + 1
        else:
            i += 1
    return out


def _dispersion(win) -> float:
    xs, ys = [s[1] for s in win], [s[2] for s in win]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def gaussian_map(points: list[tuple[float, float]], weights: list[float], sigma_px: float, map_size: tuple[int, int]) -> np.ndarray:
    """Sum of Gaussians, one per point (frame pixels), rendered at map_size and scaled to max 1."""
    W, H = map_size
    sx = W / FRAME_W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sigma = max(0.8, sigma_px * sx)
    out = np.zeros((H, W), np.float32)
    for (x, y), w in zip(points, weights):
        out += w * np.exp(-(((xx - x * sx) ** 2) + ((yy - y * sx) ** 2)) / (2 * sigma**2))
    m = float(out.max())
    return out / m if m > 0 else out


def to_map_points(points: list[tuple[float, float]], map_size: tuple[int, int]) -> list[tuple[float, float]]:
    s = map_size[0] / FRAME_W
    return [(x * s, y * s) for x, y in points]


def build_ground_truth(method: str, points: list[dict], cfg: dict) -> dict:
    """Turns a session's raw points into fixations and the human map.
    Returns {fixations, gt, nss_points, weights, warnings}. Raises ValueError when nothing usable is left."""
    map_size = tuple(cfg["map_size"])
    warnings: list[str] = []

    if method == "click":
        c = cfg["click"]
        pts = [p for p in points if _finite(p)]
        if not pts:
            raise ValueError("No usable clicks.")
        # first look first: explicit order, else the order they arrived in
        pts = sorted(pts, key=lambda p: p.get("order") or 99)
        fixations = [{"x": p["x"], "y": p["y"], "order": i + 1, "duration_ms": None} for i, p in enumerate(pts)]
        weights = [c["order_weights"].get(p.get("order") or i + 1, 0.5) for i, p in enumerate(pts)]
        xy = [(f["x"], f["y"]) for f in fixations]
        gt = gaussian_map(xy, weights, c["sigma_px"], map_size)
        return {"fixations": fixations, "gt": gt, "nss_points": xy, "weights": weights, "warnings": warnings}

    w = cfg["webgazer"]
    m = w["margin_px"]
    raw = [p for p in points if _finite(p) and p.get("t") is not None]
    inside = [p for p in raw if -m <= p["x"] <= FRAME_W + m and -m <= p["y"] <= FRAME_H + m]
    if len(raw) - len(inside) > 0.4 * max(1, len(raw)):
        warnings.append(f"{len(raw) - len(inside)} of {len(raw)} gaze samples fell outside the thumbnail and were dropped.")
    if len(inside) < w["min_samples"]:
        raise ValueError(f"Only {len(inside)} usable gaze samples (need at least {w['min_samples']}). Recalibrate and try again.")
    inside.sort(key=lambda p: p["t"])
    samples = [(p["t"], min(FRAME_W, max(0, p["x"])), min(FRAME_H, max(0, p["y"]))) for p in inside]
    fx = idt(samples, w["idt"]["dispersion_px"], w["idt"]["min_duration_ms"])
    if not fx:
        raise ValueError("No fixations detected: the gaze was moving too much. Recalibrate and keep still.")
    fixations = [{**f, "order": i + 1} for i, f in enumerate(fx)]
    xy = [(f["x"], f["y"]) for f in fixations]
    weights = [f["duration_ms"] / 1000 + 0.1 for f in fixations]  # longer looks count more
    gt = gaussian_map(xy, weights, w["sigma_px"], map_size)
    return {"fixations": fixations, "gt": gt, "nss_points": xy, "weights": None, "warnings": warnings}


def _finite(p: dict) -> bool:
    return all(isinstance(p.get(k), (int, float)) and math.isfinite(p[k]) for k in ("x", "y"))


def first_look(fixations: list[dict], scanpath: list[dict], radius_px: float) -> dict:
    """Did the viewer's first look land where we predicted the first fixation? Also checks our first three."""
    if not fixations or not scanpath:
        return {"first_look_hit": None, "first_look_distance_px": None, "first_look_hit_top3": None}
    real = fixations[0]
    pred = sorted(scanpath, key=lambda s: s["order"])
    d = [math.hypot(real["x"] - s["x"], real["y"] - s["y"]) for s in pred]
    return {
        "first_look_hit": d[0] <= radius_px,
        "first_look_distance_px": round(d[0], 1),
        "first_look_hit_top3": min(d[:3]) <= radius_px,
    }


def score_session(pred_map: np.ndarray, truth: dict, cfg: dict) -> dict:
    """CC / NSS (and SIM / KLD) of the prediction against the human map, at map resolution."""
    gt = truth["gt"]
    pred = cv2.resize(pred_map.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
    pts = to_map_points(truth["nss_points"], (gt.shape[1], gt.shape[0]))
    return {
        "cc": round(cc(pred, gt), 4),
        "nss": round(nss(pred, pts, truth["weights"]), 4),
        "sim": round(sim(pred, gt), 4),
        "kld": round(kld(pred, gt), 4),
    }
