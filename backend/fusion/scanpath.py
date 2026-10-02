"""M5. Predicted gaze order: winner-take-all with inhibition of return."""
from __future__ import annotations

import cv2
import numpy as np


def predict_scanpath(
    attention: np.ndarray,
    max_fixations: int = 5,
    min_fixations: int = 3,
    inhibition_sigma_frac: float = 0.08,
    min_ratio: float = 0.25,
    work_width: int = 320,
) -> list[dict]:
    """Repeatedly take the highest remaining peak, then suppress a Gaussian around it so the
    next winner is elsewhere. Coordinates are returned in the input map's pixel space.

    Fixations weaker than min_ratio * the first peak are dropped, but at least min_fixations
    are always returned (for a non-flat map).
    """
    H, W = attention.shape
    scale = work_width / W
    small = cv2.resize(attention.astype(np.float32), (work_width, max(1, round(H * scale))), interpolation=cv2.INTER_AREA)
    if float(small.max()) <= 1e-6:
        return []
    h, w = small.shape
    sigma = inhibition_sigma_frac * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    work = small.copy()
    first_peak = float(work.max())
    path: list[dict] = []
    for order in range(1, max_fixations + 1):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        peak = float(work[y, x])
        if order > min_fixations and peak < min_ratio * first_peak:
            break
        path.append({"x": int(round((x + 0.5) / scale)), "y": int(round((y + 0.5) / scale)), "order": order})
        work *= 1.0 - np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    for p in path:
        p["x"], p["y"] = min(p["x"], W - 1), min(p["y"], H - 1)
    return path
