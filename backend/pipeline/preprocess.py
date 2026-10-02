"""M2. Build the Context: model input, multi-scale renders, UI occlusion mask."""
from __future__ import annotations

import cv2
import numpy as np

from pipeline.context import Context


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def build_ui_mask(h: int, w: int, boxes: dict[str, list[float]]) -> np.ndarray:
    """1 where YouTube UI (duration badge, progress bar) covers the thumbnail."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes.values():
        mask[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)] = 1
    return mask


def build_context(image: np.ndarray, image_hash: str, cfg: dict) -> Context:
    h, w = image.shape[:2]
    mw, mh = cfg["model_input_size"]
    views = {name: _resize(image, vw, round(vw * h / w)) for name, vw in cfg["renders"].items()}
    return Context(
        image_hash=image_hash,
        original=image,
        model_input=_resize(image, mw, mh),
        views=views,
        ui_mask=build_ui_mask(h, w, cfg["ui_mask"]),
        config=cfg,
    )
