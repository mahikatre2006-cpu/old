"""M1. Validate an upload, normalize to 1280x720 RGB, hash it."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class IngestError(ValueError):
    """Raised for input that is not a usable thumbnail."""


@dataclass
class Ingested:
    image: np.ndarray  # H x W x 3 uint8 RGB
    sha256: str
    info: dict


def _fit_16x9(rgb: np.ndarray, target_w: int, target_h: int, tolerance: float) -> tuple[np.ndarray, str]:
    h, w = rgb.shape[:2]
    target_ar = target_w / target_h
    ar = w / h
    if abs(ar / target_ar - 1) <= 1e-3:
        method = "none"
        out = rgb
    elif abs(ar / target_ar - 1) <= tolerance:
        # slightly off 16:9: centre-crop so nothing is padded
        if ar > target_ar:
            new_w = round(h * target_ar)
            x0 = (w - new_w) // 2
            out = rgb[:, x0 : x0 + new_w]
        else:
            new_h = round(w / target_ar)
            y0 = (h - new_h) // 2
            out = rgb[y0 : y0 + new_h]
        method = "crop"
    else:
        # far from 16:9: letterbox so no content is lost
        if ar > target_ar:
            new_w, new_h = w, round(w / target_ar)
        else:
            new_w, new_h = round(h * target_ar), h
        out = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        y0, x0 = (new_h - h) // 2, (new_w - w) // 2
        out[y0 : y0 + h, x0 : x0 + w] = rgb
        method = "letterbox"
    if out.shape[:2] != (target_h, target_w):
        interp = cv2.INTER_AREA if out.shape[1] > target_w else cv2.INTER_CUBIC
        out = cv2.resize(out, (target_w, target_h), interpolation=interp)
    return out, method


def ingest(data: bytes, cfg: dict) -> Ingested:
    if not data:
        raise IngestError("Empty upload")
    if len(data) > cfg["max_bytes"]:
        raise IngestError(f"File too large (max {cfg['max_bytes'] // 1_048_576} MB)")

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in ALLOWED_FORMATS:
                raise IngestError(f"Unsupported image type {fmt!r}; use JPG, PNG or WEBP")
            if im.width * im.height > cfg["max_pixels"]:
                raise IngestError("Image has too many pixels")
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, rgba)
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as e:
        raise IngestError("Not a readable image") from e

    h, w = rgb.shape[:2]
    min_w, min_h = cfg["min_size"]
    if w < min_w or h < min_h:
        raise IngestError(f"Image too small ({w}x{h}); minimum {min_w}x{min_h}")

    tw, th = cfg["target_size"]
    norm, method = _fit_16x9(rgb, tw, th, cfg["crop_tolerance"])
    norm = np.ascontiguousarray(norm)
    digest = hashlib.sha256(norm.tobytes()).hexdigest()
    return Ingested(
        image=norm,
        sha256=digest,
        info={"source_size": [w, h], "format": fmt, "fit": method},
    )
