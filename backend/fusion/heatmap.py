"""Encode an attention map as a transparent RGBA heatmap PNG (base64) for the clients to overlay."""
from __future__ import annotations

import base64

import cv2
import numpy as np


COLORMAPS = {"inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA, "viridis": cv2.COLORMAP_VIRIDIS, "jet": cv2.COLORMAP_JET}


def encode_heatmap_png(attention: np.ndarray, size: tuple[int, int] = (640, 360), colormap: str = "inferno") -> str:
    """attention: HxW float in [0, 1]. Low values are transparent so the thumbnail shows through.
    Inferno is perceptually uniform and readable with common colour-vision deficiencies."""
    m = cv2.resize(np.clip(attention, 0, 1), size, interpolation=cv2.INTER_AREA)
    u8 = (m * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, COLORMAPS[colormap])
    alpha = np.clip((m - 0.12) / 0.5, 0, 1) * 215  # cold areas fully transparent
    ok, png = cv2.imencode(".png", np.dstack([bgr, alpha.astype(np.uint8)]))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()


def encode_preview_jpg(rgb: np.ndarray, size: tuple[int, int] = (960, 540), quality: int = 85) -> str:
    """The normalized frame the analyzers saw, so clients overlay on exactly the same pixels."""
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(jpg.tobytes()).decode()


def encode_attention_png(attention: np.ndarray, size: tuple[int, int] = (160, 90)) -> str:
    """The fused attention map itself (0..1 as 8-bit grayscale), small enough to ship with every result.
    Clients and the A/B comparison use it for difference maps and predicted-vs-real comparisons."""
    small = cv2.resize(np.clip(attention, 0, 1).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", (small * 255).round().astype(np.uint8))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()


def decode_attention_png(b64: str) -> np.ndarray:
    gray = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_GRAYSCALE)
    return gray.astype(np.float32) / 255.0


def encode_diff_png(base: np.ndarray, other: np.ndarray, size: tuple[int, int] = (640, 360)) -> str:
    """Where `other` draws more attention than `base` (orange) or less (blue). Transparent where they agree.
    The blue/orange pair stays distinguishable with common colour-vision deficiencies."""
    d = cv2.resize(other.astype(np.float32) - base.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    mag = np.clip(np.abs(d) / 0.6, 0, 1)
    out = np.zeros((size[1], size[0], 4), np.uint8)
    gain, loss = d > 0, d <= 0
    out[gain, :3] = (30, 140, 245)  # BGR orange
    out[loss, :3] = (200, 110, 30)  # BGR blue
    out[..., 3] = (np.clip((mag - 0.06) / 0.5, 0, 1) * 215).astype(np.uint8)
    ok, png = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(png.tobytes()).decode()
