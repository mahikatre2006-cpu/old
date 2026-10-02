"""M4d. Colour and contrast analyzer. Classical CV only, no models.

Attention map = blend of three cues: LAB colour-pop (rare, saturated colours), spectral-residual
saliency, and multi-scale Difference-of-Gaussians contrast. Metrics: k-means palette (k=5, LAB),
Hasler-Suesstrunk colourfulness, brightness and lightness contrast.
"""
from __future__ import annotations

import cv2
import numpy as np

from analyzers.base import Analyzer
from pipeline.context import AnalyzerResult, Context

WORK_W, WORK_H = 320, 180


def p99_norm(m: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Scale so the 99th percentile maps to 1 (robust to a few extreme pixels), clipped to [0, 1].
    Maps whose 99th percentile is below `floor` are numerical noise (e.g. a flat image) and become all zeros."""
    hi = float(np.percentile(m, 99))
    if hi < floor:
        return np.zeros_like(m, dtype=np.float32)
    return np.clip(m / hi, 0, 1).astype(np.float32)


def to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """float32 CIELAB: L in 0..100, a and b roughly -128..127."""
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def colorfulness(rgb_u8: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003). ~0 grey, ~15 slightly, ~33 moderately, ~59 averagely, ~82 quite, 109+ highly."""
    r, g, b = [rgb_u8[..., i].astype(np.float64) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def spectral_residual(gray: np.ndarray) -> np.ndarray:
    """Hou & Zhang (2007) spectral residual saliency on a small grayscale image."""
    small = cv2.resize(gray.astype(np.float64), (128, 72), interpolation=cv2.INTER_AREA)
    if small.std() < 1e-3:  # a flat image has no spectrum to be unusual against
        return np.zeros(gray.shape, np.float32)
    small -= small.mean()  # drop the DC term, which otherwise dominates the log spectrum
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-3)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.0)
    return cv2.resize(sal, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)


def dog_contrast(lab: np.ndarray, scales: tuple[float, ...] = (1.5, 4.0, 10.0)) -> np.ndarray:
    """Sum over scales of |G(s) - G(2.5 s)|, on lightness plus (half-weighted) colour channels."""
    out = np.zeros(lab.shape[:2], np.float32)
    for s in scales:
        for ch, w in ((0, 1.0), (1, 0.5), (2, 0.5)):
            c = lab[..., ch]
            out += w * np.abs(cv2.GaussianBlur(c, (0, 0), s) - cv2.GaussianBlur(c, (0, 0), 2.5 * s))
    return out


def kmeans_palette(lab_small: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centers LAB (k,3) sorted by share, shares (k,), per-pixel label map for lab_small)."""
    flat = lab_small.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)  # deterministic palettes
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    k = min(k, max(1, len(np.unique(flat.round(0), axis=0))))
    _, labels, centers = cv2.kmeans(flat, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    shares = np.bincount(labels, minlength=k) / len(labels)
    order = np.argsort(-shares)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return centers[order], shares[order], remap[labels].reshape(lab_small.shape[:2])


def lab_to_hex(lab: np.ndarray) -> str:
    rgb = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(3)
    r, g, b = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class ColorAnalyzer(Analyzer):
    name = "color"
    metrics_group = "color"

    def __init__(self, k: int = 5, w_pop: float = 0.3, w_spectral: float = 0.3, w_dog: float = 0.4, significant_share: float = 0.05):
        self.k = k
        self.weights = (w_pop, w_spectral, w_dog)
        self.significant_share = significant_share

    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult:
        H, W = img.shape[:2]
        small = cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        lab = to_lab(small)

        # palette and colour-pop: rare, saturated colours stand out
        centers, shares, label_map = kmeans_palette(lab, self.k)
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        pop = (1.0 - shares[label_map]) ** 2 * (chroma / (chroma + 25.0))
        pop = cv2.GaussianBlur(pop.astype(np.float32), (0, 0), 3.0)

        sr = spectral_residual(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
        dog = dog_contrast(lab)

        w_pop, w_sr, w_dog = self.weights
        combined = w_pop * p99_norm(pop) + w_sr * p99_norm(sr) + w_dog * p99_norm(dog)
        combined = cv2.GaussianBlur(combined, (0, 0), 2.0)
        attention = cv2.resize(p99_norm(combined), (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)

        C = colorfulness(small)
        metrics = {
            "colorfulness": round(C, 2),
            "colorfulness_score": round(min(100.0, C / 109.0 * 100.0), 1),
            "brightness": round(float(lab[..., 0].mean()), 1),  # mean L, 0..100
            "lightness_contrast": round(float(lab[..., 0].std()), 1),
            "mean_chroma": round(float(chroma.mean()), 1),
            "dominant_share": round(float(shares[0]), 3),
            "significant_colors": float((shares >= self.significant_share).sum()),
            "palette_hex": [lab_to_hex(c) for c in centers],
            "palette_share": [round(float(s), 3) for s in shares],
        }
        return AnalyzerResult(attention_map=attention.astype(np.float32), metrics=metrics)
