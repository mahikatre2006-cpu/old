"""M4c. Subject analyzer: U2-Net-lite (u2netp) salient-object mask -> area, position, background separation.

Output: subject regions (box + full-res mask), a soft subject attention map, and metrics.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from analyzers.base import Analyzer
from analyzers.saliency import minmax
from pipeline.config import MODELS_DIR
from pipeline.context import AnalyzerResult, Context, Region

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def separation_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    """How different the subject looks from its surroundings, 0..1: mean CIELAB distance between
    the subject and the background, scaled so a deltaE of 50 or more counts as fully separated."""
    if mask.sum() < 50 or (~mask).sum() < 50:
        return 0.0
    lab = cv2.cvtColor(cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    m = cv2.resize(mask.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST).astype(bool)
    if m.sum() < 20 or (~m).sum() < 20:
        return 0.0
    de = float(np.linalg.norm(lab[m].mean(axis=0) - lab[~m].mean(axis=0)))
    return round(min(1.0, de / 50.0), 3)


def edge_contrast(rgb: np.ndarray, mask: np.ndarray, radius: int = 9) -> float:
    """Colour difference (CIELAB deltaE, median) right along the subject's outline: the average colour just
    inside versus just outside the silhouette. Unlike a global subject-vs-background average, a colourful
    subject cannot cancel itself out. Roughly: under 25 is a weak edge, 30+ is clear."""
    small = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST)
    if m.sum() < 30 or (1 - m).sum() < 30:
        return 0.0
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    k = (2 * radius + 1, 2 * radius + 1)

    def local_mean(sel: np.ndarray):
        w = sel.astype(np.float32)
        den = cv2.blur(w, k)[..., None]
        return cv2.blur(lab * w[..., None], k) / np.maximum(den, 1e-3), den[..., 0]

    inner, d_in = local_mean(m)
    outer, d_out = local_mean(1 - m)
    boundary = (m == 1) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0) & (d_in > 0.05) & (d_out > 0.05)
    if boundary.sum() < 10:
        return 0.0
    return float(np.median(np.linalg.norm(inner[boundary] - outer[boundary], axis=1)))


class SubjectAnalyzer(Analyzer):
    name = "subject"
    metrics_group = "subject"

    def __init__(
        self,
        model: str = "u2netp.onnx",
        threshold: float = 0.5,
        min_area_frac: float = 0.015,  # components smaller than this are ignored
        max_regions: int = 3,
        clear_subject_range: tuple[float, float] = (0.03, 0.80),  # area fraction that reads as "a subject"
        blur_sigma: float = 12.0,
        threads: int = 2,
    ):
        self.model_path = Path(model) if Path(model).is_absolute() else MODELS_DIR / model
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.max_regions = max_regions
        self.clear_subject_range = clear_subject_range
        self.blur_sigma = blur_sigma
        self.threads = threads
        self.session = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"Subject model not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.session.run(None, {self.input_name: np.zeros((1, 3, 320, 320), np.float32)})

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Salient-object probability at full frame size, 0..1."""
        H, W = rgb.shape[:2]
        x = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = x / max(float(x.max()), 1e-6)  # rembg's U2-Net preprocessing: scale by max, then ImageNet norm
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        with self._lock:
            out = self.session.run(None, {self.input_name: x})[0]
        prob = minmax(np.squeeze(out).astype(np.float32))
        return cv2.resize(prob, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)

    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult:
        H, W = img.shape[:2]
        prob = self.predict(img)
        mask = prob >= self.threshold
        area_frac = float(mask.mean())

        # connected components: keep the biggest few as subject regions
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        comps = sorted(
            (i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] / (W * H) >= self.min_area_frac),
            key=lambda i: -stats[i, cv2.CC_STAT_AREA],
        )[: self.max_regions]
        regions = []
        for i in comps:
            x, y, w, h, _ = stats[i]
            regions.append(Region("subject", box=(int(x), int(y), int(x + w), int(y + h)), mask=labels == i))

        kept = np.zeros((H, W), bool)
        for r in regions:
            kept |= r.mask
        kept_frac = float(kept.mean())

        soft = cv2.GaussianBlur(prob * kept, (0, 0), self.blur_sigma) if kept_frac > 0 else np.zeros((H, W), np.float32)
        lo, hi = self.clear_subject_range
        metrics = {
            "count": float(len(regions)),
            "area_pct": round(100 * kept_frac, 2),
            "has_clear_subject": float(len(regions) > 0 and lo <= kept_frac <= hi),
            # how decisively the model separates subject from background: 1 = crisp, 0 = undecided
            "mask_confidence": round(float((np.abs(prob - 0.5) > 0.35).mean()), 3),
        }
        if kept_frac > 0:
            ys, xs = np.where(kept)
            metrics.update(
                cx=round(float(xs.mean()) / W, 3),
                cy=round(float(ys.mean()) / H, 3),
                background_separation=separation_score(img, kept),
                edge_contrast=round(edge_contrast(img, kept), 1),
            )
        return AnalyzerResult(attention_map=np.clip(soft, 0, 1).astype(np.float32), regions=regions, metrics=metrics)
