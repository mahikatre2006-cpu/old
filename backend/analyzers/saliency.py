"""M3. Saliency core: ONNX model -> HxW fixation probability map on the 1280x720 frame.

Model file, input size, layout, value range, letterboxing and output activation are config options, so
FastSal / MSI-Net / SimpleNet are interchangeable without touching other modules.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from analyzers.base import Analyzer
from pipeline.config import MODELS_DIR
from pipeline.context import AnalyzerResult, Context

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def minmax(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - lo) / (hi - lo)).astype(np.float32)


def letterbox(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Fit img inside w x h keeping aspect, zero-pad the rest. Returns (padded, (top, bottom, left, right))."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (h - nh) // 2, (w - nw) // 2
    out = np.zeros((h, w, 3), dtype=img.dtype)
    out[top : top + nh, left : left + nw] = small
    return out, (top, h - nh - top, left, w - nw - left)


def unpad(m: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pad
    return m[top : m.shape[0] - bottom, left : m.shape[1] - right]


class SaliencyAnalyzer(Analyzer):
    name = "saliency"
    metrics_group = "saliency"

    def __init__(
        self,
        model: str = "fastsal_salicon_A.onnx",
        input_size: tuple[int, int] = (256, 192),  # W, H the ONNX was exported with
        layout: str = "nchw",  # nchw (FastSal) | nhwc (MSI-Net)
        input_range: float = 1.0,  # 1.0 -> pixels scaled to 0..1 ; 255.0 -> raw 0..255
        mean: tuple[float, float, float] | None = IMAGENET_MEAN,  # applied after scaling; None to skip
        std: tuple[float, float, float] | None = IMAGENET_STD,
        pad_to_fit: bool = False,  # letterbox instead of stretching (MSI-Net)
        sigmoid: bool = True,
        blur_sigma: float = 6.0,  # in 1280x720 pixels, smooths upsampling blockiness
        threads: int = 2,
    ):
        self.model_path = Path(model) if Path(model).is_absolute() else MODELS_DIR / model
        self.input_size = tuple(input_size)
        self.layout = layout
        self.input_range = float(input_range)
        self.mean = None if mean is None else np.asarray(mean, np.float32).reshape(1, 1, 3)
        self.std = None if std is None else np.asarray(std, np.float32).reshape(1, 1, 3)
        self.pad_to_fit = pad_to_fit
        self.sigmoid = sigmoid
        self.blur_sigma = blur_sigma
        self.threads = threads
        self.session = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"Saliency model not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        w, h = self.input_size  # warm-up so the first real request is not slow
        self.session.run(None, {self.input_name: self._to_tensor(np.zeros((h, w, 3), np.uint8))})

    def _to_tensor(self, rgb_u8: np.ndarray) -> np.ndarray:
        x = rgb_u8.astype(np.float32)
        x = x / 255.0 * self.input_range
        if self.mean is not None:
            x = (x - self.mean) / self.std
        if self.layout == "nchw":
            x = x.transpose(2, 0, 1)
        return np.ascontiguousarray(x[None])

    def _prepare(self, ctx: Context) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        w, h = self.input_size
        if self.pad_to_fit:
            img, pad = letterbox(ctx.original, w, h)
            return self._to_tensor(img), pad
        img = ctx.model_input
        if img.shape[:2] != (h, w):
            img = cv2.resize(ctx.original, (w, h), interpolation=cv2.INTER_AREA)
        return self._to_tensor(img), None

    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult:
        x, pad = self._prepare(ctx)
        with self._lock:  # one inference at a time per session; parallelism is across analyzers
            out = self.session.run(None, {self.input_name: x})[0]
        sal = np.squeeze(out).astype(np.float32)
        if self.sigmoid:
            sal = 1.0 / (1.0 + np.exp(-sal))
        if pad is not None:
            sal = unpad(sal, pad)
        H, W = img.shape[:2]
        sal = cv2.resize(minmax(sal), (W, H), interpolation=cv2.INTER_CUBIC)
        if self.blur_sigma > 0:
            sal = cv2.GaussianBlur(sal, (0, 0), self.blur_sigma)
        return AnalyzerResult(attention_map=minmax(sal))
