"""M4a. Face analyzer: YuNet detection + HSEmotion (EfficientNet-B0, AffectNet) emotion and intensity.

Output: face regions, a Gaussian attention map weighted by face size and emotion intensity, and metrics.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from analyzers.base import Analyzer
from pipeline.config import MODELS_DIR
from pipeline.context import AnalyzerResult, Context, Region

# Class order of the 8-class AffectNet HSEmotion models. The *_va_mtl models append valence, arousal.
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def face_amplitude(area_frac: float, intensity: float | None, ref_area: float) -> float:
    """How much attention a face earns: grows with its size (saturating at ref_area of the frame)
    and with expression intensity. Ranges from ~0.25 (tiny, calm) to 1.0 (large, intense)."""
    size = min(1.0, (area_frac / ref_area) ** 0.5)
    emo = 0.7 + 0.3 * (intensity if intensity is not None else 0.0)
    return float(min(1.0, (0.35 + 0.65 * size) * emo))


def build_face_map(boxes: list[tuple[int, int, int, int]], amps: list[float], shape: tuple[int, int]) -> np.ndarray:
    """Max of one anisotropic Gaussian per face, centred on it, sized to cover the box."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    for (x0, y0, x1, y1), a in zip(boxes, amps):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy = max(4.0, 0.40 * (x1 - x0)), max(4.0, 0.40 * (y1 - y0))
        # evaluate only within 4 sigma for speed
        xa, xb = int(max(0, cx - 4 * sx)), int(min(W, cx + 4 * sx))
        ya, yb = int(max(0, cy - 4 * sy)), int(min(H, cy + 4 * sy))
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        g = a * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
        out[ya:yb, xa:xb] = np.maximum(out[ya:yb, xa:xb], g)
    return out


class FaceAnalyzer(Analyzer):
    name = "face"
    metrics_group = "face"

    def __init__(
        self,
        detector: str = "face_detection_yunet_2023mar.onnx",
        emotion_model: str | None = "enet_b0_8_va_mtl.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        min_face_px: int = 12,  # ignore detections smaller than this (shorter side)
        min_emotion_px: int = 32,  # emotion is unreliable on tinier crops, so it is skipped
        crop_margin: float = 0.15,
        ref_area: float = 0.06,  # a face covering 6% of the frame earns full size weight
        threads: int = 2,
    ):
        self.detector_path = MODELS_DIR / detector
        self.emotion_path = MODELS_DIR / emotion_model if emotion_model else None
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_px = min_face_px
        self.min_emotion_px = min_emotion_px
        self.crop_margin = crop_margin
        self.ref_area = ref_area
        self.threads = threads
        self.det = None
        self.emo = None
        self._det_lock = threading.Lock()  # cv2.FaceDetectorYN is not thread-safe

    def load(self) -> None:
        for p in (self.detector_path, self.emotion_path):
            if p is not None and not Path(p).exists():
                raise FileNotFoundError(f"Face model not found: {p}")
        self.det = cv2.FaceDetectorYN.create(
            str(self.detector_path), "", (1280, 720), self.score_threshold, self.nms_threshold, 50
        )
        if self.emotion_path:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.threads
            self.emo = ort.InferenceSession(str(self.emotion_path), opts, providers=["CPUExecutionProvider"])
            self.emo.run(None, {"input": np.zeros((1, 3, 224, 224), np.float32)})  # warm-up

    # ---- detection ----
    def detect(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        H, W = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._det_lock:
            self.det.setInputSize((W, H))
            _, faces = self.det.detect(bgr)
        out = []
        for f in faces if faces is not None else []:
            x, y, w, h = f[:4]
            x0, y0, x1, y1 = max(0, int(round(x))), max(0, int(round(y))), min(W, int(round(x + w))), min(H, int(round(y + h)))
            if min(x1 - x0, y1 - y0) >= self.min_face_px:
                out.append(((x0, y0, x1, y1), float(f[-1])))
        return sorted(out, key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))  # largest first

    # ---- emotion ----
    def _crop(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = box
        mx, my = int((x1 - x0) * self.crop_margin), int((y1 - y0) * self.crop_margin)
        crop = rgb[max(0, y0 - my) : min(H, y1 + my), max(0, x0 - mx) : min(W, x1 + mx)]
        x = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(2, 0, 1)

    def emotions(self, rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[dict | None]:
        """Per face: {probs, label, valence, arousal, intensity}, or None when the face is too small."""
        res: list[dict | None] = [None] * len(boxes)
        idx = [i for i, b in enumerate(boxes) if self.emo and min(b[2] - b[0], b[3] - b[1]) >= self.min_emotion_px]
        if not idx:
            return res
        batch = np.stack([self._crop(rgb, boxes[i]) for i in idx]).astype(np.float32)
        out = self.emo.run(None, {"input": batch})[0]  # N x 10: 8 logits, valence, arousal
        for row, i in zip(out, idx):
            probs = softmax(row[:8])
            v, a = float(np.clip(row[8], -1, 1)), float(np.clip(row[9], -1, 1))
            res[i] = {
                "probs": {e: float(p) for e, p in zip(EMOTIONS, probs)},
                "label": EMOTIONS[int(np.argmax(probs))],
                "valence": v,
                "arousal": a,
                "intensity": float(min(1.0, np.hypot(v, a))),
            }
        return res

    # ---- analyzer ----
    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult:
        H, W = img.shape[:2]
        found = self.detect(img)
        if not found:
            return AnalyzerResult(
                attention_map=np.zeros((H, W), np.float32),
                metrics={"count": 0.0, "total_area_pct": 0.0, "largest_area_pct": 0.0},
            )
        boxes = [b for b, _ in found]
        emos = self.emotions(img, boxes)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) / (W * H) for b in boxes]
        amps = [face_amplitude(a, e["intensity"] if e else None, self.ref_area) for a, e in zip(areas, emos)]

        regions = [
            Region(
                "face",
                box=b,
                attrs={
                    "emotion": e["label"] if e else None,
                    "intensity": round(e["intensity"], 3) if e else None,
                    "detection_score": round(s, 3),
                },
            )
            for (b, s), e in zip(found, emos)
        ]

        big = boxes[0]
        metrics = {
            "count": float(len(boxes)),
            "total_area_pct": round(100 * sum(areas), 2),
            "largest_area_pct": round(100 * areas[0], 2),
            "largest_cx": round((big[0] + big[2]) / 2 / W, 3),  # 0..1 position of the largest face
            "largest_cy": round((big[1] + big[3]) / 2 / H, 3),
        }
        known = [e for e in emos if e]
        if known:
            metrics["mean_intensity"] = round(float(np.mean([e["intensity"] for e in known])), 3)
            metrics["mean_valence"] = round(float(np.mean([e["valence"] for e in known])), 3)
            metrics["mean_arousal"] = round(float(np.mean([e["arousal"] for e in known])), 3)
        if emos[0]:
            metrics.update({f"p_{k}": round(v, 3) for k, v in emos[0]["probs"].items()})  # largest face

        return AnalyzerResult(attention_map=build_face_map(boxes, amps, (H, W)), regions=regions, metrics=metrics)
