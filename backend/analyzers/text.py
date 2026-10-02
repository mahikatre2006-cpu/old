"""M4b. Text analyzer: RapidOCR (PaddleOCR PP-OCRv4 mobile, ONNX) + mobile readability + WCAG contrast.

For each text line: box, string, confidence, whether OCR can still recover it from the 168 px sidebar
render (mobile_readable), and the WCAG contrast ratio between the text colour and its surroundings.
Output also includes a text attention map and metrics.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np

from analyzers.base import Analyzer
from pipeline.context import AnalyzerResult, Context, Region


# ---------- pure helpers ----------

def _lin(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def relative_luminance(rgb) -> float:
    r, g, b = _lin(np.asarray(rgb, np.float64))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def wcag_contrast(rgb1, rgb2) -> float:
    """WCAG 2.x contrast ratio, 1 (identical) to 21 (black on white)."""
    l1, l2 = relative_luminance(rgb1), relative_luminance(rgb2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def text_recovered(full_text: str, full_box, small_items: list["TextItem"], min_overlap: float = 0.3, min_cover: float = 0.7) -> bool:
    """Was this text line found again at sidebar size? The sidebar OCR must produce boxes overlapping this
    line's area whose text contains (at least min_cover of) the original string."""
    a = _norm(full_text)
    if not a:
        return False
    x0, y0, x1, y1 = full_box
    area = max(1, (x1 - x0) * (y1 - y0))
    hits = []
    for it in small_items:
        bx0, by0, bx1, by1 = it.box
        iw, ih = min(x1, bx1) - max(x0, bx0), min(y1, by1) - max(y0, by0)
        if iw > 0 and ih > 0 and iw * ih / area >= min_overlap:
            hits.append(_norm(it.text))
    if not hits:
        return False
    b = "".join(hits)
    if not b:
        return False
    # total characters of `a` found in order inside `b`, so one OCR slip does not break a match
    matched = sum(blk.size for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / len(a) >= min_cover


@dataclass
class TextItem:
    text: str
    score: float
    poly: np.ndarray  # 4x2 float, frame coordinates
    box: tuple[int, int, int, int]


def _to_item(box, text, score, W: int, H: int) -> TextItem:
    poly = np.asarray(box, np.float32)
    x0, y0 = np.clip(poly.min(axis=0), 0, [W, H]).astype(int)
    x1, y1 = np.clip(poly.max(axis=0), 0, [W, H]).astype(int)
    return TextItem(str(text), float(score), poly, (int(x0), int(y0), int(x1), int(y1)))


def estimate_colors(rgb: np.ndarray, item: TextItem) -> tuple[np.ndarray, np.ndarray] | None:
    """(text colour, background colour). Background = median of a ring just outside the text box;
    text = median of the quarter of in-box pixels furthest from that background."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = item.box
    m = max(4, int(0.3 * (y1 - y0)))
    wx0, wy0, wx1, wy1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    win = rgb[wy0:wy1, wx0:wx1]
    local = (item.poly - [wx0, wy0]).astype(np.int32)
    inside = np.zeros(win.shape[:2], np.uint8)
    cv2.fillPoly(inside, [local], 1)
    near = cv2.dilate(inside, np.ones((5, 5), np.uint8))
    ring = win[near == 0]
    px = win[inside == 1]
    if len(ring) < 20 or len(px) < 10:
        return None
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(px.astype(np.float32) - bg, axis=1)
    far = px[dist >= np.quantile(dist, 0.75)]
    return np.median(far, axis=0), bg


# ---------- analyzer ----------

class TextAnalyzer(Analyzer):
    name = "text"
    metrics_group = "text"

    def __init__(
        self,
        text_score: float = 0.5,
        det_limit_side_len: int = 768,
        sidebar_upscale: int = 3,  # sidebar render is enlarged before OCR; OCR cannot invent detail, only see it
        min_chars: int = 2,  # drops single-glyph logos and icons
        min_height_px: int = 12,  # on the 1280x720 frame
        contrast_min: float = 4.5,  # WCAG AA for normal text
        low_confidence: float = 0.75,  # mean OCR confidence below this marks the section low-confidence
        max_regions: int = 15,
        threads: int = 2,
    ):
        self.text_score = text_score
        self.det_limit_side_len = det_limit_side_len
        self.sidebar_upscale = sidebar_upscale
        self.min_chars = min_chars
        self.min_height_px = min_height_px
        self.contrast_min = contrast_min
        self.low_confidence = low_confidence
        self.max_regions = max_regions
        self.threads = threads
        self.engine = None
        self._lock = threading.Lock()

    def load(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR(
            intra_op_num_threads=self.threads,
            inter_op_num_threads=1,
            det_limit_type="max",
            det_limit_side_len=self.det_limit_side_len,
            text_score=self.text_score,
            print_verbose=False,
        )
        self.engine(np.full((96, 160, 3), 255, np.uint8), use_cls=False)  # warm-up

    def _ocr(self, rgb: np.ndarray, W: int, H: int) -> list[TextItem]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            res, _ = self.engine(bgr, use_cls=False)
        return [_to_item(b, t, s, W, H) for b, t, s in (res or [])]

    def run(self, img: np.ndarray, ctx: Context) -> AnalyzerResult:
        H, W = img.shape[:2]
        items = [
            it for it in self._ocr(img, W, H)
            if len(it.text.strip()) >= self.min_chars and (it.box[3] - it.box[1]) >= self.min_height_px
        ]
        empty_metrics = {"count": 0.0, "word_count": 0.0, "area_pct": 0.0}
        if not items:
            return AnalyzerResult(attention_map=np.zeros((H, W), np.float32), metrics=empty_metrics)

        # Mobile readability: OCR the sidebar-size render again; scale its boxes back to the frame.
        small = ctx.views["sidebar"]
        up = cv2.resize(small, None, fx=self.sidebar_upscale, fy=self.sidebar_upscale, interpolation=cv2.INTER_CUBIC)
        sx, sy = W / up.shape[1], H / up.shape[0]
        small_items = []
        for it in self._ocr(up, up.shape[1], up.shape[0]):
            x0, y0, x1, y1 = it.box
            small_items.append(TextItem(it.text, it.score, it.poly, (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))))

        canvas = np.zeros((H, W), np.float32)
        union = np.zeros((H, W), np.uint8)
        regions, contrasts, heights, readable_flags = [], [], [], []
        for it in items:
            x0, y0, x1, y1 = it.box
            h_px = y1 - y0
            readable = text_recovered(it.text, it.box, small_items)
            colors = estimate_colors(img, it)
            contrast = round(wcag_contrast(*colors), 2) if colors else None

            size = min(1.0, h_px / (0.12 * H))
            cfac = 0.7 if contrast is None else 0.5 + 0.5 * min(1.0, contrast / 7.0)
            amp = float(min(1.0, (0.3 + 0.7 * size) * cfac))
            cv2.fillPoly(canvas, [it.poly.astype(np.int32)], amp)  # later lines overwrite; fine for non-overlapping
            cv2.fillPoly(union, [it.poly.astype(np.int32)], 1)

            readable_flags.append(readable)
            heights.append(h_px / H * 100)
            if contrast is not None:
                contrasts.append(contrast)
            regions.append(
                Region(
                    "text",
                    box=it.box,
                    attrs={
                        "text": it.text,
                        "mobile_readable": readable,
                        "contrast": contrast,
                        "confidence": round(it.score, 3),
                        "height_pct": round(h_px / H * 100, 1),
                    },
                )
            )

        text_map = np.clip(cv2.GaussianBlur(canvas, (0, 0), 8), 0, 1).astype(np.float32)
        regions.sort(key=lambda r: -(r.box[3] - r.box[1]) * (r.box[2] - r.box[0]))
        mean_conf = float(np.mean([it.score for it in items]))
        words = sum(len(it.text.split()) for it in items)
        n_unread = sum(1 for f in readable_flags if not f)
        metrics = {
            "count": float(len(items)),
            "word_count": float(words),
            "area_pct": round(100 * float(union.mean()), 2),
            "largest_height_pct": round(max(heights), 1),
            "mean_confidence": round(mean_conf, 3),
            "low_confidence": float(mean_conf < self.low_confidence),
            "unreadable_mobile_count": float(n_unread),
            "mobile_readable_ratio": round(1 - n_unread / len(items), 3),
            "low_contrast_count": float(sum(1 for c in contrasts if c < self.contrast_min)),
        }
        if contrasts:
            metrics["min_contrast"] = round(min(contrasts), 2)
            metrics["mean_contrast"] = round(float(np.mean(contrasts)), 2)
        return AnalyzerResult(attention_map=text_map, regions=regions[: self.max_regions], metrics=metrics)
