"""Generates docs/mock_analysis.json: a valid example of the frozen /analyze response.

Run from backend/: .venv/Scripts/python scripts/make_mock.py
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 320, 180
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def blob(cx, cy, sx, sy, a):
    return a * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)


heat = blob(103, 52, 30, 28, 1.0) + blob(245, 40, 40, 20, 0.55) + blob(160, 130, 70, 25, 0.15)
heat = (heat / heat.max() * 255).astype(np.uint8)
bgr = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
alpha = np.clip(heat.astype(np.float32) * 1.4, 0, 200).astype(np.uint8)
ok, png = cv2.imencode(".png", np.dstack([bgr, alpha]))
assert ok

result = {
    "id": "0" * 64,
    "heatmap_png": base64.b64encode(png.tobytes()).decode(),
    "scanpath": [{"x": 412, "y": 210, "order": 1}, {"x": 980, "y": 150, "order": 2}, {"x": 640, "y": 520, "order": 3}],
    "regions": [
        {"type": "face", "box": [300, 120, 520, 400], "attention_pct": 41.2, "emotion": "surprise"},
        {"type": "text", "box": [700, 80, 1200, 260], "attention_pct": 18.5, "text": "I QUIT", "mobile_readable": False},
        {"type": "subject", "box": [250, 90, 560, 650], "attention_pct": 22.0},
        {"type": "background", "box": None, "attention_pct": 18.3},
    ],
    "scores": {"overall": 72, "focus": 80, "text_readability": 45, "face_impact": 85, "clutter": 70},
    "recommendations": [
        {"id": "text_unreadable_mobile", "severity": "high",
         "message": "Text 'I QUIT' disappears at sidebar size. Increase its height from 9% to at least 14% of the frame."},
        {"id": "ui_overlap", "severity": "medium",
         "message": "6.1% of attention falls under the duration badge. Move key content out of the bottom-right corner."},
    ],
    "metrics": {
        "face": {"count": 1, "area_pct": 9.8},
        "text": {"area_pct": 7.4, "min_contrast": 3.1},
        "color": {"colorfulness": 61.2},
        "composition": {"entropy": 0.62, "focal_points": 2},
    },
    "timing_ms": {"saliency": 310, "face": 95, "text": 420, "total": 1180},
    "unavailable": [],
    "cached": False,
}

out = Path(__file__).resolve().parent.parent.parent / "docs" / "mock_analysis.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("wrote", out)
