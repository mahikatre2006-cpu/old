"""Overlay saliency maps on a thumbnail for eyeballing, one panel per model profile.

Usage (from backend/): python scripts/saliency_demo.py [image|-] [out.png] [profile ...]
`-` renders a synthetic image. Profiles: msinet (default config), fastsal.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers.saliency import SaliencyAnalyzer  # noqa: E402
from pipeline.config import analyzers_config  # noqa: E402
from pipeline.ingest import ingest  # noqa: E402
from pipeline.preprocess import build_context  # noqa: E402

cfg = analyzers_config()
PROFILES = {
    "msinet": cfg["analyzers"]["saliency"]["options"],
    "fastsal": {"model": "fastsal_salicon_A.onnx", "input_size": [256, 192], "sigmoid": True},
}

src = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
out = sys.argv[2] if len(sys.argv) > 2 else "saliency_demo.png"
profiles = sys.argv[3:] or ["msinet", "fastsal"]

if src:
    img = ingest(Path(src).read_bytes(), cfg["ingest"]).image
else:
    from tests.test_saliency import synthetic_thumbnail

    img = synthetic_thumbnail()

bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
ctx = build_context(img, "demo", cfg["preprocess"])
panels = [bgr]
for name in profiles:
    a = SaliencyAnalyzer(**PROFILES[name])
    a.load()
    t0 = time.perf_counter()
    m = a.run(img, ctx).attention_map
    print(f"{name}: {1000 * (time.perf_counter() - t0):.0f} ms, mean {m.mean():.3f}")
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panel = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)
    cv2.putText(panel, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    panels.append(panel)

cv2.imwrite(out, cv2.resize(np.hstack(panels), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
print("wrote", out)
