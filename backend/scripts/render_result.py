"""Run the full pipeline on thumbnails and draw heatmap + numbered gaze path.

Usage (from backend/): python scripts/render_result.py out.jpg img1.jpg [img2.jpg ...]
Uses a throwaway data dir so nothing is cached or stored in the real database.
"""
import base64
import os
import sys
import tempfile
from pathlib import Path

os.environ["THUMB_DATA_DIR"] = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.ingest import ingest  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402

out, files = sys.argv[1], sys.argv[2:]
pipe = Pipeline()
rows = []
for f in files:
    data = Path(f).read_bytes()
    res = pipe.analyze(data)
    img = cv2.cvtColor(ingest(data, pipe.cfg["ingest"]).image, cv2.COLOR_RGB2BGR)
    heat = cv2.imdecode(np.frombuffer(base64.b64decode(res["heatmap_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    heat = cv2.resize(heat, (1280, 720), interpolation=cv2.INTER_CUBIC)
    a = heat[..., 3:4].astype(np.float32) / 255
    over = (img * (1 - a) + heat[..., :3] * a).astype(np.uint8)
    pts = [(p["x"], p["y"]) for p in res["scanpath"]]
    for p, q in zip(pts, pts[1:]):
        cv2.arrowedLine(over, p, q, (255, 255, 255), 4, tipLength=0.08)
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(over, (x, y), 26, (0, 0, 0), -1)
        cv2.circle(over, (x, y), 26, (255, 255, 255), 3)
        cv2.putText(over, str(i), (x - 11, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    print(Path(f).name, {k: round(v) for k, v in res["timing_ms"].items()}, res["metrics"]["attention_share"], res["unavailable"])
    rows.append(np.hstack([img, over]))
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
