"""Does the scoring engine separate good thumbnails from clearly bad ones? (PRD M6 "done when")

Takes real thumbnails and damages each in a known way; the matching score should drop.
  lowcontrast : image blended 65% toward grey   -> text readability and focus should fall
  shrunk      : content scaled to 40% in a grey canvas -> text/face get tiny (mobile readability, face impact)
  cluttered   : many random stickers + tiny text lines  -> clutter and focus should fall

Usage (from backend/): python scripts/calibrate_scores.py [n_images]
Uses a throwaway data dir. The damaged images are synthetic, so this checks direction and size of effect,
not absolute accuracy; real calibration data comes with the M12 pilot.
"""
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

THUMBS = Path(__file__).resolve().parent.parent.parent / "data" / "thumbnails"
rng = np.random.default_rng(0)


def lowcontrast(img):
    return (0.35 * img + 0.65 * 128).astype(np.uint8)


def shrunk(img):
    small = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    out = np.full_like(img, 128)
    h, w = small.shape[:2]
    out[(720 - h) // 2 : (720 - h) // 2 + h, (1280 - w) // 2 : (1280 - w) // 2 + w] = small
    return out


def cluttered(img):
    out = img.copy()
    for _ in range(45):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, 1180)), int(rng.integers(0, 640))
        if rng.random() < 0.5:
            cv2.circle(out, (x, y), int(rng.integers(15, 60)), c, -1)
        else:
            cv2.rectangle(out, (x, y), (x + int(rng.integers(30, 110)), y + int(rng.integers(20, 80))), c, -1)
    for i in range(14):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.putText(out, "free tips and tricks " * 2, (int(rng.integers(0, 600)), 40 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    return out


VARIANTS = {"lowcontrast": lowcontrast, "shrunk": shrunk, "cluttered": cluttered}
EXPECT = {  # which scores each damage should lower
    "lowcontrast": ["text_readability", "overall"],
    "shrunk": ["text_readability", "face_impact", "overall"],
    "cluttered": ["clutter", "focus", "overall"],
}
KEYS = ["overall", "focus", "text_readability", "face_impact", "clutter"]


def score(pipe, rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return pipe.analyze(buf.tobytes())["scores"]


def main(n: int) -> int:
    files = sorted(THUMBS.glob("*.jpg"))
    if not files:
        print("no thumbnails in", THUMBS)
        return 1
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]
    pipe = Pipeline()
    drops = {v: {k: [] for k in KEYS} for v in VARIANTS}
    print(f"{'image':24} {'variant':12} " + " ".join(f"{k[:7]:>8}" for k in KEYS) + "   (original -> damaged)")
    for f in files:
        img = ingest(f.read_bytes(), pipe.cfg["ingest"]).image
        base = score(pipe, img)
        print(f"{f.stem[:24]:24} {'original':12} " + " ".join(f"{'-' if base[k] is None else base[k]:>8}" for k in KEYS))
        for name, fn in VARIANTS.items():
            s = score(pipe, fn(img))
            cells = []
            for k in KEYS:
                if base[k] is None or s[k] is None:
                    cells.append(f"{'n/a':>8}")
                else:
                    drops[name][k].append(base[k] - s[k])
                    cells.append(f"{base[k]:>3}->{s[k]:<3}")
            print(f"{'':24} {name:12} " + " ".join(cells))

    print("\nMean drop (original minus damaged), positive = score fell as it should:")
    ok = True
    for name, per in drops.items():
        line = []
        for k in KEYS:
            v = per[k]
            if not v:
                continue
            fell = sum(1 for d in v if d > 0)
            mark = "*" if k in EXPECT[name] else " "
            line.append(f"{mark}{k}: {np.mean(v):+5.1f} ({fell}/{len(v)} fell)")
            if k in EXPECT[name] and (np.mean(v) < 8 or fell < 0.75 * len(v)):
                ok = False
        print(f"  {name:12} " + " | ".join(line))
    print("\n(* = the score this damage is meant to lower)")
    print("RESULT:", "separates good from bad" if ok else "WEAK separation, retune scoring.yaml")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
