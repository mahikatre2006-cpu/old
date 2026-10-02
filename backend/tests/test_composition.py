from pathlib import Path

import numpy as np
import pytest

from analyzers.base import Analyzer
from analyzers.composition import CompositionAnalyzer, find_peaks, normalized_entropy, thirds_distance
from pipeline.config import analyzers_config, load_yaml
from pipeline.context import AnalyzerResult
from pipeline.orchestrator import Pipeline
from pipeline.preprocess import build_context
from pipeline.store import SqliteStore

CFG = analyzers_config()
H, W = 720, 1280


def blob(cx, cy, sigma=50.0, amp=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))).astype(np.float32)


def run(fused, img=None):
    img = np.full((H, W, 3), 120, np.uint8) if img is None else img
    ctx = build_context(img, "c", CFG["preprocess"])
    ctx.fused = fused
    return CompositionAnalyzer().run(img, ctx).metrics


# ---------- helpers ----------

def test_thirds_distance():
    assert thirds_distance(1 / 3, 1 / 3) == pytest.approx(0)
    assert thirds_distance(0.5, 0.5) == pytest.approx(np.hypot(1 / 6, 1 / 6))
    assert thirds_distance(0.0, 0.0) > thirds_distance(0.3, 0.3)


def test_find_peaks_returns_each_separated_peak_once_strongest_first():
    m = np.maximum.reduce([blob(300, 200), blob(950, 250, amp=0.7), blob(640, 600, amp=0.4)])
    peaks = find_peaks(m)
    assert len(peaks) == 3 and [p["strength"] for p in peaks] == sorted((p["strength"] for p in peaks), reverse=True)
    assert abs(peaks[0]["x"] - 300) < 15 and abs(peaks[1]["x"] - 950) < 15 and peaks[0]["strength"] == 1.0
    assert find_peaks(np.zeros((H, W), np.float32)) == []
    assert len(find_peaks(blob(640, 360, sigma=300))) == 1  # one broad hill is one peak, not many plateau pixels


def test_entropy_is_low_for_one_hotspot_and_high_for_uniform():
    assert normalized_entropy(blob(640, 360, sigma=30)) < 0.65
    assert normalized_entropy(np.full((H, W), 0.5, np.float32)) == pytest.approx(1.0, abs=1e-3)


# ---------- analyzer ----------

def test_single_strong_focus_on_a_thirds_point_is_concentrated_and_on_thirds():
    m = run(blob(W / 3, H / 3, sigma=40))
    assert m["focal_points"] == 1 and m["rot_dist_peak"] < 0.02 and m["rot_dist_com"] < 0.03
    # a single Gaussian hotspot sits near 0.63 normalized entropy; a uniform map is 1.0
    assert m["entropy"] < 0.7 and m["concentration_top10"] > 0.5 and m["spread"] < 0.15


def test_centered_focus_is_off_thirds_and_centered():
    m = run(blob(W / 2, H / 2, sigma=40))
    assert m["com_offset_center"] < 0.02 and m["rot_dist_peak"] == pytest.approx(np.hypot(1 / 6, 1 / 6), abs=0.02)


def test_two_competing_focal_points_and_clutter_ordering():
    two = run(np.maximum(blob(300, 300), blob(950, 450)))
    one = run(blob(300, 300))
    assert two["focal_points"] == 2 and two["top_peak_ratio"] > 0.9
    assert one["focal_points"] == 1
    assert two["entropy"] > one["entropy"] and two["spread"] > one["spread"]
    busy = run(np.clip(sum(blob(x, y, 45) for x in range(150, 1200, 250) for y in range(120, 700, 200)), 0, 1))
    assert busy["focal_points"] >= 8 or busy["peak_count"] >= 8
    assert busy["entropy"] > two["entropy"]


def test_ui_overlap_detects_attention_under_the_duration_badge():
    under = run(blob(1190, 650, sigma=30))  # bottom-right, where the duration badge sits
    clear = run(blob(300, 200, sigma=30))
    assert under["ui_overlap_pct"] > 40 and clear["ui_overlap_pct"] < 1


def test_negative_space_needs_calm_and_unattended_pixels():
    plain = run(blob(300, 300, sigma=30))  # flat grey image: most of the frame is calm and ignored
    assert plain["negative_space_pct"] > 80
    noise = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)  # textured everywhere
    assert run(blob(300, 300, sigma=30), noise)["negative_space_pct"] < 5


def test_missing_or_empty_fused_map_degrades_gracefully():
    img = np.full((H, W, 3), 120, np.uint8)
    ctx = build_context(img, "c", CFG["preprocess"])
    assert CompositionAnalyzer().run(img, ctx).metrics["focal_points"] == 0
    assert run(np.zeros((H, W), np.float32))["focal_points"] == 0


# ---------- orchestration: stage 2 receives the fused map from stage 1 ----------

class Stage1(Analyzer):
    name = "s1"

    def run(self, img, ctx):
        return AnalyzerResult(attention_map=blob(400, 300))


seen = {}


class Stage2(Analyzer):
    name = "s2"
    metrics_group = "composition"

    def run(self, img, ctx):
        seen["fused"] = ctx.fused
        return AnalyzerResult(metrics={"saw_fused": float(ctx.fused is not None)})


def test_pipeline_gives_stage2_the_fused_map(tmp_path):
    import io

    from PIL import Image

    cfg = CFG | {"analyzers": {
        "s1": {"module": "tests.test_composition:Stage1", "stage": 1},
        "s2": {"module": "tests.test_composition:Stage2", "stage": 2},
    }}
    fusion = load_yaml("fusion.yaml") | {"weights": {"s1": 1.0}}
    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), (9, 9, 9)).save(buf, "PNG")
    out = Pipeline(store=SqliteStore(tmp_path), cfg=cfg, fusion_cfg=fusion).analyze(buf.getvalue())
    assert out["metrics"]["composition"]["saw_fused"] == 1.0
    fused = seen["fused"]
    assert fused.shape == (H, W) and fused[300, 400] == pytest.approx(1.0)


THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"


@pytest.mark.skipif(not (THUMBS / "tech_c347oYQO57A.jpg").exists(), reason="sample thumbnails not downloaded")
def test_full_default_pipeline_reports_every_section(tmp_path):
    p = Pipeline(store=SqliteStore(tmp_path))
    assert {k: v for k, v in p.status.items()} == {n: "loaded" for n in ("saliency", "face", "text", "subject", "color", "composition")}
    out = p.analyze((THUMBS / "tech_c347oYQO57A.jpg").read_bytes())
    assert out["unavailable"] == []
    for group in ("face", "text", "subject", "color", "composition"):
        assert out["metrics"][group], group
    assert out["metrics"]["composition"]["focal_points"] >= 1
    assert out["metrics"]["color"]["palette_hex"] and out["metrics"]["subject"]["area_pct"] > 0
    assert {r["type"] for r in out["regions"]} >= {"face", "subject", "background"}
    shares = out["metrics"]["attention_share"]
    assert sum(shares.values()) == pytest.approx(100, abs=0.3)
    assert out["timing_ms"]["total"] < 4000
