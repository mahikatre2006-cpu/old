import numpy as np
import pytest

from fusion.fuse import fuse
from fusion.scanpath import predict_scanpath
from fusion.shares import compute_shares
from pipeline.context import AnalyzerResult, Region

H, W = 720, 1280


def blob(cx, cy, sigma=60.0, amp=1.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return (amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))).astype(np.float32)


def res(m=None, regions=()):
    return AnalyzerResult(attention_map=m, regions=list(regions))


# ---------- fusion ----------

def test_fuse_weighted_sum_is_normalized():
    results = {"saliency": res(blob(300, 300)), "face": res(blob(900, 300))}
    fused, used = fuse(results, {"saliency": 0.7, "face": 0.3})
    assert fused.min() == 0 and fused.max() == pytest.approx(1.0)
    assert fused[300, 300] > fused[300, 900] > fused[600, 640]  # heavier weight wins
    assert used == pytest.approx({"saliency": 0.7, "face": 0.3})


def test_fuse_renormalizes_when_a_map_is_missing_or_failed():
    results = {"saliency": res(blob(300, 300)), "face": res(None)}  # face produced no map
    fused, used = fuse(results, {"saliency": 0.55, "face": 0.15, "text": 0.1})
    assert used == {"saliency": 1.0}  # weights renormalized over what exists
    assert np.allclose(fused, results["saliency"].attention_map, atol=1e-6)


def test_fuse_keeps_component_amplitude_so_a_weak_map_stays_weak():
    strong, weak = blob(300, 300, amp=1.0), blob(900, 300, amp=0.2)  # e.g. a big face vs a tiny face
    fused, _ = fuse({"face": res(np.maximum(strong, weak))}, {"face": 1.0})
    assert fused[300, 300] == pytest.approx(1.0) and fused[300, 900] == pytest.approx(0.2, abs=0.02)


def test_fuse_with_nothing_returns_none():
    assert fuse({}, {"saliency": 1.0}) == (None, {})


# ---------- scanpath ----------

def test_scanpath_visits_strongest_peak_first_then_others():
    fused = np.maximum.reduce([blob(300, 200, amp=1.0), blob(950, 250, amp=0.7), blob(640, 600, amp=0.5)])
    path = predict_scanpath(fused)
    assert [p["order"] for p in path] == list(range(1, len(path) + 1))
    first, second, third = path[:3]
    assert abs(first["x"] - 300) < 15 and abs(first["y"] - 200) < 15
    assert abs(second["x"] - 950) < 15 and abs(second["y"] - 250) < 15
    assert abs(third["x"] - 640) < 15 and abs(third["y"] - 600) < 15


def test_scanpath_inhibition_of_return_spreads_fixations():
    path = predict_scanpath(blob(640, 360, sigma=200))  # one broad peak
    assert 3 <= len(path) <= 5
    pts = [(p["x"], p["y"]) for p in path]
    assert len(set(pts)) == len(pts)
    dmin = min(np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1 :])
    assert dmin > 50  # no two fixations on top of each other


def test_scanpath_drops_weak_fixations_but_keeps_minimum():
    path = predict_scanpath(blob(300, 300, sigma=25), max_fixations=5, min_fixations=3)
    assert len(path) >= 3
    strong = predict_scanpath(np.maximum.reduce([blob(200, 200), blob(600, 200), blob(1000, 200), blob(400, 500), blob(900, 550)]))
    assert len(strong) == 5


def test_scanpath_coordinates_inside_frame_and_flat_map_empty():
    for p in predict_scanpath(blob(1270, 715, sigma=30)):
        assert 0 <= p["x"] < W and 0 <= p["y"] < H
    assert predict_scanpath(np.zeros((H, W), np.float32)) == []


# ---------- shares ----------

def test_shares_sum_to_100_and_respect_priority():
    fused = blob(300, 300, sigma=80) + 0.3 * blob(900, 300, sigma=80)
    face = Region("face", box=(200, 200, 400, 400), attrs={"emotion": "joy"})
    subject_mask = np.zeros((H, W), bool)
    subject_mask[100:500, 100:600] = True  # overlaps the face box entirely
    subject = Region("subject", mask=subject_mask)
    text = Region("text", box=(800, 200, 1000, 400), attrs={"text": "HELLO", "mobile_readable": False})
    results = {"face": res(None, [face]), "subject": res(None, [subject]), "text": res(None, [text])}

    regions, shares = compute_shares(fused, results, ["face", "text", "subject"])
    assert sum(shares.values()) == pytest.approx(100, abs=0.2)
    assert shares["face"] > 0 and shares["text"] > 0 and shares["subject"] > 0
    assert shares["face"] > shares["subject"]  # face pixels are not double-counted in subject

    by_type = {r["type"]: r for r in regions}
    assert by_type["face"]["emotion"] == "joy"
    assert by_type["text"]["text"] == "HELLO" and by_type["text"]["mobile_readable"] is False
    assert by_type["subject"]["box"] is not None  # derived from the mask
    assert by_type["background"]["box"] is None


def test_shares_with_no_regions_is_all_background():
    regions, shares = compute_shares(blob(640, 360), {"saliency": res(blob(640, 360))}, ["face", "text", "subject"])
    assert shares["background"] == pytest.approx(100, abs=0.1) and len(regions) == 1


def test_heatmap_png_uses_inferno_and_is_transparent_where_cold():
    import base64

    import cv2

    from fusion.heatmap import encode_heatmap_png

    m = blob(640, 360, sigma=80)
    png = cv2.imdecode(np.frombuffer(base64.b64decode(encode_heatmap_png(m)), np.uint8), cv2.IMREAD_UNCHANGED)
    hot, cold = png[180, 320], png[5, 5]  # BGRA
    assert hot[3] > 150 and cold[3] < 10  # opaque at the peak, see-through far away
    assert hot[2] > hot[0] and hot[1] > 100  # inferno's hot end is yellow/orange (R,G high, B low); jet's red has G~0
