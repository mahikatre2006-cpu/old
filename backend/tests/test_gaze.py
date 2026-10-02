import io
import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.schemas import GazeAggregate, GazeSessionOut
from evaluation.gaze import build_ground_truth, first_look, gaussian_map, idt
from evaluation.metrics import cc, kld, nss, sim
from pipeline.config import load_yaml

CFG = load_yaml("evaluation.yaml")


# ---------- metrics (known values) ----------

def test_cc_is_pearson():
    a = np.random.default_rng(0).random((20, 20))
    assert cc(a, a) == pytest.approx(1.0) and cc(a, -a) == pytest.approx(-1.0) and cc(a, 3 * a + 7) == pytest.approx(1.0)
    assert cc(a, np.full_like(a, 0.5)) == 0.0  # a flat map correlates with nothing


def test_nss_matches_a_hand_computed_value_and_clamps_out_of_range_points():
    m = np.zeros((10, 10)); m[5, 5] = 1.0
    z_peak = (1 - 0.01) / math.sqrt(0.01 * 0.99)  # mean 0.01, population std
    assert nss(m, [(5, 5)]) == pytest.approx(z_peak, rel=1e-6)
    assert nss(m, [(0, 0)]) == pytest.approx(-0.01 / math.sqrt(0.0099), rel=1e-6)  # background is below average
    assert nss(m, [(5, 5), (0, 0)]) == pytest.approx((z_peak + (-0.01 / math.sqrt(0.0099))) / 2)
    assert nss(m, [(500, -20)]) == nss(m, [(9, 0)])  # clamped to the border
    assert nss(m, []) == 0.0 and nss(np.ones((4, 4)), [(1, 1)]) == 0.0
    assert nss(m, [(5, 5), (0, 0)], weights=[1, 0]) == pytest.approx(z_peak)


def test_sim_and_kld_bounds():
    a = np.zeros((8, 8)); a[1, 1] = 1.0
    b = np.zeros((8, 8)); b[6, 6] = 1.0
    assert sim(a, a) == pytest.approx(1.0) and sim(a, b) == pytest.approx(0.0)
    assert kld(a, a) == pytest.approx(0.0, abs=1e-6) and kld(b, a) > 5
    assert 0 < sim(a, 0.5 * a + 0.5 * b) < 1


# ---------- I-DT ----------

def trace(segments, hz=30.0, jitter=4.0, seed=0):
    """segments: [(x, y, n_samples)] -> samples (t_ms, x, y) at hz with small noise."""
    rng = np.random.default_rng(seed)
    out, t = [], 0.0
    for x, y, n in segments:
        for _ in range(n):
            out.append((t, x + rng.normal(0, jitter), y + rng.normal(0, jitter)))
            t += 1000 / hz
    return out


def test_idt_finds_two_fixations_with_the_right_centres_and_durations():
    fx = idt(trace([(200, 150, 14), (900, 500, 20)]), 120, 100)
    assert len(fx) == 2
    assert abs(fx[0]["x"] - 200) < 10 and abs(fx[0]["y"] - 150) < 10 and abs(fx[1]["x"] - 900) < 10
    assert fx[0]["duration_ms"] == pytest.approx(13 * 1000 / 30, abs=40) and fx[1]["duration_ms"] > fx[0]["duration_ms"]
    assert fx[0]["start_ms"] < fx[1]["start_ms"]


def test_idt_ignores_glances_shorter_than_the_minimum_and_fast_sweeps():
    assert idt(trace([(300, 300, 2)]), 120, 100) == []  # 33 ms is a blink of the eye, not a fixation
    sweep = [(t, 100 + t * 2.0, 360.0) for t in np.arange(0, 600, 33.3)]  # constant fast drift
    assert idt(sweep, 120, 100) == []
    assert idt([], 120, 100) == []


def test_idt_respects_the_dispersion_threshold():
    pts = [(i * 33.3, 500 + (40 if i % 2 else -40), 300.0) for i in range(15)]  # +-40 px: dispersion 80
    assert len(idt(pts, 120, 100)) == 1 and idt(pts, 60, 100) == []


# ---------- ground truth ----------

def test_gaussian_map_peaks_at_the_points_and_weights_matter():
    m = gaussian_map([(320, 180), (960, 540)], [1.0, 0.5], 60, (160, 90))
    assert m.shape == (90, 160) and m.max() == 1.0
    assert m[22, 40] == pytest.approx(1.0, abs=0.02)  # (320, 180) in the frame -> (40, 22) in the 160x90 map: the heavy point
    assert m[67, 120] == pytest.approx(0.5, abs=0.02)  # (960, 540) -> (120, 67): half the weight
    assert m[80, 10] < 0.01 and m[45, 80] < 0.01  # nowhere else
    assert gaussian_map([], [], 60, (160, 90)).max() == 0


def test_click_ground_truth_orders_by_the_order_field_and_downweights_the_second_look():
    pts = [{"x": 900, "y": 400, "order": 2}, {"x": 300, "y": 200, "order": 1}]
    t = build_ground_truth("click", pts, CFG)
    assert [f["order"] for f in t["fixations"]] == [1, 2] and t["fixations"][0]["x"] == 300  # sorted, first look first
    assert t["weights"] == [1.0, 0.7]
    assert t["gt"][int(200 * 90 / 720), int(300 * 160 / 1280)] > t["gt"][int(400 * 90 / 720), int(900 * 160 / 1280)]


def test_webgazer_ground_truth_drops_wild_samples_and_complains_when_too_few_remain():
    good = [{"x": s[1], "y": s[2], "t": s[0]} for s in trace([(640, 360, 20)])]
    wild = [{"x": 5000.0, "y": 5000.0, "t": 1000 + i} for i in range(5)]
    t = build_ground_truth("webgazer", good + wild, CFG)
    assert len(t["fixations"]) == 1 and abs(t["fixations"][0]["x"] - 640) < 10
    assert t["warnings"] == []  # only a few strays: not worth a warning
    many_wild = good[:16] + [{"x": 9000.0, "y": 100.0, "t": 5000 + i} for i in range(30)]
    assert "outside the thumbnail" in build_ground_truth("webgazer", many_wild, CFG)["warnings"][0]
    with pytest.raises(ValueError, match="at least 15"):
        build_ground_truth("webgazer", good[:6], CFG)
    scattered = [{"x": 100.0 + (i % 2) * 900, "y": 100.0 + (i % 3) * 250, "t": i * 33.0} for i in range(40)]
    with pytest.raises(ValueError, match="No fixations"):
        build_ground_truth("webgazer", scattered, CFG)
    with pytest.raises(ValueError):
        build_ground_truth("click", [{"x": float("nan"), "y": 1.0}], CFG)


def test_first_look_hit_rules():
    path = [{"x": 300, "y": 200, "order": 1}, {"x": 900, "y": 300, "order": 2}, {"x": 600, "y": 600, "order": 3}]
    assert first_look([{"x": 340, "y": 220}], path, 160)["first_look_hit"] is True
    far = first_look([{"x": 880, "y": 310}], path, 160)  # landed on our SECOND fixation
    assert far["first_look_hit"] is False and far["first_look_hit_top3"] is True and far["first_look_distance_px"] > 500
    assert first_look([{"x": 20, "y": 700}], path, 160)["first_look_hit_top3"] is False
    assert first_look([], path, 160)["first_look_hit"] is None


# ---------- API, end to end ----------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["THUMB_DATA_DIR"] = str(tmp_path_factory.mktemp("gaze"))
    os.environ["DATABASE_URL"] = ""
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def analysed(client):
    """A thumbnail with one obvious focus (red disc on the left); returns the analysis incl. its predicted scanpath."""
    img = np.full((720, 1280, 3), 110, np.uint8)
    cv2.circle(img, (330, 360), 170, (255, 255, 255), -1)
    cv2.circle(img, (330, 360), 120, (220, 20, 20), -1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r = client.post("/analyze", files={"file": ("t.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    return r.json()


def dwell(x, y, n=24, t0=0.0):
    return [{"x": x + (i % 3) * 3.0, "y": y + (i % 2) * 3.0, "t": t0 + i * 33.3} for i in range(n)]


def test_webgazer_session_on_the_predicted_peak_beats_one_in_an_empty_corner(client, analysed):
    first = analysed["scanpath"][0]
    on = client.post("/gaze-session", json={"thumbnail_id": analysed["id"], "method": "webgazer", "points": dwell(first["x"], first["y"]),
                                              "screen": {"width": 1920, "height": 1080}, "participant": "p1"})
    assert on.status_code == 200, on.text
    a = on.json(); GazeSessionOut.model_validate(a)
    off = client.post("/gaze-session", json={"thumbnail_id": analysed["id"], "method": "webgazer", "points": dwell(1200, 40), "participant": "p2"}).json()

    assert a["method"] == "webgazer" and a["n_fixations"] == 1 and a["first_look_hit"] is True and a["first_look_distance_px"] < 30
    assert a["nss"] > 1.0 and a["cc"] > 0.3 and a["ground_truth_png"] and a["session_id"]
    assert off["first_look_hit"] is False and off["nss"] < a["nss"] - 1.0 and off["cc"] < a["cc"]
    assert off["nss"] < 0.5  # an empty corner is no better than chance


def test_click_session_scores_first_look_clicks(client, analysed):
    p = analysed["scanpath"]
    good = client.post("/gaze-session", json={"thumbnail_id": analysed["id"], "method": "click",
                                                "points": [{"x": p[0]["x"], "y": p[0]["y"], "order": 1}, {"x": p[1]["x"], "y": p[1]["y"], "order": 2}]}).json()
    bad = client.post("/gaze-session", json={"thumbnail_id": analysed["id"], "method": "click",
                                               "points": [{"x": 1200, "y": 650, "order": 1}, {"x": 1150, "y": 60, "order": 2}]}).json()
    assert good["first_look_hit"] is True and good["n_fixations"] == 2 and [f["order"] for f in good["fixations"]] == [1, 2]
    assert good["nss"] > bad["nss"] and bad["first_look_hit"] is False


def test_bad_requests_get_clear_errors(client, analysed):
    base = {"thumbnail_id": analysed["id"], "method": "webgazer"}
    assert client.post("/gaze-session", json={**base, "thumbnail_id": "0" * 64, "points": dwell(300, 300)}).status_code == 404
    assert client.post("/gaze-session", json={**base, "points": []}).status_code == 422  # nothing sent
    r = client.post("/gaze-session", json={**base, "points": dwell(300, 300, n=5)})
    assert r.status_code == 422 and "at least 15" in r.json()["detail"]
    assert client.post("/gaze-session", json={**base, "points": [{"x": 1, "y": 2, "t": 3}] * 3001}).status_code == 422
    assert client.post("/gaze-session", content='{"thumbnail_id":"%s","method":"webgazer","points":[{"x":NaN,"y":1,"t":0}]}' % analysed["id"],
                       headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/gaze-session", json={**base, "method": "telepathy", "points": dwell(1, 1)}).status_code == 422


def test_sessions_are_saved_and_aggregated_per_thumbnail(client, analysed):
    tid = analysed["id"]
    agg = client.get(f"/gaze-sessions/{tid}")
    assert agg.status_code == 200
    body = agg.json(); GazeAggregate.model_validate(body)
    assert body["n_sessions"] >= 4 and {s["participant"] for s in body["sessions"]} >= {"p1", "p2"}
    assert body["first_look_hit_rate"] is not None and 0 < body["first_look_hit_rate"] < 1
    assert body["aggregate_png"] and body["aggregate_cc"] is not None and body["aggregate_nss"] is not None
    assert client.get("/gaze-sessions/" + "f" * 64).status_code == 404
    empty = Image.new("RGB", (1280, 720), (5, 90, 5)); buf = io.BytesIO(); empty.save(buf, "PNG")
    other = client.post("/analyze", files={"file": ("o.png", buf.getvalue(), "image/png")}).json()["id"]
    assert client.get(f"/gaze-sessions/{other}").json()["n_sessions"] == 0
