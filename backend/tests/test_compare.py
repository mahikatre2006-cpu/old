import io
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.schemas import CompareResult
from fusion.heatmap import decode_attention_png, encode_attention_png, encode_diff_png
from pipeline.compare import explain, labels_for, pick_winner, rules_that_differ, score_deltas

THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"
needs_data = pytest.mark.skipif(not (THUMBS / "finance_gMyVHBhDvz4.jpg").exists(), reason="sample thumbnails not downloaded")
OW = {"focus": 0.3, "text_readability": 0.25, "face_impact": 0.2, "clutter": 0.25}


def result(overall, focus=70, highs=0, rec_ids=(), detail=None, rid="x"):
    return {
        "id": rid,
        "scores": {"overall": overall, "focus": focus, "text_readability": 80, "face_impact": None, "clutter": 60},
        "recommendations": [{"id": "h%d" % i, "severity": "high", "message": "m"} for i in range(highs)]
        + [{"id": i, "severity": "medium", "message": f"fix {i}"} for i in rec_ids],
        "metrics": {"scores_detail": detail or {}},
    }


def comp(cid, label, value, score, weight=0.5, goal="at least 10"):
    return {"id": cid, "label": label, "value": value, "score": score, "weight": weight, "goal": goal}


def detail(*comps, name="text_readability"):
    return {name: {"applicable": True, "value": 70, "components": list(comps)}}


# ---------- pure logic ----------

def test_labels_and_deltas_handle_missing_scores():
    assert labels_for(3) == ["A", "B", "C"]
    a, b = result(80, 70), result(70, 75)
    b["scores"]["face_impact"] = 50  # A has none for this one
    d = score_deltas([a, b])
    assert d[0] == {"overall": 0, "focus": 0, "text_readability": 0, "face_impact": None, "clutter": 0}
    assert d[1]["overall"] == -10 and d[1]["focus"] == 5 and d[1]["face_impact"] is None


def test_winner_is_highest_overall_and_close_calls_are_flagged():
    win, margin, close = pick_winner([result(60), result(75), result(70)])
    assert (win, margin, close) == (1, 5, False)
    win, margin, close = pick_winner([result(70, focus=60), result(72, focus=80)])
    assert win == 1 and margin == 2 and close is True


def test_exact_tie_is_broken_by_focus_then_by_fewer_high_severity_fixes():
    assert pick_winner([result(70, focus=60), result(70, focus=65)])[0] == 1
    assert pick_winner([result(70, focus=60, highs=2), result(70, focus=60, highs=0)])[0] == 1
    assert pick_winner([result(70, focus=60, highs=0), result(70, focus=60, highs=2)])[0] == 0


def test_rules_that_differ_ignores_shared_issues_and_generic_tips():
    rs = [result(70, rec_ids=["shared", "only_a", "improve_x"]), result(65, rec_ids=["shared", "only_b", "keep_y"])]
    diff = {d["id"]: d["present_in"] for d in rules_that_differ(rs, ["A", "B"])}
    assert diff == {"only_a": ["A"], "only_b": ["B"]}


def test_explain_names_the_components_and_issues_that_decide_it():
    w = result(85, detail=detail(comp("contrast", "Text contrast (WCAG)", 12.0, 1.0), comp("size", "Largest text is big enough", 14.0, 1.0)))
    r = result(60, rec_ids=["text_low_contrast"], detail=detail(comp("contrast", "Text contrast (WCAG)", 2.0, 0.1), comp("size", "Largest text is big enough", 13.0, 0.95)))
    reasons = explain([w, r], ["A", "B"], 0, 25, False, OW)
    assert reasons[0] == "A beats B by 25 points overall (85 vs 60)."
    assert any("text contrast (WCAG), A 12 vs B 2" in x for x in reasons)  # the big driver is named with both numbers
    assert not any("big enough" in x for x in reasons)  # a negligible difference is not listed
    assert any("B has an issue A avoids: fix text_low_contrast" in x for x in reasons)


def test_explain_says_so_when_it_is_a_close_call_and_credits_the_runner_up():
    w = result(71, detail=detail(comp("a", "Strong contrast", 30.0, 1.0)))
    r = result(70, detail=detail(comp("a", "Strong contrast", 12.0, 0.4), comp("b", "Few words", 3.0, 1.0)))
    w["metrics"]["scores_detail"]["text_readability"]["components"].append(comp("b", "Few words", 9.0, 0.2))
    reasons = explain([w, r], ["A", "B"], 0, 1, True, OW)
    assert "only 1 point overall" in reasons[0] and "close call" in reasons[0]
    assert any("B is better at" in x and "few words" in x for x in reasons)
    back = next(i for i, x in enumerate(reasons) if "B is better at" in x)
    assert all("has an issue" not in x for x in reasons[:back])  # trade-offs come before the issue list


# ---------- maps ----------

def test_attention_png_round_trips_within_8_bit_precision():
    m = np.zeros((720, 1280), np.float32)
    m[300:400, 600:700] = 1.0
    back = decode_attention_png(encode_attention_png(m))
    assert back.shape == (90, 160) and back.max() == pytest.approx(1.0, abs=0.01) and back[10, 10] == 0
    assert back[40, 80] > 0.9  # the hot block survives the downscale


def test_difference_map_colours_gain_and_loss_and_is_transparent_where_equal():
    base, other = np.zeros((90, 160), np.float32), np.zeros((90, 160), np.float32)
    base[10:30, 10:40] = 1.0  # A looks here
    other[60:80, 100:140] = 1.0  # B looks here instead
    png = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(encode_diff_png(base, other)), np.uint8), cv2.IMREAD_UNCHANGED)
    assert png.shape == (360, 640, 4)
    lost, gained, same = png[80, 100], png[280, 480], png[180, 320]  # BGRA at (A-only, B-only, neither)
    assert gained[3] > 150 and gained[2] > gained[0]   # B gains: orange (R > B)
    assert lost[3] > 150 and lost[0] > lost[2]         # B loses: blue (B > R)
    assert same[3] == 0
    assert encode_diff_png(base, base) and cv2.imdecode(np.frombuffer(__import__("base64").b64decode(encode_diff_png(base, base)), np.uint8), cv2.IMREAD_UNCHANGED)[..., 3].max() == 0


# ---------- API ----------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["THUMB_DATA_DIR"] = str(tmp_path_factory.mktemp("cmp"))
    os.environ["DATABASE_URL"] = ""
    from api.main import app

    with TestClient(app) as c:
        yield c


def png(color, size=(1280, 720)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def post(client, blobs):
    return client.post("/compare", files=[("files", (f"v{i}.png", b, "image/png")) for i, b in enumerate(blobs)])


def test_needs_between_two_and_four_images(client):
    assert post(client, [png((9, 9, 9))]).status_code == 422
    five = [png((i * 20, 5, 5)) for i in range(5)]
    r = post(client, five)
    assert r.status_code == 422 and "2 to 4" in r.json()["detail"]


def test_rejects_identical_images_and_names_them(client):
    r = post(client, [png((10, 20, 30)), png((200, 20, 30)), png((10, 20, 30))])
    assert r.status_code == 422 and "A and C" in r.json()["detail"]


def test_bad_file_error_says_which_variant(client):
    r = post(client, [png((10, 20, 30)), b"not an image"])
    assert r.status_code == 422 and r.json()["detail"].startswith("Variant B:")


@needs_data
def test_end_to_end_ab_with_a_known_winner(client):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibrate_scores as cs
    from pipeline.config import analyzers_config
    from pipeline.ingest import ingest

    img = ingest((THUMBS / "finance_gMyVHBhDvz4.jpg").read_bytes(), analyzers_config()["ingest"]).image
    enc = lambda rgb: cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
    # the damaged one is uploaded FIRST, so the winner is not simply "variant A"
    blobs = [enc(cs.cluttered(img)), enc(img), enc(cs.lowcontrast(img))]  # generated once: cluttered() is random
    r = post(client, blobs)
    assert r.status_code == 200, r.text
    body = r.json()
    CompareResult.model_validate(body)

    assert body["labels"] == ["A", "B", "C"] and body["winner_label"] == "B" and body["close"] is False and body["margin"] >= 10
    assert body["winner_id"] == body["variants"][1]["id"]
    assert body["score_deltas"][0]["overall"] == 0 and body["score_deltas"][1]["overall"] > 10
    assert len(body["difference_maps"]) == 3 and body["difference_maps"][0] is None and all(body["difference_maps"][1:])
    assert body["reasons"][0].startswith("B beats") and len(body["reasons"]) >= 3
    assert "{" not in " ".join(body["reasons"])
    assert {d["id"] for d in body["rules_differ"]}  # the variants really do need different fixes
    for v in body["variants"]:
        assert v["attention_png"] and v["scores"]["overall"] is not None

    db = sqlite3.connect(client.app.state.pipeline.store.db_path)
    row = db.execute("SELECT winner_id FROM comparisons WHERE id = ?", (body["id"],)).fetchone()
    assert row and row[0] == body["winner_id"]

    again = post(client, blobs).json()
    assert again["id"] == body["id"] and all(v["cached"] for v in again["variants"])  # repeat compare is served from cache
