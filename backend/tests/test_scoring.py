from pathlib import Path

import pytest

from api.schemas import AnalysisResult, Scores
from pipeline.config import load_yaml
from pipeline.orchestrator import Pipeline
from pipeline.store import SqliteStore
from scoring.engine import band, compute_scores, derive, ramp

CFG = load_yaml("scoring.yaml")
THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"


def good_metrics():
    return {
        "face": {"count": 1.0, "largest_area_pct": 9.0},
        "text": {"count": 2.0, "area_pct": 8.0, "word_count": 3.0, "largest_height_pct": 13.0, "mean_confidence": 0.97},
        "color": {"lightness_contrast": 28.0},
        "composition": {"concentration_top10": 0.45, "focal_points": 2.0, "top_peak_ratio": 0.4, "ui_overlap_pct": 1.0,
                        "spread": 0.33, "negative_space_pct": 40.0, "peak_count": 3.0},
        "attention_share": {"face": 30.0, "text": 25.0, "subject": 20.0, "background": 25.0},
    }


def good_regions():
    return [
        {"type": "face", "box": [100, 100, 400, 400], "intensity": 0.8},
        {"type": "text", "box": [500, 100, 1100, 250], "mobile_readable": True, "contrast": 12.0},
    ]


# ---------- ramps ----------

def test_ramp_and_band():
    assert ramp(5, 0, 10) == 0.5 and ramp(-3, 0, 10) == 0 and ramp(99, 0, 10) == 1
    assert ramp(4, 8, 2) == pytest.approx(2 / 3)  # lower is better
    assert ramp(3, 3, 3) == 1.0 and ramp(4, 3, 3) == 0.0  # degenerate range does not divide by zero
    assert band(40, 8, 30, 60, 88) == 1 and band(8, 8, 30, 60, 88) == 0 and band(88, 8, 30, 60, 88) == 0
    assert band(19, 8, 30, 60, 88) == pytest.approx(0.5) and band(74, 8, 30, 60, 88) == pytest.approx(0.5)


# ---------- derived values ----------

def test_derive_weights_readability_and_contrast_by_area():
    regions = [
        {"type": "text", "box": [0, 0, 100, 100], "mobile_readable": True, "contrast": 10.0},     # 10,000 px
        {"type": "text", "box": [0, 200, 100, 210], "mobile_readable": False, "contrast": 1.0},   # 1,000 px footnote
    ]
    d = derive({"attention_share": {"background": 40.0, "face": 5.0}}, regions)
    assert d["text_mobile_readable_ratio"] == pytest.approx(10 / 11)
    assert d["text_weighted_contrast"] == pytest.approx((10 * 10 + 1 * 1) / 11)
    assert d["background_share"] == 40.0 and d["face_share"] == 5.0
    assert "face_intensity" not in d


def test_unknown_emotion_counts_as_mildly_expressive_not_zero():
    assert derive({}, [{"type": "face", "intensity": None}])["face_intensity"] == 0.3


# ---------- scores ----------

def test_strong_thumbnail_scores_high_everywhere():
    s, d = compute_scores(good_metrics(), good_regions(), CFG)
    assert all(s[k] >= 85 for k in ("overall", "focus", "text_readability", "face_impact", "clutter")), s
    assert d["overall"]["penalties"] == []


def test_each_weakness_lowers_its_own_score():
    base, _ = compute_scores(good_metrics(), good_regions(), CFG)

    m = good_metrics(); m["composition"].update(concentration_top10=0.2, focal_points=8, top_peak_ratio=0.95)
    assert compute_scores(m, good_regions(), CFG)[0]["focus"] < base["focus"] - 25

    regions = [dict(r, mobile_readable=False, contrast=1.4) if r["type"] == "text" else r for r in good_regions()]
    assert compute_scores(good_metrics(), regions, CFG)[0]["text_readability"] < base["text_readability"] - 40

    m = good_metrics(); m["face"]["largest_area_pct"] = 0.6; m["attention_share"]["face"] = 3.0
    assert compute_scores(m, good_regions(), CFG)[0]["face_impact"] < base["face_impact"] - 40

    m = good_metrics(); m["text"].update(area_pct=40.0, word_count=20.0, count=14.0); m["composition"].update(peak_count=9.0, negative_space_pct=6.0)
    assert compute_scores(m, good_regions(), CFG)[0]["clutter"] < base["clutter"] - 50


def test_not_applicable_scores_are_none_and_left_out_of_overall():
    m = good_metrics(); m["face"]["count"] = 0.0
    s, d = compute_scores(m, [r for r in good_regions() if r["type"] != "face"], CFG)
    assert s["face_impact"] is None and d["face_impact"]["applicable"] is False and d["face_impact"]["note"] == "No face found"
    parts = {k: w for k, w in CFG["overall_weights"].items() if k != "face_impact"}
    expect = sum(s[k] * w for k, w in parts.items()) / sum(parts.values())
    assert s["overall"] == round(expect)

    m = good_metrics(); m["text"]["count"] = 0.0
    assert compute_scores(m, [r for r in good_regions() if r["type"] != "text"], CFG)[0]["text_readability"] is None


def test_missing_metrics_are_skipped_not_invented():
    m = good_metrics(); del m["color"]  # colour analyzer failed
    s, d = compute_scores(m, good_regions(), CFG)
    punch = next(c for c in d["focus"]["components"] if c["id"] == "punch")
    assert punch["value"] is None and s["focus"] is not None  # remaining components are re-weighted
    s_none, d_none = compute_scores({}, [], CFG)
    assert s_none["focus"] is None and s_none["clutter"] is None and s_none["overall"] is None
    assert d_none["focus"]["note"] == "Not enough data"


def test_empty_frame_penalty_only_hits_overall_and_is_explained():
    m = good_metrics(); m["composition"]["negative_space_pct"] = 90.0
    s, d = compute_scores(m, good_regions(), CFG)
    base, _ = compute_scores(good_metrics(), good_regions(), CFG)
    pen = d["overall"]["penalties"]
    assert pen and pen[0]["id"] == "empty_frame" and pen[0]["points"] == 30.0
    assert s["overall"] < base["overall"] - 25 and s["focus"] >= base["focus"] - 5


# ---------- config sanity ----------

def test_weights_are_complete_and_sum_to_one():
    assert sum(CFG["overall_weights"].values()) == pytest.approx(1.0)
    for name, spec in CFG["scores"].items():
        assert sum(c["weight"] for c in spec["components"]) == pytest.approx(1.0), name
        assert len({c["id"] for c in spec["components"]}) == len(spec["components"]), f"duplicate component id in {name}"
        for c in spec["components"]:
            assert ("bad" in c and "good" in c) or all(k in c for k in ("bad_low", "good_low", "good_high", "bad_high")), c["id"]
    assert set(CFG["overall_weights"]) == set(CFG["scores"])


def test_api_schema_accepts_missing_scores():
    s = Scores(overall=70, focus=80, text_readability=None, face_impact=None, clutter=60)
    assert s.text_readability is None
    AnalysisResult(id="x", scores=s)


@pytest.mark.skipif(not (THUMBS / "tech_c347oYQO57A.jpg").exists(), reason="sample thumbnails not downloaded")
def test_every_component_has_real_data_in_the_live_pipeline(tmp_path):
    """Guards against a typo in a metric path silently dropping a component."""
    out = Pipeline(store=SqliteStore(tmp_path)).analyze((THUMBS / "tech_c347oYQO57A.jpg").read_bytes())
    assert out["unavailable"] == []
    s, d = out["scores"], out["metrics"]["scores_detail"]
    assert all(s[k] is not None for k in ("overall", "focus", "text_readability", "face_impact", "clutter"))
    for name, det in d.items():
        for c in det["components"]:
            assert c["value"] is not None, f"{name}.{c['id']} ({c['metric']}) has no data"
    AnalysisResult.model_validate(out)
