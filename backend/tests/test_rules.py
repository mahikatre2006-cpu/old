import copy
from pathlib import Path

import numpy as np
import pytest

from analyzers.subject import edge_contrast
from api.schemas import AnalysisResult
from pipeline.config import load_yaml
from pipeline.context import AnalyzerResult, Region
from pipeline.orchestrator import Pipeline
from pipeline.store import SqliteStore
from scoring.engine import compute_scores
from scoring.rules import (
    MissingValue,
    annotate_regions,
    evaluate,
    flatten_metrics,
    focal_description,
    safe_eval,
    validate_rules,
)

RULES = load_yaml("rules.yaml")
SCORING = load_yaml("scoring.yaml")
THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"


# ---------- safe expression evaluator ----------

def test_safe_eval_supports_the_documented_language():
    ns = {"a": 4.0, "b": None, "t": "high"}
    assert safe_eval("a > 3 and a < 5", ns) is True
    assert safe_eval("min(a, 2) + abs(-1)", ns) == 3
    assert safe_eval("'high' if a > 3 else 'low'", ns) == "high"
    assert safe_eval("b is not None and b > 1", ns) is False
    assert safe_eval("not (a == 4)", ns) is False
    assert safe_eval(0.5, ns) == 0.5


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "().__class__",
        "a.real",
        "[1, 2][0]",
        "(lambda: 1)()",
        "[x for x in (1, 2)]",
        "open('x')",
        "a ** 2 ** 9",
        "f'{a}'",
    ],
)
def test_safe_eval_rejects_everything_outside_the_whitelist(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr, {"a": 1.0})


def test_missing_names_raise_and_comparing_with_none_is_a_type_error():
    with pytest.raises(MissingValue):
        safe_eval("face_count > 1", {})
    with pytest.raises(TypeError):
        safe_eval("b > 1", {"b": None})


# ---------- namespace ----------

def test_flatten_metrics_keeps_only_finite_numbers():
    flat = flatten_metrics({"text": {"count": 3, "x": float("nan"), "ok": True, "palette": ["#fff"]}, "scores_detail": {"focus": {"a": 1}}, "weird": 5})
    assert flat == {"text_count": 3.0}


def test_focal_description_names_what_each_peak_sits_on():
    metrics = {"composition": {"peaks": [
        {"x": 300, "y": 300, "strength": 1.0}, {"x": 900, "y": 200, "strength": 0.9},
        {"x": 640, "y": 650, "strength": 0.8}, {"x": 100, "y": 650, "strength": 0.7}, {"x": 1000, "y": 650, "strength": 0.3}]}}
    regions = [{"type": "face", "box": [200, 200, 400, 400]}, {"type": "text", "box": [800, 100, 1100, 300], "text": "JUNGLE"}]
    d = focal_description(metrics, regions)
    assert d.startswith("the face") and 'the text "JUNGLE"' in d and "the background (2 spots)" in d
    assert focal_description({"composition": {"peaks": []}}, regions) == "several places"


def test_annotate_regions_computes_text_overlap_with_the_subject():
    mask = np.zeros((720, 1280), bool)
    mask[:, :640] = True  # subject = left half
    results = {"subject": AnalyzerResult(regions=[Region("subject", box=(0, 0, 640, 720), mask=mask)])}
    regions = [{"type": "text", "box": [100, 100, 300, 200]}, {"type": "text", "box": [500, 100, 780, 200]}, {"type": "text", "box": [800, 100, 900, 200]},
               {"type": "face", "box": [0, 0, 10, 10]}]
    annotate_regions(regions, results)
    assert [r.get("subject_overlap") for r in regions] == [1.0, pytest.approx(140 / 280, abs=0.01), 0.0, None]
    annotate_regions(regions, {})  # no subject analyzer: nothing is "scene text"
    assert regions[0]["subject_overlap"] == 0.0


# ---------- scenario helpers ----------

def clean_metrics():
    return {
        "face": {"count": 1.0, "largest_area_pct": 9.0, "mean_intensity": 0.8},
        "text": {"count": 2.0, "area_pct": 8.0, "word_count": 3.0, "largest_height_pct": 14.0, "mean_confidence": 0.97},
        "subject": {"count": 1.0, "area_pct": 25.0, "has_clear_subject": 1.0, "edge_contrast": 45.0, "background_separation": 0.8},
        "color": {"lightness_contrast": 28.0, "colorfulness_score": 60.0},
        "composition": {"concentration_top10": 0.45, "focal_points": 2.0, "top_peak_ratio": 0.4, "ui_overlap_pct": 1.0, "spread": 0.33,
                        "negative_space_pct": 40.0, "peak_count": 3.0,
                        "peaks": [{"x": 300, "y": 300, "strength": 1.0}, {"x": 900, "y": 200, "strength": 0.6}]},
        "attention_share": {"face": 30.0, "text": 25.0, "subject": 20.0, "background": 25.0},
    }


def clean_regions():
    return [
        {"type": "face", "box": [200, 200, 400, 400], "intensity": 0.8, "emotion": "happiness", "attention_pct": 30.0},
        {"type": "text", "box": [700, 100, 1200, 300], "text": "BIG TITLE", "mobile_readable": True, "contrast": 12.0,
         "height_pct": 14.0, "attention_pct": 25.0, "subject_overlap": 0.0},
    ]


def run(metrics=None, regions=None, cfg=None):
    m, r = metrics or clean_metrics(), regions or clean_regions()
    scores, details = compute_scores(m, r, SCORING)
    return evaluate(cfg or RULES, m, scores, r, details, SCORING["overall_weights"])


def real_rules(recs):
    return [x for x in recs if not x["id"].startswith(("improve_", "keep_"))]


def text_region(**kw):
    base = {"type": "text", "box": [700, 100, 1200, 300], "text": "tiny words", "mobile_readable": True, "contrast": 12.0,
            "height_pct": 14.0, "attention_pct": 25.0, "subject_overlap": 0.0}
    return {**base, **kw}


def test_a_clean_thumbnail_triggers_no_real_rules():
    assert real_rules(run()) == []


# each case: rule id -> (metrics mutation, regions) that must make exactly that rule fire with the key numbers in the message
def mut(group, **kw):
    def f(m, r):
        m[group].update(kw)
        return m, r
    return f


CASES = {
    "text_unreadable_mobile": (lambda m, r: (m, [r[0], text_region(mobile_readable=False, height_pct=6.0, text="tiny words")]), ["tiny words", "6%", "12%"]),
    "text_font_unreadable": (lambda m, r: (m, [r[0], text_region(mobile_readable=False, height_pct=20.0, text="Fancy Font")]), ["Fancy Font", "20%"]),
    "text_low_contrast": (lambda m, r: (m, [r[0], text_region(contrast=1.8, text="grey on grey")]), ["grey on grey", "1.8:1", "3:1"]),
    "too_many_words": (mut("text", word_count=11.0), ["11 words", "5 or fewer"]),
    "text_covers_too_much": (mut("text", area_pct=38.0), ["38%", "25%"]),
    "face_too_small": (mut("face", largest_area_pct=1.4), ["1.4%", "30%", "8%"]),
    "face_flat_expression": (lambda m, r: (m, [dict(r[0], intensity=0.1, emotion="neutral"), r[1]]), ["0.10", "neutral"]),
    "competing_focal_points": (mut("composition", focal_points=6.0), ["6 spots", "the face"]),
    "weak_primary_focus": (mut("composition", top_peak_ratio=0.95, concentration_top10=0.22), ["95%"]),
    "attention_under_ui": (mut("composition", ui_overlap_pct=9.0), ["9%", "duration badge"]),
    "low_contrast_image": (mut("color", lightness_contrast=8.0), ["8", "22"]),
    "empty_frame": (mut("composition", negative_space_pct=82.0), ["82%"]),
    "cramped_background": (mut("composition", negative_space_pct=6.0, peak_count=8.0), ["6%", "8 attention peaks"]),
    "subject_blends_in": (mut("subject", edge_contrast=14.0), ["25%", "14", "30"]),
    "low_colorfulness": (mut("color", colorfulness_score=9.0), ["9 of 100"]),
}


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_each_rule_fires_alone_with_real_numbers(rule_id):
    mutate, must_contain = CASES[rule_id]
    m, r = mutate(clean_metrics(), clean_regions())
    fired = {x["id"]: x for x in real_rules(run(m, r))}
    assert rule_id in fired, f"{rule_id} did not fire; got {sorted(fired)}"
    msg = fired[rule_id]["message"]
    for s in must_contain:
        assert s in msg, f"{s!r} missing from: {msg}"
    assert "{" not in msg and "None" not in msg and "nan" not in msg.lower()
    assert fired[rule_id]["severity"] in {"high", "medium", "low"} and 0 <= fired[rule_id]["impact"] <= 1


def test_every_rule_in_rules_yaml_has_a_test_scenario():
    assert {r["id"] for r in RULES["rules"]} == set(CASES)
    assert 10 <= len(RULES["rules"]) <= 15  # PRD: 10 to 15 rules


# ---------- item selection, scene text, ranking, fallback ----------

def test_rule_names_the_worst_item_and_returns_its_box():
    regions = [clean_regions()[0],
               text_region(text="small one", mobile_readable=False, height_pct=6.0, box=[0, 0, 100, 50]),
               text_region(text="the big miss", mobile_readable=False, height_pct=9.0, box=[300, 300, 900, 450])]
    rec = next(x for x in run(regions=regions) if x["id"] == "text_unreadable_mobile")
    assert "the big miss" in rec["message"] and rec["box"] == [300, 300, 900, 450]  # largest failing line wins


def test_text_inside_the_subject_and_unseen_text_are_ignored():
    regions = [clean_regions()[0],
               text_region(text="4:08", mobile_readable=False, height_pct=6.0, subject_overlap=0.95),
               text_region(text="watermark", mobile_readable=False, height_pct=4.0, attention_pct=0.3, box=[0, 650, 100, 700])]
    assert "text_unreadable_mobile" not in {x["id"] for x in run(regions=regions)}


def test_severity_expression_and_ranking_by_severity_times_impact():
    m = clean_metrics(); m["composition"].update(focal_points=7.0)
    rec = {x["id"]: x for x in run(m)}["competing_focal_points"]
    assert rec["severity"] == "high"
    m2 = clean_metrics(); m2["composition"].update(focal_points=6.0)
    assert {x["id"]: x for x in run(m2)}["competing_focal_points"]["severity"] == "medium"
    m3 = clean_metrics(); m3["composition"].update(focal_points=5.0)
    assert "competing_focal_points" not in {x["id"] for x in run(m3)}  # 5 foci is common in good thumbnails

    m3 = clean_metrics(); m3["text"].update(word_count=7.0); m3["color"].update(colorfulness_score=15.0)  # medium vs low
    ranks = [x["id"] for x in real_rules(run(m3))]
    assert ranks.index("too_many_words") < ranks.index("low_colorfulness")


def test_at_most_five_and_ordered_by_rank():
    m = clean_metrics()
    m["text"].update(word_count=12.0, area_pct=40.0)
    m["face"].update(largest_area_pct=1.0)
    m["composition"].update(focal_points=7.0, ui_overlap_pct=10.0, negative_space_pct=80.0, top_peak_ratio=0.95, concentration_top10=0.2)
    m["color"].update(lightness_contrast=6.0, colorfulness_score=5.0)
    m["subject"].update(edge_contrast=10.0)
    recs = run(m)
    assert len(recs) == 5
    w = {"high": 3, "medium": 2, "low": 1}
    ranks = [w[x["severity"]] * x["impact"] for x in recs]
    assert ranks == sorted(ranks, reverse=True)
    assert len({x["id"] for x in recs}) == 5


def test_fallback_tips_fill_up_to_two_for_a_strong_thumbnail():
    recs = run()
    assert len(recs) == 2 and all(x["severity"] == "low" for x in recs)
    assert recs[0]["id"].startswith("improve_") and "target" in recs[0]["message"]  # the one real gap first
    assert recs[1]["id"].startswith("keep_")  # nothing else to fix, so say what works
    assert all("{" not in x["message"] and "None" not in x["message"] for x in recs)


def test_missing_analyzer_data_skips_rules_instead_of_failing():
    m = clean_metrics(); del m["face"]; del m["composition"]
    recs = run(m, [r for r in clean_regions() if r["type"] != "face"])
    assert isinstance(recs, list)  # nothing raised
    assert not {"face_too_small", "attention_under_ui"} & {x["id"] for x in recs}


# ---------- config validation ----------

def test_validate_rules_accepts_the_shipped_file_and_rejects_bad_ones():
    validate_rules(RULES)
    unquoted = copy.deepcopy(RULES); unquoted["rules"][0]["severity"] = "high if text_count > 1 else medium"
    with pytest.raises(ValueError, match="quote"):
        validate_rules(unquoted)
    bad = copy.deepcopy(RULES); bad["rules"][0]["when"] = "__import__('os')"
    with pytest.raises(ValueError):
        validate_rules(bad)
    dup = copy.deepcopy(RULES); dup["rules"][1]["id"] = dup["rules"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rules(dup)
    typo = copy.deepcopy(RULES); typo["rules"][2]["impact"] = "min(1, 2 +"
    with pytest.raises(SyntaxError):
        validate_rules(typo)


def test_a_broken_rule_is_skipped_at_runtime_not_fatal():
    cfg = copy.deepcopy(RULES)
    cfg["rules"].append({"id": "bad", "severity": "high", "when": "text_count.real > 1", "message": "x"})
    assert "bad" not in {x["id"] for x in run(cfg=cfg)}


# ---------- subject outline contrast ----------

def test_edge_contrast_separates_a_popping_subject_from_a_camouflaged_one():
    mask = np.zeros((720, 1280), bool)
    mask[250:470, 400:700] = True
    pop = np.full((720, 1280, 3), 120, np.uint8); pop[mask] = (255, 20, 20)
    camo = np.full((720, 1280, 3), 120, np.uint8); camo[mask] = (124, 120, 118)
    assert edge_contrast(pop, mask) > 40 and edge_contrast(camo, mask) < 5
    assert edge_contrast(pop, np.zeros_like(mask)) == 0.0


# ---------- live pipeline ----------

@pytest.mark.skipif(not (THUMBS / "tech_c347oYQO57A.jpg").exists(), reason="sample thumbnails not downloaded")
def test_live_pipeline_gives_two_to_five_clean_numbered_fixes(tmp_path):
    p = Pipeline(store=SqliteStore(tmp_path))
    for name in ("tech_c347oYQO57A", "vlog_bwDjCBUsXYM", "education_UrsmFxEIp5k", "gaming_Rd9ehQBj47k"):
        out = p.analyze((THUMBS / f"{name}.jpg").read_bytes())
        recs = out["recommendations"]
        assert 2 <= len(recs) <= 5, (name, len(recs))
        assert len({r["id"] for r in recs}) == len(recs)
        for r in recs:
            assert r["message"].strip() and "{" not in r["message"] and "None" not in r["message"] and "nan" not in r["message"].lower()
        AnalysisResult.model_validate(out)
