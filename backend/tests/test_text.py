from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzers.text import TextAnalyzer, TextItem, estimate_colors, relative_luminance, text_recovered, wcag_contrast
from pipeline.config import analyzers_config
from pipeline.ingest import ingest
from pipeline.preprocess import build_context

CFG = analyzers_config()
THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"
needs_data = pytest.mark.skipif(not (THUMBS / "education_KorR0O7kiEk.jpg").exists(), reason="sample thumbnails not downloaded")


def item(text, box, score=0.99):
    x0, y0, x1, y1 = box
    return TextItem(text, score, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), box)


def frame(bg=(20, 20, 20)):
    return np.full((720, 1280, 3), bg, np.uint8)


@pytest.fixture(scope="module")
def analyzer():
    a = TextAnalyzer(**CFG["analyzers"]["text"]["options"])
    a.load()
    return a


def ctx_for(img):
    return build_context(img, "t", CFG["preprocess"])


# ---------- WCAG ----------

def test_wcag_known_values():
    assert wcag_contrast((0, 0, 0), (255, 255, 255)) == pytest.approx(21.0)
    assert wcag_contrast((128, 128, 128), (128, 128, 128)) == pytest.approx(1.0)
    assert wcag_contrast((118, 118, 118), (255, 255, 255)) == pytest.approx(4.54, abs=0.02)  # #767676, the AA boundary
    assert wcag_contrast((255, 255, 255), (0, 0, 0)) == wcag_contrast((0, 0, 0), (255, 255, 255))
    assert relative_luminance((255, 255, 255)) == pytest.approx(1.0)


# ---------- mobile-readability matching ----------

def test_text_recovered_matching_rules():
    box = (300, 120, 800, 220)
    assert text_recovered("PYTHON", box, [item("PYTHON", (310, 125, 790, 215))])
    assert text_recovered("PYTHON", box, [item("PYTH0N", (310, 125, 790, 215))])  # one OCR slip is fine
    assert text_recovered("PYTHON", box, [item("FULL COURSE PYTHON", (200, 100, 900, 230))])  # merged lines
    assert not text_recovered("PYTHON", box, [item("PYTHON", (900, 500, 1200, 600))])  # same text, elsewhere
    assert not text_recovered("PYTHON", box, [item("JAVA", (310, 125, 790, 215))])  # different text, same place
    assert not text_recovered("PYTHON", box, [])
    assert not text_recovered("!!", box, [item("!!", box)])  # nothing alphanumeric to match


# ---------- colour / contrast estimation ----------

def test_estimate_colors_light_text_on_dark():
    img = frame((20, 20, 20))
    img[300:360, 400:800] = (250, 250, 250)
    text, bg = estimate_colors(img, item("HI", (400, 300, 800, 360)))
    assert text.min() > 240 and bg.max() < 30
    assert wcag_contrast(text, bg) > 15


def test_estimate_colors_low_contrast_text_is_flagged():
    img = frame((120, 120, 120))
    img[300:360, 400:800] = (140, 140, 140)
    text, bg = estimate_colors(img, item("HI", (400, 300, 800, 360)))
    assert wcag_contrast(text, bg) < 1.5


# ---------- analyzer with a faked OCR (deterministic) ----------

def test_analyzer_flags_unreadable_and_low_contrast_without_a_real_ocr(analyzer, monkeypatch):
    img = frame((20, 20, 20))
    img[100:190, 100:700] = (250, 250, 250)  # big, high contrast
    img[500:530, 100:400] = (30, 30, 30)  # tiny and near-invisible against the background
    big, tiny = item("BIG TITLE", (100, 100, 700, 190)), item("fine print", (100, 500, 400, 530))
    calls = iter([[big, tiny], [item("BIG TITLE", (30, 30, 230, 65))]])  # sidebar OCR only recovers the title
    monkeypatch.setattr(analyzer, "_ocr", lambda rgb, W, H: next(calls))

    r = analyzer.run(img, ctx_for(img))
    by_text = {g.attrs["text"]: g for g in r.regions}
    assert by_text["BIG TITLE"].attrs["mobile_readable"] is True
    assert by_text["fine print"].attrs["mobile_readable"] is False
    assert by_text["BIG TITLE"].attrs["contrast"] > 15 and by_text["fine print"].attrs["contrast"] < 1.5
    assert r.metrics["count"] == 2 and r.metrics["unreadable_mobile_count"] == 1
    assert r.metrics["mobile_readable_ratio"] == 0.5 and r.metrics["low_contrast_count"] == 1
    assert r.metrics["word_count"] == 4
    assert r.attention_map[145, 400] > r.attention_map[515, 250] > 0  # big contrasty text draws more attention
    assert r.attention_map.max() <= 1.0 and r.attention_map[650, 1100] == 0


def test_analyzer_filters_single_glyphs_and_tiny_text(analyzer, monkeypatch):
    img = frame()
    junk = [item("E", (50, 50, 80, 90)), item("tiny words", (50, 200, 300, 206))]  # 1 char; 6 px high
    monkeypatch.setattr(analyzer, "_ocr", lambda rgb, W, H: junk)
    r = analyzer.run(img, ctx_for(img))
    assert r.metrics["count"] == 0 and r.regions == [] and r.attention_map.max() == 0


# ---------- real OCR ----------

def test_real_ocr_reads_big_synthetic_text_and_not_blank_frames(analyzer):
    img = frame((30, 30, 120))
    cv2.putText(img, "HELLO WORLD", (120, 360), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 255, 255), 14, cv2.LINE_AA)
    r = analyzer.run(img, ctx_for(img))
    assert r.regions, "OCR found nothing in large clean text"
    joined = "".join(g.attrs["text"] for g in r.regions).replace(" ", "").upper()
    assert "HELLO" in joined or "WORLD" in joined
    assert all(g.attrs["mobile_readable"] for g in r.regions)  # ~35% frame height survives the sidebar

    blank = analyzer.run(frame(), ctx_for(frame()))
    assert blank.metrics["count"] == 0 and blank.attention_map.max() == 0


@needs_data
def test_real_thumbnail_title_is_readable_and_small_text_is_not(analyzer):
    def run(name):
        img = ingest((THUMBS / f"{name}.jpg").read_bytes(), CFG["ingest"]).image
        return analyzer.run(img, ctx_for(img))

    edu = run("education_KorR0O7kiEk")
    py = next(g for g in edu.regions if "PYTHON" in g.attrs["text"].upper())
    assert py.attrs["mobile_readable"] is True and py.attrs["contrast"] > 7

    vlog = run("vlog_OqVRBFqal2o")
    small = [g for g in vlog.regions if g.attrs["height_pct"] < 7]
    assert small and all(g.attrs["mobile_readable"] is False for g in small)
    assert vlog.metrics["unreadable_mobile_count"] >= 2
