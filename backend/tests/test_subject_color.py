from pathlib import Path

import cv2
import numpy as np
import pytest

from analyzers.color import (
    ColorAnalyzer,
    colorfulness,
    dog_contrast,
    kmeans_palette,
    p99_norm,
    spectral_residual,
    to_lab,
)
from analyzers.subject import SubjectAnalyzer, separation_score
from pipeline.config import analyzers_config
from pipeline.ingest import ingest
from pipeline.preprocess import build_context

CFG = analyzers_config()
THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"
needs_data = pytest.mark.skipif(not (THUMBS / "tech_c347oYQO57A.jpg").exists(), reason="sample thumbnails not downloaded")


def ctx_for(img):
    return build_context(img, "t", CFG["preprocess"])


def real(name):
    img = ingest((THUMBS / f"{name}.jpg").read_bytes(), CFG["ingest"]).image
    return img, ctx_for(img)


def square_on_grey(color=(255, 0, 0), grey=120):
    img = np.full((720, 1280, 3), grey, np.uint8)
    img[300:420, 200:360] = color
    return img


# =================== M4c subject ===================

@pytest.fixture(scope="module")
def subject():
    a = SubjectAnalyzer(**CFG["analyzers"]["subject"]["options"])
    a.load()
    return a


def test_separation_score_distinguishes_contrast_from_camouflage():
    img = square_on_grey((255, 0, 0), 120)
    mask = np.zeros((720, 1280), bool)
    mask[300:420, 200:360] = True
    assert separation_score(img, mask) > 0.8  # red on grey
    assert separation_score(square_on_grey((122, 120, 118), 120), mask) < 0.1  # nearly the same colour
    assert separation_score(img, np.zeros((720, 1280), bool)) == 0.0  # empty mask


def test_subject_probability_contract(subject):
    img = square_on_grey()
    p = subject.predict(img)
    assert p.shape == (720, 1280) and p.dtype == np.float32 and 0 <= p.min() and p.max() <= 1.0


def test_subject_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        SubjectAnalyzer(model="nope.onnx").load()


@needs_data
def test_subject_finds_person_with_consistent_mask_box_and_metrics(subject):
    img, ctx = real("tech_c347oYQO57A")
    r = subject.run(img, ctx)
    assert r.metrics["count"] >= 1 and r.metrics["has_clear_subject"] == 1.0
    assert 0.1 < r.metrics["area_pct"] / 100 < 0.8
    reg = r.regions[0]
    x0, y0, x1, y1 = reg.box
    ys, xs = np.where(reg.mask)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (x0, y0, x1, y1)  # box matches mask
    assert reg.mask.shape == (720, 1280) and reg.mask.dtype == bool
    assert 0 <= r.metrics["background_separation"] <= 1 and 0 <= r.metrics["cx"] <= 1
    m = r.attention_map
    assert m.shape == (720, 1280) and m.max() <= 1.0
    far_outside = ~cv2.dilate(reg.mask.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    assert m[reg.mask].mean() > 0.7 and m[far_outside].max() < 0.05  # soft map follows the mask, not the box


# =================== M4d colour ===================

@pytest.fixture(scope="module")
def color():
    return ColorAnalyzer(**CFG["analyzers"]["color"]["options"])


def test_colorfulness_orders_grey_pastel_vivid():
    grey = np.full((90, 160, 3), 128, np.uint8)
    pastel = np.full((90, 160, 3), (200, 170, 170), np.uint8)
    vivid = np.zeros((90, 160, 3), np.uint8)
    vivid[:, :80] = (255, 0, 0)
    vivid[:, 80:] = (0, 0, 255)
    assert colorfulness(grey) == pytest.approx(0.0, abs=0.01)
    assert colorfulness(grey) < colorfulness(pastel) < colorfulness(vivid)
    assert colorfulness(vivid) > 80  # in the "highly colourful" band


def test_palette_is_sorted_by_share_and_sums_to_one():
    img = np.zeros((180, 320, 3), np.uint8)
    img[:, :200] = (255, 0, 0)  # 62.5 %
    img[:, 200:] = (0, 0, 255)  # 37.5 %
    centers, shares, labels = kmeans_palette(to_lab(img), 5)
    assert shares[0] == pytest.approx(0.625, abs=0.02) and shares.sum() == pytest.approx(1.0)
    assert list(shares) == sorted(shares, reverse=True) and labels.shape == (180, 320)


def test_spectral_residual_and_dog_peak_on_the_odd_one_out():
    img = np.full((180, 320, 3), 100, np.uint8)
    img[80:100, 150:170] = (255, 255, 255)
    sr = spectral_residual(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    dog = dog_contrast(to_lab(img))
    for m in (sr, dog):
        assert m[60:120, 130:190].max() > 5 * m[:40, :60].mean()
    assert p99_norm(np.zeros((4, 4), np.float32)).max() == 0


def test_flat_image_has_no_attention_and_one_colour(color):
    img = np.full((720, 1280, 3), 128, np.uint8)
    r = color.run(img, ctx_for(img))
    assert r.attention_map.max() == pytest.approx(0, abs=1e-3)
    assert r.metrics["colorfulness"] == pytest.approx(0, abs=0.1) and r.metrics["dominant_share"] > 0.99
    assert r.metrics["significant_colors"] == 1.0


def test_colour_patch_on_grey_draws_attention_and_is_deterministic(color):
    img = square_on_grey((255, 0, 0), 120)
    r = color.run(img, ctx_for(img))
    m = r.attention_map
    assert m.shape == (720, 1280) and m.dtype == np.float32 and 0 <= m.min() and m.max() <= 1.0
    assert m[300:420, 200:360].mean() > 5 * m[500:700, 700:1200].mean()
    again = color.run(img, ctx_for(img))
    assert np.array_equal(m, again.attention_map) and again.metrics["palette_hex"] == r.metrics["palette_hex"]
    assert all(h.startswith("#") and len(h) == 7 for h in r.metrics["palette_hex"])
    assert sum(r.metrics["palette_share"]) == pytest.approx(1.0, abs=0.01)


@needs_data
def test_colour_metrics_rank_real_thumbnails_sensibly(color):
    vivid = color.run(*(lambda i, c: (i, c))(*real("fitness_vsw8_8ANJlc"))).metrics
    muted = color.run(*(lambda i, c: (i, c))(*real("vlog_OqVRBFqal2o"))).metrics
    assert vivid["colorfulness"] > muted["colorfulness"] + 30
