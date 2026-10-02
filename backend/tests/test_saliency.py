import cv2
import numpy as np
import pytest

from analyzers.saliency import SaliencyAnalyzer, letterbox, minmax, unpad
from pipeline.config import analyzers_config
from pipeline.ingest import Ingested
from pipeline.preprocess import build_context

CFG = analyzers_config()


def synthetic_thumbnail() -> np.ndarray:
    """Plain grey frame with one high-contrast red disc with a white ring, left of centre."""
    img = np.full((720, 1280, 3), 110, np.uint8)
    cv2.circle(img, (380, 360), 170, (255, 255, 255), -1)
    cv2.circle(img, (380, 360), 120, (220, 20, 20), -1)
    return img


@pytest.fixture(scope="module")
def analyzer():
    opts = CFG["analyzers"]["saliency"]["options"]
    a = SaliencyAnalyzer(**opts)
    a.load()
    return a


def ctx_for(img):
    return build_context(img, "h", CFG["preprocess"])


def test_output_contract(analyzer):
    img = synthetic_thumbnail()
    res = analyzer.run(img, ctx_for(img))
    m = res.attention_map
    assert m.shape == (720, 1280) and m.dtype == np.float32
    assert m.min() >= 0 and m.max() == pytest.approx(1.0)
    assert np.isfinite(m).all()


def test_object_attracts_more_than_empty_background(analyzer):
    img = synthetic_thumbnail()
    m = analyzer.run(img, ctx_for(img)).attention_map
    on_object = m[190:530, 210:550].mean()
    empty = m[:150, 800:].mean()
    assert on_object > 2 * empty


def test_deterministic(analyzer):
    img = synthetic_thumbnail()
    a = analyzer.run(img, ctx_for(img)).attention_map
    b = analyzer.run(img, ctx_for(img)).attention_map
    assert np.array_equal(a, b)


def test_flat_image_does_not_crash(analyzer):
    img = np.full((720, 1280, 3), 128, np.uint8)
    m = analyzer.run(img, ctx_for(img)).attention_map
    assert m.shape == (720, 1280) and np.isfinite(m).all()


def test_missing_model_raises_on_load():
    with pytest.raises(FileNotFoundError):
        SaliencyAnalyzer(model="does_not_exist.onnx").load()


def test_minmax_constant_map_is_zero():
    assert minmax(np.ones((4, 4), np.float32)).max() == 0


def test_letterbox_16x9_into_4x3_and_unpad_roundtrip():
    img = np.full((720, 1280, 3), 200, np.uint8)
    boxed, pad = letterbox(img, 320, 240)
    assert boxed.shape == (240, 320, 3)
    assert pad == (30, 30, 0, 0)  # 320x180 content, 30 px bars top and bottom
    assert boxed[0].max() == 0 and boxed[120].min() == 200
    assert unpad(boxed[..., 0], pad).shape == (180, 320)


def test_msinet_style_tensor_layout_and_range():
    a = SaliencyAnalyzer(model="x.onnx", input_size=(320, 240), layout="nhwc", input_range=255.0,
                         mean=None, std=None, pad_to_fit=True, sigmoid=False)
    img = np.full((720, 1280, 3), 255, np.uint8)
    boxed, _ = letterbox(img, 320, 240)
    t = a._to_tensor(boxed)
    assert t.shape == (1, 240, 320, 3) and t.dtype == np.float32
    assert t.max() == pytest.approx(255.0)
