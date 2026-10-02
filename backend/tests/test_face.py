from pathlib import Path

import numpy as np
import pytest

from analyzers.face import FaceAnalyzer, build_face_map, face_amplitude, softmax
from pipeline.config import analyzers_config
from pipeline.ingest import ingest
from pipeline.preprocess import build_context

CFG = analyzers_config()
THUMBS = Path(__file__).resolve().parents[2] / "data" / "thumbnails"
needs_data = pytest.mark.skipif(not (THUMBS / "tech_c347oYQO57A.jpg").exists(), reason="sample thumbnails not downloaded")


def load(name):
    img = ingest((THUMBS / f"{name}.jpg").read_bytes(), CFG["ingest"]).image
    return img, build_context(img, name, CFG["preprocess"])


@pytest.fixture(scope="module")
def analyzer():
    a = FaceAnalyzer(**CFG["analyzers"]["face"]["options"])
    a.load()
    return a


# ---------- pure functions ----------

def test_softmax_sums_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert p.sum(axis=1) == pytest.approx([1, 1]) and p[1, 0] == pytest.approx(1 / 3)


def test_amplitude_grows_with_size_and_intensity_and_is_bounded():
    small, big = face_amplitude(0.005, 0.5, 0.06), face_amplitude(0.08, 0.5, 0.06)
    calm, intense = face_amplitude(0.03, 0.0, 0.06), face_amplitude(0.03, 1.0, 0.06)
    assert small < big <= 1.0 and calm < intense <= 1.0
    assert face_amplitude(1e-6, 0.0, 0.06) == pytest.approx(0.245, abs=0.01)  # tiny + calm still counts a little
    assert face_amplitude(0.5, 1.0, 0.06) == 1.0  # capped
    assert face_amplitude(0.03, None, 0.06) > 0  # emotion unknown (face too small) is fine


def test_face_map_peaks_in_each_face_and_scales_with_amplitude():
    m = build_face_map([(100, 100, 300, 300), (800, 200, 900, 300)], [1.0, 0.4], (720, 1280))
    assert m[200, 200] == pytest.approx(1.0, abs=0.01)
    assert m[250, 850] == pytest.approx(0.4, abs=0.01)
    assert m[600, 640] < 0.01 and m.dtype == np.float32 and m.max() <= 1.0


def test_face_map_handles_faces_touching_the_frame_edge():
    m = build_face_map([(0, 0, 80, 80), (1200, 640, 1280, 720)], [0.8, 0.8], (720, 1280))
    assert m.shape == (720, 1280) and np.isfinite(m).all() and m[40, 40] > 0.7


# ---------- real thumbnails ----------

@needs_data
def test_detects_the_one_face_with_emotion(analyzer):
    img, ctx = load("tech_c347oYQO57A")
    r = analyzer.run(img, ctx)
    assert r.metrics["count"] == 1
    face = r.regions[0]
    x0, y0, x1, y1 = face.box
    assert (x1 - x0) > 200 and x0 < 500  # big face on the left half
    assert face.attrs["emotion"] in {"anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"}
    assert 0 <= face.attrs["intensity"] <= 1
    assert r.attention_map.max() <= 1.0 and r.attention_map[(y0 + y1) // 2, (x0 + x1) // 2] > 0.5
    probs = [r.metrics[f"p_{k}"] for k in ("anger", "happiness", "neutral", "sadness", "surprise", "contempt", "disgust", "fear")]
    assert sum(probs) == pytest.approx(1.0, abs=0.02)


@needs_data
def test_no_faces_gives_empty_map_and_zero_metrics(analyzer):
    img, ctx = load("gaming_Rd9ehQBj47k")
    r = analyzer.run(img, ctx)
    assert r.metrics["count"] == 0 and r.regions == [] and r.attention_map.max() == 0


@needs_data
def test_tiny_faces_are_detected_but_skip_emotion(analyzer):
    img, ctx = load("vlog_OqVRBFqal2o")
    r = analyzer.run(img, ctx)
    sizes = {i: r_.box[2] - r_.box[0] for i, r_ in enumerate(r.regions)}
    tiny = [r_ for i, r_ in enumerate(r.regions) if sizes[i] < analyzer.min_emotion_px]
    assert tiny and all(t.attrs["emotion"] is None and t.attrs["intensity"] is None for t in tiny)
    assert r.regions[0].attrs["emotion"] is not None  # largest first, big enough


def test_missing_models_raise_on_load():
    with pytest.raises(FileNotFoundError):
        FaceAnalyzer(detector="nope.onnx").load()
    with pytest.raises(FileNotFoundError):
        FaceAnalyzer(emotion_model="nope.onnx").load()


@needs_data
def test_pipeline_end_to_end_reports_face_share(tmp_path, monkeypatch):
    from pipeline.orchestrator import Pipeline
    from pipeline.store import SqliteStore

    p = Pipeline(store=SqliteStore(tmp_path))
    assert p.status["face"] == "loaded"
    out = p.analyze((THUMBS / "tech_c347oYQO57A.jpg").read_bytes())
    faces = [r for r in out["regions"] if r["type"] == "face"]
    assert len(faces) == 1 and faces[0]["attention_pct"] > 5
    assert out["metrics"]["face"]["count"] == 1
    shares = out["metrics"]["attention_share"]
    assert sum(shares.values()) == pytest.approx(100, abs=0.3) and shares["face"] > 5
    assert out["unavailable"] == []
