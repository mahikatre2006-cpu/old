import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.schemas import AnalysisResult


def png(color=(10, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("THUMB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "")  # never hit a real database from the default tests
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_then_cache_then_fetch(client):
    files = {"file": ("t.png", png(), "image/png")}
    first = client.post("/analyze", files=files)
    assert first.status_code == 200
    body = first.json()
    AnalysisResult.model_validate(body)
    assert len(body["id"]) == 64 and body["cached"] is False
    assert body["timing_ms"]["total"] > 0

    second = client.post("/analyze", files={"file": ("t.png", png(), "image/png")}).json()
    assert second["cached"] is True and second["id"] == body["id"]

    stored = client.get(f"/analysis/{body['id']}")
    assert stored.status_code == 200 and stored.json()["id"] == body["id"]


def test_bad_upload_is_422(client):
    r = client.post("/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_unknown_analysis_is_404(client):
    assert client.get("/analysis/nope").status_code == 404


def test_mock_fixture_matches_contract(client, monkeypatch):
    from api.main import MOCK_PATH

    AnalysisResult.model_validate(json.loads(MOCK_PATH.read_text(encoding="utf-8")))
    monkeypatch.setenv("THUMB_MOCK", "1")
    r = client.post("/analyze", files={"file": ("t.png", png(), "image/png")})
    assert r.json()["scores"]["overall"] == 72


def test_analyze_returns_heatmap_png(client):
    import base64

    import cv2
    import numpy as np

    img = np.full((720, 1280, 3), 110, np.uint8)
    cv2.circle(img, (380, 360), 170, (255, 255, 255), -1)
    cv2.circle(img, (380, 360), 120, (220, 20, 20), -1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    body = client.post("/analyze", files={"file": ("t.png", buf.tobytes(), "image/png")}).json()
    assert body["unavailable"] == [] and "saliency" in body["timing_ms"]
    heat = cv2.imdecode(np.frombuffer(base64.b64decode(body["heatmap_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert heat.shape == (360, 640, 4)
    assert heat[180, 190, 3] > heat[20, 600, 3]  # more opaque over the object than the empty corner
    assert client.get("/health").json()["analyzers"]["saliency"] == "loaded"


def test_analyze_returns_aligned_preview(client):
    import base64

    import cv2
    import numpy as np

    sq = np.full((720, 720, 3), 90, np.uint8)  # not 16:9 -> backend letterboxes
    ok, buf = cv2.imencode(".png", sq)
    body = client.post("/analyze", files={"file": ("sq.png", buf.tobytes(), "image/png")}).json()
    prev = cv2.imdecode(np.frombuffer(base64.b64decode(body["preview_jpg"]), np.uint8), cv2.IMREAD_COLOR)
    assert prev.shape == (540, 960, 3)  # always the 16:9 frame the model saw
    assert prev[270, 5].max() < 10 and prev[270, 480].min() > 60  # black side bars, content in the middle
