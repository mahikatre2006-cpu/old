import io

import numpy as np
import pytest
from PIL import Image

from pipeline.config import analyzers_config
from pipeline.ingest import IngestError, ingest
from pipeline.preprocess import build_context

CFG = analyzers_config()


def encode(size, fmt="PNG", mode="RGB", color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_formats_normalize_to_1280x720(fmt):
    out = ingest(encode((640, 360), fmt), CFG["ingest"])
    assert out.image.shape == (720, 1280, 3) and out.image.dtype == np.uint8
    assert len(out.sha256) == 64


def test_same_image_same_hash_different_image_different_hash():
    a = ingest(encode((1280, 720)), CFG["ingest"])
    b = ingest(encode((1280, 720)), CFG["ingest"])
    c = ingest(encode((1280, 720), color=(0, 0, 255)), CFG["ingest"])
    assert a.sha256 == b.sha256 != c.sha256


def test_slightly_off_aspect_is_cropped_not_padded():
    out = ingest(encode((1200, 720)), CFG["ingest"])  # 5% narrower than 16:9
    assert out.info["fit"] == "crop"
    assert out.image.shape == (720, 1280, 3)
    assert (out.image == 0).sum() == 0  # no black bars


def test_far_off_aspect_is_letterboxed():
    out = ingest(encode((720, 720)), CFG["ingest"])  # square
    assert out.info["fit"] == "letterbox"
    assert out.image[360, 0].tolist() == [0, 0, 0]  # side bar
    assert out.image[360, 640].tolist() == [200, 30, 30]  # content kept


def test_transparent_png_composited_on_white():
    out = ingest(encode((1280, 720), mode="RGBA", color=(0, 0, 0, 0)), CFG["ingest"])
    assert out.image[0, 0].tolist() == [255, 255, 255]


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image", encode((10, 10)), encode((64, 36), "GIF", mode="P", color=1)],
)
def test_rejects_bad_input(data):
    with pytest.raises(IngestError):
        ingest(data, CFG["ingest"])


def test_rejects_oversize():
    with pytest.raises(IngestError, match="too large"):
        ingest(b"x" * (CFG["ingest"]["max_bytes"] + 1), CFG["ingest"])


def test_context_views_and_mask():
    ing = ingest(encode((1280, 720)), CFG["ingest"])
    ctx = build_context(ing.image, ing.sha256, CFG["preprocess"])
    assert ctx.model_input.shape == (192, 256, 3)
    assert ctx.views["desktop"].shape == (202, 360, 3)
    assert ctx.views["mobile"].shape == (180, 320, 3)
    assert ctx.views["sidebar"].shape == (94, 168, 3)
    assert ctx.ui_mask.shape == (720, 1280)
    assert ctx.ui_mask[700, 1200] == 1  # duration badge
    assert ctx.ui_mask[715, 10] == 1  # progress bar
    assert ctx.ui_mask[300, 600] == 0
    assert 0 < ctx.ui_mask.mean() < 0.05
