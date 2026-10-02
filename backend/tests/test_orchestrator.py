"""One analyzer failing returns partial results, never a failed request."""
import io

import numpy as np
from PIL import Image

from analyzers.base import Analyzer
from pipeline.config import analyzers_config, load_yaml
from pipeline.context import AnalyzerResult
from pipeline.orchestrator import Pipeline
from pipeline.store import SqliteStore


class Good(Analyzer):
    name = "good"
    metrics_group = "color"

    def run(self, img, ctx):
        return AnalyzerResult(attention_map=np.zeros((720, 1280), np.float32), metrics={"x": 1.0})


class Boom(Analyzer):
    name = "boom"

    def run(self, img, ctx):
        raise RuntimeError("model exploded")


def png():
    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), (1, 2, 3)).save(buf, "PNG")
    return buf.getvalue()


def test_failing_analyzer_marks_section_unavailable(tmp_path):
    cfg = analyzers_config() | {
        "analyzers": {
            "good": {"module": "tests.test_orchestrator:Good", "stage": 1},
            "boom": {"module": "tests.test_orchestrator:Boom", "stage": 1},
            "missing": {"module": "tests.nope:Nothing", "stage": 1},
        }
    }
    fusion_cfg = load_yaml("fusion.yaml") | {"weights": {"good": 1.0}}
    p = Pipeline(store=SqliteStore(tmp_path), cfg=cfg, fusion_cfg=fusion_cfg)
    result = p.analyze(png())
    assert result["unavailable"] == ["boom", "missing"]
    assert result["metrics"]["color"] == {"x": 1.0}
    assert p.status["missing"].startswith("error")


def test_cache_is_invalidated_when_the_pipeline_changes(tmp_path):
    store = SqliteStore(tmp_path)
    base = analyzers_config() | {"analyzers": {"good": {"module": "tests.test_orchestrator:Good", "stage": 1}}}
    fusion = load_yaml("fusion.yaml") | {"weights": {"good": 1.0}}

    p1 = Pipeline(store=store, cfg=base, fusion_cfg=fusion)
    first = p1.analyze(png())
    assert first["cached"] is False
    assert p1.analyze(png())["cached"] is True  # same pipeline -> cache hit

    p2 = Pipeline(store=store, cfg=base, fusion_cfg=fusion | {"weights": {"good": 0.5}})  # edited a weight
    assert p2.signature != p1.signature
    again = p2.analyze(png())
    assert again["cached"] is False  # recomputed, not served stale
    assert p2.analyze(png())["cached"] is True  # and the fresh result is what gets cached now
    assert again["id"] == first["id"]
