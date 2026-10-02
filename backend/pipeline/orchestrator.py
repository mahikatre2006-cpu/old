"""Runs ingest -> preprocess -> analyzers (stage 1 parallel, stage 2 after) and assembles the response.

A failing analyzer never fails the request: its section is listed in `unavailable`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from analyzers.base import Analyzer
from fusion.fuse import fuse
from fusion.heatmap import encode_attention_png, encode_heatmap_png, encode_preview_jpg
from fusion.scanpath import predict_scanpath
from fusion.shares import compute_shares
from pipeline.config import MODELS_DIR, analyzers_config, load_yaml
from pipeline.context import AnalyzerResult, Context
from pipeline.ingest import ingest
from pipeline.preprocess import build_context
from pipeline.store import Store, make_store
from scoring.engine import compute_scores
from scoring.rules import annotate_regions, evaluate, validate_rules

log = logging.getLogger("pipeline")

RESULT_FORMAT = 2  # bump when the shape of a stored result changes, so old cache entries are recomputed


class Pipeline:
    def __init__(self, store: Store | None = None, cfg: dict | None = None, fusion_cfg: dict | None = None,
                 scoring_cfg: dict | None = None, rules_cfg: dict | None = None):
        self.cfg = cfg or analyzers_config()
        self.fusion_cfg = fusion_cfg or load_yaml("fusion.yaml")
        self.scoring_cfg = scoring_cfg or load_yaml("scoring.yaml")
        self.rules_cfg = rules_cfg or load_yaml("rules.yaml")
        validate_rules(self.rules_cfg)
        self.store = store or make_store()
        self.analyzers: dict[str, Analyzer] = {}
        self.stages: dict[str, int] = {}
        self.status: dict[str, str] = {}
        self.pool = ThreadPoolExecutor(max_workers=self.cfg["pipeline"]["max_workers"])
        self.compare_pool = ThreadPoolExecutor(max_workers=4)  # one analysis per A/B variant, run side by side
        self._load_analyzers()
        self.signature = self._signature()

    def _signature(self) -> str:
        """Fingerprint of everything that changes results: configs, which analyzers loaded, model files.
        Cached results are only reused when it matches, so swapping a model or editing a weight
        never serves stale analyses."""
        models = {f.name: f.stat().st_size for f in sorted(MODELS_DIR.glob("*.onnx"))}
        blob = json.dumps(
            {"cfg": self.cfg, "fusion": self.fusion_cfg, "scoring": self.scoring_cfg, "rules": self.rules_cfg, "status": self.status, "models": models, "format": RESULT_FORMAT},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _load_analyzers(self) -> None:
        for name, spec in (self.cfg.get("analyzers") or {}).items():
            if not spec.get("enabled", True):
                self.status[name] = "disabled"
                continue
            try:
                module, cls = spec["module"].split(":")
                options = spec.get("options") or {}
                analyzer: Analyzer = getattr(importlib.import_module(module), cls)(**options)
                analyzer.load()
                self.analyzers[name] = analyzer
                self.stages[name] = spec.get("stage", 1)
                self.status[name] = "loaded"
            except Exception as e:  # model missing, import error, ...
                log.exception("analyzer %s failed to load", name)
                self.status[name] = f"error: {e}"

    def _run_one(self, name: str, ctx: Context) -> tuple[str, AnalyzerResult | None, float]:
        t0 = time.perf_counter()
        try:
            res = self.analyzers[name].run(ctx.original, ctx)
        except Exception:
            log.exception("analyzer %s failed", name)
            res = None
        return name, res, (time.perf_counter() - t0) * 1000

    def analyze(self, data: bytes) -> dict:
        t_start = time.perf_counter()
        timing: dict[str, float] = {}

        t0 = time.perf_counter()
        ing = ingest(data, self.cfg["ingest"])
        timing["ingest"] = (time.perf_counter() - t0) * 1000

        cached = self.store.get_analysis(ing.sha256)
        if cached is not None and cached.get("pipeline_sig") == self.signature:
            cached["cached"] = True
            cached["timing_ms"] = {"total": (time.perf_counter() - t_start) * 1000}
            return cached

        t0 = time.perf_counter()
        ctx = build_context(ing.image, ing.sha256, self.cfg["preprocess"])
        timing["preprocess"] = (time.perf_counter() - t0) * 1000

        unavailable = [n for n, s in self.status.items() if s.startswith("error")]
        for i, stage in enumerate(sorted(set(self.stages.values()))):
            if i > 0:  # stage-2 analyzers (composition) read the fused map built from earlier stages
                ctx.fused, _ = fuse(ctx.results, self.fusion_cfg["weights"])
            names = [n for n, s in self.stages.items() if s == stage]
            for name, res, ms in self.pool.map(lambda n: self._run_one(n, ctx), names):
                timing[name] = ms
                if res is None:
                    unavailable.append(name)
                else:
                    ctx.results[name] = res

        t0 = time.perf_counter()
        fusion = self._fuse(ctx)
        timing["fusion"] = (time.perf_counter() - t0) * 1000

        result = self._assemble(ctx, unavailable, fusion)
        result["timing_ms"] = {**timing, "total": (time.perf_counter() - t_start) * 1000}
        self.store.save_analysis(result)
        return result

    def _fuse(self, ctx: Context) -> dict | None:
        """M5: fused heatmap, gaze order, attention share per region."""
        fcfg = self.fusion_cfg
        fused, used = fuse(ctx.results, fcfg["weights"])
        if fused is None:
            return None
        regions, shares = compute_shares(fused, ctx.results, fcfg["shares"]["priority"])
        return {
            "map": fused,
            "weights": used,
            "scanpath": predict_scanpath(fused, **fcfg["scanpath"]),
            "regions": regions,
            "shares": shares,
        }

    def _assemble(self, ctx: Context, unavailable: list[str], fusion: dict | None) -> dict:
        """Response skeleton. Scoring (M6) and rules (M7) fill in scores and recommendations."""
        metrics: dict[str, dict] = {"face": {}, "text": {}, "color": {}, "composition": {}}
        for name, res in ctx.results.items():
            group = getattr(self.analyzers[name], "metrics_group", name)
            metrics.setdefault(group, {}).update(res.metrics)
        if fusion is not None:
            metrics["attention_share"] = fusion["shares"]
            metrics["fusion"] = {"version": self.fusion_cfg["version"], **{k: round(v, 3) for k, v in fusion["weights"].items()}}
            annotate_regions(fusion["regions"], ctx.results)
            scores, details = compute_scores(metrics, fusion["regions"], self.scoring_cfg)
            metrics["scores_detail"] = details  # why each score is what it is (used by the rules engine and UI)
            recommendations = evaluate(self.rules_cfg, metrics, scores, fusion["regions"], details, self.scoring_cfg["overall_weights"])
        else:
            scores, recommendations = None, []
            unavailable = [*unavailable, "fusion"]
        return {
            "id": ctx.image_hash,
            "heatmap_png": encode_heatmap_png(fusion["map"], colormap=self.fusion_cfg.get("heatmap", {}).get("colormap", "inferno")) if fusion else None,
            "preview_jpg": encode_preview_jpg(ctx.original),
            "attention_png": encode_attention_png(fusion["map"]) if fusion else None,
            "scanpath": fusion["scanpath"] if fusion else [],
            "regions": fusion["regions"] if fusion else [],
            "scores": scores,
            "recommendations": recommendations,
            "metrics": metrics,
            "timing_ms": {},
            "unavailable": sorted(set(unavailable)),
            "cached": False,
            "pipeline_sig": self.signature,  # stored for cache validation; not part of the API schema
        }
