"""M6. Scoring engine: metrics -> Focus, Text readability, Face impact, Clutter, Overall (0-100).

Pure functions over the metrics and regions the analyzers produced; thresholds live in scoring.yaml.
Each score also returns the breakdown that explains it, which the recommendation engine (M7) reuses.
"""
from __future__ import annotations

from typing import Any


def ramp(value: float, bad: float, good: float) -> float:
    """0 at `bad`, 1 at `good`, linear in between, clipped. Works when good < bad (lower is better)."""
    if good == bad:
        return 1.0 if value == good else 0.0
    return float(min(1.0, max(0.0, (value - bad) / (good - bad))))


def band(value: float, bad_low: float, good_low: float, good_high: float, bad_high: float) -> float:
    """1 inside [good_low, good_high], falling linearly to 0 at bad_low and bad_high: for metrics where
    both too little and too much are bad (e.g. negative space: cramped vs. an under-filled frame)."""
    return min(ramp(value, bad_low, good_low), ramp(value, bad_high, good_high))


def _goal(c: dict) -> str:
    if "bad_low" in c:
        return f"{c['good_low']:g} to {c['good_high']:g}"
    return f"{'at most' if c['good'] < c['bad'] else 'at least'} {c['good']:g}"


def _component_score(c: dict, value: float) -> float:
    if "bad_low" in c:
        return band(value, c["bad_low"], c["good_low"], c["good_high"], c["bad_high"])
    return ramp(value, c["bad"], c["good"])


def _get(view: dict, path: str) -> float | None:
    cur: Any = view
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return float(cur) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def derive(metrics: dict, regions: list[dict]) -> dict[str, float]:
    """Values that need the region list or a combination of metrics. Missing inputs are simply absent."""
    d: dict[str, float] = {}
    share = metrics.get("attention_share") or {}
    if "background" in share:
        d["background_share"] = float(share["background"])
    if "face" in share:
        d["face_share"] = float(share["face"])

    text = [r for r in regions if r.get("type") == "text" and r.get("box")]
    # small text sitting inside the main subject (a phone screen, a product label) is part of the photo
    text = [r for r in text if not (r.get("subject_overlap", 0) >= 0.7 and r.get("height_pct", 100) < 10)]
    if text:
        def area(r):
            x0, y0, x1, y1 = r["box"]
            return max(1, (x1 - x0) * (y1 - y0))

        flags = [(area(r), r.get("mobile_readable")) for r in text if r.get("mobile_readable") is not None]
        if flags:
            d["text_mobile_readable_ratio"] = sum(a for a, ok in flags if ok) / sum(a for a, _ in flags)
        cs = [(area(r), r["contrast"]) for r in text if r.get("contrast") is not None]
        if cs:
            # area-weighted so a tiny low-contrast footnote matters less than the headline
            d["text_weighted_contrast"] = sum(a * c for a, c in cs) / sum(a for a, _ in cs)

    faces = [r for r in regions if r.get("type") == "face"]
    known = [r["intensity"] for r in faces if r.get("intensity") is not None]
    if faces:
        # faces too small for an emotion read count as mildly expressive rather than as zero
        d["face_intensity"] = sum(known) / len(known) if known else 0.3
    return d


def compute_scores(metrics: dict, regions: list[dict], cfg: dict) -> tuple[dict[str, int | None], dict[str, dict]]:
    """Returns ({score: 0-100 or None}, {score: {value, applicable, note, components[]}})."""
    view = {**metrics, "derived": derive(metrics, regions)}
    scores: dict[str, int | None] = {}
    details: dict[str, dict] = {}

    for name, spec in cfg["scores"].items():
        req = spec.get("requires")
        if req:
            have = _get(view, req["metric"])
            if have is None or have < req.get("min", 1):
                scores[name] = None
                details[name] = {"value": None, "applicable": False, "note": spec.get("na_note", "Not applicable"), "components": []}
                continue

        comps, num, den = [], 0.0, 0.0
        for c in spec["components"]:
            v = _get(view, c["metric"])
            if v is None:
                comps.append({"id": c["id"], "label": c["label"], "metric": c["metric"], "value": None, "score": None, "weight": c["weight"]})
                continue
            s = _component_score(c, v)
            num += c["weight"] * s
            den += c["weight"]
            comps.append({"id": c["id"], "label": c["label"], "metric": c["metric"], "value": round(v, 3), "score": round(s, 3),
                          "weight": c["weight"], "goal": _goal(c)})

        if den == 0:  # every input missing (analyzers failed): no score rather than a made-up one
            scores[name] = None
            details[name] = {"value": None, "applicable": True, "note": "Not enough data", "components": comps}
        else:
            scores[name] = int(round(100 * num / den))
            details[name] = {"value": scores[name], "applicable": True, "note": "", "components": comps}

    ow = {k: w for k, w in cfg["overall_weights"].items() if scores.get(k) is not None}
    if ow:
        blended = sum(scores[k] * w for k, w in ow.items()) / sum(ow.values())
        applied = []
        for pen in cfg.get("penalties", []):
            v = _get(view, pen["metric"])
            pts = pen["max_points"] * ramp(v, pen["start"], pen["full"]) if v is not None else 0.0
            if pts > 0:
                applied.append({"id": pen["id"], "label": pen["label"], "metric": pen["metric"], "value": round(v, 3), "points": round(pts, 1)})
        scores["overall"] = int(round(max(0.0, blended - sum(p["points"] for p in applied))))
        details["overall"] = {"value": scores["overall"], "applicable": True, "note": "", "penalties": applied, "components": [
            {"id": k, "label": k.replace("_", " "), "metric": k, "value": scores[k], "score": scores[k] / 100, "weight": w} for k, w in ow.items()
        ]}
    else:
        scores["overall"] = None
        details["overall"] = {"value": None, "applicable": False, "note": "Not enough data", "components": []}
    return scores, details
