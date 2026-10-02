"""M8. A/B comparison: run the pipeline on 2-4 variants, then say which wins and why.

The "why" comes from the same score breakdown and rules the single-image report uses, so it never
contradicts it: the biggest score components that differ, and the fixes that one variant needs and the
other does not.
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from fusion.heatmap import decode_attention_png, encode_diff_png
from pipeline.ingest import IngestError, ingest

log = logging.getLogger("compare")

MIN_VARIANTS, MAX_VARIANTS = 2, 4
MIN_DRIVER = 0.01  # a component must be worth about one overall point to be named as a reason
TIE_MARGIN = 3  # overall points; closer than this is a coin flip, so we say so
SCORE_KEYS = ["overall", "focus", "text_readability", "face_impact", "clutter"]
SCORE_NAMES = {"focus": "focus", "text_readability": "text readability", "face_impact": "face impact", "clutter": "cleanliness"}
TIPS = ("improve_", "keep_")  # generic tips are not "issues" worth contrasting


class CompareError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def labels_for(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def score_deltas(results: list[dict]) -> list[dict[str, int | None]]:
    """Each variant's scores minus variant A's. A itself is all zeros."""
    base = results[0]["scores"] or {}
    out = []
    for r in results:
        s = r["scores"] or {}
        out.append({k: (s[k] - base[k]) if s.get(k) is not None and base.get(k) is not None else None for k in SCORE_KEYS})
    return out


def pick_winner(results: list[dict]) -> tuple[int, int, bool]:
    """(winner index, margin over the runner-up, close?). Highest overall wins; a near tie is broken by
    focus and then by having fewer high-severity fixes, and is flagged as close."""
    def key(i: int):
        r = results[i]
        s = r["scores"]
        highs = sum(1 for x in r["recommendations"] if x["severity"] == "high")
        return (s["overall"], s["focus"] or 0, -highs)

    order = sorted(range(len(results)), key=key, reverse=True)
    top, second = order[0], order[1]
    margin = results[top]["scores"]["overall"] - results[second]["scores"]["overall"]
    return top, margin, margin < TIE_MARGIN


def _issue_ids(result: dict) -> dict[str, dict]:
    return {r["id"]: r for r in result["recommendations"] if not r["id"].startswith(TIPS)}


def rules_that_differ(results: list[dict], labels: list[str]) -> list[dict]:
    """Which real fixes apply to which variants, for the ones that are not shared by everyone."""
    per = [_issue_ids(r) for r in results]
    all_ids = sorted({i for p in per for i in p})
    out = []
    for rid in all_ids:
        present = [labels[i] for i, p in enumerate(per) if rid in p]
        if len(present) != len(results):
            msg = next(p[rid]["message"] for p in per if rid in p)
            out.append({"id": rid, "present_in": present, "example": msg})
    return out


def _component_gaps(win: dict, other: dict, overall_weights: dict) -> list[tuple[float, str, dict, dict]]:
    """Components where `win` beats `other`, largest weighted gap first: (gap, score name, win comp, other comp)."""
    wd = win["metrics"].get("scores_detail", {})
    od = other["metrics"].get("scores_detail", {})
    gaps = []
    for name, ow in overall_weights.items():
        a, b = wd.get(name, {}), od.get(name, {})
        if not a.get("applicable") or not b.get("applicable"):
            continue
        bmap = {c["id"]: c for c in b.get("components", [])}
        for c in a.get("components", []):
            o = bmap.get(c["id"])
            if o is None or c.get("score") is None or o.get("score") is None:
                continue
            gaps.append(((c["score"] - o["score"]) * c["weight"] * ow, name, c, o))
    return sorted(gaps, key=lambda t: -t[0])


def explain(results: list[dict], labels: list[str], win: int, margin: int, close: bool, overall_weights: dict) -> list[str]:
    runner = max((i for i in range(len(results)) if i != win), key=lambda i: results[i]["scores"]["overall"])
    W, R = labels[win], labels[runner]
    ws, rs = results[win]["scores"]["overall"], results[runner]["scores"]["overall"]
    reasons = []
    if close:
        reasons.append(f"{W} edges {R} by only {margin} point{'s' if margin != 1 else ''} overall ({ws} vs {rs}). Treat this as a close call and check the details below.")
    else:
        reasons.append(f"{W} beats {R} by {margin} points overall ({ws} vs {rs}).")

    gaps = _component_gaps(results[win], results[runner], overall_weights)
    for gap, name, wc, rc in [g for g in gaps if g[0] >= MIN_DRIVER][:3]:
        goal = f" (target {wc['goal']})" if wc.get("goal") else ""
        reasons.append(f"{SCORE_NAMES.get(name, name).capitalize()}: {wc['label'][0].lower() + wc['label'][1:]}, {W} {wc['value']:g} vs {R} {rc['value']:g}{goal}.")

    if close:  # a near tie is decided by trade-offs, so say what the runner-up does better before the issue list
        for gap, name, wc, rc in [g for g in gaps[::-1] if g[0] <= -MIN_DRIVER][:2]:
            reasons.append(f"{R} is better at {SCORE_NAMES.get(name, name)}: {rc['label'][0].lower() + rc['label'][1:]}, {R} {rc['value']:g} vs {W} {wc['value']:g}.")

    w_issues, r_issues = _issue_ids(results[win]), _issue_ids(results[runner])
    for rid in [i for i in r_issues if i not in w_issues][:2]:
        reasons.append(f"{R} has an issue {W} avoids: {r_issues[rid]['message']}")
    return reasons[:7]


def run_compare(pipeline, datas: list[bytes]) -> dict:
    n = len(datas)
    if not MIN_VARIANTS <= n <= MAX_VARIANTS:
        raise CompareError(422, f"Compare needs {MIN_VARIANTS} to {MAX_VARIANTS} images, got {n}.")
    labels = labels_for(n)

    hashes = []
    for lab, data in zip(labels, datas):  # validate everything up front so a bad file fails fast with its label
        try:
            hashes.append(ingest(data, pipeline.cfg["ingest"]).sha256)
        except IngestError as e:
            raise CompareError(422, f"Variant {lab}: {e}") from e
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] == hashes[j]:
                raise CompareError(422, f"Variants {labels[i]} and {labels[j]} are the same image.")

    futures = [pipeline.compare_pool.submit(pipeline.analyze, d) for d in datas]
    results = [f.result() for f in futures]
    for lab, r in zip(labels, results):
        if not r.get("scores") or r["scores"].get("overall") is None:
            raise CompareError(503, f"Variant {lab} could not be scored (an analyzer is unavailable: {', '.join(r.get('unavailable', [])) or 'unknown'}).")

    win, margin, close = pick_winner(results)
    base_map = decode_attention_png(results[0]["attention_png"]) if results[0].get("attention_png") else None
    diffs = [None] + [
        encode_diff_png(base_map, decode_attention_png(r["attention_png"])) if base_map is not None and r.get("attention_png") else None
        for r in results[1:]
    ]
    comparison_id = hashlib.sha256("|".join(sorted(r["id"] for r in results)).encode()).hexdigest()
    pipeline.store.save_comparison(comparison_id, [r["id"] for r in results], results[win]["id"])

    return {
        "id": comparison_id,
        "labels": labels,
        "variants": results,
        "score_deltas": score_deltas(results),
        "winner_id": results[win]["id"],
        "winner_label": labels[win],
        "margin": margin,
        "close": close,
        "reasons": explain(results, labels, win, margin, close, pipeline.scoring_cfg["overall_weights"]),
        "rules_differ": rules_that_differ(results, labels),
        "difference_maps": diffs,
    }
