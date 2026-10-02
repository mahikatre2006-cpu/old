"""M5. Attention share per region and per element type (face / text / subject / background)."""
from __future__ import annotations

import numpy as np

from pipeline.context import AnalyzerResult, Region


def region_mask(r: Region, shape: tuple[int, int]) -> np.ndarray:
    if r.mask is not None:
        return r.mask.astype(bool)
    m = np.zeros(shape, dtype=bool)
    if r.box is not None:
        x0, y0, x1, y1 = r.box
        m[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
    return m


def _box_of(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def compute_shares(
    fused: np.ndarray,
    results: dict[str, AnalyzerResult],
    priority: list[str],
) -> tuple[list[dict], dict[str, float]]:
    """Each pixel belongs to the first matching region in `priority` order, the rest is background,
    so the per-type shares sum to 100. Returns (region dicts for the API, {type: pct})."""
    total = float(fused.sum())
    if total <= 0:
        return [], {t: 0.0 for t in [*priority, "background"]}

    taken = np.zeros(fused.shape, dtype=bool)
    regions: list[dict] = []
    by_type = {t: 0.0 for t in priority}

    for rtype in priority:
        for res in results.values():
            for r in res.regions:
                if r.type != rtype:
                    continue
                own = region_mask(r, fused.shape) & ~taken
                taken |= own
                pct = float(fused[own].sum()) / total * 100
                by_type[rtype] += pct
                box = list(r.box) if r.box is not None else _box_of(own)
                regions.append({"type": rtype, "box": box, "attention_pct": round(pct, 1), **r.attrs})

    background = float(fused[~taken].sum()) / total * 100
    regions.append({"type": "background", "box": None, "attention_pct": round(background, 1)})
    shares = {**{t: round(v, 1) for t, v in by_type.items()}, "background": round(background, 1)}
    return regions, shares
