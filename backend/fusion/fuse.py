"""M5. Fuse analyzer attention maps into the final heatmap."""
from __future__ import annotations

import numpy as np

from analyzers.saliency import minmax
from pipeline.context import AnalyzerResult


def fuse(results: dict[str, AnalyzerResult], weights: dict[str, float]) -> tuple[np.ndarray | None, dict[str, float]]:
    """normalize(sum_i w_i * map_i) over the maps that exist. Returns (map, weights actually used).

    Component maps are only clipped to [0, 1], never rescaled: an analyzer encodes how much a
    region matters in its amplitude (a small face peaks lower than a large one). Each analyzer is
    responsible for its own normalization; only the final fused map is min-max normalized."""
    parts = {
        name: results[name].attention_map
        for name in weights
        if weights[name] > 0 and name in results and results[name].attention_map is not None
    }
    if not parts:
        return None, {}
    total = sum(weights[n] for n in parts)
    used = {n: weights[n] / total for n in parts}
    fused = sum(used[n] * np.clip(m.astype(np.float32), 0.0, 1.0) for n, m in parts.items())
    return minmax(fused), used
