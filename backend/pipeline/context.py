"""Shared types: Context, Region, AnalyzerResult (PRD section 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Region:
    type: str  # face | text | subject | background
    box: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 on the 1280x720 frame
    mask: np.ndarray | None = None  # HxW bool, optional
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    attention_map: np.ndarray | None = None  # HxW float32, normalized 0-1
    regions: list[Region] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Context:
    """Everything analyzers need. Built once by M2, read-only for analyzers."""

    image_hash: str
    original: np.ndarray  # 720x1280x3 uint8 RGB
    model_input: np.ndarray  # saliency model input, uint8 RGB
    views: dict[str, np.ndarray]  # desktop / mobile / sidebar renders, uint8 RGB
    ui_mask: np.ndarray  # 720x1280 uint8, 1 where YouTube UI covers the thumbnail
    config: dict[str, Any] = field(default_factory=dict)
    results: dict[str, AnalyzerResult] = field(default_factory=dict)  # filled for stage-2 analyzers
    fused: np.ndarray | None = None  # fused attention map from stage-1 results, set before stage 2 runs
