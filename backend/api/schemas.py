"""Frozen API contract (PRD section 9). The frontend builds against this."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScanpathPoint(BaseModel):
    x: int
    y: int
    order: int


class RegionOut(BaseModel):
    type: Literal["face", "text", "subject", "background"]
    box: list[int] | None = None  # [x0, y0, x1, y1] in 1280x720 space
    attention_pct: float = 0.0
    emotion: str | None = None
    intensity: float | None = None  # 0..1 expression strength (from valence/arousal)
    text: str | None = None
    mobile_readable: bool | None = None
    contrast: float | None = None  # WCAG contrast ratio of text vs its surroundings
    height_pct: float | None = None  # text line height as % of frame height


class Scores(BaseModel):
    """0-100. text_readability / face_impact are None when there is no text / no face to judge."""

    overall: int | None = None
    focus: int | None = None
    text_readability: int | None = None
    face_impact: int | None = None
    clutter: int | None = None  # 100 = clean


class Recommendation(BaseModel):
    id: str
    severity: Literal["high", "medium", "low"]
    message: str
    impact: float | None = None  # 0..1, how much fixing it should matter (used for ranking)
    box: list[int] | None = None  # [x0, y0, x1, y1] of the element the fix is about, if there is one


class AnalysisResult(BaseModel):
    id: str
    heatmap_png: str | None = None  # base64 PNG, same aspect as the 1280x720 frame
    preview_jpg: str | None = None  # base64 JPEG of the normalized frame the model saw (overlay-aligned)
    attention_png: str | None = None  # base64 8-bit grayscale 160x90 fused attention map (0..1)
    scanpath: list[ScanpathPoint] = Field(default_factory=list)
    regions: list[RegionOut] = Field(default_factory=list)
    scores: Scores | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    metrics: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {"face": {}, "text": {}, "color": {}, "composition": {}}
    )
    timing_ms: dict[str, float] = Field(default_factory=dict)
    # Reliability: a failed analyzer yields a partial result, never a failed request.
    unavailable: list[str] = Field(default_factory=list)
    cached: bool = False


class RuleDiffer(BaseModel):
    id: str
    present_in: list[str]  # variant labels that need this fix; the rest do not
    example: str


class CompareResult(BaseModel):
    """POST /compare: 2-4 variants analysed side by side."""

    id: str
    labels: list[str]  # "A", "B", ...
    variants: list[AnalysisResult]
    score_deltas: list[dict[str, int | None]]  # each variant minus variant A
    winner_id: str
    winner_label: str
    margin: int  # overall points ahead of the runner-up
    close: bool  # true when the margin is too small to call confidently
    reasons: list[str]
    rules_differ: list[RuleDiffer] = Field(default_factory=list)
    difference_maps: list[str | None] = Field(default_factory=list)  # base64 RGBA PNG vs variant A; [0] is None


class GazePoint(BaseModel):
    """One gaze sample or click, in 1280x720 thumbnail-frame pixels (the client maps screen to frame)."""

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    t: float | None = Field(default=None, allow_inf_nan=False)  # ms since the thumbnail appeared (webgazer)
    order: int | None = Field(default=None, ge=1, le=9)  # 1 = first look, 2 = second look (click test)


class GazeSessionIn(BaseModel):
    """Only gaze points leave the browser; webcam frames never do."""

    thumbnail_id: str = Field(min_length=8, max_length=128)
    method: Literal["click", "webgazer"]
    points: list[GazePoint] = Field(min_length=1, max_length=3000)
    screen: dict[str, int] = Field(default_factory=dict)  # viewer's screen size, kept for the record
    participant: str | None = Field(default=None, max_length=40)  # anonymous label, optional


class FixationOut(BaseModel):
    x: float
    y: float
    order: int
    start_ms: float | None = None
    duration_ms: float | None = None


class GazeSessionOut(BaseModel):
    session_id: str
    method: Literal["click", "webgazer"]
    cc: float
    nss: float
    sim: float
    kld: float
    n_points: int
    n_fixations: int
    fixations: list[FixationOut]
    first_look_hit: bool | None = None  # real first look within the hit radius of our predicted fixation 1
    first_look_distance_px: float | None = None
    first_look_hit_top3: bool | None = None  # ...or of any of our first three
    warnings: list[str] = Field(default_factory=list)
    ground_truth_png: str | None = None  # base64 RGBA heatmap of the human map


class GazeSessionSummary(BaseModel):
    id: str
    method: str
    cc: float | None = None
    nss: float | None = None
    participant: str | None = None
    first_look_hit: bool | None = None


class GazeAggregate(BaseModel):
    thumbnail_id: str
    n_sessions: int
    sessions: list[GazeSessionSummary]
    mean_cc: float | None = None
    mean_nss: float | None = None
    first_look_hit_rate: float | None = None
    aggregate_cc: float | None = None  # prediction vs all viewers pooled
    aggregate_nss: float | None = None
    aggregate_png: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    analyzers: dict[str, str]  # name -> "loaded" | "disabled" | "error: ..."
    version: str = "0.1.0"
