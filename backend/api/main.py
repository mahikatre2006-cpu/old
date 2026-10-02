"""M13 routes. Run with: uvicorn api.main:app --reload (from backend/)."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import AnalysisResult, CompareResult, GazeAggregate, GazeSessionIn, GazeSessionOut, Health
from evaluation.session import GazeError, aggregate, run_gaze_session
from pipeline.compare import CompareError, run_compare
from pipeline.ingest import IngestError
from pipeline.config import load_yaml
from pipeline.orchestrator import Pipeline

MOCK_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "mock_analysis.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline()  # models load once, here
    yield
    app.state.pipeline.pool.shutdown(wait=False)
    app.state.pipeline.compare_pool.shutdown(wait=False)


app = FastAPI(title="Thumbnail Attention Heatmap", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error(_, exc: RequestValidationError):
    """FastAPI's default 422 echoes the offending input back, which crashes the response when that input is
    NaN or Infinity (not valid JSON). Return only where and why."""
    return JSONResponse(status_code=422, content={"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]})


@app.get("/health", response_model=Health)
def health():
    status = app.state.pipeline.status
    ok = not any(v.startswith("error") for v in status.values())
    return Health(status="ok" if ok else "degraded", analyzers=status)


@app.post("/analyze", response_model=AnalysisResult)
def analyze(file: UploadFile = File(...)):
    if os.environ.get("THUMB_MOCK") == "1":  # lets the frontend build against the frozen contract
        return json.loads(MOCK_PATH.read_text(encoding="utf-8"))
    data = file.file.read()
    try:
        return app.state.pipeline.analyze(data)
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/compare", response_model=CompareResult)
def compare(files: list[UploadFile] = File(...)):
    """2 to 4 images -> results per variant, score deltas vs A, a winner with reasons, difference maps."""
    datas = [f.file.read() for f in files]
    try:
        return run_compare(app.state.pipeline, datas)
    except CompareError as e:
        raise HTTPException(status_code=e.status, detail=e.message)


@app.post("/gaze-session", response_model=GazeSessionOut)
def gaze_session(body: GazeSessionIn):
    """Real viewing data (clicks or WebGazer samples) for a thumbnail -> CC / NSS against our prediction."""
    try:
        return run_gaze_session(app.state.pipeline.store, load_yaml("evaluation.yaml"), body.model_dump())
    except GazeError as e:
        raise HTTPException(status_code=e.status, detail=e.message)


@app.get("/gaze-sessions/{thumbnail_id}", response_model=GazeAggregate)
def gaze_sessions(thumbnail_id: str):
    """Every saved session for one thumbnail, with pooled-viewer metrics and the pooled human map."""
    try:
        return aggregate(app.state.pipeline.store, load_yaml("evaluation.yaml"), thumbnail_id)
    except GazeError as e:
        raise HTTPException(status_code=e.status, detail=e.message)


@app.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str):
    result = app.state.pipeline.store.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown analysis id")
    return result
