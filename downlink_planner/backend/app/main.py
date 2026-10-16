from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image

from .algorithms import AnalysisConfig, AnalysisResult, analyze
from .rww import RWWMeta, inspect_rww, open_cube


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Hyperspectral Downlink Planner", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET: RWWMeta | None = None
ANALYSES: dict[str, AnalysisResult] = {}


class AnalyzeRequest(BaseModel):
    channel_mbps: float = Field(10.0, gt=0, le=10000)
    session_seconds: float = Field(5.0, gt=0, le=86400)
    analysis_stride: int = Field(4, ge=1, le=16)
    local_tile: int = Field(32, ge=8, le=128)
    seed_percentile: float = Field(99.5, gt=90, lt=100)
    growth_percentile: float = Field(85.0, gt=80, lt=100)
    spectral_angle_deg: float = Field(12.0, gt=0.1, le=45)
    min_region_pixels: int = Field(10, ge=1, le=5000)
    max_regions: int = Field(200, ge=1, le=500)
    regularization: float = Field(0.05, gt=0, le=1)


def _try_default_dataset() -> None:
    global DATASET
    candidates = [os.getenv("RWW_PATH"), str(DATA_DIR / "file.rww")]
    for raw in candidates:
        if raw and Path(raw).exists():
            DATASET = inspect_rww(raw)
            return


_try_default_dataset()


def _require_dataset() -> RWWMeta:
    if DATASET is None:
        raise HTTPException(status_code=404, detail="No RWW dataset loaded")
    return DATASET


def _stretch_u8(a: np.ndarray, lo: float = 1.0, hi: float = 98.0) -> np.ndarray:
    """Robust percentile normalization used only for display."""
    a = np.asarray(a, dtype=np.float32)
    p0, p1 = np.percentile(a, [lo, hi])
    if p1 <= p0:
        return np.zeros_like(a, dtype=np.uint8)
    return (np.clip((a - p0) / (p1 - p0), 0, 1) * 255).astype(np.uint8)


def _png(arr: np.ndarray, mode: str | None = None) -> Response:
    image = Image.fromarray(arr, mode=mode) if mode else Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "dataset_loaded": DATASET is not None}


@app.get("/api/dataset")
def dataset_info() -> dict:
    m = _require_dataset()
    return {
        "path": m.path,
        "file_size": m.file_size,
        "header_bytes": m.header_bytes,
        "frames": m.frames,
        "width": m.width,
        "bands": m.bands,
        "shape": list(m.shape),
        "dtype": "uint8",
        "interpretation": "[scan_line, spatial_sample, spectral_channel]",
    }


@app.post("/api/dataset/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict:
    global DATASET
    name = Path(file.filename or "scene.rww").name
    if not name.lower().endswith(".rww"):
        raise HTTPException(400, "Expected .rww file")
    dest = DATA_DIR / "uploaded.rww"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)
    try:
        DATASET = inspect_rww(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    ANALYSES.clear()
    return dataset_info()


@app.get("/api/preview.png")
def preview_png(
    r: int = Query(20, ge=0, le=63),
    g: int = Query(7, ge=0, le=63),
    b: int = Query(0, ge=0, le=63),
    band: int | None = Query(None, ge=0, le=63),
) -> Response:
    m = _require_dataset()
    cube = open_cube(m)
    if band is not None:
        return _png(_stretch_u8(cube[:, :, band]))
    rgb = np.stack([_stretch_u8(cube[:, :, i]) for i in (r, g, b)], axis=-1)
    return _png(rgb)


@app.post("/api/analyze")
def run_analysis(req: AnalyzeRequest) -> dict:
    m = _require_dataset()
    cfg = AnalysisConfig(**req.model_dump())
    result = analyze(m, cfg)
    analysis_id = uuid.uuid4().hex[:12]
    ANALYSES.clear()
    ANALYSES[analysis_id] = result
    return {
        "analysis_id": analysis_id,
        "capacity_bytes": result.capacity_bytes,
        "used_bytes": result.used_bytes,
        "capacity_mbit": result.capacity_bytes * 8 / 1_000_000,
        "used_mbit": result.used_bytes * 8 / 1_000_000,
        "stride": result.stride,
        "region_count": len(result.regions),
        "selected_count": sum(r.selected for r in result.regions),
        "regions": [r.to_dict() for r in result.regions],
    }


def _analysis_or_404(analysis_id: str) -> AnalysisResult:
    if analysis_id not in ANALYSES:
        raise HTTPException(404, "Analysis not found; run analysis again")
    return ANALYSES[analysis_id]


@app.get("/api/analysis/{analysis_id}/lrx.png")
def lrx_png(analysis_id: str) -> Response:
    result = _analysis_or_404(analysis_id)
    img = _stretch_u8(np.log1p(np.maximum(result.score, 0)), 1, 99.7)
    m = _require_dataset()
    pil = Image.fromarray(img).resize((m.width, m.frames), Image.Resampling.BILINEAR)
    buf = io.BytesIO(); pil.save(buf, "PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/analysis/{analysis_id}/regions.png")
def regions_png(analysis_id: str) -> Response:
    result = _analysis_or_404(analysis_id)
    labels = result.labels
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    rgba[labels > 0] = [255, 255, 255, 85]
    m = _require_dataset()
    pil = Image.fromarray(rgba, "RGBA").resize((m.width, m.frames), Image.Resampling.NEAREST)
    buf = io.BytesIO(); pil.save(buf, "PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/analysis/{analysis_id}/region/{region_id}.png")
def region_overlay_png(analysis_id: str, region_id: int) -> Response:
    from scipy.ndimage import binary_dilation

    result = _analysis_or_404(analysis_id)
    mask = result.labels == region_id
    if not mask.any():
        raise HTTPException(404, "Region not found")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = [255, 80, 40, 205]
    halo = binary_dilation(mask, iterations=1) & ~mask
    rgba[halo] = [255, 230, 80, 240]
    m = _require_dataset()
    pil = Image.fromarray(rgba, "RGBA").resize((m.width, m.frames), Image.Resampling.NEAREST)
    buf = io.BytesIO(); pil.save(buf, "PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png")
