from __future__ import annotations

from dataclasses import dataclass, asdict
import heapq
import math
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA

from .rww import RWWMeta, open_cube


@dataclass
class AnalysisConfig:
    channel_mbps: float = 10.0
    session_seconds: float = 5.0
    analysis_stride: int = 4
    pca_components: int = 8
    local_tile: int = 32
    seed_percentile: float = 99.5
    growth_percentile: float = 85.0
    spectral_angle_deg: float = 12.0
    min_region_pixels: int = 10
    max_regions: int = 200
    regularization: float = 0.05


@dataclass
class RegionInfo:
    id: int
    lowres_pixels: int
    estimated_pixels: int
    max_lrx: float
    mean_lrx: float
    spectral_coherence: float
    utility: float
    bytes: int
    bbox: list[int]
    selected: bool = False
    transmission_order: int | None = None
    transmission_start_s: float | None = None
    transmission_end_s: float | None = None
    utility_per_mbit: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    score: np.ndarray
    labels: np.ndarray
    regions: list[RegionInfo]
    stride: int
    capacity_bytes: int
    used_bytes: int
    pca_explained_variance: float


def _fit_pca(sample_cube: np.ndarray, n_components: int, max_samples: int = 60000) -> tuple[PCA, np.ndarray]:
    h, w, bands = sample_cube.shape
    X = sample_cube.reshape(-1, bands).astype(np.float32)
    n_components = max(2, min(n_components, bands))
    if X.shape[0] > max_samples:
        ids = np.linspace(0, X.shape[0] - 1, max_samples, dtype=np.int64)
        fit_x = X[ids]
    else:
        fit_x = X
    pca = PCA(n_components=n_components, whiten=False, svd_solver="randomized", random_state=0)
    pca.fit(fit_x)
    Z = pca.transform(X).astype(np.float32).reshape(h, w, n_components)
    return pca, Z


def _block_local_rx(z: np.ndarray, tile: int, regularization: float) -> np.ndarray:
    """Scalable LRXD approximation: score each tile against a 3x3-tile local background."""
    h, w, d = z.shape
    out = np.empty((h, w), dtype=np.float32)
    tile = max(8, int(tile))
    for y0 in range(0, h, tile):
        y1 = min(h, y0 + tile)
        ny0, ny1 = max(0, y0 - tile), min(h, y1 + tile)
        for x0 in range(0, w, tile):
            x1 = min(w, x0 + tile)
            nx0, nx1 = max(0, x0 - tile), min(w, x1 + tile)
            bg = z[ny0:ny1, nx0:nx1].reshape(-1, d).astype(np.float64)
            mu = bg.mean(axis=0)
            centered = bg - mu
            cov = centered.T @ centered / max(1, len(bg) - 1)
            scale = float(np.trace(cov) / d) if np.isfinite(cov).all() else 1.0
            cov += np.eye(d) * max(1e-6, regularization * max(scale, 1e-6))
            inv = np.linalg.pinv(cov, hermitian=True)
            core = z[y0:y1, x0:x1].reshape(-1, d).astype(np.float64) - mu
            rx = np.einsum("ij,jk,ik->i", core, inv, core)
            out[y0:y1, x0:x1] = rx.reshape(y1 - y0, x1 - x0).astype(np.float32)
    return out


def _spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False)
    bb = b.astype(np.float64, copy=False)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom <= 1e-12:
        return 180.0
    c = float(np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))
    return math.degrees(math.acos(c))


def _neighbors(y: int, x: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                yield yy, xx


def _grow_regions(
    spectra: np.ndarray,
    score: np.ndarray,
    seed_percentile: float,
    growth_percentile: float,
    angle_threshold: float,
    min_pixels: int,
    max_regions: int,
) -> tuple[np.ndarray, list[dict]]:
    h, w, _ = spectra.shape
    labels = np.zeros((h, w), dtype=np.int32)
    seed_t = float(np.percentile(score, seed_percentile))
    growth_t = float(np.percentile(score, growth_percentile))
    candidates = score >= growth_t
    seed_flat = np.flatnonzero(score.ravel() >= seed_t)
    seed_flat = seed_flat[np.argsort(score.ravel()[seed_flat])[::-1]]
    region_stats: list[dict] = []
    label_id = 0
    rejected = np.zeros_like(labels, dtype=bool)

    for flat in seed_flat:
        if len(region_stats) >= max_regions:
            break
        sy, sx = divmod(int(flat), w)
        if labels[sy, sx] != 0 or rejected[sy, sx]:
            continue
        heap: list[tuple[float, int, int]] = [(-float(score[sy, sx]), sy, sx)]
        queued = {(sy, sx)}
        pixels: list[tuple[int, int]] = []
        mean = spectra[sy, sx].astype(np.float64)
        mean_n = 0
        angles: list[float] = []

        while heap:
            _, y, x = heapq.heappop(heap)
            if labels[y, x] != 0:
                continue
            if not candidates[y, x] and (y, x) != (sy, sx):
                continue
            s = spectra[y, x]
            angle = 0.0 if mean_n == 0 else _spectral_angle_deg(s, mean)
            if mean_n and angle > angle_threshold:
                continue
            pixels.append((y, x))
            if mean_n == 0:
                mean = s.astype(np.float64)
            else:
                mean += (s - mean) / (mean_n + 1)
                angles.append(angle)
            mean_n += 1
            labels[y, x] = -1
            for yy, xx in _neighbors(y, x, h, w):
                if labels[yy, xx] == 0 and candidates[yy, xx] and (yy, xx) not in queued:
                    queued.add((yy, xx))
                    heapq.heappush(heap, (-float(score[yy, xx]), yy, xx))

        if len(pixels) < min_pixels:
            for y, x in pixels:
                labels[y, x] = 0
                rejected[y, x] = True
            continue

        label_id += 1
        ys = np.fromiter((p[0] for p in pixels), dtype=np.int32)
        xs = np.fromiter((p[1] for p in pixels), dtype=np.int32)
        labels[ys, xs] = label_id
        values = score[ys, xs]
        coherence = 1.0 if not angles else float(np.clip(1.0 - np.mean(angles) / max(angle_threshold, 1e-6), 0.0, 1.0))
        region_stats.append({
            "id": label_id,
            "pixels": len(pixels),
            "max_lrx": float(values.max()),
            "mean_lrx": float(values.mean()),
            "coherence": coherence,
            "bbox_low": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
        })
    return labels, region_stats


def _robust01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    lo, hi = np.percentile(values, [5, 95]) if len(values) > 2 else (values.min(), values.max())
    if hi <= lo + 1e-12:
        return np.ones_like(values) * 0.5
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _knapsack(regions: list[RegionInfo], capacity_bytes: int, quantum: int = 4096) -> set[int]:
    if capacity_bytes <= 0 or not regions:
        return set()
    capacity = max(1, capacity_bytes // quantum)
    weights = [max(1, math.ceil(r.bytes / quantum)) for r in regions]
    if capacity > 100_000:
        quantum = math.ceil(capacity_bytes / 100_000)
        capacity = max(1, capacity_bytes // quantum)
        weights = [max(1, math.ceil(r.bytes / quantum)) for r in regions]
    dp = np.zeros(capacity + 1, dtype=np.float64)
    take = np.full((len(regions), capacity + 1), False, dtype=bool)
    for i, (r, wt) in enumerate(zip(regions, weights)):
        if wt > capacity:
            continue
        prev = dp.copy()
        cand = prev[:-wt] + float(r.utility)
        better = cand > dp[wt:] + 1e-12
        dp[wt:][better] = cand[better]
        take[i, wt:][better] = True
    c = int(np.argmax(dp))
    chosen: set[int] = set()
    for i in range(len(regions) - 1, -1, -1):
        wt = weights[i]
        if c >= wt and take[i, c]:
            chosen.add(regions[i].id)
            c -= wt
    return chosen


def analyze(meta: RWWMeta, cfg: AnalysisConfig) -> AnalysisResult:
    cube = open_cube(meta)
    stride = max(1, int(cfg.analysis_stride))
    small = np.asarray(cube[::stride, ::stride, :], dtype=np.uint8)
    pca, z = _fit_pca(small, cfg.pca_components)
    score = _block_local_rx(z, cfg.local_tile, cfg.regularization)
    labels, raw_regions = _grow_regions(
        small, score, cfg.seed_percentile, cfg.growth_percentile,
        cfg.spectral_angle_deg, cfg.min_region_pixels, cfg.max_regions,
    )
    capacity_bytes = int(cfg.channel_mbps * 1_000_000 / 8 * cfg.session_seconds)
    if not raw_regions:
        return AnalysisResult(score, labels, [], stride, capacity_bytes, 0, float(pca.explained_variance_ratio_.sum()))

    max_rx = _robust01(np.array([r["max_lrx"] for r in raw_regions]))
    mean_rx = _robust01(np.array([r["mean_lrx"] for r in raw_regions]))
    coherence = np.array([r["coherence"] for r in raw_regions], dtype=np.float64)
    regions: list[RegionInfo] = []
    for idx, raw in enumerate(raw_regions):
        est_pixels = int(raw["pixels"] * stride * stride)
        x0, y0, x1, y1 = raw["bbox_low"]
        bbox_full = [x0 * stride, y0 * stride, min(meta.width, x1 * stride), min(meta.frames, y1 * stride)]
        bbox_area = max(1, (bbox_full[2] - bbox_full[0]) * (bbox_full[3] - bbox_full[1]))
        mask_bytes = math.ceil(bbox_area / 8)
        bytes_cost = int(est_pixels * meta.bands + mask_bytes + 512)
        utility = float(0.50 * max_rx[idx] + 0.30 * mean_rx[idx] + 0.20 * coherence[idx])
        mbit = max(bytes_cost * 8 / 1_000_000, 1e-9)
        regions.append(RegionInfo(
            id=raw["id"], lowres_pixels=raw["pixels"], estimated_pixels=est_pixels,
            max_lrx=raw["max_lrx"], mean_lrx=raw["mean_lrx"], spectral_coherence=raw["coherence"],
            utility=utility, bytes=bytes_cost, bbox=bbox_full, utility_per_mbit=utility / mbit,
        ))

    chosen = _knapsack(regions, capacity_bytes)
    selected = [r for r in regions if r.id in chosen]
    selected.sort(key=lambda r: (r.utility_per_mbit, r.max_lrx), reverse=True)
    elapsed = 0.0
    bytes_per_second = max(cfg.channel_mbps * 1_000_000 / 8, 1e-9)
    for order, r in enumerate(selected, start=1):
        r.selected = True
        r.transmission_order = order
        r.transmission_start_s = elapsed
        elapsed += r.bytes / bytes_per_second
        r.transmission_end_s = elapsed
    used = sum(r.bytes for r in selected)
    regions.sort(key=lambda r: (0 if r.selected else 1, r.transmission_order or 10**9, -r.utility))
    return AnalysisResult(score, labels, regions, stride, capacity_bytes, used, float(pca.explained_variance_ratio_.sum()))
