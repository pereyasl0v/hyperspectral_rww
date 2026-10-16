# Hyperspectral Downlink Planner

FastAPI + React research prototype for the chain:

**RWW → local RX anomaly score → priority spectral-spatial region growing → region utility → knapsack downlink selection → transmission queue**.

## RWW format used by the prototype

- first 1024 bytes: service/header data, ignored;
- payload: repeated uint8 frames;
- one frame: 696 spatial samples × 64 spectral channels;
- cube layout: `[scan_line, 696, 64]`.

The supplied development file was validated as exactly `4971 × 696 × 64` after removing the 1024-byte header.

## Algorithm

1. Memory-map the RWW payload.
2. Downsample spatially by an adjustable analysis stride.
3. Reduce spectral covariance dimensionality with PCA.
4. Compute a scalable block-local RXD approximation: each tile is scored against statistics from its local 3×3-tile neighborhood.
5. Rank high-RX pixels and use them as seeds.
6. Grow spatially connected regions in priority order. Candidate pixels must satisfy both the RX growth threshold and a Spectral Angle Mapper (SAM) threshold relative to the current running region spectrum.
7. Compute region utility from robustly normalized max LRXD, mean LRXD, and spectral coherence.
8. Estimate transmission cost as full 64-channel uint8 region samples plus mask/metadata overhead.
9. Solve a 0/1 knapsack problem using the manually specified downlink budget `Mbit/s × session seconds`.
10. The selected subset is ordered for transmission by utility per transmitted Mbit. Hovering a queue item highlights its region on the scene.

The knapsack determines **which** regions are transmitted; queue ordering is a separate explicit policy.

## Important scientific limitation

Only raw ADC counts were supplied. No wavelength calibration, radiometric calibration, bad-pixel map, dark-frame correction, or geometric metadata is assumed. Therefore the RGB view is a **pseudo-RGB** assembled from arbitrary ADC channels and should not be interpreted as natural color.

The current LRXD implementation is a computationally practical block-local approximation for interactive prototyping, not yet a validated flight implementation.

## Run

Requirements: Python 3.11–3.13, Poetry, Node.js 20+.

```bash
cd downlink_planner
poetry install
cd frontend
npm install
cd ..
```

Either place the local raw file at `downlink_planner/data/file.rww` or point directly to it:

```bash
export RWW_PATH=/absolute/path/to/file.rww
```

Backend:

```bash
poetry run uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

The UI can also upload an `.rww` file directly.

## Default demonstration parameters

- downlink: 10 Mbit/s for 5 s = 50 Mbit;
- analysis stride: 4;
- PCA dimensions: 8;
- local RX tile: 32 analysis pixels;
- seed percentile: 99.5;
- growth percentile: 85;
- SAM threshold: 12°;
- minimum region: 10 analysis pixels;
- maximum regions: 200.

For the supplied RWW test file, these defaults produced 157 candidate regions; the 50 Mbit knapsack budget selected 130 regions using about 47.83 Mbit.

## API

- `GET /api/dataset` — parsed RWW metadata
- `POST /api/dataset/upload` — load RWW
- `GET /api/preview.png` — pseudo-RGB or individual ADC channel
- `POST /api/analyze` — LRXD, grouping and downlink optimization
- `GET /api/analysis/{id}/lrx.png` — anomaly map
- `GET /api/analysis/{id}/regions.png` — region overlay
- `GET /api/analysis/{id}/region/{region_id}.png` — hovered-region overlay
