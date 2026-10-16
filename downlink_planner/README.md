# Hyperspectral Downlink Planner

FastAPI + React research prototype for:

**RWW → LRXD on raw spectral vectors → Mahalanobis-priority spectral-spatial grouping → region size/deviation → 0/1 knapsack → transmission queue**.

## RWW format

- first 1024 bytes: service/header data, ignored;
- payload: repeated uint8 frames;
- one raw frame is stored as `[K, X] = [64, 696]`;
- the analysis cube is formed as `[Y, X, K]` by transposing every frame;
- the supplied file contains 4971 frames, so the cube is `4971 × 696 × 64`.

This byte order is important: reshaping the stream directly as `[Y, 696, 64]` mixes spatial samples and spectral channels and produces vertical striping.

## Preview

Preview normalization is intentionally separated from analysis.

For display only, each selected channel is robustly stretched by the 1st and 98th percentiles and clipped to `[0, 255]`. The default pseudo-RGB channels are `R=20`, `G=7`, `B=0`, corresponding approximately to the RGB choice used by the existing project for an assumed 490–1000 nm range.

**LRXD, SAM grouping and the optimizer always use raw uint8 ADC counts, not preview-normalized values.**

## Algorithm

1. Read the raw 64-channel ADC cube without PCA or spectral projection.
2. Compute local RXD/Mahalanobis scores directly in the original 64-dimensional spectral space.
3. Rank high-score pixels; the highest unprocessed score initializes the next region.
4. Grow a spatially connected region while enforcing spectral similarity with Spectral Angle Mapper (SAM). Neighbours are processed in decreasing Mahalanobis priority.
5. For every region keep its maximum and mean LRXD score and estimate the number of bytes required to transmit its 64-channel samples plus mask/metadata overhead.
6. Use the region mean Mahalanobis deviation as the knapsack value and the transmission byte count as the knapsack weight.
7. Solve the 0/1 knapsack under the manually specified budget `Mbit/s × session seconds`.
8. Order the selected regions for transmission by decreasing Mahalanobis deviation. Hovering a queue item highlights its region on the image.

`analysis_stride` is only a spatial speed control for the interactive prototype; it does not reduce the number of spectral channels.

## Run

Requirements: Python 3.11–3.13, Poetry, Node.js 20+.

```bash
cd downlink_planner
poetry install
cd frontend
npm install
cd ..
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

Open `http://127.0.0.1:5173` or `http://localhost:5173`.

The UI can upload an `.rww` file directly. Alternatively put it at `downlink_planner/data/file.rww` or set `RWW_PATH`.

## Current limitations

Only raw ADC values and the frame geometry are known. No wavelength table, radiometric calibration, dark frame, bad-pixel map or geometric metadata is assumed. The RGB image is therefore a visualization, not calibrated reflectance.

The current local RX implementation uses local tile neighbourhoods for computational practicality. It performs the Mahalanobis calculation directly in all 64 channels; no PCA is used.
