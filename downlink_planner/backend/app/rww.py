from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RWWMeta:
    path: str
    file_size: int
    header_bytes: int
    width: int
    bands: int
    frames: int
    payload_bytes: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.frames, self.width, self.bands)


def inspect_rww(path: str | Path, header_bytes: int = 1024, width: int = 696, bands: int = 64) -> RWWMeta:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    payload = size - header_bytes
    frame_bytes = width * bands
    if payload <= 0:
        raise ValueError("RWW file is smaller than the service header")
    if payload % frame_bytes != 0:
        raise ValueError(
            f"RWW payload ({payload} bytes) is not divisible by one frame "
            f"({frame_bytes} bytes = {width} x {bands} x uint8)."
        )
    frames = payload // frame_bytes
    return RWWMeta(
        path=str(path.resolve()),
        file_size=size,
        header_bytes=header_bytes,
        width=width,
        bands=bands,
        frames=frames,
        payload_bytes=payload,
    )


def open_cube(meta: RWWMeta) -> np.ndarray:
    """Open RWW payload as [Y, X, K] without copying.

    One raw frame is stored on disk as [K, X]: 64 complete spectral rows,
    each containing 696 spatial samples. The on-disk view is therefore
    [Y, K, X] and is transposed to the analysis cube [Y, X, K].
    """
    raw = np.memmap(
        meta.path,
        dtype=np.uint8,
        mode="r",
        offset=meta.header_bytes,
        shape=(meta.frames, meta.bands, meta.width),
        order="C",
    )
    return raw.transpose(0, 2, 1)
