"""Сохранение результатов анализа Махаланобиса."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_distance_map(distance_visualization: np.ndarray, path: str | Path) -> Path:
    """Сохранить нормализованную карту D² как 8-битный PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.rint(np.clip(distance_visualization, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)
    return path


def save_anomaly_overlay(overlay: np.ndarray, path: str | Path) -> Path:
    """Сохранить RGB-визуализацию аномалий."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.rint(np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)
    return path
