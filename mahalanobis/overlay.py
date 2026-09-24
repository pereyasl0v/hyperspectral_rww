"""Наложение маски отклонений красным цветом."""

from __future__ import annotations

import numpy as np


def highlight_anomalies(
    base_image: np.ndarray,
    anomaly_mask: np.ndarray,
    alpha: float = 0.85,
) -> np.ndarray:
    """Наложить аномальные пиксели красным поверх серого изображения."""
    base_image = np.asarray(base_image, dtype=np.float32)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    if base_image.shape != anomaly_mask.shape:
        raise ValueError("base_image и anomaly_mask должны иметь одинаковую форму")
    if base_image.ndim != 2:
        raise ValueError("base_image должен быть двумерным")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha должен быть в диапазоне [0, 1]")

    gray = np.clip(base_image, 0.0, 1.0)
    rgb = np.stack([gray, gray, gray], axis=-1)

    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    rgb[anomaly_mask] = alpha * red[anomaly_mask] + (1.0 - alpha) * rgb[anomaly_mask]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
