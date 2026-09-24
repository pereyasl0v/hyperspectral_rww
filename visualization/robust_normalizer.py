"""Устойчивая нормализация только для визуализации."""

from __future__ import annotations

import numpy as np


def robust_normalize_image(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Нормализовать изображение [0, 1] через процентильный диапазон."""
    image = np.asarray(image)
    if image.size == 0:
        raise ValueError("Изображение не должно быть пустым")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Процентили должны удовлетворять 0 <= lower < upper <= 100")

    values = image.astype(np.float32, copy=False)
    low, high = np.percentile(values, [lower_percentile, upper_percentile])
    low = float(low)
    high = float(high)

    if high <= low:
        return np.zeros_like(values, dtype=np.float32)

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def robust_normalize_channels(
    channels: list[np.ndarray],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> list[np.ndarray]:
    """Устойчиво нормализовать каждый спектральный канал для просмотра."""
    return [
        robust_normalize_image(channel, lower_percentile, upper_percentile)
        for channel in channels
    ]
