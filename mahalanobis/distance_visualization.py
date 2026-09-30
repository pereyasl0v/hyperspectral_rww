"""Визуализация карты расстояний Махаланобиса."""


from __future__ import annotations

import numpy as np


def stretch_distance_map(
    distance_map: np.ndarray,
    lower_percentile: float = 0.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Привести карту D² к [0, 1] для отображения.

    Верхний процентиль ограничивает влияние редких экстремальных значений.
    Самые большие расстояния при этом отображаются как 255.
    """
    distance_map = np.asarray(distance_map, dtype=np.float64)
    if distance_map.size == 0:
        raise ValueError("Карта расстояний не должна быть пустой")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Некорректные процентили")

    low, high = np.percentile(distance_map, [lower_percentile, upper_percentile])
    low = float(low)
    high = float(high)
    if high <= low:
        return np.zeros_like(distance_map, dtype=np.float32)

    return np.clip((distance_map - low) / (high - low), 0.0, 1.0).astype(np.float32)
