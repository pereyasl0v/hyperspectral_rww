"""Выделение наиболее отклонённых пикселей."""

from __future__ import annotations

import numpy as np


def create_anomaly_mask(
    distance_map: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    """Выделить пиксели выше заданного процентиля D².

    Процентиль здесь является только параметром визуального выделения.
    Научный порог детектирования можно будет заменить отдельной функцией позже.
    """
    distance_map = np.asarray(distance_map, dtype=np.float64)
    if distance_map.size == 0:
        raise ValueError("Карта расстояний не должна быть пустой")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile должен быть между 0 и 100")

    threshold = float(np.percentile(distance_map, percentile))
    return distance_map > threshold
