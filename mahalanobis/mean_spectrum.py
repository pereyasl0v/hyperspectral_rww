"""Оценка среднего спектра фона."""

from __future__ import annotations

import numpy as np


def calculate_mean_spectrum(cube: np.ndarray) -> np.ndarray:
    """Рассчитать mu_k по всем пространственным координатам X, Y."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube должен иметь форму [Y, X, K], получено {cube.shape}")
    return cube.mean(axis=(0, 1), dtype=np.float64)
