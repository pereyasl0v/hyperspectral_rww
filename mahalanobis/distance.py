"""Расчёт расстояния Махаланобиса."""

from __future__ import annotations

import numpy as np


def calculate_mahalanobis_distance_map(
    cube: np.ndarray,
    mean_spectrum: np.ndarray,
    covariance_matrix: np.ndarray,
    block_size: int = 100_000,
) -> np.ndarray:
    """Рассчитать D²_M для каждого пикселя и вернуть карту [Y, X].

    Используется псевдообратная матрица Sigma, что позволяет устойчиво
    работать и с сильно коррелированными спектральными каналами.
    """
    cube = np.asarray(cube)
    mean_spectrum = np.asarray(mean_spectrum, dtype=np.float64)
    covariance_matrix = np.asarray(covariance_matrix, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError("cube должен иметь форму [Y, X, K]")
    k_size = cube.shape[2]
    if mean_spectrum.shape != (k_size,):
        raise ValueError("Размер mean_spectrum не соответствует K")
    if covariance_matrix.shape != (k_size, k_size):
        raise ValueError("Размер covariance_matrix не соответствует K")
    if block_size <= 0:
        raise ValueError("block_size должен быть положительным")

    inverse_covariance = np.linalg.pinv(covariance_matrix)
    samples = cube.reshape(-1, k_size)
    distances = np.empty(samples.shape[0], dtype=np.float64)

    for start in range(0, samples.shape[0], block_size):
        stop = min(start + block_size, samples.shape[0])
        block = samples[start:stop].astype(np.float64, copy=False)
        centered = block - mean_spectrum
        distances[start:stop] = np.einsum(
            "bi,ij,bj->b",
            centered,
            inverse_covariance,
            centered,
            optimize=True,
        )

    return np.maximum(distances, 0.0).reshape(cube.shape[0], cube.shape[1])
