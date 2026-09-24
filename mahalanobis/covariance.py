"""Оценка ковариационной матрицы спектральных признаков."""

from __future__ import annotations

import numpy as np


def calculate_covariance_matrix(
    cube: np.ndarray,
    mean_spectrum: np.ndarray,
    block_size: int = 100_000,
) -> np.ndarray:
    """Рассчитать выборочную ковариационную матрицу Sigma по всем пикселям."""
    cube = np.asarray(cube)
    mean_spectrum = np.asarray(mean_spectrum, dtype=np.float64)

    if cube.ndim != 3:
        raise ValueError("cube должен иметь форму [Y, X, K]")
    if mean_spectrum.shape != (cube.shape[2],):
        raise ValueError("Размер mean_spectrum не соответствует K")
    if block_size <= 0:
        raise ValueError("block_size должен быть положительным")

    samples = cube.reshape(-1, cube.shape[2])
    sample_count = samples.shape[0]
    if sample_count < 2:
        raise ValueError("Для ковариации нужны минимум два пикселя")

    covariance = np.zeros((cube.shape[2], cube.shape[2]), dtype=np.float64)
    for start in range(0, sample_count, block_size):
        block = samples[start : start + block_size].astype(np.float64, copy=False)
        centered = block - mean_spectrum
        covariance += centered.T @ centered

    covariance /= sample_count - 1
    return covariance
