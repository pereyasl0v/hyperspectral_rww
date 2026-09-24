"""Построение изображений отдельных спектральных каналов."""

from __future__ import annotations

import numpy as np


def build_channel_images(cube: np.ndarray) -> list[np.ndarray]:
    """Вернуть K изображений формы [Y, X] из куба [Y, X, K]."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"cube должен иметь 3 измерения, получено {cube.shape}")
    return [cube[:, :, k].copy() for k in range(cube.shape[2])]
