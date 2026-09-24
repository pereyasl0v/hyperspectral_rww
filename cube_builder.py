"""Построение гиперспектрального куба P[X, Y, K]."""

from __future__ import annotations

import numpy as np


def frames_to_cube(
    raw_frames: list[np.ndarray],
    x_size: int = 696,
    k_size: int = 64,
) -> np.ndarray:
    """Преобразовать raw_frames в куб формы [Y, X, K].

    raw_frames содержит последовательность кадров [K, X].
    Номер кадра является второй пространственной координатой Y.
    """
    if not raw_frames:
        raise ValueError("Список кадров не должен быть пустым")
    if x_size <= 0 or k_size <= 0:
        raise ValueError("x_size и k_size должны быть положительными")

    y_size = len(raw_frames)
    cube = np.empty((y_size, x_size, k_size), dtype=np.asarray(raw_frames[0]).dtype)

    expected_size = x_size * k_size
    for y, frame in enumerate(raw_frames):
        frame_array = np.asarray(frame)
        if frame_array.ndim != 1 or frame_array.size != expected_size:
            raise ValueError(
                f"Кадр {y} имеет неверный размер: shape={frame_array.shape}, "
                f"size={frame_array.size}; ожидалось {expected_size}"
            )
        cube[y] = frame_array.reshape(k_size, x_size).T

    return cube
