"""Разбиение последовательного RAW-потока на кадры."""

from __future__ import annotations

import numpy as np


def split_frames(
    payload: np.ndarray,
    x_size: int = 696,
    k_size: int = 64,
) -> list[np.ndarray]:
    """Разбить RAW-поток на последовательность 1D-кадров [K*X].

    Геометрия одного кадра: [K, X],
    где X — пространственная координата, K — спектральный канал.
    Номер кадра становится координатой Y.
    """

    payload = np.asarray(payload)
    if payload.ndim != 1:
        raise ValueError("payload должен быть одномерным массивом")

    values_per_frame = x_size * k_size
    if values_per_frame <= 0:
        raise ValueError("x_size и k_size должны быть положительными")

    if payload.size % values_per_frame != 0:
        raise ValueError(
            f"Размер RAW ({payload.size}) не кратен размеру кадра "
            f"({values_per_frame})"
        )

    return [
        payload[start : start + values_per_frame].copy()
        for start in range(0, payload.size, values_per_frame)
    ]


def frame_to_matrix(
    frame: np.ndarray,
    x_size: int = 696,
    k_size: int = 64,
) -> np.ndarray:
    """Преобразовать 1D-кадр в матрицу [K, X]."""

    frame = np.asarray(frame)
    expected_size = x_size * k_size

    if frame.ndim != 1 or frame.size != expected_size:
        raise ValueError(
            f"Ожидался 1D-кадр длиной {expected_size}, "
            f"получено: shape={frame.shape}, size={frame.size}"
        )

    return frame.reshape(k_size, x_size)
