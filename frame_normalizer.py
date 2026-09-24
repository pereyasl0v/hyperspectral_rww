"""Min-Max нормализация кадров."""

from __future__ import annotations

import numpy as np


def min_max_normalize(frame: np.ndarray) -> np.ndarray:
    """Нормализовать один кадр в диапазон [0, 1]."""

    frame = np.asarray(frame)
    if frame.size == 0:
        raise ValueError("Нельзя нормализовать пустой кадр")

    frame = frame.astype(np.float32, copy=False)
    minimum = float(frame.min())
    maximum = float(frame.max())

    if maximum == minimum:
        return np.zeros_like(frame, dtype=np.float32)

    return (frame - minimum) / (maximum - minimum)


def normalize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Нормализовать каждый кадр независимо."""

    return [min_max_normalize(frame) for frame in frames]
