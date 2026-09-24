"""Построение сводной визуализации спектральных каналов."""

from __future__ import annotations

import numpy as np


def build_mean_visualization(channels: list[np.ndarray]) -> np.ndarray:
    """Усреднить нормализованные каналы в одно изображение [Y, X]."""
    if not channels:
        raise ValueError("Список каналов не должен быть пустым")
    stack = np.stack([np.asarray(channel, dtype=np.float32) for channel in channels], axis=0)
    return np.clip(stack.mean(axis=0), 0.0, 1.0).astype(np.float32)
