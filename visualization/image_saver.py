"""Сохранение визуализаций."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_grayscale_image(image: np.ndarray, path: str | Path) -> Path:
    """Сохранить изображение из диапазона [0, 1] как 8-битный PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)
    return path


def save_channel_images(
    channels: list[np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    """Сохранить нормализованные спектральные каналы как PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, channel in enumerate(channels, start=1):
        saved.append(save_grayscale_image(channel, output_dir / f"channel_{index:02d}.png"))
    return saved
