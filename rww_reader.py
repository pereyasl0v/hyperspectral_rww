"""Чтение RWW-контейнера."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class RWWConfig:
    """Геометрия и формат RAW-части RWW."""

    header_size: int = 1024
    x_size: int = 696          # первая пространственная координата X
    k_size: int = 64           # спектральная координата K
    dtype: np.dtype = np.dtype(np.uint8)

    @property
    def elements_per_frame(self) -> int:
        return self.x_size * self.k_size

    @property
    def frame_bytes(self) -> int:
        return self.elements_per_frame * self.dtype.itemsize


@dataclass
class RWWData:
    """Результат чтения RWW."""

    header: bytes
    payload: np.ndarray
    config: RWWConfig

    @property
    def y_size(self) -> int:
        """Y — вторая пространственная координата, равная числу кадров."""
        return self.payload.size // self.config.elements_per_frame


def read_rww(path: str | Path, config: RWWConfig | None = None) -> RWWData:
    """Считать RWW и вернуть служебную часть и RAW-поток."""

    path = Path(path)
    config = config or RWWConfig()

    if not path.is_file():
        raise FileNotFoundError(f"RWW-файл не найден: {path}")

    if config.header_size < 0:
        raise ValueError("Размер заголовка не может быть отрицательным")

    file_size = path.stat().st_size
    if file_size < config.header_size:
        raise ValueError(
            f"Размер файла ({file_size} байт) меньше заголовка "
            f"({config.header_size} байт)"
        )

    with path.open("rb") as file:
        header = file.read(config.header_size)
        raw = file.read()

    if len(raw) % config.frame_bytes != 0:
        raise ValueError(
            "RAW-область не кратна размеру полного кадра: "
            f"RAW={len(raw)} байт, кадр={config.frame_bytes} байт"
        )

    payload = np.frombuffer(raw, dtype=config.dtype).copy()

    return RWWData(
        header=header,
        payload=payload,
        config=config,
    )
