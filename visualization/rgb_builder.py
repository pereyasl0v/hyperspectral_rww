from __future__ import annotations

import numpy as np


def build_rgb_image(
    cube: np.ndarray,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
) -> np.ndarray:
    """
    Построить RGB из гиперспектрального куба [Y, X, K].

    Каналы RGB выбираются как ближайшие спектральные каналы
    к 450 нм (B), 550 нм (G) и 650 нм (R).
    """

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Ожидался cube [Y, X, K], получено {cube.shape}"
        )

    y_size, x_size, k_size = cube.shape

    if k_size < 2:
        raise ValueError("Количество спектральных каналов должно быть >= 2")

    if wavelength_min_nm >= wavelength_max_nm:
        raise ValueError(
            "Минимальная длина волны должна быть меньше максимальной"
        )

    # Спектральные длины волн всех каналов.
    wavelengths = np.linspace(
        wavelength_min_nm,
        wavelength_max_nm,
        k_size,
    )

    # Целевые длины волн RGB.
    blue_target = 450.0
    green_target = 550.0
    red_target = 650.0

    # Если целевая длина волны за пределами диапазона камеры,
    # используем ближайшую доступную границу.
    blue_target = np.clip(
        blue_target,
        wavelength_min_nm,
        wavelength_max_nm,
    )

    green_target = np.clip(
        green_target,
        wavelength_min_nm,
        wavelength_max_nm,
    )

    red_target = np.clip(
        red_target,
        wavelength_min_nm,
        wavelength_max_nm,
    )

    # Ищем ближайшие каналы.
    blue_channel = np.argmin(
        np.abs(wavelengths - blue_target)
    )

    green_channel = np.argmin(
        np.abs(wavelengths - green_target)
    )

    red_channel = np.argmin(
        np.abs(wavelengths - red_target)
    )

    print(
        f"B: канал {blue_channel + 1}, "
        f"{wavelengths[blue_channel]:.2f} нм"
    )

    print(
        f"G: канал {green_channel + 1}, "
        f"{wavelengths[green_channel]:.2f} нм"
    )

    print(
        f"R: канал {red_channel + 1}, "
        f"{wavelengths[red_channel]:.2f} нм"
    )

    # Формируем RGB.
    red = cube[:, :, red_channel]
    green = cube[:, :, green_channel]
    blue = cube[:, :, blue_channel]

    rgb = np.stack(
        [red, green, blue],
        axis=-1,
    )

    # Контроль.
    expected_shape = (y_size, x_size, 3)

    if rgb.shape != expected_shape:
        raise RuntimeError(
            f"Получена неправильная форма RGB: {rgb.shape}, "
            f"ожидалось {expected_shape}"
        )

    return rgb