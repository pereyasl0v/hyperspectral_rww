"""Главная точка запуска гиперспектрального проекта.

main.py только управляет последовательностью вызовов модулей.
Сами вычисления находятся в отдельных функциях.
"""

from pathlib import Path
import json

import numpy as np

from channel_builder import build_channel_images
from cube_builder import frames_to_cube
from frame_splitter import split_frames
from mahalanobis.anomaly_mask import create_anomaly_mask
from mahalanobis.covariance import calculate_covariance_matrix
from mahalanobis.distance import calculate_mahalanobis_distance_map
from mahalanobis.distance_visualization import stretch_distance_map
from mahalanobis.image_saver import save_anomaly_overlay, save_distance_map
from mahalanobis.mean_spectrum import calculate_mean_spectrum
from mahalanobis.overlay import highlight_anomalies
from rww_reader import RWWConfig, read_rww
from visualization.composite import build_mean_visualization
from visualization.image_saver import save_channel_images as save_visual_channel_images
from visualization.image_saver import save_grayscale_image
from visualization.robust_normalizer import robust_normalize_channels
from visualization.rgb_builder import build_rgb_image
from visualization.rgb_image_saver import save_rgb_image

# -----------------------------------------------------------------------------
# НАСТРОЙКИ
# -----------------------------------------------------------------------------

INPUT_FILE = Path("data/input/file.rww")
OUTPUT_DIR = Path("data/output")

X = 1392
K = 128
HEADER_SIZE = 1024

# Робастная нормализация используется только для визуализации.
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 98.0

# Параметр визуального выделения наиболее отклонённых пикселей.
ANOMALY_PERCENTILE = 98.0

# Размер блока вычислений Mahalanobis.
BLOCK_SIZE = 100_000


def main() -> None:
    # 1. Считывание RWW -------------------------------------------------------
    rww = read_rww(
        INPUT_FILE,
        config=RWWConfig(
            header_size=HEADER_SIZE,
            x_size=X,
            k_size=K,
        ),
    )

    Y = rww.y_size
    print(f"RWW прочитан: X={X}, Y={Y}, K={K}")

    # 2. RAW -> последовательность кадров -----------------------------------
    raw_frames = split_frames(
        rww.payload,
        x_size=X,
        k_size=K,
    )
    print(f"Кадров получено: {len(raw_frames)}")

    # 3. Кадры -> гиперспектральный куб P[Y, X, K] -----------------------------
    cube = frames_to_cube(
        raw_frames,
        x_size=X,
        k_size=K,
    )
    print(f"Куб построен: shape={cube.shape} (Y, X, K)")

    # -------------------------------------------------------------------------
    # ПОТОК 1. ВИЗУАЛИЗАЦИЯ С ROBUST NORMALIZATION
    # -------------------------------------------------------------------------

    raw_channels = build_channel_images(cube)
    visual_channels = robust_normalize_channels(
        raw_channels,
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
    )

    visualization_dir = OUTPUT_DIR / "visualization" / "channels"
    saved_visual_channels = save_visual_channel_images(
        visual_channels,
        visualization_dir,
    )

    mean_visualization = build_mean_visualization(visual_channels)
    save_grayscale_image(
        mean_visualization,
        OUTPUT_DIR / "visualization" / "mean_spectrum.png",
    )

    print(f"Визуализация каналов сохранена: {len(saved_visual_channels)}")

    # -------------------------------------------------------------------------
    # ПОТОК 2. MAHALANOBIS НА НЕНОРМАЛИЗОВАННЫХ ДАННЫХ
    # -------------------------------------------------------------------------

    mahalanobis_dir = OUTPUT_DIR / "mahalanobis"
    mahalanobis_dir.mkdir(parents=True, exist_ok=True)

    # 4. mu = [mu_1, ..., mu_K] ----------------------------------------------
    mean_spectrum = calculate_mean_spectrum(cube)
    np.save(mahalanobis_dir / "mean_spectrum.npy", mean_spectrum)
    print("Средний спектр mu рассчитан")

    # 5. Sigma ---------------------------------------------------------------
    covariance_matrix = calculate_covariance_matrix(
        cube,
        mean_spectrum,
        block_size=BLOCK_SIZE,
    )
    np.save(mahalanobis_dir / "covariance_matrix.npy", covariance_matrix)
    print("Ковариационная матрица Sigma рассчитана")

    # 6. D²_M для каждого XY -------------------------------------------------
    distance_map = calculate_mahalanobis_distance_map(
        cube,
        mean_spectrum,
        covariance_matrix,
        block_size=BLOCK_SIZE,
    )
    np.save(mahalanobis_dir / "distance_map.npy", distance_map)
    print(
        "Расстояние Махаланобиса рассчитано: "
        f"shape={distance_map.shape}, "
        f"min={distance_map.min():.4f}, max={distance_map.max():.4f}"
    )

    # 7. Карта D², растянутая для отображения -------------------------------
    distance_visualization = stretch_distance_map(distance_map)
    save_distance_map(
        distance_visualization,
        mahalanobis_dir / "mahalanobis_distance.png",
    )

    # 8. Маска наиболее отклонённых пикселей --------------------------------
    anomaly_mask = create_anomaly_mask(
        distance_map,
        percentile=ANOMALY_PERCENTILE,
    )
    np.save(mahalanobis_dir / "anomaly_mask.npy", anomaly_mask)

    # 9. Выделение отклонений красным ----------------------------------------
    anomaly_overlay = highlight_anomalies(
        mean_visualization,
        anomaly_mask,
    )
    save_anomaly_overlay(
        anomaly_overlay,
        mahalanobis_dir / "mahalanobis_anomalies_red.png",
    )

    print(
        f"Пикселей выделено как отклонения: "
        f"{int(anomaly_mask.sum())} из {anomaly_mask.size}"
    )

  
    # -------------------------------------------------------------------------
    # RGB-ВИЗУАЛИЗАЦИЯ
    # -------------------------------------------------------------------------

    rgb_image = build_rgb_image(
        cube,
        wavelength_min_nm=490,
        wavelength_max_nm=1000,
    )

    print(f"RGB shape до нормализации: {rgb_image.shape}")
    print(
        f"RGB диапазон до нормализации: "
        f"{rgb_image.min()} ... {rgb_image.max()}"
    )

    # Разделяем RGB на три отдельных изображения
    rgb_channels = [
        rgb_image[:, :, 0],  # R
        rgb_image[:, :, 1],  # G
        rgb_image[:, :, 2],  # B
    ]

    # Нормализуем каждый канал отдельно
    rgb_channels = robust_normalize_channels(
        rgb_channels,
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
    )

    # Снова собираем RGB
    rgb_image = np.stack(
        rgb_channels,
        axis=-1,
    )

    print(f"RGB shape после нормализации: {rgb_image.shape}")
    print(
        f"RGB диапазон после нормализации: "
        f"{rgb_image.min():.3f} ... {rgb_image.max():.3f}"
    )

    save_rgb_image(
        rgb_image,
        OUTPUT_DIR / "visualization" / "rgb.png",
    )

    print(f"Результат: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
