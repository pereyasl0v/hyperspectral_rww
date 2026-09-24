from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb_image(
    rgb_image: np.ndarray,
    output_path: str | Path = "data/output/visualization/rgb.png",
) -> Path:
    """
    Сохранить RGB-изображение в PNG.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb_image = np.asarray(
        rgb_image,
        dtype=np.float32,
    )

    rgb_image = np.clip(
        rgb_image,
        0.0,
        1.0,
    )

    rgb_uint8 = np.rint(
        rgb_image * 255.0
    ).astype(np.uint8)

    image = Image.fromarray(
        rgb_uint8,
        mode="RGB",
    )

    image.save(output_path)

    return output_path