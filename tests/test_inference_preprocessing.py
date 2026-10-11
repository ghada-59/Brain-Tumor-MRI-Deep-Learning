import numpy as np
import pytest
from PIL import Image

from src.brain_tumor_dl.inference import preprocess_pil_image


def test_preprocessing_returns_expected_float32_rgb_batch() -> None:
    image = Image.new("L", (31, 19), color=128)
    batch = preprocess_pil_image(image, image_size=32)

    assert batch.shape == (1, 32, 32, 3)
    assert batch.dtype == np.float32
    assert np.all(np.isfinite(batch))
    assert np.all(batch >= 0)
    assert np.all(batch <= 255)


@pytest.mark.parametrize("image_size", [0, -1])
def test_preprocessing_rejects_invalid_image_size(image_size: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        preprocess_pil_image(Image.new("RGB", (8, 8)), image_size=image_size)
