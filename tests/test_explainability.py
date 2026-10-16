import numpy as np
from PIL import Image

from src.brain_tumor_dl.explainability import (
    _salient_region_box,
    make_explanation_views,
)


def test_salient_region_box_uses_high_activation_pixels():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[3:7, 4:8] = 0.9

    box = _salient_region_box(heatmap)

    assert box == (4, 3, 7, 6)


def test_salient_region_box_returns_none_for_empty_map():
    assert _salient_region_box(np.zeros((8, 8), dtype=np.float32)) is None


def test_explanation_views_keep_original_dimensions():
    image = Image.new("RGB", (120, 80), color="gray")
    heatmap = np.zeros((7, 7), dtype=np.float32)
    heatmap[2:5, 2:5] = 1.0

    colored, overlay, box = make_explanation_views(image, heatmap)

    assert colored.size == image.size
    assert overlay.size == image.size
    assert box is not None
