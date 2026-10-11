"""Shared image preprocessing for CLI and Streamlit inference."""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> np.ndarray:
    """Convert a PIL image to a float32 model batch with shape (1, H, W, 3)."""
    if image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    rgb = image.convert("RGB").resize((image_size, image_size))
    batch = np.asarray(rgb, dtype=np.float32)[None, ...]
    if batch.shape != (1, image_size, image_size, 3):
        raise ValueError(f"Unexpected preprocessed image shape: {batch.shape}")
    if not np.all(np.isfinite(batch)):
        raise ValueError("Preprocessed image contains non-finite values.")
    return batch
