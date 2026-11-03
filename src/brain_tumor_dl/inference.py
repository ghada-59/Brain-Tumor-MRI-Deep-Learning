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


def preprocess_image_bytes(payload: bytes, image_size: int = 224) -> np.ndarray:
    """Decode encoded image bytes exactly as the training pipeline does.

    Uses ``tf.io.decode_image`` (3 channels) and bilinear ``tf.image.resize``, the same
    operations as ``data._image_dataset``, so served pixels equal training pixels.
    Pixels stay in the raw 0-255 range; the model rescales them internally.
    """
    import tensorflow as tf

    if image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    image = tf.io.decode_image(payload, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.cast(tf.image.resize(image, (image_size, image_size)), tf.float32)
    batch = image.numpy()[None, ...]
    if batch.shape != (1, image_size, image_size, 3) or not np.all(np.isfinite(batch)):
        raise ValueError(f"Unexpected preprocessed image: shape {batch.shape}.")
    return batch
