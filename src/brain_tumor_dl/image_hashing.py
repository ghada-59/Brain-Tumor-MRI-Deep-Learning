"""Image fingerprints used by the duplicate and near-duplicate audits.

Three levels, from strictest to loosest. None of them identifies a patient:

* ``file_sha256``  - identical bytes on disk (same file, possibly renamed).
* ``pixel_sha256`` - identical decoded RGB pixels and size (same image, re-encoded).
* ``perceptual_hash`` - 64-bit DCT hash; a small Hamming distance means the images
  look alike (re-exports, resized copies, adjacent slices, or unrelated look-alikes).

``perceptual_hash`` is the same algorithm as ``audit_near_duplicates.py`` at the
repository root, so distances computed here are comparable with
``reports_near_duplicates_20261021``. A test checks that the two stay identical.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from src.brain_tumor_dl.experiment_record import file_sha256

__all__ = ["file_sha256", "pixel_sha256", "perceptual_hash", "hamming_distance"]

_DCT_SIZE = 32


def _dct_matrix(n: int = _DCT_SIZE) -> np.ndarray:
    positions = np.arange(n)
    dct = np.cos(np.pi * (2 * positions[None, :] + 1) * positions[:, None] / (2 * n))
    dct[0, :] *= np.sqrt(1 / n)
    dct[1:, :] *= np.sqrt(2 / n)
    return dct


_DCT = _dct_matrix()


def pixel_sha256(path: Path) -> str:
    """Hash of the decoded RGB pixels and the image size, not of the file encoding."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = sha256()
        digest.update(f"{rgb.width}x{rgb.height}:".encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """64-bit DCT perceptual hash (grayscale, 32x32 Lanczos, top-left 8x8, median threshold)."""
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    coefficients = _DCT @ pixels @ _DCT.T
    low_frequency = coefficients[:8, :8].flatten()
    threshold = np.median(low_frequency[1:])
    result = 0
    for bit in low_frequency > threshold:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
