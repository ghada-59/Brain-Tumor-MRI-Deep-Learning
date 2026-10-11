"""Audit exact duplicate images across dataset splits without changing files."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .data import SUPPORTED_EXTENSIONS


def image_fingerprints(path: Path) -> tuple[str, str]:
    """Return byte-level and decoded-pixel fingerprints for an image."""
    byte_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:".encode("ascii") + rgb.tobytes()
            ).hexdigest()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode image '{path}': {exc}") from exc
    return byte_hash, pixel_hash


def find_exact_duplicates(
    split_paths: dict[str, Iterable[Path]],
) -> dict[str, list[tuple[str, str]]]:
    """Find exact file or decoded-pixel duplicates; do not infer near-duplicates."""
    byte_groups: dict[str, list[str]] = defaultdict(list)
    pixel_groups: dict[str, list[str]] = defaultdict(list)
    for split_name, paths in split_paths.items():
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            byte_hash, pixel_hash = image_fingerprints(path)
            label = f"{split_name}:{path.as_posix()}"
            byte_groups[byte_hash].append(label)
            pixel_groups[pixel_hash].append(label)

    byte_duplicates = [
        tuple(sorted(items)) for items in byte_groups.values() if len(items) > 1
    ]
    pixel_duplicates = [
        tuple(sorted(items)) for items in pixel_groups.values() if len(items) > 1
    ]
    return {
        "identical_files": sorted(byte_duplicates),
        "identical_decoded_pixels": sorted(pixel_duplicates),
    }
