
"""Resolve the MobileNetV2 model artifact."""

from __future__ import annotations

from pathlib import Path


def resolve_mobilenet_path(root: Path) -> Path:
    """Return the repaired model when available, otherwise the canonical model."""
    artifacts = root / "artifacts"
    canonical = artifacts / "brain_tumor_mobilenetv2.keras"
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"

    if repaired.is_file():
        return repaired

    return canonical
