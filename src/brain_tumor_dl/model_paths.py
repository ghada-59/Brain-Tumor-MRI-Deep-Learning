"""Resolve the canonical MobileNetV2 artifact with a local repair fallback."""
from __future__ import annotations

from pathlib import Path


def resolve_mobilenet_path(root: Path) -> Path:
    """Prefer the canonical trained artifact; fall back to a repaired local copy."""
    artifacts = root / "artifacts"
    canonical = artifacts / "brain_tumor_mobilenetv2.keras"
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"
    if canonical.is_file():
        return canonical
    if repaired.is_file():
        return repaired
    return canonical
