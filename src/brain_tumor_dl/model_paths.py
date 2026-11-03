"""Resolve which model checkpoint the API serves."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ENV = "BRAIN_TUMOR_MODEL_PATH"
# The checkpoint behind every metric reported in the README and the dashboard.
EVALUATED_RELATIVE = Path("artifacts_finetune30_leakagecontrolled_20261026") / "brain_tumor_mobilenetv2.keras"
EVALUATED_SHA256 = "8b7b4c2e423d18d72d72bb25773b906b03c2a9a6a8cf40b060de1e56a0d6f63f"


@dataclass(frozen=True)
class ModelResolution:
    path: Path
    variant: str  # "override" | "evaluated" | "repaired" | "canonical"
    stale_repair: bool
    root: Path


def resolve_mobilenet_path(root: Path) -> Path:
    """Return the repaired model when available, otherwise the canonical model."""
    artifacts = root / "artifacts"
    canonical = artifacts / "brain_tumor_mobilenetv2.keras"
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"

    if repaired.is_file():
        return repaired

    return canonical


def resolve_model(root: Path, env: dict | None = None) -> ModelResolution:
    """Choose the checkpoint to serve.

    Order: ``BRAIN_TUMOR_MODEL_PATH`` override, the evaluated fine-tuned checkpoint
    (if present locally), then the legacy repaired/canonical files in ``artifacts/``.
    Nothing is substituted silently: the chosen variant is reported by /api/status.
    """
    env = os.environ if env is None else env
    override = (env.get(MODEL_ENV) or "").strip()
    if override:
        path = Path(override).expanduser()
        return ModelResolution(path if path.is_absolute() else root / path, "override", False, root)
    evaluated = root / EVALUATED_RELATIVE
    if evaluated.is_file():
        return ModelResolution(evaluated, "evaluated", False, root)
    legacy = resolve_mobilenet_path(root)
    canonical = legacy.parent / "brain_tumor_mobilenetv2.keras"
    variant = "repaired" if legacy.name.endswith("_repaired.keras") else "canonical"
    stale = variant == "repaired" and canonical.is_file() and canonical.stat().st_mtime > legacy.stat().st_mtime
    return ModelResolution(legacy, variant, stale, root)


@dataclass(frozen=True)
class ModelChoice:
    path: Path
    class_names_path: Path
    variant: str


def resolve_model_choice(root: Path, env: dict | None = None) -> ModelChoice:
    """Checkpoint plus the class-name file to use with it (adjacent file preferred)."""
    resolution = resolve_model(root, env)
    adjacent = resolution.path.parent / "class_names.json"
    classes = adjacent if adjacent.is_file() else root / "artifacts" / "class_names.json"
    return ModelChoice(resolution.path, classes, resolution.variant)
