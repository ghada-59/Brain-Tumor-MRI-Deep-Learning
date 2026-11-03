"""Reproducible Grad-CAM figure for the dashboard, with a provenance sidecar.

The figure previously shown on the Explainability page came from the historical
"repaired" checkpoint and no script in the repository records how it was made. This
script fixes both: it uses the checkpoint you pass (by default the one the API
serves), selects images deterministically (first file of each class in sorted order
unless --images is given), preprocesses them exactly like training, and writes
<output>.json next to the PNG with the checkpoint SHA-256, the image hashes, the
true and predicted labels and the raw scores.

Grad-CAM is an approximate attribution map for the predicted class. It is not a
segmentation mask, not a tumour boundary and not validated against annotations.

Example (Windows PowerShell):
    python make_gradcam_figure.py --data-dir data_leakage_controlled_distance0\\Testing `
        --output reports_finetune30_leakagecontrolled_20261026\\dashboard_assets\\gradcam_examples.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.brain_tumor_dl.data import SUPPORTED_EXTENSIONS
from src.brain_tumor_dl.experiment_record import file_sha256
from src.brain_tumor_dl.model_paths import resolve_model_choice

ROOT = Path(__file__).resolve().parent


def _relative(path: Path) -> str:
    """Repository-relative POSIX path, so the sidecar carries no local absolute path."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    resolution = resolve_model_choice(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=resolution.path)
    parser.add_argument("--classes", type=Path, default=resolution.class_names_path)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with one sub-folder per class (e.g. .../Testing).")
    parser.add_argument("--images", type=Path, nargs="*", default=None, help="Explicit images instead of the first per class.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def select_images(data_dir: Path, class_names: list[str]) -> list[tuple[Path, str]]:
    chosen = []
    for name in class_names:
        files = sorted(p for p in (data_dir / name).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise SystemExit(f"No images for class '{name}' in {data_dir}.")
        chosen.append((files[0], name))
    return chosen


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from PIL import Image

    from src.brain_tumor_dl.explainability import compute_gradcam, make_explanation_views
    from src.brain_tumor_dl.inference import preprocess_image_bytes

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model, compile=False)
    size = int(model.input_shape[1])
    if args.images:
        selected = [(p, p.parent.name) for p in args.images]
    else:
        selected = select_images(args.data_dir, class_names)

    rows = []
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)), squeeze=False)
    for row, (path, true_label) in enumerate(selected):
        batch = preprocess_image_bytes(path.read_bytes(), size)
        scores = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)
        predicted = int(scores.argmax())
        heatmap = compute_gradcam(model, batch, predicted)
        image = Image.open(path).convert("RGB")
        heat_view, overlay, _ = make_explanation_views(image, heatmap)
        for col, (view, title) in enumerate(((image, "Input"), (heat_view, "Grad-CAM (predicted class)"), (overlay, "Overlay"))):
            axes[row][col].imshow(view)
            axes[row][col].set_title(title, fontsize=9)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(true_label)
        axes[row][1].set_xlabel("")
        axes[row][0].text(
            0, -0.08, f"true: {true_label} | predicted: {class_names[predicted]} ({scores[predicted]:.3f})",
            transform=axes[row][0].transAxes, fontsize=8, va="top",
        )
        rows.append({
            "image": _relative(path),
            "image_sha256": file_sha256(path),
            "true_label": true_label,
            "predicted_label": class_names[predicted],
            "scores": dict(zip(class_names, scores.round(6).tolist())),
        })
    sha = file_sha256(args.model)
    figure.suptitle(f"Grad-CAM, checkpoint {args.model.name} (SHA-256 {sha[:12]}...)", fontsize=10)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=120)
    plt.close(figure)
    sidecar = {
        "figure": args.output.name,
        "model_path": _relative(args.model),
        "model_sha256": sha,
        "selection": "explicit --images" if args.images else "first file of each class in sorted order",
        "preprocessing": "inference.preprocess_image_bytes (identical to training)",
        "rows": rows,
        "note": "Grad-CAM is an approximate attribution map, not a segmentation or validated localisation.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
