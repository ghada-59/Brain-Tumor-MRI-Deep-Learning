"""Generate a T-PAL attention/saliency visualization for one MRI image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.brain_tumor_dl.tpal import TPALClassifier
from src.brain_tumor_dl.tpal_explainability import save_explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a T-PAL explanation image.")
    parser.add_argument("image", type=Path, help="Path to an MRI image.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts_tpal/tpal_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports_tpal/explanations/example.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train T-PAL first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        class_names = checkpoint["class_names"]
        model = TPALClassifier(
            num_classes=len(class_names),
            image_size=int(checkpoint["image_size"]),
            dimension=int(checkpoint["dimension"]),
            heads=int(checkpoint["heads"]),
            depth=int(checkpoint["depth"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not load a compatible T-PAL checkpoint: {exc}") from exc

    result = save_explanation(args.image, model, checkpoint, args.output, device, class_names)
    print(json.dumps(result, indent=2))
    print("Explanations are model visualizations, not validated tumor localization.")


if __name__ == "__main__":
    main()
