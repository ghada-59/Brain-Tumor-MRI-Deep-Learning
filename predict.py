"""Command-line prediction for one brain MRI image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a class for one MRI image.")
    parser.add_argument("image", type=Path, help="Path to an MRI image file (JPG/PNG).")
    parser.add_argument("--model", type=Path, default=Path("artifacts/brain_tumor_mobilenetv2.keras"))
    parser.add_argument("--classes", type=Path, default=Path("artifacts/class_names.json"))
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if not args.model.is_file() or not args.classes.is_file():
        raise SystemExit("Trained model or class_names.json not found. Run train.py first.")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    try:
        image = Image.open(args.image).convert("RGB").resize((args.image_size, args.image_size))
    except (UnidentifiedImageError, OSError) as exc:
        raise SystemExit(f"Cannot read image file: {exc}") from exc

    model = tf.keras.models.load_model(args.model)
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(pixels, verbose=0)[0]
    ranked = np.argsort(probabilities)[::-1]

    print("Educational model output — not a medical diagnosis")
    print(f"Image: {args.image}")
    print(f"Predicted class: {class_names[int(ranked[0])]}")
    print("Top class probabilities:")
    for index in ranked[: min(4, len(ranked))]:
        print(f"  {class_names[int(index)]:15s} {float(probabilities[index]):.2%}")


if __name__ == "__main__":
    main()
