
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from src.brain_tumor_dl.inference import preprocess_pil_image
from src.brain_tumor_dl.explainability import (
    compute_gradcam,
    make_explanation_views,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "Testing"
MODEL_PATH = ROOT / "artifacts" / "brain_tumor_mobilenetv2_repaired.keras"
CLASS_NAMES_PATH = ROOT / "artifacts" / "class_names.json"
OUTPUT_DIR = ROOT / "artifacts" / "gradcam_testing"
IMAGE_SIZE = 224

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Testing folder not found: {DATA_DIR}")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

class_names = json.loads(
    CLASS_NAMES_PATH.read_text(encoding="utf-8")
)

if (
    not isinstance(class_names, list)
    or not class_names
    or not all(isinstance(name, str) for name in class_names)
    or len(set(class_names)) != len(class_names)
):
    raise ValueError("class_names.json must contain unique class names.")

class_dirs = sorted(
    (p for p in DATA_DIR.iterdir() if p.is_dir()),
    key=lambda p: p.name,
)

if not class_dirs:
    raise RuntimeError(f"No class folders found in {DATA_DIR}")

folder_names = {p.name for p in class_dirs}

if folder_names != set(class_names):
    raise ValueError(
        "Testing folders do not match class_names.json. "
        f"Model classes: {class_names}; testing folders: {sorted(folder_names)}"
    )

images = [
    (class_dir.name, path)
    for class_dir in class_dirs
    for path in sorted(class_dir.rglob("*"))
    if path.is_file() and path.suffix.lower() in EXTENSIONS
]

if not images:
    raise RuntimeError(f"No supported images found in {DATA_DIR}")

print(f"Images trouvees : {len(images)}")
print(f"Classes du modele : {class_names}")
print("Chargement du modele...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

output_shape = model.output_shape
if (
    not isinstance(output_shape, tuple)
    or len(output_shape) != 2
    or output_shape[-1] != len(class_names)
):
    raise ValueError(
        f"Model output shape {output_shape} does not match "
        f"{len(class_names)} classes."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT_DIR / "predictions_gradcam.csv"
rows = []
errors = []

for index, (true_class, image_path) in enumerate(images, start=1):
    try:
        with Image.open(image_path) as source:
            original = source.convert("RGB")

        batch = preprocess_pil_image(original, IMAGE_SIZE)
        probabilities = np.asarray(
            model.predict(batch, verbose=0)[0]
        )

        if probabilities.ndim != 1 or len(probabilities) != len(class_names):
            raise ValueError(
                "Model prediction count does not match the class list."
            )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned non-finite prediction values.")

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]

        heatmap = compute_gradcam(model, batch, predicted_index)
        heatmap_image, overlay, box = make_explanation_views(
            original, heatmap
        )

        relative_path = image_path.relative_to(DATA_DIR)
        output_folder = OUTPUT_DIR / relative_path.parent
        output_folder.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        original.save(output_folder / f"{stem}_original.png")
        heatmap_image.save(output_folder / f"{stem}_heatmap.png")

        draw = ImageDraw.Draw(overlay)
        label = f"Pred: {predicted_class}"
        draw.rectangle((0, 0, min(overlay.width, 220), 28), fill="black")
        draw.text((6, 6), label, fill="white")
        overlay.save(output_folder / f"{stem}_overlay.png")

        rows.append({
            "image": str(relative_path),
            "true_class": true_class,
            "predicted_class": predicted_class,
            "correct": true_class == predicted_class,
            "confidence": float(probabilities[predicted_index]),
            "red_box_found": box is not None,
            "red_box_coordinates": str(box) if box is not None else "",
            **{
                f"prob_{name}": float(probabilities[i])
                for i, name in enumerate(class_names)
            },
        })

    except Exception as exc:
        errors.append((str(image_path), str(exc)))

    if index % 25 == 0 or index == len(images):
        print(f"Progression : {index}/{len(images)}")

fieldnames = [
    "image",
    "true_class",
    "predicted_class",
    "correct",
    "confidence",
    "red_box_found",
    "red_box_coordinates",
    *[f"prob_{name}" for name in class_names],
]

with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n--- Bilan ---")
print(f"Images trouvees : {len(images)}")
print(f"Images traitees : {len(rows)}")
print(f"Erreurs : {len(errors)}")
print(f"CSV : {csv_path}")
print(f"Resultats : {OUTPUT_DIR}")

if rows:
    accuracy = np.mean([row["correct"] for row in rows])
    boxes = sum(row["red_box_found"] for row in rows)

    print(f"Accuracy sur les images traitees : {accuracy:.2%}")
    print(f"Images avec rectangle rouge : {boxes}/{len(rows)}")
    print(
        "Attention : les rectangles sont des regions d'activation "
        "heuristiques, pas des localisations tumorales validees."
    )

if errors:
    error_path = OUTPUT_DIR / "errors.txt"
    error_path.write_text(
        "\n".join(f"{path}: {error}" for path, error in errors),
        encoding="utf-8",
    )
    print(f"Details des erreurs : {error_path}")

