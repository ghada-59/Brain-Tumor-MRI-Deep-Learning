
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data_leakage_free" / "Testing"

MODELS = {
    "Baseline": ROOT / "artifacts_clean" / "brain_tumor_mobilenetv2.keras",
    "Fine-tuned": ROOT / "artifacts_finetune" / "brain_tumor_mobilenetv2.keras",
}

IMAGE_SIZE = (224, 224)
IMAGES_PER_CLASS = 10
SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    random.seed(SEED)

    if not TEST_DIR.is_dir():
        raise SystemExit(f"Test directory not found: {TEST_DIR}")

    for name, path in MODELS.items():
        if not path.is_file():
            raise SystemExit(f"{name} model not found: {path}")

    class_names = sorted(
        folder.name for folder in TEST_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_names:
        raise SystemExit("No class folders found in the test directory.")

    models = {
        name: tf.keras.models.load_model(path)
        for name, path in MODELS.items()
    }

    selected = []
    for class_name in class_names:
        folder = TEST_DIR / class_name
        images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
        random.shuffle(images)

        if len(images) < IMAGES_PER_CLASS:
            raise SystemExit(
                f"Not enough images in class {class_name}: {len(images)}"
            )

        selected.extend(
            (class_name, image_path)
            for image_path in images[:IMAGES_PER_CLASS]
        )

    print(f"\nImages selected: {len(selected)}")
    print("Predictions are qualitative checks, not a new independent test.\n")

    correct = {name: 0 for name in models}

    for index, (true_class, image_path) in enumerate(selected, start=1):
        image = load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
        array = img_to_array(image)
        batch = np.expand_dims(array, axis=0)

        print(f"{index:02d}. Actual: {true_class} | {image_path.name}")

        for model_name, model in models.items():
            probabilities = model.predict(batch, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])
            is_correct = predicted_class == true_class

            correct[model_name] += int(is_correct)
            status = "OK" if is_correct else "ERROR"

            print(
                f"    {model_name:10s}: {predicted_class:12s} "
                f"| confidence={confidence:.1%} | {status}"
            )

        print()

    print("SUMMARY")
    for model_name in models:
        print(
            f"{model_name}: {correct[model_name]}/{len(selected)} "
            f"correct on these selected images"
        )


if __name__ == "__main__":
    main()
