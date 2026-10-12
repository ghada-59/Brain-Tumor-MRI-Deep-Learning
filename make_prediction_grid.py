
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data_leakage_free" / "Testing"
OUTPUT_DIR = ROOT / "reports_finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
IMAGES_PER_CLASS = 10
SEED = 42

MODELS = {
    "Baseline": ROOT / "artifacts_clean" / "brain_tumor_mobilenetv2.keras",
    "Fine-tuned": ROOT / "artifacts_finetune" / "brain_tumor_mobilenetv2.keras",
}

random.seed(SEED)
np.random.seed(SEED)

class_names = sorted(
    p.name for p in TEST_DIR.iterdir() if p.is_dir()
)

models = {
    name: tf.keras.models.load_model(path)
    for name, path in MODELS.items()
}

samples = []

for class_name in class_names:
    files = sorted(
        p for p in (TEST_DIR / class_name).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    selected = random.sample(
        files, min(IMAGES_PER_CLASS, len(files))
    )

    for path in selected:
        image = load_img(path, target_size=IMAGE_SIZE)
        array = img_to_array(image)
        batch = np.expand_dims(array, axis=0)

        predictions = {
            name: model.predict(batch, verbose=0)[0]
            for name, model in models.items()
        }

        samples.append({
            "path": path,
            "actual": class_name,
            "image": image,
            "predictions": predictions,
        })

# 5 rows x 8 columns = 40 image panels
cols = 8
rows = int(np.ceil(len(samples) / cols))
fig, axes = plt.subplots(
    rows, cols, figsize=(24, rows * 3.2)
)
axes = np.array(axes).reshape(-1)

baseline_correct = 0
finetuned_correct = 0

for ax, item in zip(axes, samples):
    actual = item["actual"]
    baseline_idx = int(np.argmax(item["predictions"]["Baseline"]))
    tuned_idx = int(np.argmax(item["predictions"]["Fine-tuned"]))

    baseline_pred = class_names[baseline_idx]
    tuned_pred = class_names[tuned_idx]

    baseline_correct += baseline_pred == actual
    finetuned_correct += tuned_pred == actual

    ax.imshow(item["image"])
    ax.axis("off")

    baseline_mark = "OK" if baseline_pred == actual else "ERROR"
    tuned_mark = "OK" if tuned_pred == actual else "ERROR"

    ax.set_title(
        f"True: {actual}\n"
        f"Base: {baseline_pred} ({baseline_mark})\n"
        f"Fine: {tuned_pred} ({tuned_mark})",
        fontsize=8,
    )

for ax in axes[len(samples):]:
    ax.axis("off")

fig.suptitle(
    "Brain MRI Classification — Baseline vs Fine-tuned\n"
    f"Selected sample: {len(samples)} images | "
    f"Baseline: {baseline_correct}/{len(samples)} | "
    f"Fine-tuned: {finetuned_correct}/{len(samples)}",
    fontsize=16,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

output_path = OUTPUT_DIR / "prediction_grid_40.png"
fig.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Grid saved to: {output_path}")
print(f"Images displayed: {len(samples)}")
print(f"Baseline: {baseline_correct}/{len(samples)}")
print(f"Fine-tuned: {finetuned_correct}/{len(samples)}")
