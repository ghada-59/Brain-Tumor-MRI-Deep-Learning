
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

reports_dir = Path("reports_finetune")

baseline = pd.read_csv(reports_dir / "training_history.csv")
fine_tune = pd.read_csv(reports_dir / "training_history_finetune.csv")

metrics = [
    ("accuracy", "val_accuracy", "Accuracy"),
    ("loss", "val_loss", "Loss"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for row, (train_metric, val_metric, title) in enumerate(metrics):
    for col, (history, phase) in enumerate(
        [
            (baseline, "Frozen backbone"),
            (fine_tune, "Fine-tuning"),
        ]
    ):
        ax = axes[row, col]
        epochs = range(1, len(history) + 1)

        ax.plot(
            epochs,
            history[train_metric],
            label="Training",
        )
        ax.plot(
            epochs,
            history[val_metric],
            label="Validation",
        )
        ax.set_title(f"{phase} — {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.25)

fig.suptitle("MobileNetV2 Training and Fine-tuning History")
fig.tight_layout()

output_path = reports_dir / "training_curves_combined.png"
fig.savefig(output_path, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Combined training curves saved to: {output_path}")
print(f"Baseline epochs: {len(baseline)}")
print(f"Fine-tuning epochs: {len(fine_tune)}")
