"""Evaluation metrics and result visualizations computed from held-out test images."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def save_training_curves(history: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_data = history.history
    epochs = range(1, len(history_data.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history_data.get("accuracy", []), label="Training")
    axes[0].plot(epochs, history_data.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history_data.get("loss", []), label="Training")
    axes[1].plot(epochs, history_data.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Training history — MobileNetV2")
    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_and_visualize(
    model: tf.keras.Model,
    test_ds: Any,
    class_names: list[str],
    reports_dir: Path,
    artifacts_dir: Path,
    max_examples: int = 12,
) -> dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    true_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    for images, labels in test_ds:
        true_batches.append(labels.numpy())
        prob_batches.append(model.predict(images, verbose=0))
    y_true = np.concatenate(true_batches).astype(int)
    probabilities = np.concatenate(prob_batches)
    y_pred = probabilities.argmax(axis=1)

    labels_idx = list(range(len(class_names)))
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_image_count": int(len(y_true)),
    }
    (artifacts_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    (artifacts_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    report_frame = pd.DataFrame(report).transpose()
    report_frame.to_csv(reports_dir / "classification_report.csv", index_label="class")

    matrix = confusion_matrix(y_true, y_pred, labels=labels_idx)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=25
    )
    ax.set_title("Confusion matrix — held-out test set")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0
    )
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_xticks(x, class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class performance — held-out test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports_dir / "class_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # A separate shuffled stream is used only to select illustrative examples;
    # metric computation above always uses the deterministic held-out test order.
    preview_stream = test_ds.unbatch().shuffle(1000, seed=42).batch(max_examples).take(1)
    for images, labels in preview_stream:
        preview_probs = model.predict(images, verbose=0)
        preview_pred = preview_probs.argmax(axis=1)
        n = min(len(images), max_examples)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3.2))
        axes = np.array(axes).reshape(-1)
        for i, ax in enumerate(axes):
            if i >= n:
                ax.axis("off")
                continue
            image = np.clip(images[i].numpy(), 0, 255).astype(np.uint8)
            actual = class_names[int(labels[i].numpy())]
            predicted = class_names[int(preview_pred[i])]
            confidence = float(preview_probs[i, preview_pred[i]])
            ax.imshow(image)
            ax.set_title(f"True: {actual}\nPred: {predicted} ({confidence:.1%})", fontsize=9)
            ax.axis("off")
        fig.suptitle("Example predictions (confidence is not clinical certainty)")
        fig.tight_layout()
        fig.savefig(reports_dir / "prediction_examples.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        break

    return metrics
