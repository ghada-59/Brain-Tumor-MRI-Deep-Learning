"""Train and evaluate the standalone T-PAL classifier."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

from src.brain_tumor_dl.tpal import TPALClassifier, association_regularization, spatial_association_prior
from src.brain_tumor_dl.tpal_data import load_tpal_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the T-PAL-style brain MRI classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_leakage_free"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_tpal"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports_tpal"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--association-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    return parser.parse_args()


def evaluate(model, loader, device, class_names):
    model.eval()
    true_labels, predicted_labels, probability_rows = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probability_rows.append(probabilities)
            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(probabilities.argmax(axis=1).tolist())

    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(predicted_labels, dtype=np.int64)
    probabilities = np.concatenate(probability_rows, axis=0)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_image_count": int(len(y_true)),
    }
    per_class = [
        {
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    ]
    return metrics, report, per_class, confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    ), y_true, y_pred, probabilities


def save_evaluation(reports_dir, metrics, report, per_class, matrix, class_names):
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (reports_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (reports_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("T-PAL confusion matrix — held-out test set")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise SystemExit("--epochs, --batch-size and --patience must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("--learning-rate must be positive and --weight-decay non-negative.")
    if args.association_weight < 0 or args.diversity_weight < 0:
        raise SystemExit("Regularization weights cannot be negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader, class_names, mean, std = load_tpal_datasets(
        args.data_dir,
        image_size=args.image_size,
        validation_split=args.validation_split,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TPALClassifier(
        num_classes=len(class_names),
        image_size=args.image_size,
        dimension=args.dimension,
        heads=args.heads,
        depth=args.depth,
    ).to(device)
    prior = spatial_association_prior(model.grid_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = args.output_dir / "tpal_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        correct = 0
        sample_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(images, return_aux=True)
            classification_loss = criterion(logits, labels)
            association_loss, diversity_loss = association_regularization(auxiliary, prior)
            loss = (
                classification_loss
                + args.association_weight * association_loss
                + args.diversity_weight * diversity_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            count = labels.size(0)
            loss_total += float(loss.detach()) * count
            correct += int((logits.argmax(dim=1) == labels).sum())
            sample_count += count

        model.eval()
        val_true, val_pred = [], []
        val_loss_total = 0.0
        val_count = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device))
                labels_device = labels.to(device)
                loss = criterion(logits, labels_device)
                val_loss_total += float(loss) * labels.size(0)
                val_count += labels.size(0)
                val_true.extend(labels.numpy().tolist())
                val_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

        val_macro_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        val_accuracy = float(accuracy_score(val_true, val_pred))
        scheduler.step(val_macro_f1)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, sample_count),
            "train_accuracy": correct / max(1, sample_count),
            "val_loss": val_loss_total / max(1, val_count),
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_accuracy={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "mean": mean,
                    "std": std,
                    "dimension": args.dimension,
                    "heads": args.heads,
                    "depth": args.depth,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_macro_f1,
                    "seed": args.seed,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    with (args.reports_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Validation macro-F1")
    axes[1].set_title("Classification performance")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.reports_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, report, per_class, matrix, y_true, y_pred, probabilities = evaluate(
        model, test_loader, device, class_names
    )
    metrics.update({
        "model": "TPALClassifier",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "image_size": args.image_size,
        "seed": args.seed,
        "class_names": class_names,
        "normalization_mean": mean,
        "normalization_std": std,
        "test_data_dir": str(args.data_dir),
    })
    save_evaluation(args.reports_dir, metrics, report, per_class, matrix, class_names)
    print(json.dumps(metrics, indent=2))
    print("Saved checkpoint:", best_path)
    print("Saved reports:", args.reports_dir)


if __name__ == "__main__":
    main()
