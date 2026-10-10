"""Train the MRI classifier and save reproducible evaluation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.brain_tumor_dl.data import load_datasets
from src.brain_tumor_dl.evaluation import evaluate_and_visualize, save_training_curves
from src.brain_tumor_dl.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 brain MRI classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder containing Training/ and Testing/.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for model and machine-readable metrics.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory for plots and CSV reports.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs for the frozen-backbone baseline.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet", help="Use pretrained ImageNet weights, or train from random initialization.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_ds, val_ds, test_ds, class_names = load_datasets(
            args.data_dir,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Dataset setup error: {exc}") from exc

    print("Class order:", class_names)
    model = build_model(
        num_classes=len(class_names),
        image_size=(args.image_size, args.image_size),
        weights=None if args.weights == "none" else "imagenet",
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(args.reports_dir / "training_history.csv")),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model_path = args.output_dir / "brain_tumor_mobilenetv2.keras"
    model.save(model_path)
    (args.output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    config = {
        "architecture": "MobileNetV2 transfer learning",
        "image_size": [args.image_size, args.image_size],
        "batch_size": args.batch_size,
        "requested_epochs": args.epochs,
        "actual_epochs": len(history.history.get("loss", [])),
        "weights": args.weights,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "class_names": class_names,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    save_training_curves(history, args.reports_dir)
    metrics = evaluate_and_visualize(
        model, test_ds, class_names, args.reports_dir, args.output_dir
    )

    print("\nHeld-out test results (computed from Testing/):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\nModel saved to: {model_path}")
    print(f"Plots and reports saved to: {args.reports_dir.resolve()}")
    print("Educational project only — not for clinical decision-making.")


if __name__ == "__main__":
    main()
