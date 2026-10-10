"""Dataset loading and validation helpers for the MRI classification project."""
from __future__ import annotations

from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def count_images_by_class(directory: Path) -> dict[str, int]:
    """Count supported image files directly inside each class directory."""
    if not directory.is_dir():
        return {}
    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1 for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return counts


def load_datasets(
    data_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, Any, list[str]]:
    """Load Training/ into train/validation subsets and keep Testing/ held out."""
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Dataset folders were not found. Expected data/Training/<class>/ and "
            "data/Testing/<class>/. See README.md for setup instructions."
        )

    import tensorflow as tf

    common = {
        "image_size": image_size,
        "batch_size": batch_size,
        "label_mode": "int",
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        shuffle=True,
        **common,
    )
    class_names = list(train_ds.class_names)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        shuffle=False,
        class_names=class_names,
        **common,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        class_names=class_names,
        **common,
    )

    test_classes = list(test_ds.class_names)
    if test_classes != class_names:
        raise ValueError(
            f"Class folder mismatch. Training classes: {class_names}; "
            f"Testing classes: {test_classes}. Ensure names match exactly."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names
