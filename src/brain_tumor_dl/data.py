"""Dataset loading and stratified train/validation splitting helpers."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def count_images_by_class(directory: Path) -> dict[str, int]:
    """Count supported image files recursively within each class directory."""
    if not directory.is_dir():
        return {}
    counts: dict[str, int] = {}
    for class_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1 for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return counts


def split_files_by_class(
    directory: Path,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[int], list[Path], list[int], list[str]]:
    """Create a deterministic, class-stratified split of images in class folders.

    Unlike splitting the final N paths from one globally sorted directory
    listing, this keeps examples from every class in both subsets.
    """
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be strictly between 0 and 1.")
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    class_dirs = sorted(path for path in directory.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(f"Expected at least two class folders in {directory}.")

    rng = random.Random(seed)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    val_paths: list[Path] = []
    val_labels: list[int] = []
    class_names = [path.name for path in class_dirs]

    for label, class_dir in enumerate(class_dirs):
        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(files) < 2:
            raise ValueError(
                f"Class '{class_dir.name}' needs at least two supported images "
                "to create non-empty training and validation subsets."
            )
        rng.shuffle(files)
        val_count = min(len(files) - 1, max(1, round(len(files) * validation_split)))
        val_class_files = files[:val_count]
        train_class_files = files[val_count:]
        train_paths.extend(train_class_files)
        train_labels.extend([label] * len(train_class_files))
        val_paths.extend(val_class_files)
        val_labels.extend([label] * len(val_class_files))

    return train_paths, train_labels, val_paths, val_labels, class_names


def _image_dataset(
    paths: list[Path],
    labels: list[int],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    import tensorflow as tf

    path_strings = [str(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices((path_strings, labels))

    def load_image(path: Any, label: Any) -> tuple[Any, Any]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_datasets(
    data_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, Any, list[str]]:
    """Load stratified Training/ subsets and keep Testing/ held out."""
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Dataset folders were not found. Expected data/Training/<class>/ and "
            "data/Testing/<class>/. See README.md for setup instructions."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if image_size[0] < 1 or image_size[1] < 1:
        raise ValueError("image_size dimensions must be positive.")

    train_paths, train_labels, val_paths, val_labels, class_names = split_files_by_class(
        train_dir, validation_split=validation_split, seed=seed
    )

    test_class_dirs = sorted(path for path in test_dir.iterdir() if path.is_dir())
    test_class_names = [path.name for path in test_class_dirs]
    if test_class_names != class_names:
        raise ValueError(
            f"Class folder mismatch. Training classes: {class_names}; "
            f"Testing classes: {test_class_names}. Ensure names match exactly."
        )

    test_paths: list[Path] = []
    test_labels: list[int] = []
    for label, class_dir in enumerate(test_class_dirs):
        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No supported images found in test class '{class_dir.name}'.")
        test_paths.extend(files)
        test_labels.extend([label] * len(files))

    train_ds = _image_dataset(
        train_paths, train_labels, image_size, batch_size, shuffle=True, seed=seed
    )
    val_ds = _image_dataset(
        val_paths, val_labels, image_size, batch_size, shuffle=False, seed=seed
    )
    test_ds = _image_dataset(
        test_paths, test_labels, image_size, batch_size, shuffle=False, seed=seed
    )
    return train_ds, val_ds, test_ds, class_names
