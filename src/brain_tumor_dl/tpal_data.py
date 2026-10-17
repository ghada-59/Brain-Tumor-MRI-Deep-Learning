"""Dataset loading for the standalone T-PAL experiment."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _images_by_class(split_dir: Path, expected_classes: list[str] | None = None):
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    if expected_classes is not None and class_names != expected_classes:
        raise ValueError(
            f"Class folders in {split_dir} do not match the training classes. "
            f"Expected {expected_classes}; found {class_names}."
        )
    if len(class_names) < 2:
        raise ValueError(f"Expected at least two class folders in {split_dir}.")

    paths: list[Path] = []
    labels: list[int] = []
    for label, class_dir in enumerate(class_dirs):
        class_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(f"No supported images found for class '{class_dir.name}'.")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    return paths, np.asarray(labels, dtype=np.int64), class_names


def compute_normalization_stats(
    paths: list[Path], image_size: int = 128
) -> tuple[float, float]:
    """Compute grayscale mean and standard deviation using training images only."""
    pixel_count = 0
    pixel_sum = 0.0
    pixel_square_sum = 0.0

    for path in paths:
        with Image.open(path) as source:
            image = source.convert("L").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixel_count += pixels.size
        pixel_sum += float(pixels.sum(dtype=np.float64))
        pixel_square_sum += float(np.square(pixels, dtype=np.float64).sum())

    if pixel_count == 0:
        raise ValueError("Cannot calculate normalization statistics from no images.")

    mean = pixel_sum / pixel_count
    variance = max(pixel_square_sum / pixel_count - mean * mean, 0.0)
    std = max(variance ** 0.5, 1e-6)
    return float(mean), float(std)


class BrainMRIImageDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: np.ndarray,
        image_size: int,
        mean: float,
        std: float,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("Image paths and labels must have the same length.")
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.image_size = image_size
        self.mean = mean
        self.std = max(std, 1e-6)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as source:
            image = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )

        if self.augment:
            rng = random.Random(self.seed + random.randrange(1_000_000) + index)
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
            if rng.random() < 0.5:
                image = image.rotate(rng.uniform(-10.0, 10.0), resample=Image.Resampling.BILINEAR)

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        tensor = torch.from_numpy(pixels.copy()).unsqueeze(0)
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def load_tpal_datasets(
    data_dir: Path,
    image_size: int = 128,
    validation_split: float = 0.15,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], float, float]:
    """Load Training/ and held-out Testing/ with a stratified validation split."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be strictly between 0 and 1.")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative.")
    if image_size < 8 or image_size % 8 != 0:
        raise ValueError("image_size must be a positive multiple of 8 (at least 8).")

    train_paths, train_labels, class_names = _images_by_class(data_dir / "Training")
    test_paths, test_labels, _ = _images_by_class(
        data_dir / "Testing", expected_classes=class_names
    )
    train_indices, val_indices = train_test_split(
        np.arange(len(train_paths)),
        test_size=validation_split,
        random_state=seed,
        stratify=train_labels,
    )

    fit_paths = [train_paths[int(index)] for index in train_indices]
    fit_labels = train_labels[train_indices]
    val_paths = [train_paths[int(index)] for index in val_indices]
    val_labels = train_labels[val_indices]
    mean, std = compute_normalization_stats(fit_paths, image_size)

    train_dataset = BrainMRIImageDataset(
        fit_paths, fit_labels, image_size, mean, std, augment=True, seed=seed
    )
    val_dataset = BrainMRIImageDataset(
        val_paths, val_labels, image_size, mean, std, augment=False, seed=seed
    )
    test_dataset = BrainMRIImageDataset(
        test_paths, test_labels, image_size, mean, std, augment=False, seed=seed
    )

    loader_options = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **loader_options
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_options
    )
    return train_loader, val_loader, test_loader, class_names, mean, std
