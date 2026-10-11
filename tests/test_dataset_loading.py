from pathlib import Path

import pytest
from PIL import Image

from src.brain_tumor_dl.data import load_datasets


def _make_dataset(root: Path, classes=("glioma", "notumor")) -> None:
    for split in ("Training", "Testing"):
        for class_index, class_name in enumerate(classes):
            folder = root / split / class_name
            folder.mkdir(parents=True)
            count = 6 if split == "Training" else 2
            for index in range(count):
                color = (class_index * 100, index * 10, 40)
                Image.new("RGB", (24, 18), color=color).save(folder / f"{index}.png")


def test_load_datasets_returns_expected_shapes_and_all_splits(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    train_ds, val_ds, test_ds, class_names = load_datasets(
        tmp_path, image_size=(16, 16), batch_size=2, validation_split=0.25, seed=4
    )

    assert class_names == ["glioma", "notumor"]
    train_images, train_labels = next(iter(train_ds))
    val_images, val_labels = next(iter(val_ds))
    test_images, test_labels = next(iter(test_ds))
    assert tuple(train_images.shape[1:]) == (16, 16, 3)
    assert tuple(val_images.shape[1:]) == (16, 16, 3)
    assert tuple(test_images.shape[1:]) == (16, 16, 3)
    assert set(train_labels.numpy().tolist()) <= {0, 1}
    assert set(val_labels.numpy().tolist()) == {0, 1}
    assert set(test_labels.numpy().tolist()) == {0, 1}


def test_load_datasets_rejects_class_folder_mismatch(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    (tmp_path / "Testing" / "notumor").rename(tmp_path / "Testing" / "other")
    with pytest.raises(ValueError, match="Class folder mismatch"):
        load_datasets(tmp_path, image_size=(16, 16), batch_size=2)


def test_load_datasets_rejects_missing_split(tmp_path: Path) -> None:
    (tmp_path / "Training").mkdir()
    with pytest.raises(FileNotFoundError, match="Dataset folders were not found"):
        load_datasets(tmp_path)
