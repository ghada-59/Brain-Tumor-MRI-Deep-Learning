"""Tests for T-PAL dataset loading and train-only normalization."""
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.brain_tumor_dl.tpal_data import load_tpal_datasets


def _write_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((20, 24), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def test_tpal_loader_keeps_testing_separate_and_stratifies(tmp_path):
    for class_index, class_name in enumerate(("glioma", "meningioma")):
        for image_index in range(4):
            _write_image(
                tmp_path / "Training" / class_name / f"train_{image_index}.png",
                40 + class_index * 80 + image_index,
            )
        for image_index in range(2):
            _write_image(
                tmp_path / "Testing" / class_name / f"test_{image_index}.png",
                50 + class_index * 80 + image_index,
            )

    train_loader, val_loader, test_loader, classes, mean, std = load_tpal_datasets(
        tmp_path,
        image_size=16,
        validation_split=0.25,
        seed=42,
        batch_size=2,
    )

    assert classes == ["glioma", "meningioma"]
    assert len(train_loader.dataset) == 6
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 4
    assert 0.0 < mean < 1.0
    assert std > 0.0

    images, labels = next(iter(train_loader))
    assert images.shape == (2, 1, 16, 16)
    assert labels.shape == (2,)
    assert torch.isfinite(images).all()


def test_tpal_loader_rejects_class_folder_mismatch(tmp_path):
    _write_image(tmp_path / "Training" / "glioma" / "train.png", 80)
    _write_image(tmp_path / "Training" / "meningioma" / "train.png", 120)
    _write_image(tmp_path / "Testing" / "glioma" / "test.png", 80)
    _write_image(tmp_path / "Testing" / "other" / "test.png", 120)

    with pytest.raises(ValueError, match="do not match"):
        load_tpal_datasets(tmp_path, image_size=16, batch_size=2)
