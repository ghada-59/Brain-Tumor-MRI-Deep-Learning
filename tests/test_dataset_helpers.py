from pathlib import Path

import pytest

from src.brain_tumor_dl.data import count_images_by_class, split_files_by_class


def test_count_images_by_class_counts_only_supported_images(tmp_path: Path) -> None:
    (tmp_path / "glioma").mkdir()
    (tmp_path / "meningioma").mkdir()
    (tmp_path / "glioma" / "scan.jpg").write_bytes(b"fixture")
    (tmp_path / "glioma" / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "meningioma" / "scan.png").write_bytes(b"fixture")

    assert count_images_by_class(tmp_path) == {"glioma": 1, "meningioma": 1}


def test_missing_directory_returns_empty_mapping(tmp_path: Path) -> None:
    assert count_images_by_class(tmp_path / "not-here") == {}


def test_stratified_split_is_deterministic_disjoint_and_covers_each_class(
    tmp_path: Path,
) -> None:
    for class_name in ("glioma", "meningioma", "notumor"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(20):
            (class_dir / f"image_{index:02d}.jpg").write_bytes(b"fixture")

    first = split_files_by_class(tmp_path, validation_split=0.2, seed=17)
    second = split_files_by_class(tmp_path, validation_split=0.2, seed=17)
    train_paths, train_labels, val_paths, val_labels, class_names = first

    assert first == second
    assert class_names == ["glioma", "meningioma", "notumor"]
    assert set(train_paths).isdisjoint(set(val_paths))
    assert len(train_paths) + len(val_paths) == 60
    assert set(train_labels) == {0, 1, 2}
    assert set(val_labels) == {0, 1, 2}
    for label in range(len(class_names)):
        assert train_labels.count(label) == 16
        assert val_labels.count(label) == 4


@pytest.mark.parametrize("split", [0, 1, -0.1, 1.1])
def test_stratified_split_rejects_invalid_validation_fraction(
    tmp_path: Path, split: float
) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        split_files_by_class(tmp_path, validation_split=split)
