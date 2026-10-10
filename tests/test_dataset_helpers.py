from pathlib import Path

from src.brain_tumor_dl.data import count_images_by_class


def test_count_images_by_class_counts_only_supported_images(tmp_path: Path) -> None:
    (tmp_path / "glioma").mkdir()
    (tmp_path / "meningioma").mkdir()
    (tmp_path / "glioma" / "scan.jpg").write_bytes(b"not-an-image-but-a-jpg-fixture")
    (tmp_path / "glioma" / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "meningioma" / "scan.png").write_bytes(b"fixture")

    assert count_images_by_class(tmp_path) == {"glioma": 1, "meningioma": 1}


def test_missing_directory_returns_empty_mapping(tmp_path: Path) -> None:
    assert count_images_by_class(tmp_path / "not-here") == {}
