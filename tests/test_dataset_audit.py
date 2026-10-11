from pathlib import Path

from PIL import Image

from src.brain_tumor_dl.audit import find_exact_duplicates, image_fingerprints


def test_detects_same_file_bytes_across_splits(tmp_path: Path) -> None:
    first = tmp_path / "train.jpg"
    second = tmp_path / "test.jpg"
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")

    # The audit is image-aware; invalid image bytes should be reported as a clear error.
    import pytest
    with pytest.raises(ValueError, match="Cannot decode image"):
        find_exact_duplicates({"train": [first], "test": [second]})


def test_detects_pixel_identical_images_with_different_file_encodings(tmp_path: Path) -> None:
    first = tmp_path / "train.png"
    second = tmp_path / "test.bmp"
    image = Image.new("RGB", (12, 8), color=(20, 40, 60))
    image.save(first, format="PNG")
    image.save(second, format="BMP")

    assert image_fingerprints(first)[0] != image_fingerprints(second)[0]
    assert image_fingerprints(first)[1] == image_fingerprints(second)[1]
    result = find_exact_duplicates({"train": [first], "test": [second]})
    assert result["identical_files"] == []
    assert len(result["identical_decoded_pixels"]) == 1


def test_no_duplicates_returns_empty_groups(tmp_path: Path) -> None:
    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(first)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(second)

    result = find_exact_duplicates({"train": [first], "test": [second]})
    assert result == {"identical_files": [], "identical_decoded_pixels": []}



def test_finds_byte_identical_image_copied_between_splits(tmp_path: Path) -> None:
    first = tmp_path / "training.png"
    second = tmp_path / "testing.png"
    Image.new("RGB", (10, 10), color=(15, 25, 35)).save(first, format="PNG")
    second.write_bytes(first.read_bytes())

    result = find_exact_duplicates({"train": [first], "test": [second]})
    assert len(result["identical_files"]) == 1
    assert len(result["identical_decoded_pixels"]) == 1
