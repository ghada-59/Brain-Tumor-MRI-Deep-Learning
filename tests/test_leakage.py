from pathlib import Path

import pytest

from src.brain_tumor_dl.leakage import UnionFind, build_duplicate_groups


def test_union_find_groups_transitively_connected_nodes() -> None:
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    uf.union("d", "e")
    assert uf.find("a") == uf.find("c")
    assert uf.find("a") != uf.find("d")


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = ["hamming_distance", "split_1", "class_1", "image_1", "split_2", "class_2", "image_2"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_duplicate_groups_identifies_test_touching_and_pure_training_components(
    tmp_path: Path,
) -> None:
    testing_root = tmp_path / "Testing"
    training_root = tmp_path / "Training"
    (testing_root / "glioma").mkdir(parents=True)
    (training_root / "glioma").mkdir(parents=True)

    test_image = testing_root / "glioma" / "test_a.jpg"
    train_image_1 = training_root / "glioma" / "train_a.jpg"
    train_image_2 = training_root / "glioma" / "train_b.jpg"
    train_image_3 = training_root / "glioma" / "train_c.jpg"
    train_image_4 = training_root / "glioma" / "train_d.jpg"
    for image in (test_image, train_image_1, train_image_2, train_image_3, train_image_4):
        image.write_bytes(b"fixture")

    csv_path = tmp_path / "near_duplicate_candidates.csv"
    _write_csv(
        csv_path,
        [
            # test_a <-> train_a: a test-touching component (train_a must be excluded).
            {
                "hamming_distance": "0", "split_1": "train", "class_1": "glioma",
                "image_1": str(train_image_1), "split_2": "test", "class_2": "glioma",
                "image_2": str(test_image),
            },
            # train_b <-> train_c: a pure-training component (must stay grouped).
            {
                "hamming_distance": "0", "split_1": "train", "class_1": "glioma",
                "image_1": str(train_image_2), "split_2": "validation", "class_2": "glioma",
                "image_2": str(train_image_3),
            },
            # train_d <-> test_a at distance 4: excluded by the distance-0 threshold.
            {
                "hamming_distance": "4", "split_1": "train", "class_1": "glioma",
                "image_1": str(train_image_4), "split_2": "test", "class_2": "glioma",
                "image_2": str(test_image),
            },
        ],
    )

    groups = build_duplicate_groups(csv_path, testing_root, max_distance=0)

    excluded = groups.excluded_training_paths()
    assert str(train_image_1.resolve()) in excluded
    assert str(train_image_4.resolve()) not in excluded  # distance 4 > threshold 0

    pure_groups = groups.pure_training_groups()
    assert len(pure_groups) == 1
    (members,) = pure_groups.values()
    assert members == sorted([str(train_image_2.resolve()), str(train_image_3.resolve())])


def test_build_duplicate_groups_respects_max_distance(tmp_path: Path) -> None:
    testing_root = tmp_path / "Testing"
    training_root = tmp_path / "Training"
    (testing_root / "glioma").mkdir(parents=True)
    (training_root / "glioma").mkdir(parents=True)
    train_image = training_root / "glioma" / "a.jpg"
    test_image = testing_root / "glioma" / "b.jpg"
    train_image.write_bytes(b"fixture")
    test_image.write_bytes(b"fixture")

    csv_path = tmp_path / "near_duplicate_candidates.csv"
    _write_csv(
        csv_path,
        [
            {
                "hamming_distance": "2", "split_1": "train", "class_1": "glioma",
                "image_1": str(train_image), "split_2": "test", "class_2": "glioma",
                "image_2": str(test_image),
            },
        ],
    )
    strict = build_duplicate_groups(csv_path, testing_root, max_distance=0)
    assert strict.excluded_training_paths() == {}

    loose = build_duplicate_groups(csv_path, testing_root, max_distance=2)
    assert len(loose.excluded_training_paths()) == 1
