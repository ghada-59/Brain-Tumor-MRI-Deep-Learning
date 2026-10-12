
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import shutil

from PIL import Image

SOURCE = Path("data")
OUTPUT = Path("data_leakage_free")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def image_hash(path: Path) -> str:
    """Hash decoded RGB pixels, not the original file encoding."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = sha256()
        digest.update(f"{rgb.width}x{rgb.height}:".encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def main():
    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists. Rename or inspect it first; "
            "this script never deletes an existing output folder."
        )

    train_root = SOURCE / "Training"
    test_root = SOURCE / "Testing"

    if not train_root.is_dir() or not test_root.is_dir():
        raise SystemExit("Expected data/Training/ and data/Testing/.")

    train_classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    test_classes = sorted(p.name for p in test_root.iterdir() if p.is_dir())

    if train_classes != test_classes:
        raise SystemExit(
            f"Class mismatch:\nTraining={train_classes}\nTesting={test_classes}"
        )

    records = []
    hash_labels = defaultdict(set)

    for split, root in (("Training", train_root), ("Testing", test_root)):
        for class_name in sorted(train_classes):
            for path in image_files(root / class_name):
                try:
                    digest = image_hash(path)
                except Exception as exc:
                    raise SystemExit(f"Cannot read image {path}: {exc}") from exc

                records.append((split, class_name, path, digest))
                hash_labels[digest].add(class_name)

    conflicts = {
        digest: labels
        for digest, labels in hash_labels.items()
        if len(labels) > 1
    }

    if conflicts:
        print("ERROR: identical decoded images have conflicting class labels.")
        for digest, labels in list(conflicts.items())[:20]:
            examples = [
                str(path) for _, _, path, item_hash in records
                if item_hash == digest
            ]
            print(f"\nHash: {digest[:16]} | labels: {sorted(labels)}")
            for example in examples[:6]:
                print(f"  {example}")
        raise SystemExit(
            "No output dataset was created. Investigate these label conflicts."
        )

    test_hashes = {
        digest for split, _, _, digest in records if split == "Testing"
    }
    seen_test = set()
    seen_train = set()

    counts = {
        "source_training_images": sum(r[0] == "Training" for r in records),
        "source_testing_images": sum(r[0] == "Testing" for r in records),
        "training_removed_due_to_test_overlap": 0,
        "training_duplicates_removed": 0,
        "testing_duplicates_removed": 0,
        "clean_training_images": 0,
        "clean_testing_images": 0,
    }

    for split, class_name, path, digest in records:
        if split == "Testing":
            if digest in seen_test:
                counts["testing_duplicates_removed"] += 1
                continue
            seen_test.add(digest)
        else:
            if digest in test_hashes:
                counts["training_removed_due_to_test_overlap"] += 1
                continue
            if digest in seen_train:
                counts["training_duplicates_removed"] += 1
                continue
            seen_train.add(digest)

        destination_dir = OUTPUT / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{digest}{path.suffix.lower()}"
        shutil.copy2(path, destination)
        counts[f"clean_{split.lower()}_images"] += 1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "deduplication_summary.json"
    summary_path.write_text(
        json.dumps(counts, indent=2),
        encoding="utf-8",
    )

    print("\nDataset preparation complete.")
    for name, value in counts.items():
        print(f"{name}: {value}")
    print(f"\nOutput: {OUTPUT.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print("Original data was not modified.")
    print(
        "Note: exact pixel duplicates were checked; near-duplicates "
        "and patient-level independence are not guaranteed."
    )


if __name__ == "__main__":
    main()
