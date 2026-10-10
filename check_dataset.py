"""Check that the downloaded dataset matches the expected folder layout."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.brain_tumor_dl.data import count_images_by_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MRI dataset folder layout.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    ok = True
    for split in ("Training", "Testing"):
        counts = count_images_by_class(args.data_dir / split)
        print(f"\n{split}/")
        if not counts:
            print("  MISSING: no class subfolders found")
            ok = False
            continue
        for class_name, count in counts.items():
            status = "OK" if count else "WARNING: no supported images"
            print(f"  {class_name:15} {count:5} images  {status}")
            if count == 0:
                ok = False
    train_classes = set(count_images_by_class(args.data_dir / "Training"))
    test_classes = set(count_images_by_class(args.data_dir / "Testing"))
    if train_classes != test_classes:
        print(f"\nERROR: Training and Testing classes differ: {train_classes} vs {test_classes}")
        ok = False
    if not ok:
        raise SystemExit("Dataset check failed. See the README dataset instructions.")
    print("\nDataset folders look structurally valid. This check does not verify image labels or quality.")


if __name__ == "__main__":
    main()
