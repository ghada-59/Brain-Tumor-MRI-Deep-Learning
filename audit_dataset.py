"""Report exact duplicate images across train, validation and held-out test splits."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.brain_tumor_dl.audit import find_exact_duplicates
from src.brain_tumor_dl.data import SUPPORTED_EXTENSIONS, split_files_by_class


def images_in_class_folders(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check byte-identical and pixel-identical images across data splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    training = args.data_dir / "Training"
    testing = args.data_dir / "Testing"
    if not training.is_dir() or not testing.is_dir():
        raise SystemExit(
            "Dataset folders missing. Expected data/Training/<class>/ and data/Testing/<class>/."
        )
    try:
        train_paths, _, val_paths, _, class_names = split_files_by_class(
            training, validation_split=args.validation_split, seed=args.seed
        )
        test_classes = sorted(path.name for path in testing.iterdir() if path.is_dir())
        if test_classes != class_names:
            raise ValueError(
                f"Class folders differ: Training={class_names}, Testing={test_classes}"
            )
        test_paths = images_in_class_folders(testing)
        result = find_exact_duplicates({
            "train": train_paths,
            "validation": val_paths,
            "test": test_paths,
        })
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Dataset audit failed: {exc}") from exc

    print("Exact duplicate audit (read-only; no files changed)")
    print(f"Classes: {', '.join(class_names)}")
    for category, groups in result.items():
        print(f"\n{category}: {len(groups)} duplicate group(s)")
        for group in groups[:20]:
            print("  - " + " | ".join(group))
        if len(groups) > 20:
            print(f"  ... and {len(groups) - 20} more group(s)")
    print(
        "\nLimit: this checks identical file bytes and identical decoded pixels only. "
        "It does not detect near-duplicates or prove patient-level separation."
    )
    if result["identical_files"] or result["identical_decoded_pixels"]:
        raise SystemExit(
            "Potential leakage detected. Review the listed groups before reporting final metrics."
        )
    print("No exact duplicates detected by these checks.")


if __name__ == "__main__":
    main()
