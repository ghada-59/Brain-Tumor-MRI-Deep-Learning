"""Build a near-duplicate-screened copy of data_leakage_free/.

data_leakage_free/ only removes EXACT (byte/pixel-identical) duplicates. The
near-duplicate audit (audit_near_duplicates.py -> reports_near_duplicates_20261021/)
found Hamming-distance-0 perceptual-hash matches that are NOT exact duplicates but
are, on visual inspection, the same anatomical slice (re-exported, re-compressed,
or differently windowed copies) — see README.md, "Dataset integrity: near-duplicate
leakage". This script builds a new, separate dataset copy with those matches
controlled for, without ever modifying data/ or data_leakage_free/.

Policy (conservative; threshold and rationale documented in
src/brain_tumor_dl/leakage.py and README.md):

1. Connected components are computed over near_duplicate_candidates.csv pairs at
   or below --max-distance (default 0 — the only threshold visually verified in
   this project so far; see README for why looser thresholds are not used by
   default).
2. Any component that includes a Testing/-folder image is "test-touching": every
   Training/-folder member of that component is EXCLUDED from the output
   Training/ folder. Testing/ is copied through unmodified — it remains the
   fixed, citable held-out benchmark used by every historical test_metrics.json.
3. Any component entirely within Training/ ("pure training") is recorded as an
   atomic group in a group manifest. This script does not itself split
   train/validation (that happens at train time); the manifest lets
   split_files_by_class_grouped (src/brain_tumor_dl/data.py) keep each group on
   one side of a future train/validation split instead of letting chance
   separate near-duplicate images across it.

This does NOT establish patient-level independence: image-level near-duplicate
grouping only prevents the specific matches this audit found from crossing
train/validation/test boundaries. It cannot detect near-duplicates this audit
missed, and image similarity alone cannot prove or disprove shared patient
identity. Patient identifiers are not available in this dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.brain_tumor_dl.data import SUPPORTED_EXTENSIONS, count_images_by_class
from src.brain_tumor_dl.leakage import build_duplicate_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("data_leakage_free"))
    parser.add_argument("--output", type=Path, default=Path("data_leakage_controlled_distance0"))
    parser.add_argument(
        "--near-duplicates-csv",
        type=Path,
        default=Path("reports_near_duplicates_20261021/near_duplicate_candidates.csv"),
    )
    parser.add_argument("--max-distance", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists. Rename or inspect it first; this "
            "script never deletes or overwrites an existing output folder."
        )
    if not args.source.is_dir():
        raise SystemExit(f"Source dataset not found: {args.source}")
    if not args.near_duplicates_csv.is_file():
        raise SystemExit(
            f"Near-duplicate candidates CSV not found: {args.near_duplicates_csv}. "
            "Run audit_near_duplicates.py first."
        )

    training_root = args.source / "Training"
    testing_root = args.source / "Testing"
    if not training_root.is_dir() or not testing_root.is_dir():
        raise SystemExit(f"Expected {training_root} and {testing_root}.")

    before_counts = {
        "Training": count_images_by_class(training_root),
        "Testing": count_images_by_class(testing_root),
    }

    groups = build_duplicate_groups(args.near_duplicates_csv, testing_root, max_distance=args.max_distance)
    excluded_training_paths = groups.excluded_training_paths()  # resolved path (str) -> component index
    pure_training_groups = groups.pure_training_groups()  # group_id -> sorted resolved paths (str)

    # Copy Testing/ through unmodified.
    for class_dir in sorted(p for p in testing_root.iterdir() if p.is_dir()):
        destination_dir = args.output / "Testing" / class_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                shutil.copy2(path, destination_dir / path.name)

    # Copy Training/, skipping excluded (test-touching) images.
    excluded_records = []
    kept_by_class: Counter[str] = Counter()
    excluded_by_class: Counter[str] = Counter()
    for class_dir in sorted(p for p in training_root.iterdir() if p.is_dir()):
        destination_dir = args.output / "Training" / class_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(class_dir.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS):
                continue
            resolved = str(path.resolve())
            if resolved in excluded_training_paths:
                excluded_by_class[class_dir.name] += 1
                excluded_records.append({
                    "excluded_training_image": str(path.relative_to(args.source)),
                    "class": class_dir.name,
                    "component_index": excluded_training_paths[resolved],
                    "reason": f"near-duplicate (Hamming distance <= {args.max_distance}) of a Testing image",
                })
                continue
            shutil.copy2(path, destination_dir / path.name)
            kept_by_class[class_dir.name] += 1

    args.output.mkdir(parents=True, exist_ok=True)

    # Excluded-image manifest.
    excluded_path = args.output / "excluded_training_images.json"
    excluded_path.write_text(json.dumps(excluded_records, indent=2), encoding="utf-8")

    # Atomic train/validation group manifest, keyed by resolved path IN THE OUTPUT
    # directory (so it is directly usable with split_files_by_class_grouped on
    # args.output without the caller having to translate paths).
    relative_groups: dict[str, list[str]] = {}
    for group_id, members in pure_training_groups.items():
        output_members = []
        for member in members:
            relative = Path(member).resolve().relative_to(args.source.resolve())
            output_members.append(str((args.output / relative).resolve()))
        relative_groups[group_id] = output_members
    group_manifest_path = args.output / "train_validation_groups.json"
    group_manifest_path.write_text(json.dumps(relative_groups, indent=2), encoding="utf-8")

    # path -> group_id mapping, the format split_files_by_class_grouped expects directly.
    path_to_group = {path: group_id for group_id, paths in relative_groups.items() for path in paths}
    path_to_group_path = args.output / "train_validation_groups_by_path.json"
    path_to_group_path.write_text(json.dumps(path_to_group, indent=2), encoding="utf-8")

    after_counts = {
        "Training": count_images_by_class(args.output / "Training"),
        "Testing": count_images_by_class(args.output / "Testing"),
    }
    class_balance = {
        "before": before_counts,
        "after": after_counts,
        "training_images_lost_per_class": dict(excluded_by_class),
        "training_images_kept_per_class": dict(kept_by_class),
    }

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "near_duplicates_csv": str(args.near_duplicates_csv),
        "max_hamming_distance_excluded": args.max_distance,
        "policy": (
            "Any connected component (at or below max_hamming_distance) touching a "
            "Testing/ image has all its Training/ members excluded from Training/ "
            "in the output. Testing/ is never modified. Components confined to "
            "Training/ are recorded as atomic train/validation groups rather than "
            "being split, via train_validation_groups_by_path.json."
        ),
        "total_connected_components_at_threshold": len(groups.components),
        "test_touching_components": sum(1 for c in groups.components if groups.touches_testing(c)),
        "pure_training_components_with_2plus_members": len(pure_training_groups),
        "total_training_images_excluded": len(excluded_records),
        "class_balance": class_balance,
        "limitations": [
            "Grouping is based only on the specific near-duplicate pairs this audit "
            "found at the chosen threshold; it cannot detect near-duplicates the "
            "audit missed (including anything above --max-distance).",
            "Image-level grouping does not establish or verify patient-level "
            "independence; this dataset has no patient identifiers.",
            "The near-duplicate audit only compared images across DIFFERENT splits "
            "(train/validation/test); it does not detect near-duplicate clusters "
            "entirely within one split beyond what the train-vs-validation "
            "comparison happened to surface, so a small residual risk of "
            "undetected within-split duplication remains.",
        ],
    }
    summary_path = args.output / "leakage_control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Leakage-controlled dataset preparation complete.")
    print(f"Output: {args.output.resolve()}")
    print(f"Connected components at distance<={args.max_distance}: {len(groups.components)}")
    print(f"  test-touching: {summary['test_touching_components']}")
    print(f"  pure-training (grouped for train/validation): {summary['pure_training_components_with_2plus_members']}")
    print(f"Training images excluded: {len(excluded_records)}")
    print(f"  by class: {dict(excluded_by_class)}")
    print("Class balance before -> after (Training):")
    for class_name in before_counts["Training"]:
        print(
            f"  {class_name}: {before_counts['Training'][class_name]} -> "
            f"{after_counts['Training'].get(class_name, 0)}"
        )
    print(f"\nManifests: {excluded_path.name}, {group_manifest_path.name}, "
          f"{path_to_group_path.name}, {summary_path.name}")
    print("Testing/ was copied unmodified. data_leakage_free/ and data/ were not touched.")


if __name__ == "__main__":
    main()
