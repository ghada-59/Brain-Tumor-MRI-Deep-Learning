
from pathlib import Path
from itertools import combinations

from src.brain_tumor_dl.audit import find_exact_duplicates
from src.brain_tumor_dl.data import (
    SUPPORTED_EXTENSIONS,
    split_files_by_class,
)


def main():
    train, _, validation, _, _ = split_files_by_class(
        Path("data/Training"),
        validation_split=0.15,
        seed=42,
    )

    test = sorted(
        p
        for p in Path("data/Testing").rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    result = find_exact_duplicates({
        "train": train,
        "validation": validation,
        "test": test,
    })

    for kind, groups in result.items():
        pair_counts = {
            ("train", "validation"): 0,
            ("train", "test"): 0,
            ("validation", "test"): 0,
        }

        for group in groups:
            splits = {
                item.split(":", 1)[0]
                for item in group
            }

            for pair in combinations(
                ["train", "validation", "test"], 2
            ):
                if set(pair).issubset(splits):
                    pair_counts[pair] += 1

        print(f"\n=== {kind} ===")
        for pair, count in pair_counts.items():
            print(f"{pair[0]} <-> {pair[1]} : {count} groupes")

    print(
        "\nAudit terminé. Aucune image n'a été supprimée "
        "ni modifiée."
    )


if __name__ == "__main__":
    main()
