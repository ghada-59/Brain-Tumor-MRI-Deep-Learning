
from pathlib import Path

root = Path("data_leakage_free")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

for split in ("Training", "Testing"):
    split_dir = root / split

    print(f"\n=== {split} ===")

    if not split_dir.is_dir():
        print("ERREUR : dossier introuvable")
        continue

    total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        print(f"{class_dir.name}: {count} images")
        total += count

    print(f"TOTAL {split}: {total} images")

print("\nVérification terminée.")
