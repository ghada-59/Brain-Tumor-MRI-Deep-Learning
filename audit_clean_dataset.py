
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image

ROOT = Path("data_leakage_free")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def get_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = sha256()
        digest.update(f"{image.width}x{image.height}:".encode())
        digest.update(image.tobytes())
        return digest.hexdigest()


def scan_split(split):
    hashes = defaultdict(list)
    folder = ROOT / split

    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            hashes[get_hash(path)].append(path)

    return hashes


training = scan_split("Training")
testing = scan_split("Testing")

train_duplicates = sum(
    len(paths) - 1 for paths in training.values() if len(paths) > 1
)
test_duplicates = sum(
    len(paths) - 1 for paths in testing.values() if len(paths) > 1
)

overlap = set(training) & set(testing)

print("=== Audit du dataset nettoyé ===")
print(f"Images uniques dans Training : {len(training)}")
print(f"Images uniques dans Testing : {len(testing)}")
print(f"Doublons internes Training : {train_duplicates}")
print(f"Doublons internes Testing : {test_duplicates}")
print(f"Images identiques entre Training et Testing : {len(overlap)}")

if train_duplicates == 0 and test_duplicates == 0 and len(overlap) == 0:
    print("\nSUCCES : aucun doublon exact détecté.")
    print("Le dataset peut passer à l'étape d'entraînement.")
else:
    print("\nATTENTION : des doublons exacts restent à examiner.")
