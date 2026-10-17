"""Compare recorded held-out metrics without inventing missing results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_metrics(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} metrics file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a table from already-recorded MobileNetV2 and T-PAL metrics."
    )
    parser.add_argument("--baseline", type=Path, default=Path("artifacts_clean/test_metrics.json"))
    parser.add_argument("--finetuned", type=Path, default=Path("artifacts_finetune/test_metrics.json"))
    parser.add_argument("--tpal", type=Path, default=Path("reports_tpal/test_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports_model_comparison/model_comparison.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for label, path in (
        ("MobileNetV2 baseline", args.baseline),
        ("MobileNetV2 fine-tuned", args.finetuned),
        ("T-PAL", args.tpal),
    ):
        try:
            metrics = read_metrics(path, label)
        except FileNotFoundError:
            if label == "T-PAL":
                print("T-PAL metrics are not available yet; train and evaluate T-PAL first.")
                continue
            raise
        rows.append({
            "model": label,
            "test_accuracy": metrics.get("test_accuracy"),
            "test_macro_f1": metrics.get("test_macro_f1"),
            "test_weighted_f1": metrics.get("test_weighted_f1"),
            "test_balanced_accuracy": metrics.get("test_balanced_accuracy"),
            "test_image_count": metrics.get("test_image_count"),
            "best_epoch": metrics.get("best_epoch"),
            "source_metrics_file": str(path),
        })

    if not rows:
        raise SystemExit("No metrics were found; run the training/evaluation workflows first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved comparison table: {args.output}")
    print(
        "Interpret the table as a valid model comparison only when all models were "
        "evaluated on the same held-out images and class order. Matching image counts "
        "alone does not prove that the test samples are identical."
    )


if __name__ == "__main__":
    main()
