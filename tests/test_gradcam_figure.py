"""make_gradcam_figure.py writes a figure and a provenance sidecar (untrained fixture model)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import make_gradcam_figure
from src.brain_tumor_dl.experiment_record import file_sha256
from src.brain_tumor_dl.model import build_model


def test_figure_and_sidecar_record_the_checkpoint_and_images(tmp_path: Path) -> None:
    classes = ["glioma", "meningioma"]
    rng = np.random.default_rng(0)
    for name in classes:
        (tmp_path / "Testing" / name).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(rng.integers(0, 255, size=(60, 60, 3), dtype=np.uint8)).save(tmp_path / "Testing" / name / f"{i}.jpg")
    model_path = tmp_path / "model.keras"
    build_model(2, (224, 224), weights=None).save(model_path)
    (tmp_path / "classes.json").write_text(json.dumps(classes))
    output = tmp_path / "out" / "gradcam.png"
    make_gradcam_figure.main([
        "--model", str(model_path), "--classes", str(tmp_path / "classes.json"),
        "--data-dir", str(tmp_path / "Testing"), "--output", str(output),
    ])
    sidecar = json.loads(output.with_suffix(".json").read_text())
    assert output.is_file()
    assert sidecar["model_sha256"] == file_sha256(model_path)
    assert [Path(r["image"]).name for r in sidecar["rows"]] == ["0.jpg", "0.jpg"]
    assert [r["true_label"] for r in sidecar["rows"]] == classes
