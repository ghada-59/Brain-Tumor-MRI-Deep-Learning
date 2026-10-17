"""Local FastAPI adapter for the Brain MRI classification practice project."""
from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.brain_tumor_dl.inference import preprocess_pil_image
from src.brain_tumor_dl.model_paths import resolve_mobilenet_path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = resolve_mobilenet_path(ROOT)
CLASSES_PATH = ROOT / "artifacts" / "class_names.json"
IMAGE_SIZE = 224
app = FastAPI(title="NeuroVision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache(maxsize=1)
def _load_assets():
    if not MODEL_PATH.is_file():
        return None, None
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError("artifacts/class_names.json is missing.")
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names.json must contain a non-empty list.")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, class_names


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "NeuroVision API"}


@app.get("/api/status")
def status():
    return {
        "model_available": MODEL_PATH.is_file(),
        "class_names_available": CLASSES_PATH.is_file(),
        "expected_model": MODEL_PATH.relative_to(ROOT).as_posix(),
        "image_size": IMAGE_SIZE,
        "classes": json.loads(CLASSES_PATH.read_text(encoding="utf-8")) if CLASSES_PATH.is_file() else [],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename or "brain-mri"
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, WEBP or BMP image.")
    try:
        payload = await file.read()
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file could not be decoded as an image.") from exc
        model, class_names = _load_assets()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Trained model weights are not included in this repository. Train the model locally and place artifacts/brain_tumor_mobilenetv2.keras in the expected location.",
            )
        pixels = preprocess_pil_image(image, IMAGE_SIZE)
        probabilities = np.asarray(model.predict(pixels, verbose=0)[0], dtype=float)
        if probabilities.ndim != 1 or len(probabilities) != len(class_names):
            raise ValueError("Model output size does not match class_names.json.")
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned non-finite scores.")
        order = np.argsort(probabilities)[::-1]
        return {
            "filename": filename,
            "preview": _image_data_url(image),
            "image_size": IMAGE_SIZE,
            "top_class": class_names[int(order[0])],
            "classes": [{"label": class_names[int(i)], "score": float(probabilities[i])} for i in order[: min(4, len(order))]],
            "note": "Educational image-level classification only. Not a medical diagnosis. Scores are model outputs and are not clinically calibrated confidence.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Inference failed: " + str(exc)) from exc
