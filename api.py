"""FastAPI service for the Brain MRI classification project (image-level classification only)."""
from __future__ import annotations

import logging
from pathlib import Path, PureWindowsPath

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.brain_tumor_dl import serving
from src.brain_tumor_dl.experiment_record import file_sha256
from src.brain_tumor_dl.model_paths import resolve_model

logger = logging.getLogger("neurovision.api")

ROOT = Path(__file__).resolve().parent
MODEL_RESOLUTION = resolve_model(ROOT)
MODEL_PATH = MODEL_RESOLUTION.path
_adjacent = MODEL_PATH.parent / "class_names.json"
CLASSES_PATH = _adjacent if _adjacent.is_file() else ROOT / "artifacts" / "class_names.json"
CLASSIFIER = serving.ClassifierService(MODEL_RESOLUTION, CLASSES_PATH, ROOT)
MAX_REQUEST_BYTES = serving.MAX_UPLOAD_BYTES + 1024 * 1024  # multipart framing overhead
API_VERSION = "1.1.0"

app = FastAPI(title="NeuroVision API", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173",
                   "http://127.0.0.1:4173", "http://localhost:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    declared = request.headers.get("content-length", "")
    if request.method == "POST" and declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
        mb = MAX_REQUEST_BYTES // (1024 * 1024)
        return JSONResponse({"detail": f"Request body is larger than {mb} MB."}, status_code=413)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    fields = sorted({str(err["loc"][-1]) for err in exc.errors() if err.get("loc")}) or ["file"]
    names = ", ".join(f"'{f}'" for f in fields)
    return JSONResponse({"detail": f"Invalid request: send the image as multipart form field {names}."},
                        status_code=400)


_disk_hash_cache: dict[tuple, str] = {}


def _loaded_model_sha256() -> str | None:
    """SHA-256 of the loaded weights, or of the file on disk when not loaded yet (cached by size/mtime)."""
    described = CLASSIFIER.describe()
    if described["model_sha256"]:
        return described["model_sha256"]
    try:
        st = MODEL_RESOLUTION.path.stat()
    except OSError:
        return None
    key = (str(MODEL_RESOLUTION.path), st.st_size, st.st_mtime_ns)
    if key not in _disk_hash_cache:
        _disk_hash_cache.clear()
        _disk_hash_cache[key] = file_sha256(MODEL_RESOLUTION.path)
    return _disk_hash_cache[key]


def _runtime() -> dict:
    import keras
    import tensorflow as tf

    return {"tensorflow": tf.__version__, "keras": keras.__version__}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "NeuroVision API"}


@app.get("/api/status")
def status():
    described = CLASSIFIER.describe()
    try:
        classes = CLASSIFIER.read_class_names()
    except (OSError, ValueError):
        classes = []
    return {
        "api_version": API_VERSION,
        **described,
        "model_available": described["model_state"] in (serving.READY, serving.NOT_LOADED),
        "model_variant": MODEL_RESOLUTION.variant,
        "expected_model": CLASSIFIER.display_path(MODEL_PATH),
        "model_sha256": _loaded_model_sha256(),
        "model_sha256_of": "loaded_weights" if described["model_ready"] else "file_on_disk",
        "class_names_available": CLASSES_PATH.is_file(),
        "image_size": CLASSIFIER.image_size or 224,
        "classes": classes,
        "runtime": _runtime(),
    }


def _client_filename(name: str | None) -> str:
    base = PureWindowsPath(name or "").name.replace("/", "\\").split("\\")[-1]
    return base or "brain-mri"


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    filename = _client_filename(file.filename)
    payload = await file.read(serving.MAX_UPLOAD_BYTES + 1)
    try:
        upload = serving.inspect_upload(payload, filename)
        if not CLASSIFIER.ensure_loaded():
            raise serving.ModelUnavailableError(CLASSIFIER.describe()["model_error"] or "Model unavailable.")
        pixels, decoder = serving.preprocess_upload(payload, upload, CLASSIFIER.image_size)
        scores = CLASSIFIER.predict(pixels)
    except serving.InvalidUploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except serving.ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except serving.InvalidModelOutputError as exc:
        logger.error("Invalid model output: %s", exc)
        raise HTTPException(status_code=500, detail="The model returned an invalid output.") from exc
    names = CLASSIFIER.class_names
    order = sorted(range(len(names)), key=lambda i: -scores[i])
    description = CLASSIFIER.describe()
    return {
        "filename": filename,
        "top_class": names[order[0]],
        "classes": [{"label": names[i], "score": float(scores[i])} for i in order],
        "scores": {names[i]: float(scores[i]) for i in range(len(names))},
        "preprocessing": {
            "decoder": decoder,
            "image_size": CLASSIFIER.image_size,
            "original_size": [upload.width, upload.height],
            "input_range": "0-255 (model rescales internally)",
        },
        "model": {
            "sha256": description["model_sha256"],
            "variant": MODEL_RESOLUTION.variant,
            "matches_experiment_record": description["model_matches_experiment_record"],
        },
        "note": "Educational image-level classification only. Not a medical diagnosis. "
                "Scores are softmax outputs, not calibrated probabilities.",
    }
