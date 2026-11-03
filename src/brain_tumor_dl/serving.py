"""Model lifecycle and input validation for the classification API.

``ClassifierService`` owns one checkpoint and reports exactly one of these states:

* ``missing``      - the checkpoint file does not exist;
* ``not_loaded``   - the file exists but has not been loaded yet (lazy mode);
* ``ready``        - loaded and validated; inference can run;
* ``load_failed``  - the file exists but could not be loaded or failed validation.

Validation at load time: the class list is a non-empty list of unique strings; the
model takes (None, H, W, 3) with H == W; its output has one value per class; and the
final layer is a softmax, so outputs are probabilities and no extra softmax is applied.
Every prediction is checked again (shape, finite values, probabilities summing to 1).

Loading is guarded by a lock so concurrent first requests load the model once, and a
failed load is not retried until the checkpoint file changes. Error messages returned
to clients never contain server file paths; the full error goes to the server log.
"""
from __future__ import annotations

import io
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.brain_tumor_dl.experiment_record import file_sha256
from src.brain_tumor_dl.model_paths import ModelResolution

logger = logging.getLogger("neurovision.serving")

MISSING, NOT_LOADED, READY, LOAD_FAILED = "missing", "not_loaded", "ready", "load_failed"

# Formats whose decoding matches the training pipeline (tf.io.decode_image).
TF_DECODED_FORMATS = {"JPEG", "PNG", "BMP"}
# Accepted but decoded with Pillow; reported to the client as such.
PILLOW_DECODED_FORMATS = {"WEBP"}
ACCEPTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # same limit as the dashboard
MAX_IMAGE_PIXELS = 25_000_000
PROBABILITY_SUM_TOLERANCE = 1e-3


class ModelUnavailableError(RuntimeError):
    """No usable model: missing file or failed load (HTTP 503)."""


class InvalidModelOutputError(RuntimeError):
    """The model returned something that is not one probability per class (HTTP 500)."""


class InvalidUploadError(ValueError):
    """The upload cannot be used as an image (HTTP 400/413/415, see ``status_code``)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DecodedUpload:
    format: str
    width: int
    height: int
    mode: str
    image: Image.Image


def inspect_upload(payload: bytes, filename: str) -> DecodedUpload:
    """Check size, extension and actual content of an uploaded image."""
    if not filename.lower().endswith(ACCEPTED_EXTENSIONS):
        raise InvalidUploadError("Upload a PNG, JPG, BMP or WEBP image.", 415)
    if not payload:
        raise InvalidUploadError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InvalidUploadError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
    try:
        image = Image.open(io.BytesIO(payload))
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise InvalidUploadError(f"The image has more than {MAX_IMAGE_PIXELS // 1_000_000} million pixels.", 413)
        image.load()  # full decode: catches truncated or corrupt files
    except InvalidUploadError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise InvalidUploadError("The uploaded file could not be decoded as an image.") from exc
    fmt = (image.format or "").upper()
    if fmt not in TF_DECODED_FORMATS | PILLOW_DECODED_FORMATS:
        raise InvalidUploadError(f"Unsupported image format '{fmt or 'unknown'}'. Use PNG, JPG, BMP or WEBP.", 415)
    return DecodedUpload(format=fmt, width=width, height=height, mode=image.mode, image=image)


def preprocess_upload(payload: bytes, upload: DecodedUpload, image_size: int) -> tuple[np.ndarray, str]:
    """Model input batch (1, N, N, 3) for an inspected upload, and the decoder used.

    JPEG/PNG/BMP go through the training pipeline's own decode and bilinear resize
    (``tensorflow``); WEBP, which ``tf.io.decode_image`` cannot read, is decoded with
    Pillow and then resized the same way (``pillow``).
    """
    from src.brain_tumor_dl.inference import preprocess_image_bytes, preprocess_pil_image

    if upload.format in PILLOW_DECODED_FORMATS:
        return preprocess_pil_image(upload.image, image_size), "pillow"
    return preprocess_image_bytes(payload, image_size), "tensorflow"


def _default_loader(path: Path) -> Any:
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def _final_activation(model: Any) -> str | None:
    try:
        activation = getattr(model.layers[-1], "activation", None)
    except (AttributeError, IndexError):
        return None
    return getattr(activation, "__name__", None)


class ClassifierService:
    def __init__(self, resolution: ModelResolution, class_names_path: Path, root: Path,
                 loader: Callable[[Path], Any] | None = None) -> None:
        self.resolution = resolution
        self.path = resolution.path
        self.class_names_path = class_names_path
        self.root = root
        self._loader = loader or _default_loader
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._state = NOT_LOADED
        self._error: str | None = None
        self._model: Any = None
        self._class_names: list[str] = []
        self._image_size: int | None = None
        self._loaded_stat: tuple[int, int] | None = None
        self._failed_stat: tuple[int, int] | None = None
        self._sha256: str | None = None

    # -- helpers ---------------------------------------------------------------
    def display_path(self, path: Path) -> str:
        """Repository-relative path, or just the file name: never an absolute path."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.name

    def _stat(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    def _fail(self, public: str, exc: BaseException | None = None) -> None:
        self._state, self._error, self._model = LOAD_FAILED, public, None
        self._failed_stat = self._stat()
        if exc is not None:
            # Keras nests the real cause under several "Exception encountered:" layers; log that
            # cause on one line, and the full chained traceback only at DEBUG level.
            cause = str(exc).rsplit("Exception encountered:", 1)[-1].strip().splitlines()
            logger.error("Model load failed for %s: %s: %s", self.display_path(self.path), type(exc).__name__,
                         cause[0][:300] if cause else "no message")
            logger.debug("Full model-load traceback", exc_info=exc)
        else:
            logger.error("Model load failed for %s: %s", self.display_path(self.path), public)

    def read_class_names(self) -> list[str]:
        """The class list from class_names.json (validated); raises OSError/ValueError."""
        names = json.loads(self.class_names_path.read_text(encoding="utf-8"))
        if (not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("class_names.json must be a non-empty list of unique class names.")
        return names

    # -- lifecycle -------------------------------------------------------------
    @property
    def state(self) -> str:
        if self._state == READY:
            return READY  # weights are in memory; a later file change is reported separately
        if not self.path.is_file():
            return MISSING
        if self._state == MISSING:
            return NOT_LOADED  # the file has appeared since the last check
        if self._state == LOAD_FAILED and self._stat() != self._failed_stat:
            return NOT_LOADED  # the file changed since the failure: a new load will be attempted
        return self._state

    def ensure_loaded(self) -> bool:
        """Load and validate once; return True when the model is ready."""
        if self.state == READY:
            return True
        with self._load_lock:
            state = self.state
            if state == READY:
                return True
            if state == MISSING:
                self._state, self._error = MISSING, "Model weights file not found."
                return False
            if state == LOAD_FAILED:
                return False
            try:
                class_names = self.read_class_names()
            except FileNotFoundError as exc:
                self._fail("class_names.json not found next to the checkpoint or in artifacts/.", exc)
                return False
            except (ValueError, json.JSONDecodeError) as exc:
                self._fail("class_names.json is invalid.", exc)
                return False
            stat_before = self._stat()
            try:
                sha = file_sha256(self.path)
                model = self._loader(self.path)
            except Exception as exc:  # any loader failure is reported, never hidden
                self._fail("The checkpoint could not be loaded (corrupt file or incompatible Keras/TensorFlow version).", exc)
                return False
            input_shape = tuple(getattr(model, "input_shape", ()) or ())
            output_shape = tuple(getattr(model, "output_shape", ()) or ())
            if len(input_shape) != 4 or input_shape[3] != 3 or input_shape[1] != input_shape[2] or not input_shape[1]:
                self._fail(f"Unexpected model input shape {input_shape}; expected (None, N, N, 3).")
                return False
            if len(output_shape) != 2 or output_shape[1] != len(class_names):
                self._fail(f"Model output shape {output_shape} does not match {len(class_names)} class names.")
                return False
            if _final_activation(model) != "softmax":
                self._fail("The model's final layer is not a softmax; this API only serves probability outputs.")
                return False
            if self._stat() != stat_before:
                self._fail("The checkpoint changed while it was being loaded; retry.")
                return False
            self._model, self._class_names, self._image_size = model, class_names, int(input_shape[1])
            self._sha256, self._loaded_stat = sha, stat_before
            self._state, self._error = READY, None
            logger.info("Model ready: %s (variant %s, sha256 %s, input %s, %d classes)",
                        self.display_path(self.path), self.resolution.variant, sha[:12], input_shape, len(class_names))
            return True

    # -- inference -------------------------------------------------------------
    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def image_size(self) -> int | None:
        return self._image_size

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Probabilities for one preprocessed image, validated before they are returned."""
        if not self.ensure_loaded():
            raise ModelUnavailableError(self._error or "Model unavailable.")
        expected = (1, self._image_size, self._image_size, 3)
        if batch.shape != expected:
            raise ValueError(f"Preprocessed batch has shape {batch.shape}, expected {expected}.")
        with self._predict_lock:
            raw = self._model.predict(batch, verbose=0)
        scores = np.asarray(raw, dtype=np.float64)
        if scores.shape != (1, len(self._class_names)):
            raise InvalidModelOutputError(f"Model returned shape {scores.shape}, expected (1, {len(self._class_names)}).")
        scores = scores[0]
        if not np.all(np.isfinite(scores)):
            raise InvalidModelOutputError("Model returned non-finite scores.")
        if np.any(scores < -1e-6) or np.any(scores > 1 + 1e-6) or abs(scores.sum() - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise InvalidModelOutputError("Model scores are not probabilities (outside [0, 1] or not summing to 1).")
        return scores

    # -- status ----------------------------------------------------------------
    def _experiment_record_match(self) -> bool | None:
        record = self.path.parent / "experiment_record.json"
        if self._sha256 is None or not record.is_file():
            return None
        try:
            expected = json.loads(record.read_text(encoding="utf-8")).get("checkpoint_sha256")
        except (OSError, json.JSONDecodeError):
            return None
        return None if not expected else expected == self._sha256

    def describe(self) -> dict:
        state = self.state
        return {
            "model_state": state,
            "model_ready": state == READY,
            "model_error": self._error if state in (LOAD_FAILED, MISSING) else None,
            "model_sha256": self._sha256 if state == READY else None,
            "model_matches_experiment_record": self._experiment_record_match() if state == READY else None,
            "model_file_changed_since_load": (self._stat() != self._loaded_stat) if state == READY else None,
            "model_input_shape": [None, self._image_size, self._image_size, 3] if state == READY else None,
        }
