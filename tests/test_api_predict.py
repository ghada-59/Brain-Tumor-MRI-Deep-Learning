"""/api/predict and /api/status through the real ASGI stack (multipart parsing, exception
handlers, middleware), without a server or httpx.

Most tests use a FAKE model (fixed outputs) to check plumbing: status codes, error
messages, validation of model outputs, model lifecycle. The fake's numbers are not
results. The last test is an integration check with the real checkpoint: the API's
answer must equal a direct ``model.predict`` on the training pipeline's own
preprocessing of the same file. It is skipped when the checkpoint or the test images
are not present.
"""
from __future__ import annotations

import asyncio
import io
import json
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

api = pytest.importorskip("api")
from src.brain_tumor_dl import serving  # noqa: E402
from src.brain_tumor_dl.experiment_record import file_sha256  # noqa: E402
from src.brain_tumor_dl.model_paths import ModelResolution  # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


# --- helpers ----------------------------------------------------------------------

class _Activation:
    def __init__(self, name: str) -> None:
        self.__name__ = name


class _Layer:
    def __init__(self, activation: str) -> None:
        self.activation = _Activation(activation)


class FakeModel:
    """Stands in for a Keras model; records what it was asked to predict."""

    def __init__(self, output, *, activation="softmax", input_shape=(None, 224, 224, 3), n_out=4) -> None:
        self.output = np.asarray(output, dtype=np.float32)
        self.layers = [_Layer(activation)]
        self.input_shape = input_shape
        self.output_shape = (None, n_out)
        self.batches: list[np.ndarray] = []

    def predict(self, batch, verbose=0):
        self.batches.append(np.array(batch))
        return self.output


def _service(tmp_path: Path, model=None, *, write_checkpoint=True, loader=None) -> serving.ClassifierService:
    checkpoint = tmp_path / "model.keras"
    if write_checkpoint:
        checkpoint.write_bytes(b"fake checkpoint bytes")
    (tmp_path / "class_names.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    resolution = ModelResolution(path=checkpoint, variant="override", stale_repair=False, root=tmp_path)
    return serving.ClassifierService(resolution, tmp_path / "class_names.json", tmp_path,
                                     loader=loader or (lambda _path: model))


@pytest.fixture
def use_service(monkeypatch):
    def install(service: serving.ClassifierService) -> serving.ClassifierService:
        monkeypatch.setattr(api, "CLASSIFIER", service)
        monkeypatch.setattr(api, "MODEL_RESOLUTION", service.resolution)
        monkeypatch.setattr(api, "MODEL_PATH", service.path)
        monkeypatch.setattr(api, "CLASSES_PATH", service.class_names_path)
        return service
    return install


def _image(mode="RGB", fmt="PNG", size=(64, 48)) -> bytes:
    if mode == "P":  # a real palette image (quantised colours), not an all-black placeholder
        rgb = np.random.default_rng(1).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(rgb).quantize(16)
    else:
        image = Image.new(mode, size, {"L": 120, "RGB": (120, 60, 30), "RGBA": (120, 60, 30, 128)}[mode])
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _multipart(fields: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b""
    for name, (filename, data) in fields.items():
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                 "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def call(method: str, path: str, fields: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    """One request through api.app (ASGI), returning (status, JSON body)."""
    body, content_type = _multipart(fields) if fields is not None else (b"", None)
    raw_headers = {"content-length": str(len(body)), **({"content-type": content_type} if content_type else {}),
                   **(headers or {})}
    messages: list[dict] = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(k.encode(), v.encode()) for k, v in raw_headers.items()],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    asyncio.run(api.app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(payload or b"null")


def predict(filename: str, data: bytes) -> tuple[int, dict]:
    return call("POST", "/api/predict", {"file": (filename, data)})


PROBS = [0.15, 0.6, 0.2, 0.05]


# --- valid inputs -----------------------------------------------------------------

VALID = [
    ("scan.png", _image("RGB", "PNG"), "tensorflow"),
    ("scan.jpg", _image("RGB", "JPEG"), "tensorflow"),
    ("gray.png", _image("L", "PNG"), "tensorflow"),
    ("alpha.png", _image("RGBA", "PNG"), "tensorflow"),
    ("palette.png", _image("P", "PNG"), "tensorflow"),
    ("scan.bmp", _image("RGB", "BMP"), "tensorflow"),
    ("scan.webp", _image("RGB", "WEBP"), "pillow"),
]


@pytest.mark.parametrize("name,data,decoder", VALID, ids=[case[0] for case in VALID])
def test_valid_images_reach_the_model_as_one_224x224x3_float_batch(tmp_path, use_service, name, data, decoder):
    model = FakeModel([PROBS])
    use_service(_service(tmp_path, model))
    status, body = predict(name, data)
    assert status == 200, body
    batch = model.batches[-1]
    assert batch.shape == (1, 224, 224, 3) and batch.dtype == np.float32
    # Raw 0-255 pixels: the [-1, 1] scaling is a layer inside the real model, so it must not happen here.
    assert batch.min() >= 0 and batch.max() <= 255 and batch.max() > 1
    assert body["preprocessing"]["decoder"] == decoder
    assert body["top_class"] == "meningioma"
    assert [c["label"] for c in body["classes"]] == ["meningioma", "notumor", "glioma", "pituitary"]
    assert body["scores"] == pytest.approx(dict(zip(CLASSES, PROBS)))  # training class order, no re-softmax
    assert body["model"]["sha256"] == file_sha256(tmp_path / "model.keras")
    assert body["filename"] == name


def test_grayscale_is_replicated_to_three_identical_channels(tmp_path, use_service):
    model = FakeModel([PROBS])
    use_service(_service(tmp_path, model))
    assert predict("gray.png", _image("L", "PNG"))[0] == 200
    batch = model.batches[-1][0]
    assert np.array_equal(batch[..., 0], batch[..., 1]) and np.array_equal(batch[..., 1], batch[..., 2])


def test_the_api_pixels_are_exactly_the_training_pipeline_pixels(tmp_path, use_service):
    from src.brain_tumor_dl.data import _image_dataset

    rng = np.random.default_rng(0)
    image = tmp_path / "noise.jpg"
    Image.fromarray(rng.integers(0, 256, (173, 211, 3), dtype=np.uint8)).save(image, quality=90)
    model = FakeModel([PROBS])
    use_service(_service(tmp_path, model))
    assert predict("noise.jpg", image.read_bytes())[0] == 200
    training_batch, _ = next(iter(_image_dataset([image], [0], (224, 224), 1, False, 0)))
    assert np.array_equal(model.batches[-1], training_batch.numpy())


def test_client_path_in_the_filename_is_not_echoed(tmp_path, use_service):
    use_service(_service(tmp_path, FakeModel([PROBS])))
    status, body = predict("C:\\Users\\someone\\scan.png", _image())
    assert status == 200 and body["filename"] == "scan.png"


# --- invalid uploads --------------------------------------------------------------

INVALID = [
    ("notes.txt", b"hello", 415, "PNG, JPG, BMP or WEBP"),
    ("empty.png", b"", 400, "empty"),
    ("broken.jpg", b"\xff\xd8\xff not really a jpeg", 400, "could not be decoded"),
    ("truncated.png", _image("RGB", "PNG", (300, 300))[:200], 400, "could not be decoded"),
    ("animation.png", _image("RGB", "GIF"), 415, "Unsupported image format 'GIF'"),
]


@pytest.mark.parametrize("name,data,code,fragment", INVALID, ids=[case[0] for case in INVALID])
def test_unusable_uploads_are_client_errors_and_never_reach_the_model(tmp_path, use_service, name, data, code, fragment):
    model = FakeModel([PROBS])
    use_service(_service(tmp_path, model))
    status, body = predict(name, data)
    assert status == code and fragment in body["detail"]
    assert model.batches == []


def test_oversized_file_is_413(tmp_path, use_service):
    use_service(_service(tmp_path, FakeModel([PROBS])))
    status, body = predict("big.png", b"\x89PNG" + b"0" * (serving.MAX_UPLOAD_BYTES + 10))
    assert status == 413 and "15 MB" in body["detail"]


def test_too_many_pixels_is_413_before_decoding(tmp_path, use_service, monkeypatch):
    monkeypatch.setattr(serving, "MAX_IMAGE_PIXELS", 100 * 100)
    use_service(_service(tmp_path, FakeModel([PROBS])))
    status, body = predict("wide.png", _image("L", "PNG", (200, 100)))
    assert status == 413 and "million pixels" in body["detail"]


def test_declared_body_over_the_limit_is_refused_before_parsing(tmp_path, use_service):
    use_service(_service(tmp_path, FakeModel([PROBS])))
    status, body = call("POST", "/api/predict", {"file": ("x.png", _image())},
                        headers={"content-length": str(api.MAX_REQUEST_BYTES + 1)})
    assert status == 413 and "Request body is larger" in body["detail"]


def test_missing_file_field_is_a_readable_400(tmp_path, use_service):
    use_service(_service(tmp_path, FakeModel([PROBS])))
    status, body = call("POST", "/api/predict", {"image": ("x.png", _image())})
    assert status == 400 and "'file'" in body["detail"] and isinstance(body["detail"], str)


# --- model lifecycle and output validation ----------------------------------------

def test_missing_checkpoint_is_503_and_reported_by_status(tmp_path, use_service):
    service = use_service(_service(tmp_path, FakeModel([PROBS]), write_checkpoint=False))
    status, body = predict("x.png", _image())
    assert status == 503 and "not found" in body["detail"] and str(tmp_path) not in body["detail"]
    state = call("GET", "/api/status")[1]
    assert state["model_state"] == "missing" and state["model_available"] is False and state["model_ready"] is False
    # The file appears later: the next request loads it.
    service.path.write_bytes(b"now present")
    assert service.state == "not_loaded"
    assert predict("x.png", _image())[0] == 200


def test_corrupt_checkpoint_is_503_without_paths_and_is_not_reloaded_every_request(tmp_path, use_service):
    attempts = []

    def failing_loader(path):
        attempts.append(path)
        raise ValueError(f"File not found: filepath={path}. Please ensure the file is an accessible .keras zip file.")

    service = use_service(_service(tmp_path, loader=failing_loader))
    for _ in range(3):
        status, body = predict("x.png", _image())
        assert status == 503 and str(tmp_path) not in body["detail"] and "could not be loaded" in body["detail"]
    assert len(attempts) == 1
    state = call("GET", "/api/status")[1]
    assert state["model_state"] == "load_failed" and state["model_available"] is False
    assert str(tmp_path) not in json.dumps(state)
    # Replacing the file allows a new attempt.
    service.path.write_bytes(b"a different, longer checkpoint file")
    predict("x.png", _image())
    assert len(attempts) == 2


@pytest.mark.parametrize("model,fragment", [
    (FakeModel([PROBS], activation="linear"), "not a softmax"),
    (FakeModel([PROBS], n_out=3), "does not match 4 class names"),
    (FakeModel([PROBS], input_shape=(None, 224, 224, 1)), "Unexpected model input shape"),
])
def test_incompatible_models_are_refused_at_load(tmp_path, use_service, model, fragment):
    use_service(_service(tmp_path, model))
    status, body = predict("x.png", _image())
    assert status == 503 and fragment in body["detail"]
    assert model.batches == []


@pytest.mark.parametrize("output", [
    [[np.nan, 0.5, 0.25, 0.25]],
    [[0.5, 0.5, 0.5, 0.5]],           # does not sum to 1
    [[2.0, -1.0, 0.0, 0.0]],          # sums to 1 but outside [0, 1]
    [[0.25, 0.25, 0.25, 0.25]] * 2,   # wrong batch size
])
def test_outputs_that_are_not_one_probability_per_class_are_500(tmp_path, use_service, output):
    use_service(_service(tmp_path, FakeModel(output)))
    status, body = predict("x.png", _image())
    assert status == 500 and "invalid output" in body["detail"]


def test_concurrent_first_requests_load_the_model_once(tmp_path):
    calls = []
    gate = threading.Event()

    def slow_loader(_path):
        calls.append(1)
        gate.wait(1)
        return FakeModel([PROBS])

    service = _service(tmp_path, loader=slow_loader)
    threads = [threading.Thread(target=service.ensure_loaded) for _ in range(8)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join()
    assert len(calls) == 1 and service.state == "ready"


def test_invalid_class_names_file_fails_the_load(tmp_path, use_service):
    service = _service(tmp_path, FakeModel([PROBS]))
    service.class_names_path.write_text(json.dumps(["glioma", "glioma", "notumor", "pituitary"]), encoding="utf-8")
    use_service(service)
    status, body = predict("x.png", _image())
    assert status == 503 and "class_names.json is invalid" in body["detail"]


# --- health vs status ---------------------------------------------------------------

def test_health_is_liveness_only_and_status_reports_readiness(tmp_path, use_service):
    use_service(_service(tmp_path, FakeModel([PROBS])))
    assert call("GET", "/api/health") == (200, {"status": "ok", "service": "NeuroVision API"})
    before = call("GET", "/api/status")[1]
    assert before["model_state"] == "not_loaded" and before["model_ready"] is False
    assert before["model_available"] is True and before["model_sha256_of"] == "file_on_disk"
    assert before["classes"] == CLASSES
    assert set(before["runtime"]) == {"tensorflow", "keras"}
    predict("x.png", _image())
    after = call("GET", "/api/status")[1]
    assert after["model_state"] == "ready" and after["model_ready"] is True
    assert after["model_sha256_of"] == "loaded_weights" and after["model_input_shape"] == [None, 224, 224, 3]
    assert after["model_file_changed_since_load"] is False


def test_status_never_contains_absolute_paths(tmp_path, use_service):
    use_service(_service(tmp_path, FakeModel([PROBS])))
    assert str(tmp_path) not in json.dumps(call("GET", "/api/status")[1])


# --- integration with the real checkpoint -------------------------------------------

def _first_test_image() -> Path | None:
    testing = api.ROOT / "data_leakage_controlled_distance0" / "Testing"
    if not testing.is_dir():
        return None
    for class_dir in sorted(p for p in testing.iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if files:
            return files[0]
    return None


def test_api_matches_a_direct_prediction_with_the_real_checkpoint():
    import tensorflow as tf
    from src.brain_tumor_dl.data import _image_dataset

    image = _first_test_image()
    if not api.MODEL_PATH.is_file() or image is None:
        pytest.skip("Real checkpoint or leakage-controlled test images not present.")

    try:
        model = tf.keras.models.load_model(api.MODEL_PATH, compile=False)
    except (TypeError, ValueError) as exc:
        import keras

        reason = str(exc).rsplit("Exception encountered:", 1)[-1].strip().splitlines()[0][:160]
        pytest.skip(f"Keras {keras.__version__} in this environment cannot load {api.MODEL_PATH.name} "
                    f"({reason}); the API cannot serve it here either. Run this check in the environment "
                    "that serves the API.")
    size = int(model.input_shape[1])
    batch, _ = next(iter(_image_dataset([image], [0], (size, size), 1, False, 0)))
    expected = model.predict(batch, verbose=0)[0]
    class_names = json.loads(api.CLASSES_PATH.read_text(encoding="utf-8"))

    status, body = predict(image.name, image.read_bytes())
    assert status == 200, body
    assert body["preprocessing"]["decoder"] == "tensorflow"
    got = np.array([body["scores"][name] for name in class_names])
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert body["top_class"] == class_names[int(np.argmax(expected))]
    assert body["model"]["sha256"] == file_sha256(api.MODEL_PATH)
    if api.MODEL_RESOLUTION.variant == "evaluated":
        record = json.loads((api.MODEL_PATH.parent / "experiment_record.json").read_text(encoding="utf-8"))
        assert body["model"]["sha256"] == record["checkpoint_sha256"]
        assert call("GET", "/api/status")[1]["model_matches_experiment_record"] is True
