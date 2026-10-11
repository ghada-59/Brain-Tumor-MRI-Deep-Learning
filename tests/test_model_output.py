import numpy as np
import tensorflow as tf

from src.brain_tumor_dl.model import build_model


def test_model_output_has_one_score_per_class() -> None:
    model = build_model(num_classes=4, image_size=(64, 64), weights=None)
    images = np.zeros((2, 64, 64, 3), dtype=np.float32)

    scores = model.predict(images, verbose=0)

    assert scores.shape == (2, 4)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    np.testing.assert_allclose(scores.sum(axis=1), np.ones(2), atol=1e-5)
