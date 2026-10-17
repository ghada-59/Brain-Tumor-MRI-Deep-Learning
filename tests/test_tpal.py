"""Unit tests for T-PAL model construction and explanation utilities."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.brain_tumor_dl.tpal import TPALClassifier, association_regularization, spatial_association_prior
from src.brain_tumor_dl.tpal_explainability import activation_box, attention_rollout, input_saliency, normalize_map


def test_spatial_prior_is_normalized():
    prior = spatial_association_prior(grid_size=4, sigma=2.0)
    assert prior.shape == (16, 16)
    assert torch.allclose(prior.sum(dim=-1), torch.ones(16), atol=1e-6)
    assert torch.isfinite(prior).all()


def test_model_logits_and_auxiliary_shapes():
    model = TPALClassifier(num_classes=4, image_size=32, dimension=32, heads=4, depth=2)
    images = torch.randn(2, 1, 32, 32)
    logits = model(images)
    assert logits.shape == (2, 4)
    logits_aux, auxiliary = model(images, return_aux=True)
    assert logits_aux.shape == (2, 4)
    assert len(auxiliary["attention"]) == 2
    assert auxiliary["attention"][0].shape == (2, 4, 16, 16)
    assert auxiliary["combined_attention"][0].shape == (2, 4, 16, 16)
    assert torch.isfinite(logits).all()


def test_association_regularization_is_finite():
    model = TPALClassifier(num_classes=4, image_size=32, dimension=32, heads=4, depth=2)
    images = torch.randn(2, 1, 32, 32)
    _, auxiliary = model(images, return_aux=True)
    prior = spatial_association_prior(model.grid_size)
    association, diversity = association_regularization(auxiliary, prior)
    assert association.ndim == 0
    assert diversity.ndim == 0
    assert torch.isfinite(association)
    assert torch.isfinite(diversity)


def test_saliency_and_attention_maps_have_expected_shape():
    model = TPALClassifier(num_classes=4, image_size=32, dimension=32, heads=4, depth=1)
    images = torch.randn(1, 1, 32, 32)
    saliency = input_saliency(model, images, class_index=1)
    attention = attention_rollout(model, images)
    assert saliency.shape == (32, 32)
    assert attention.shape == (32, 32)
    assert np.isfinite(saliency).all()
    assert np.isfinite(attention).all()
    assert saliency.min() >= 0 and saliency.max() <= 1
    assert attention.min() >= 0 and attention.max() <= 1


def test_activation_box_and_flat_map():
    assert activation_box(np.zeros((8, 8), dtype=np.float32)) is None
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[2:5, 3:7] = 1
    assert activation_box(heatmap) == (3, 2, 6, 4)
    with pytest.raises(ValueError):
        activation_box(heatmap, threshold_fraction=0)


def test_invalid_model_dimensions_are_rejected():
    with pytest.raises(ValueError):
        TPALClassifier(image_size=30)
    with pytest.raises(ValueError):
        TPALClassifier(dimension=30, heads=4)
