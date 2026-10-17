
"""Grad-CAM visual explanations for the MobileNetV2 MRI classifier."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import colormaps


def compute_gradcam(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Return a normalized Grad-CAM map for one image and one output class."""
    if image_batch.ndim != 4 or image_batch.shape[0] != 1:
        raise ValueError("image_batch must have shape (1, height, width, channels).")

    backbone = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = backbone.get_layer("out_relu")

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output],
    )

    inputs = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    augmentation = model.get_layer("training_augmentation")
    scaling = model.get_layer("mobilenet_scaling")
    pooling = model.get_layer("global_average_pooling")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("class_probabilities")

    if (
        not isinstance(class_index, (int, np.integer))
        or class_index < 0
        or class_index >= int(classifier.units)
    ):
        raise ValueError("class_index is outside the model output range.")

    with tf.GradientTape() as tape:
        augmented = augmentation(inputs, training=False)
        scaled = scaling(augmented)
        conv_features, backbone_output = feature_model(
            scaled, training=False
        )

        pooled = pooling(backbone_output)
        dropped = dropout(pooled, training=False)
        scores = classifier(dropped)
        target_score = scores[:, class_index]

    gradients = tape.gradient(target_score, conv_features)
    if gradients is None:
        raise RuntimeError("Could not calculate gradients for the selected class.")

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_features, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)

    return np.asarray(heatmap.numpy(), dtype=np.float32)


def _salient_region_box(
    heatmap: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Estimate a box around high-activation pixels; this is not tumor detection."""
    if heatmap.ndim != 2 or not np.all(np.isfinite(heatmap)):
        raise ValueError("heatmap must be a finite two-dimensional array.")

    peak = float(heatmap.max(initial=0.0))
    if peak <= 0:
        return None

    threshold = max(0.35, float(np.quantile(heatmap, 0.90)))
    ys, xs = np.where(heatmap >= threshold)

    if len(xs) < 4:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_explanation_views(
    image: Image.Image,
    heatmap: np.ndarray,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int] | None]:
    """Create a colored heatmap and an overlay with an approximate red region box."""
    if heatmap.ndim != 2 or not np.all(np.isfinite(heatmap)):
        raise ValueError("heatmap must be a finite two-dimensional array.")

    original = image.convert("RGB")
    width, height = original.size

    resized_heatmap = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize((width, height), Image.Resampling.BILINEAR)

    normalized = np.asarray(resized_heatmap, dtype=np.float32) / 255.0

    colors = colormaps["jet"](normalized)[..., :3]
    heatmap_image = Image.fromarray(np.uint8(colors * 255), mode="RGB")

    base = np.asarray(original, dtype=np.float32)
    colored = np.asarray(heatmap_image, dtype=np.float32)
    overlay = Image.fromarray(
        np.uint8(np.clip(0.55 * base + 0.45 * colored, 0, 255))
    )

    draw = ImageDraw.Draw(overlay)
    box = _salient_region_box(normalized)

    if box is not None:
        draw.rectangle(
            box,
            outline=(255, 0, 0),
            width=max(2, round(min(width, height) / 150)),
        )

    return heatmap_image, overlay, box
