"""MobileNetV2 transfer-learning model for four-class brain MRI image classification."""
from __future__ import annotations

import tensorflow as tf


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build a lightweight transfer-learning classifier.

    Images are supplied in the 0..255 range; rescaling is included in the model
    so training, CLI prediction, and Streamlit inference share preprocessing.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="training_augmentation",
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights=weights,
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3), name="mri_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_scaling")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="brain_mri_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
