# MRI explainability and model comparison

## Why add visual explanations?

The main model in this repository performs image-level classification into four dataset classes: glioma, meningioma, no tumor, and pituitary. Classification predicts a label for the whole image; it does not provide a verified tumor location.

Grad-CAM provides a visual explanation of the trained MobileNetV2 classifier by highlighting regions that contribute to a selected class score. The Streamlit application displays the original image, a colored activation map, and an overlay. A red rectangle is drawn around a high-activation region as a **rough visualization aid**.

The red rectangle is not ground truth, a tumor segmentation, or a validated object-detection prediction. A high-activation region may reflect anatomy, artifacts, image borders, or other patterns learned by the classifier. It must not be interpreted as a confirmed tumor boundary.

## Purpose of comparing MobileNetV2 and T-PAL

The supplied T-PAL notebook describes a second, experimental approach: a compact convolutional stem followed by transformer-style pixel-association blocks. Its goal is still four-class MRI classification, while also examining learned associations between image patches.

The comparison is useful for two different questions:

- **Classification:** which approach performs better on the same held-out test set, using the same class definitions and data split?
- **Explainability:** what patterns do Grad-CAM and T-PAL's attention/association maps highlight, and are those patterns stable and relevant?

These are not interchangeable visualizations. Grad-CAM summarizes gradients with respect to a class score. T-PAL attention/association maps show relationships between spatial tokens inside that model. Neither is automatically a tumor segmentation mask.

## Fair experimental protocol

1. Use the same deduplicated data and fixed train/validation/test split for both approaches.
2. Keep the test set untouched during model selection.
3. Record the exact preprocessing, seed, training configuration, and model checkpoint for each approach.
4. Report per-class precision, recall and F1, macro-F1, confusion matrix, and one-vs-rest ROC-AUC when appropriate.
5. Generate explanation examples from the same held-out images, including correct and incorrect predictions.
6. Do not publish a claimed T-PAL accuracy until the implementation has been run and its outputs reproduced locally.

The supplied PDF includes a T-PAL implementation and reported experimental outputs, but these numbers should not be treated as results from this repository. The repository's current deployed classifier remains MobileNetV2 until the T-PAL experiment is independently implemented, run, and checked.

## Current status

- MobileNetV2: existing classification model.
- Grad-CAM: implemented as an optional Streamlit visualization.
- Red activation box: approximate threshold-based box derived from the Grad-CAM map; not a detector.
- T-PAL: comparison target requiring its own reproducible implementation and training run.
- Clinical use: not intended; neither model nor visual explanation is clinically validated.
