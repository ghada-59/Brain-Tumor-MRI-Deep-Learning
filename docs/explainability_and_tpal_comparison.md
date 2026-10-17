# Explainability and T-PAL comparison

## Task definition

Both model paths classify a complete MRI image into one of four dataset classes: glioma, meningioma, notumor, or pituitary. They do not directly perform tumour segmentation or validated object detection.

## MobileNetV2 explanations

Grad-CAM uses gradients of a selected class score to highlight spatial activations in the convolutional network. The Streamlit interface displays the input, a heatmap, and an approximate box around high-activation pixels.

## T-PAL-style architecture

The separate PyTorch implementation in \`src/brain_tumor_dl/tpal.py\` contains a convolutional stem, spatial positional embeddings, multi-head pixel association, a Gaussian spatial prior, gated residual updates, and a classification head. Optional association and diversity regularizers can be enabled during training. Auxiliary attention matrices are exposed for analysis.

The implementation is inspired by the supplied notebook; it should not be described as an exact reproduction until differences in data splitting, preprocessing, training, and evaluation have been reconciled.

## T-PAL explanation views

\`explain_tpal.py\` exports three panels: original MRI, a blue map combining input-gradient saliency with attention rollout, and an overlay with a red threshold-derived activation box. These views explain model behavior only. Neither attention nor saliency is a validated tumour mask, and the red box is not produced by a trained detector.

## Evaluation protocol

1. Prepare one cleaned dataset and document the exact train/validation/test file lists.
2. Use the same held-out test images and labels for both model families.
3. Keep test data out of model selection and hyperparameter tuning.
4. Report accuracy, balanced accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, and the confusion matrix.
5. Save configuration, seed, preprocessing statistics, and selected checkpoint metadata.
6. Inspect both correct and incorrect predictions; do not select examples only because they look persuasive.
7. Report compute time and inference latency only when measured using a documented procedure.

The T-PAL paper/notebook's reported score is not a result of this repository. The repository does not claim a T-PAL score until a local run has completed and its artifacts have been reviewed. A matching number of test images is not proof that two models used the same examples.

## Limitations

- Dataset labels and provenance may contain errors.
- Exact duplicate audits cannot prove patient-level independence.
- Domain shift across hospitals, scanners, sequences, and populations is not assessed.
- The dataset is not a substitute for clinical data or external validation.
- Neither model nor visualization is intended for diagnosis or treatment decisions.
