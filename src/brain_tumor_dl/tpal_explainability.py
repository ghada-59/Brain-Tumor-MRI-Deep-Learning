"""Visual explanations for a trained T-PAL classifier.

Heatmaps describe model activations/associations; boxes are threshold-based
visual aids, not validated tumor detections or segmentation masks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from src.brain_tumor_dl.tpal import TPALClassifier


def normalize_map(values: torch.Tensor | np.ndarray) -> np.ndarray:
    array = values.detach().float().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def input_saliency(model: TPALClassifier, image_tensor: torch.Tensor, class_index: int) -> np.ndarray:
    """Return a gradient-based input saliency map for the selected logit."""
    model.eval()
    inputs = image_tensor.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Expected classifier logits.")
    if not 0 <= class_index < logits.shape[1]:
        raise ValueError("class_index is outside the model output range.")
    logits[0, class_index].backward()
    saliency = inputs.grad.detach().abs().mean(dim=1)[0]
    return normalize_map(saliency)


@torch.inference_mode()
def attention_rollout(model: TPALClassifier, image_tensor: torch.Tensor) -> np.ndarray:
    """Aggregate the model's spatial association matrices into a token map."""
    model.eval()
    output = model(image_tensor, return_aux=True)
    if not isinstance(output, tuple):
        raise RuntimeError("Expected auxiliary attention outputs.")
    _, auxiliary = output
    attentions = auxiliary["combined_attention"]
    token_count = attentions[0].shape[-1]
    rollout = torch.eye(token_count, device=image_tensor.device)
    for attention in attentions:
        matrix = attention[0].mean(dim=0)
        matrix = matrix + torch.eye(token_count, device=image_tensor.device)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rollout = matrix @ rollout
    relevance = rollout.mean(dim=0).reshape(model.grid_size, model.grid_size)
    upsampled = F.interpolate(
        relevance[None, None],
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return normalize_map(upsampled)


def activation_box(heatmap: np.ndarray, threshold_fraction: float = 0.65):
    """Return a box around the thresholded heatmap, or None for a flat map."""
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in (0, 1].")
    heatmap = normalize_map(heatmap)
    if float(heatmap.max()) < 1e-8:
        return None
    mask = heatmap >= threshold_fraction * float(heatmap.max())
    rows, columns = np.where(mask)
    if len(rows) == 0:
        return None
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def save_explanation(
    image_path: Path,
    model: TPALClassifier,
    checkpoint: dict,
    output_path: Path,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """Save original, blue heatmap, and red-box overlay for one image."""
    image_size = int(checkpoint["image_size"])
    with Image.open(image_path) as source:
        original = source.convert("RGB")
        gray = source.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
    normalized = (pixels - float(checkpoint["mean"])) / max(float(checkpoint["std"]), 1e-6)
    tensor = torch.from_numpy(normalized.copy()).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    class_index = int(probabilities.argmax())
    saliency = input_saliency(model, tensor, class_index)
    attention = attention_rollout(model, tensor)
    heatmap = normalize_map(0.5 * saliency + 0.5 * attention)
    box = activation_box(heatmap)

    original_resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    heatmap_rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    intensity = (heatmap * 255).astype(np.uint8)
    heatmap_rgb[..., 2] = intensity
    heatmap_rgb[..., 1] = (intensity * 0.35).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
    overlay = Image.blend(original_resized, heatmap_image, alpha=0.45)
    if box is not None:
        ImageDraw.Draw(overlay).rectangle(box, outline=(255, 0, 0), width=max(2, image_size // 64))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_resized, cmap="gray")
    axes[0].set_title("Original MRI")
    axes[1].imshow(heatmap_image)
    axes[1].set_title("Blue saliency + attention map")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay and approximate activation box")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"Predicted class: {class_names[class_index]} "
        f"(model score {probabilities[class_index]:.3f})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": str(image_path),
        "predicted_class": class_names[class_index],
        "model_score": float(probabilities[class_index]),
        "class_scores": {name: float(probabilities[i]) for i, name in enumerate(class_names)},
        "activation_box_xyxy": list(box) if box is not None else None,
        "explanation_note": (
            "Blue map combines input-gradient saliency and attention rollout. "
            "The red box is a threshold-based visualization aid, not a validated tumor location."
        ),
        "output_image": str(output_path),
    }
