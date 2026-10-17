"""T-PAL-style pixel-association Transformer for grayscale brain MRI classification.

This is a standalone PyTorch experiment inspired by the supplied T-PAL notebook.
It does not replace the existing TensorFlow/Keras MobileNetV2 model.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def spatial_association_prior(grid_size: int = 16, sigma: float = 3.0) -> torch.Tensor:
    """Build a normalized Gaussian spatial prior between all grid locations."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    distances = torch.cdist(coordinates, coordinates)
    prior = torch.exp(-(distances ** 2) / (2.0 * sigma ** 2))
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class PixelAssociationBlock(nn.Module):
    """Transformer block combining learned attention with a spatial prior."""

    def __init__(
        self,
        dimension: int,
        heads: int,
        spatial_prior: torch.Tensor,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("dimension must be divisible by heads.")

        self.heads = heads
        self.head_dimension = dimension // heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output_projection = nn.Linear(dimension, dimension)
        self.gate = nn.Linear(2 * dimension, dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )
        self.norm_attention = nn.LayerNorm(dimension)
        self.norm_feed_forward = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("spatial_prior", spatial_prior.clone())

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, token_count, dimension = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size, token_count, 3, self.heads, self.head_dimension
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention = torch.softmax(
            (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dimension),
            dim=-1,
        )
        combined = attention * self.spatial_prior[None, None, :, :]
        combined = combined / combined.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        attended = (self.dropout(combined) @ value).transpose(1, 2).reshape(
            batch_size, token_count, dimension
        )
        attended = self.output_projection(attended)
        gate = torch.sigmoid(self.gate(torch.cat((tokens, attended), dim=-1)))
        updated = self.norm_attention(gate * attended + (1.0 - gate) * tokens)
        output = self.norm_feed_forward(updated + self.dropout(self.feed_forward(updated)))
        return output, attention, combined, gate


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class TPALClassifier(nn.Module):
    """Compact convolutional stem plus pixel-association Transformer blocks.

    Input shape is (batch, 1, image_size, image_size), with normalized grayscale
    pixels. The default 128-pixel input produces a 16 x 16 token grid.
    """

    def __init__(
        self,
        num_classes: int = 4,
        image_size: int = 128,
        dimension: int = 128,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        spatial_sigma: float = 3.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if image_size < 8 or image_size % 8 != 0:
            raise ValueError("image_size must be a positive multiple of 8 (at least 8).")
        if dimension < 1 or heads < 1 or depth < 1:
            raise ValueError("dimension, heads, and depth must be positive integers.")
        if dimension % heads != 0:
            raise ValueError("dimension must be divisible by heads.")

        self.image_size = image_size
        self.grid_size = image_size // 8
        self.num_classes = num_classes
        self.dimension = dimension
        self.heads = heads
        self.depth = depth

        self.stem = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, dimension),
        )
        token_count = self.grid_size * self.grid_size
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count, dimension))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        prior = spatial_association_prior(self.grid_size, spatial_sigma)
        self.blocks = nn.ModuleList(
            [
                PixelAssociationBlock(dimension, heads, prior, dropout)
                for _ in range(depth)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, num_classes),
        )

    def forward(
        self, images: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected input shape (batch, 1, height, width).")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} input images."
            )

        tokens = self.stem(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding
        raw_attention = []
        combined_attention = []
        gates = []

        for block in self.blocks:
            tokens, attention, combined, gate = block(tokens)
            raw_attention.append(attention)
            combined_attention.append(combined)
            gates.append(gate)

        features = tokens.mean(dim=1)
        logits = self.classifier(features)
        if not return_aux:
            return logits

        return logits, {
            "tokens": tokens,
            "attention": raw_attention,
            "combined_attention": combined_attention,
            "gates": gates,
            "features": features,
        }


def association_regularization(
    auxiliary: dict[str, object], spatial_prior: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return association-prior and cross-head diversity penalties."""
    attentions = auxiliary["attention"]
    combined_attentions = auxiliary["combined_attention"]
    if not isinstance(attentions, list) or not isinstance(combined_attentions, list):
        raise ValueError("Auxiliary attention data is missing.")

    prior = spatial_prior.to(device=combined_attentions[0].device, dtype=torch.float32)
    association_loss = prior.new_zeros(())
    diversity_loss = prior.new_zeros(())

    for attention, combined in zip(attentions, combined_attentions):
        association_loss = association_loss + (
            combined.float() - prior
        ).square().sum(dim=-1).mean()

        head_count = attention.shape[1]
        layer_diversity = prior.new_zeros(())
        pair_count = 0
        for first in range(head_count):
            for second in range(first + 1, head_count):
                overlap = attention[:, first].float() @ attention[:, second].float().transpose(-1, -2)
                layer_diversity = layer_diversity + overlap.square().sum(dim=(-1, -2)).mean()
                pair_count += 1
        if pair_count:
            diversity_loss = diversity_loss + layer_diversity / pair_count

    layer_count = max(1, len(attentions))
    return association_loss / layer_count, diversity_loss / layer_count
