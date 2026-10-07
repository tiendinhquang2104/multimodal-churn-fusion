"""Trainable projection for precomputed product-image vectors."""

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Project precomputed CLIP vectors in a later trainable model.

    Frozen CLIP extraction for the feature table lives in data/image.py.
    """

    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        self.projection = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, image_input: torch.Tensor) -> torch.Tensor:
        """Project a [batch, input_dim] precomputed image embedding tensor."""
        return self.projection(image_input)
