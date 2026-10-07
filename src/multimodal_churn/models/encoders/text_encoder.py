"""Trainable projection for precomputed product-text vectors."""

import torch
from torch import nn


class TextEncoder(nn.Module):
    """Project precomputed MiniLM vectors in a later trainable model.

    Frozen MiniLM extraction for the feature table lives in data/text.py.
    """

    def __init__(self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        self.projection = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, text_input: torch.Tensor) -> torch.Tensor:
        """Project a [batch, input_dim] precomputed text embedding tensor."""
        return self.projection(text_input)
