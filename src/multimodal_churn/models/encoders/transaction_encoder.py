"""Transaction encoders for tabular features and purchase sequences."""

from collections.abc import Sequence

import torch
from torch import nn


class TransactionEncoder(nn.Module):
    """MLP tabular encoder retained as a simple ablation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All encoder dimensions must be positive.")
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, transaction_features: torch.Tensor) -> torch.Tensor:
        """Encode a [batch, input_dim] feature tensor."""
        return self.network(transaction_features)


class TransactionTransformerEncoder(nn.Module):
    """Encode an observed purchase sequence to one customer vector.

    Weights are randomly initialized. Train this module before treating its
    vectors as predictive churn features.
    """

    def __init__(
        self,
        input_dim: int = 3,
        model_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_sequence_length: int = 32,
    ) -> None:
        super().__init__()
        if min(input_dim, model_dim, output_dim, num_heads, num_layers, max_sequence_length) <= 0:
            raise ValueError("Transformer dimensions and layer counts must be positive.")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.max_sequence_length = max_sequence_length
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, output_dim)
        self.output_dim = output_dim

    def forward(
        self, values: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return [batch, output_dim] from [batch, sequence, input_dim].

        valid_mask is True for real transactions and False for padding.
        """
        if values.ndim != 3 or values.shape[1] > self.max_sequence_length:
            raise ValueError("Expected [batch, sequence, features] within max_sequence_length.")
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if valid_mask.shape != values.shape[:2] or not valid_mask.any(dim=1).all():
            raise ValueError("Each customer needs at least one valid transaction.")
        positions = torch.arange(values.shape[1], device=values.device)
        hidden = self.input_projection(values) + self.position_embedding(positions)
        encoded = self.encoder(hidden, src_key_padding_mask=~valid_mask)
        mask = valid_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
        return self.output_projection(pooled)
