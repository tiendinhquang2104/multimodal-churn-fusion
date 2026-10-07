"""Binary churn prediction head."""

from collections.abc import Sequence

import torch
from torch import nn


class ChurnClassifier(nn.Module):
    """Map a fused embedding to one logit per customer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All classifier dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """Return [batch] logits; apply sigmoid only for probabilities."""
        return self.network(fused_embedding).squeeze(-1)
