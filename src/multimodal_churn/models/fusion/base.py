"""Common interface for modality fusion."""

from abc import abstractmethod

import torch
from torch import nn


class BaseFusion(nn.Module):
    """Fuse ordered modality embeddings into one [batch, features] tensor.

    Embeddings are ordered as transaction, text, image. Inactive modalities
    may be passed as None; concrete fusion modules define how to handle them.
    """

    @abstractmethod
    def forward(
        self,
        transaction_embedding: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        image_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a fused embedding for the provided modalities."""
