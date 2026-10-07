"""Feature-level concatenation baseline."""

import torch

from multimodal_churn.models.fusion.base import BaseFusion


class ConcatFusion(BaseFusion):
    """Concatenate available modality embeddings in transaction/text/image order."""

    def forward(
        self,
        transaction_embedding: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        image_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse one to three aligned [batch, features] tensors."""
        embeddings = [
            embedding
            for embedding in (transaction_embedding, text_embedding, image_embedding)
            if embedding is not None
        ]
        if not embeddings:
            raise ValueError("ConcatFusion needs at least one modality embedding.")
        if any(embedding.ndim != 2 for embedding in embeddings):
            raise ValueError("Each embedding must have shape [batch, features].")
        if len({embedding.shape[0] for embedding in embeddings}) != 1:
            raise ValueError("Modality batch sizes must match.")
        return torch.cat(embeddings, dim=-1)
