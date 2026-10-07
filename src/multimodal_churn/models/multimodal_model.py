"""Composition of encoders, fusion, and churn classifier."""

import torch
from torch import nn

from multimodal_churn.models.fusion.base import BaseFusion


class MultimodalChurnModel(nn.Module):
    """Run configured modality encoders, then fusion and classification.

    Omit encoders for inactive modalities. The classifier input dimension must
    match the fused output width (128 times the number of active modalities
    for the initial concatenation baseline).
    """

    def __init__(
        self,
        fusion: BaseFusion,
        classifier: nn.Module,
        transaction_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        image_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if all(encoder is None for encoder in (transaction_encoder, text_encoder, image_encoder)):
            raise ValueError("Configure at least one modality encoder.")
        self.transaction_encoder = transaction_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion
        self.classifier = classifier

    def forward(
        self,
        transaction_features: torch.Tensor | None = None,
        text_input: torch.Tensor | None = None,
        image_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return churn logits for available and configured inputs."""
        pairs = (
            (self.transaction_encoder, transaction_features),
            (self.text_encoder, text_input),
            (self.image_encoder, image_input),
        )
        if any((encoder is None) != (value is None) for encoder, value in pairs):
            raise ValueError("Provide one input for each configured encoder and no extra inputs.")
        embeddings = [
            encoder(value) if encoder is not None and value is not None else None
            for encoder, value in pairs
        ]
        fused = self.fusion(*embeddings)
        return self.classifier(fused)
