"""Placeholder for a future weighted fusion experiment."""

from multimodal_churn.models.fusion.base import BaseFusion


class WeightedFusion(BaseFusion):
    """Reserved for learned or configured modality weights."""

    def forward(self, transaction_embedding=None, text_embedding=None, image_embedding=None):
        """Raise until the weighted fusion design is specified."""
        raise NotImplementedError("Weighted fusion is a future experiment.")
