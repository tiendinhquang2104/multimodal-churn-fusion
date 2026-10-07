"""Placeholder for a future cross-attention fusion experiment."""

from multimodal_churn.models.fusion.base import BaseFusion


class CrossAttentionFusion(BaseFusion):
    """Reserved for attention across modality representations."""

    def forward(self, transaction_embedding=None, text_embedding=None, image_embedding=None):
        """Raise until the cross-attention design is specified."""
        raise NotImplementedError("Cross-attention fusion is a future experiment.")
