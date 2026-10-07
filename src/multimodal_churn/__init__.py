"""Multimodal customer churn research package.

Data utilities can be imported without loading PyTorch model modules.
"""

__all__ = ["MultimodalChurnModel"]


def __getattr__(name: str):
    """Load the composed model only when explicitly requested."""
    if name == "MultimodalChurnModel":
        from multimodal_churn.models.multimodal_model import MultimodalChurnModel

        return MultimodalChurnModel
    raise AttributeError(name)
