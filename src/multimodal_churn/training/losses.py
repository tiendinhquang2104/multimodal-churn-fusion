"""Loss helper for binary churn logits."""

import torch
from torch.nn import functional as F


def binary_churn_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute binary cross entropy from logits and float 0/1 targets."""
    return F.binary_cross_entropy_with_logits(logits, targets.float())
