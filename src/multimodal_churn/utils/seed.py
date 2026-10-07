"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and all available CUDA devices.

    Some GPU operations can still be nondeterministic unless deterministic
    algorithms and compatible CUDA settings are configured separately.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
