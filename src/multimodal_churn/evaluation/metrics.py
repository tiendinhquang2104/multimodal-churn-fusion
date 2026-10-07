"""Binary churn metrics computed from probabilities."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC/AP, F1, precision, and recall.

    Undefined ranking metrics are returned as NaN if only one target class is
    present. Precision, recall, and F1 use zero_division=0.
    """
    y_true = np.asarray(targets).reshape(-1)
    y_score = np.asarray(probabilities).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_score.shape:
        raise ValueError("Targets and probabilities must be nonempty and aligned.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary 0/1 values.")
    if not np.isfinite(y_score).all() or ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    predictions = (y_score >= threshold).astype(int)
    has_both_classes = np.unique(y_true).size == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score))
        if has_both_classes
        else float("nan"),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }
