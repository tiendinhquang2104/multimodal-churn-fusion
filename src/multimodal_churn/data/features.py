"""Customer-level pooling and feature-table assembly."""

import numpy as np
import pandas as pd


def mean_purchased_article_vectors(
    transactions: pd.DataFrame,
    customer_ids: list[str],
    article_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean available article vectors over each customer's observed purchases.

    Repeated purchases contribute repeatedly. Missing products are ignored;
    customers with no available products receive a zero vector.
    """
    dimension = len(next(iter(article_vectors.values()))) if article_vectors else 0
    sums = np.zeros((len(customer_ids), dimension), dtype=np.float32)
    counts = np.zeros(len(customer_ids), dtype=np.int32)
    customer_rows = {customer_id: row for row, customer_id in enumerate(customer_ids)}
    for customer_id, article_id in transactions[["customer_id", "article_id"]].itertuples(
        index=False, name=None
    ):
        vector = article_vectors.get(str(article_id))
        if vector is not None:
            row = customer_rows[str(customer_id)]
            sums[row] += vector
            counts[row] += 1
    available = counts > 0
    if available.any():
        sums[available] /= counts[available, None]
    return sums, available


def random_projection(
    vectors: np.ndarray, output_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a seeded Gaussian projection to frozen pretrained vectors.

    This is an untrained dimensionality reduction for the feature-table
    milestone. Save the returned matrix to reproduce the exact features.
    """
    if output_dim <= 0 or vectors.ndim != 2:
        raise ValueError("Expected a 2D vector matrix and positive output_dim.")
    if vectors.shape[1] == 0:
        return (
            np.zeros((vectors.shape[0], output_dim), dtype=np.float32),
            np.empty((0, output_dim), dtype=np.float32),
        )
    rng = np.random.default_rng(seed)
    matrix = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(output_dim),
        size=(vectors.shape[1], output_dim),
    ).astype(np.float32)
    return vectors.astype(np.float32) @ matrix, matrix


def concat_feature_table(
    customer_ids: list[str],
    transaction_vectors: np.ndarray,
    text_vectors: np.ndarray,
    image_vectors: np.ndarray,
) -> pd.DataFrame:
    """Create one row per customer with transaction, text, then image columns."""
    blocks = {
        "transaction": transaction_vectors,
        "text": text_vectors,
        "image": image_vectors,
    }
    n_customers = len(customer_ids)
    if len(set(customer_ids)) != n_customers:
        raise ValueError("customer_ids must be unique.")
    if any(block.ndim != 2 or block.shape[0] != n_customers for block in blocks.values()):
        raise ValueError("All modality blocks must be 2D and customer-aligned.")
    columns = {
        f"{name}_{index:03d}": block[:, index]
        for name, block in blocks.items()
        for index in range(block.shape[1])
    }
    return pd.DataFrame({"customer_id": customer_ids, **columns})
