"""Frozen MiniLM product-text embedding extraction."""

import numpy as np
import pandas as pd


def encode_article_texts(
    articles: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode each selected H&M product's name and description once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rows = articles.fillna("").copy()
    rows["text"] = (
        rows["prod_name"].str.strip() + ". " + rows["detail_desc"].str.strip()
    ).str.strip(" .")
    rows = rows.loc[rows["text"].ne("")]
    if rows.empty:
        return {}
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        rows["text"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {
        str(article_id): vector.astype(np.float32, copy=False)
        for article_id, vector in zip(rows["article_id"], vectors, strict=True)
    }
