"""Run the first H&M customer feature-table baseline."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from multimodal_churn.data.features import (
    concat_feature_table,
    mean_purchased_article_vectors,
    random_projection,
)
from multimodal_churn.data.hm import (
    hm_image_members,
    read_hm_articles,
    read_hm_transactions,
)
from multimodal_churn.data.image import encode_article_images
from multimodal_churn.data.text import encode_article_texts
from multimodal_churn.data.transaction import transaction_sequences
from multimodal_churn.models.encoders.transaction_encoder import (
    TransactionTransformerEncoder,
)
from multimodal_churn.utils.seed import set_seed


def build_hm_feature_table(
    config: dict,
    archive_path: str | Path,
    observation_end: str,
    output_dir: str | Path,
    observation_start: str | None = None,
    max_customers: int | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Export one row/customer: 128 transaction + 128 text + 128 image features.

    This feature extraction milestone has no churn labels or classifier.
    The transaction Transformer and Gaussian projections are untrained;
    their outputs validate the pipeline but are not predictive evidence.
    """
    if config["dataset"]["name"] != "hm":
        raise ValueError("This source adapter supports only the H&M dataset.")
    if config["transaction_encoder"]["type"] != "transformer":
        raise ValueError("This baseline requires transaction_encoder.type=transformer.")
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"H&M archive not found: {archive_path}")
    output_dir = Path(output_dir)
    common_dim = int(config["model"]["common_embedding_dim"])
    encoder_config = config["transaction_encoder"]
    for branch in ("transaction_encoder", "text_encoder", "image_encoder"):
        if int(config[branch]["output_dim"]) != common_dim:
            raise ValueError(f"{branch}.output_dim must equal common_embedding_dim.")
    feature_config = config["features"]
    customer_limit = (
        int(feature_config["max_customers"]) if max_customers is None else max_customers
    )
    sequence_length = int(encoder_config["max_sequence_length"])
    if int(encoder_config["input_dim"]) != 3:
        raise ValueError("The H&M transaction sequence has three input features.")
    seed = int(config["training"]["seed"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    with ZipFile(archive_path) as archive:
        transactions = read_hm_transactions(
            archive,
            observation_end=observation_end,
            observation_start=observation_start,
            max_customers=customer_limit,
            max_sequence_length=sequence_length,
        )
        article_ids = set(transactions["article_id"].dropna().astype(str))
        articles = read_hm_articles(archive, article_ids)
        image_members = hm_image_members(archive, article_ids)
        text_article_vectors = encode_article_texts(
            articles,
            model_name=config["text_encoder"]["model_name"],
            batch_size=int(feature_config["text_batch_size"]),
            device=device,
        )
        image_article_vectors = encode_article_images(
            archive,
            image_members=image_members,
            model_name=config["image_encoder"]["model_name"],
            batch_size=int(feature_config["image_batch_size"]),
            device=device,
        )

    customer_ids, transaction_values, valid_mask = transaction_sequences(
        transactions,
        observation_end=observation_end,
        max_sequence_length=sequence_length,
    )
    encoder = TransactionTransformerEncoder(
        input_dim=transaction_values.shape[-1],
        model_dim=int(encoder_config["model_dim"]),
        output_dim=common_dim,
        num_heads=int(encoder_config["num_heads"]),
        num_layers=int(encoder_config["num_layers"]),
        max_sequence_length=sequence_length,
    ).to(device).eval()
    transaction_batches = []
    batch_size = int(feature_config["transaction_batch_size"])
    if batch_size <= 0:
        raise ValueError("transaction_batch_size must be positive.")
    with torch.inference_mode():
        for offset in range(0, len(customer_ids), batch_size):
            values = torch.from_numpy(transaction_values[offset : offset + batch_size]).to(device)
            mask = torch.from_numpy(valid_mask[offset : offset + batch_size]).to(device)
            transaction_batches.append(encoder(values, mask).cpu().numpy())
    transaction_vectors = np.concatenate(transaction_batches).astype(np.float32)

    text_native, text_available = mean_purchased_article_vectors(
        transactions, customer_ids, text_article_vectors
    )
    image_native, image_available = mean_purchased_article_vectors(
        transactions, customer_ids, image_article_vectors
    )
    text_vectors, text_matrix = random_projection(text_native, common_dim, seed + 1)
    image_vectors, image_matrix = random_projection(image_native, common_dim, seed + 2)
    table = concat_feature_table(
        customer_ids, transaction_vectors, text_vectors, image_vectors
    )
    expected_columns = 1 + 3 * common_dim
    if table.shape[1] != expected_columns or not np.isfinite(table.iloc[:, 1:].to_numpy()).all():
        raise ValueError("Feature table has an unexpected shape or non-finite values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "features.csv", index=False)
    np.savez_compressed(
        output_dir / "projection_matrices.npz",
        text=text_matrix,
        image=image_matrix,
    )
    torch.save(encoder.cpu().state_dict(), output_dir / "transaction_encoder.pt")
    manifest = {
        "dataset": "hm",
        "archive_path": str(archive_path),
        "observation_start": observation_start,
        "observation_end": observation_end,
        "sample_customers": customer_limit,
        "selection_note": "First encountered customers; smoke sample, not a research cohort.",
        "max_sequence_length": sequence_length,
        "customer_rows": len(customer_ids),
        "observed_transaction_rows": len(transactions),
        "text_articles_encoded": len(text_article_vectors),
        "image_articles_encoded": len(image_article_vectors),
        "text_customers_available": int(text_available.sum()),
        "image_customers_available": int(image_available.sum()),
        "feature_order": ["transaction", "text", "image"],
        "feature_dim_per_modality": common_dim,
        "has_churn_labels": False,
        "transaction_encoder": "untrained Transformer, seeded random initialization",
        "text_encoder": config["text_encoder"]["model_name"],
        "image_encoder": config["image_encoder"]["model_name"],
        "projection": "untrained seeded Gaussian matrices",
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return table
