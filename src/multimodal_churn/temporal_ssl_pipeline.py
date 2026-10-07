"""Build reusable 384D customer features without churn labels."""

import hashlib
import json
import logging
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from multimodal_churn.data.hm import hm_image_members, read_hm_articles
from multimodal_churn.data.image import encode_article_images
from multimodal_churn.data.temporal_store import (
    MODALITIES, TemporalHistoryStore, build_history_arrays, observed_article_ids,
    stage_observed_transactions,
)
from multimodal_churn.data.text import encode_article_texts
from multimodal_churn.models.temporal_ssl import TemporalModalityBranch
from multimodal_churn.training.temporal_ssl import (
    customer_split, train_branches, transaction_normalization,
)
from multimodal_churn.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _product_vectors(archive: ZipFile, article_ids: set[str], modality: str,
                     model_name: str, revision: str, batch_size: int, device: str,
                     cache_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Cache every attempted article, including those with missing text/images."""
    from huggingface_hub import HfApi, snapshot_download

    resolved_revision = HfApi().model_info(model_name, revision=revision).sha
    preprocessing = ("name+description; normalized MiniLM sentence vector" if modality == "text"
                     else "RGB AutoProcessor; normalized CLIP image vector")
    dimension = 384 if modality == "text" else 512
    identity = {"model_name": model_name, "revision": resolved_revision,
                "preprocessing": preprocessing, "dimension": dimension}
    key = _hash(identity)[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{modality}_{key}.npz"
    attempted: set[str] = set()
    vectors: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            attempted = set(cached["attempted"].tolist())
            ids = cached["article_ids"].tolist()
            matrix = cached["vectors"]
            if matrix.ndim != 2 or matrix.shape[1] != dimension:
                raise ValueError(f"Invalid {modality} product cache dimension")
            vectors = dict(zip(ids, matrix, strict=True))
    missing = article_ids - attempted
    if missing:
        snapshot = snapshot_download(repo_id=model_name, revision=resolved_revision)
        if modality == "text":
            articles = read_hm_articles(archive, missing)
            new = encode_article_texts(articles, snapshot, batch_size, device)
        else:
            members = hm_image_members(archive, missing)
            new = encode_article_images(archive, members, snapshot, batch_size, device)
        if any(vector.shape != (dimension,) for vector in new.values()):
            raise ValueError(f"Unexpected {modality} embedding dimension")
        vectors.update(new)
        attempted.update(missing)
        ordered = sorted(vectors)
        matrix = (np.stack([vectors[key] for key in ordered]).astype("float32")
                  if ordered else np.empty((0, dimension), dtype="float32"))
        temporary = cache_path.with_name(cache_path.name + ".partial")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, attempted=np.array(sorted(attempted), dtype=str),
                                    article_ids=np.array(ordered, dtype=str), vectors=matrix)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return {key: value for key, value in vectors.items() if key in article_ids}, {
        **identity, "cache": str(cache_path), "available_articles": len(vectors),
        "requested_articles": len(article_ids),
    }


def _export(store: TemporalHistoryStore, checkpoint_dir: Path, output_dir: Path,
            settings: dict, normalization: dict[str, float], device: str,
            csv_compat: bool) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    branches = {}
    for modality, depth in (("transaction", 3), ("text", 2), ("image", 2)):
        checkpoint = torch.load(checkpoint_dir / f"best_{modality}_ssl.pt",
                                map_location=device, weights_only=False)
        branch = TemporalModalityBranch(
            modality, depth, int(settings["max_history"]),
            normalization if modality == "transaction" else None,
        ).to(device)
        branch.load_state_dict(checkpoint["branch"])
        branches[modality] = branch.eval()
    feature_names = [f"{modality}_{index:03d}" for modality in MODALITIES
                     for index in range(128)]
    feature_schema = pa.schema([pa.field("customer_id", pa.string())]
                               + [pa.field(name, pa.float32()) for name in feature_names])
    sidecar_names = [f"{modality}_{suffix}" for modality in MODALITIES
                     for suffix in ("available", "original_length", "used_length")]
    sidecar_schema = pa.schema([pa.field("customer_id", pa.string())] + [
        pa.field(name, pa.bool_() if name.endswith("available") else pa.int32())
        for name in sidecar_names
    ])
    feature_path = output_dir / "features_temporal_ssl.parquet"
    sidecar_path = output_dir / "features_temporal_ssl_metadata.parquet"
    csv_path = output_dir / "features_temporal_ssl.csv"
    if csv_compat and csv_path.exists():
        csv_path.unlink()
    count = 0
    with pq.ParquetWriter(feature_path, feature_schema, compression="zstd") as feature_writer, \
         pq.ParquetWriter(sidecar_path, sidecar_schema, compression="zstd") as sidecar_writer, \
         torch.inference_mode():
        for start in range(0, len(store), int(settings["export_batch_size"])):
            rows = np.arange(start, min(start + int(settings["export_batch_size"]), len(store)))
            features = {"customer_id": store.customer_ids[start:start + len(rows)]}
            sidecar = {"customer_id": features["customer_id"]}
            for modality in MODALITIES:
                values, mask = store.batch(modality, rows)
                available = mask.any(axis=1)
                h = np.zeros((len(rows), 128), dtype="float32")
                if available.any():
                    selected = np.flatnonzero(available)
                    encoded = branches[modality](
                        torch.from_numpy(values[selected]).to(device),
                        torch.from_numpy(mask[selected]).to(device),
                    ).cpu().numpy()
                    h[selected] = encoded.astype("float32")
                if not np.isfinite(h).all():
                    raise ValueError(f"Non-finite {modality} features")
                for index in range(128):
                    features[f"{modality}_{index:03d}"] = h[:, index]
                sidecar[f"{modality}_available"] = available
                sidecar[f"{modality}_original_length"] = np.asarray(
                    store.arrays[f"{modality}_original"][rows], dtype="int32")
                sidecar[f"{modality}_used_length"] = np.asarray(
                    store.arrays[f"{modality}_used"][rows], dtype="int32")
            feature_table = pa.Table.from_pydict(features, schema=feature_schema)
            sidecar_table = pa.Table.from_pydict(sidecar, schema=sidecar_schema)
            feature_writer.write_table(feature_table)
            sidecar_writer.write_table(sidecar_table)
            if csv_compat:
                import pandas as pd
                feature_table.to_pandas().to_csv(csv_path, mode="a", index=False,
                                                 header=count == 0)
            count += len(rows)
            if count % 100_000 < len(rows):
                LOGGER.info("Exported %d/%d customer feature rows", count, len(store))
    return count


def build_temporal_ssl_feature_table(
    config: dict, archive_path: str | Path, observation_end: str,
    output_dir: str | Path, observation_start: str | None = None,
    max_customers: int | None = None, device: str | None = None,
    cache_dir: str | Path | None = None, csv_compat: bool = False,
    product_vectors: dict[str, dict[str, np.ndarray]] | None = None,
    resume_dir: str | Path | None = None,
) -> Path:
    """Pretrain three branches and export raw h, with optional injected test vectors."""
    if config["dataset"]["name"] != "hm":
        raise ValueError("Temporal SSL currently supports H&M only")
    settings = dict(config["temporal_ssl"])
    settings["seed"] = int(config["training"]["seed"])
    if int(settings["max_history"]) != 32:
        raise ValueError("V1 requires max_history=32")
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate = output_dir / "intermediate"
    intermediate.mkdir(exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(settings["seed"])
    db_path = intermediate / "observed.sqlite"
    with ZipFile(archive_path) as archive:
        staging = stage_observed_transactions(archive, db_path, observation_end,
                                               observation_start)
        article_ids = observed_article_ids(db_path, max_customers)
        if product_vectors is None:
            cache_path = Path(cache_dir) if cache_dir is not None else output_dir / "product_cache"
            text_vectors, text_meta = _product_vectors(
                archive, article_ids, "text", config["text_encoder"]["model_name"],
                settings["minilm_revision"], int(config["features"]["text_batch_size"]),
                device, cache_path)
            image_vectors, image_meta = _product_vectors(
                archive, article_ids, "image", config["image_encoder"]["model_name"],
                settings["clip_revision"], int(config["features"]["image_batch_size"]),
                device, cache_path)
        else:
            text_vectors = product_vectors["text"]
            image_vectors = product_vectors["image"]
            text_meta = {"model_name": "injected", "revision": "test",
                         "preprocessing": "injected", "dimension": 384}
            image_meta = {"model_name": "injected", "revision": "test",
                          "preprocessing": "injected", "dimension": 512}
    customer_count = build_history_arrays(db_path, intermediate, text_vectors,
                                          image_vectors, 32, max_customers)
    db_path.unlink()
    store = TemporalHistoryStore(intermediate)
    try:
        train_rows, _, _ = customer_split(
            store.customer_ids, settings["seed"], int(settings["validation_percent"]))
        normalization = transaction_normalization(store, train_rows)
        checkpoint_dir = output_dir / "checkpoints"
        run_identity = {
            "archive_name": archive_path.name,
            "archive_size": archive_path.stat().st_size,
            "observation_start": observation_start, "observation_end": observation_end,
            "max_customers": max_customers, "staging": staging,
            "text_revision": text_meta["revision"],
            "image_revision": image_meta["revision"],
        }
        summaries, split_meta = train_branches(store, checkpoint_dir, settings,
                                               normalization, device,
                                               Path(resume_dir) if resume_dir is not None else None,
                                               run_identity)
        exported = _export(store, checkpoint_dir, output_dir, settings,
                           normalization, device, csv_compat)
        if exported != customer_count:
            raise ValueError("Exported row count differs from staged customers")
    finally:
        store.close()
    architecture = {
        "name": "three modality-specific Temporal Transformers",
        "shared_implementation": True, "shared_parameters": False,
        "d_model": 128, "n_heads": 4, "ffn_dimension": 256,
        "transaction_layers": 3, "text_layers": 2, "image_layers": 2,
        "pooling": "masked_attention", "ssl_projector": [128, 128, 64],
    }
    manifest = {
        "run_id": _hash({"settings": settings, **run_identity})[:16],
        "mode": "temporal_ssl", "pretraining_scope": "all_customers_debug",
        "has_churn_labels": False, "observation_start": observation_start,
        "observation_end": observation_end, "seed": settings["seed"],
        "max_history_length": 32, "feature_dimension": 384,
        "feature_dtype": "float32", "customer_rows": exported,
        "staging": staging, "settings": settings, "architecture": architecture,
        "source_archive": str(archive_path),
        "source_archive_size": archive_path.stat().st_size,
        "encoder_config_hash": _hash({"architecture": architecture, "settings": settings,
                                      "text_revision": text_meta["revision"],
                                      "image_revision": image_meta["revision"]}),
        "augmentation_policy": {
            "long_history": "distinct ordered event drops; latest in at least one view",
            "short_history": "distinct latent feature masks; no event removal",
        },
        "normalization": normalization,
        "backbones": {"text": text_meta, "image": image_meta},
        "split": split_meta, "checkpoints": summaries,
        "checkpoint_backup_dir": str(Path(resume_dir) / "checkpoints")
        if resume_dir is not None else None,
        "feature_file": "features_temporal_ssl.parquet",
        "metadata_file": "features_temporal_ssl_metadata.parquet",
        "intermediate_dir": "intermediate",
        "note": "Unlabeled customer representations; not evidence of churn prediction quality.",
    }
    (output_dir / "manifest_temporal_ssl.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir / "features_temporal_ssl.parquet"
