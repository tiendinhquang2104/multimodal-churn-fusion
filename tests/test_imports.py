"""Import and tensor smoke checks without pretrained downloads."""

import pytest

torch = pytest.importorskip("torch")


def test_main_components_import() -> None:
    """Main package modules can be imported after an editable install."""
    from multimodal_churn import MultimodalChurnModel
    from multimodal_churn.evaluation.metrics import compute_binary_metrics
    from multimodal_churn.models.encoders import (
        ImageEncoder,
        TextEncoder,
        TransactionEncoder,
    )
    from multimodal_churn.models.fusion import BaseFusion, ConcatFusion
    from multimodal_churn.models.heads import ChurnClassifier
    from multimodal_churn.utils.config import load_config
    from multimodal_churn.utils.seed import set_seed

    assert all(
        component is not None
        for component in (
            MultimodalChurnModel,
            compute_binary_metrics,
            ImageEncoder,
            TextEncoder,
            TransactionEncoder,
            BaseFusion,
            ConcatFusion,
            ChurnClassifier,
            load_config,
            set_seed,
        )
    )


def test_mlp_ablation_smoke() -> None:
    """Legacy MLP ablation composition still agrees on tensor shapes."""
    from multimodal_churn import MultimodalChurnModel
    from multimodal_churn.models.encoders import (
        ImageEncoder,
        TextEncoder,
        TransactionEncoder,
    )
    from multimodal_churn.models.fusion import ConcatFusion
    from multimodal_churn.models.heads import ChurnClassifier

    batch_size = 4
    transaction = TransactionEncoder(input_dim=6, output_dim=128)
    text = TextEncoder(input_dim=384, output_dim=128)
    image = ImageEncoder(input_dim=512, output_dim=128)
    model = MultimodalChurnModel(
        transaction_encoder=transaction,
        text_encoder=text,
        image_encoder=image,
        fusion=ConcatFusion(),
        classifier=ChurnClassifier(input_dim=384),
    )
    logits = model(
        transaction_features=torch.randn(batch_size, 6),
        text_input=torch.randn(batch_size, 384),
        image_input=torch.randn(batch_size, 512),
    )
    assert logits.shape == (batch_size,)


def test_config_merge_and_metrics() -> None:
    """Config override and probability metrics expose the baseline contract."""
    from pathlib import Path

    from multimodal_churn.evaluation.metrics import compute_binary_metrics
    from multimodal_churn.utils.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "baseline.yaml",
        experiment_path=root / "configs" / "experiments" / "concat.yaml",
    )
    assert config["model"]["common_embedding_dim"] == 128
    assert config["experiment"]["modalities"] == ["transaction", "text", "image"]
    metrics = compute_binary_metrics([0, 1], [0.1, 0.9])
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_dataset_path_config_is_portable() -> None:
    """One runtime root and dataset YAML select the H&M archive."""
    from pathlib import Path

    from multimodal_churn.utils.config import load_config, resolve_dataset_paths

    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "baseline.yaml",
        dataset_path=root / "configs" / "datasets" / "hm.yaml",
    )
    paths = resolve_dataset_paths(config, Path("mounted_datasets"))
    assert paths["folder"] == Path("mounted_datasets") / "02_hm_fashion"
    assert paths["archive"] == paths["folder"] / "h-and-m-personalized-fashion-recommendations.zip"

    other = {"dataset": {"name": "other", "relative_path": "other_dataset"}}
    assert resolve_dataset_paths(other, "mounted_datasets") == {
        "folder": Path("mounted_datasets") / "other_dataset"
    }

    with pytest.raises(ValueError):
        resolve_dataset_paths(
            {"dataset": {"relative_path": "../outside"}}, "mounted_datasets"
        )


def test_transaction_transformer_vector_shape() -> None:
    """A padded purchase sequence produces one vector per customer."""
    from multimodal_churn.models.encoders import TransactionTransformerEncoder

    encoder = TransactionTransformerEncoder(
        input_dim=3, model_dim=16, output_dim=8,
        num_heads=4, num_layers=1, max_sequence_length=4,
    ).eval()
    values = torch.randn(2, 4, 3)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    with torch.inference_mode():
        vectors = encoder(values, valid)
    assert vectors.shape == (2, 8)
    assert torch.isfinite(vectors).all()
