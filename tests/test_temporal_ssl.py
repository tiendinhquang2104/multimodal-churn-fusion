"""Architecture, augmentation, and unlabeled feature-table checks."""

import json
from copy import deepcopy
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from multimodal_churn.models.temporal_ssl import (
    SSLProjectionHead, TemporalHistoryTransformer, TemporalModalityBranch, nt_xent,
)
from multimodal_churn.training.temporal_ssl import augment_views, customer_split


@pytest.fixture
def workspace_tmp():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        yield Path(directory)


def test_temporal_mask_and_gradients() -> None:
    model = TemporalHistoryTransformer(2, max_history=4)
    values = torch.randn(2, 4, 128, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    h, attention = model(values, mask, return_attention=True)
    assert h.shape == (2, 128)
    assert model.encode_sequence(values, mask).shape == (2, 4, 128)
    torch.testing.assert_close(attention[~mask], torch.zeros_like(attention[~mask]))
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(2))
    altered = values.detach().clone()
    altered[~mask] = 10_000
    torch.testing.assert_close(h, model(altered, mask), atol=1e-5, rtol=1e-5)
    h.square().sum().backward()
    assert model.attention.weight.grad is not None
    assert model.blocks.layers[0].self_attn.in_proj_weight.grad is not None
    assert values.grad is not None
    with pytest.raises(ValueError, match="all-invalid"):
        model(values.detach(), torch.zeros_like(mask))


@pytest.mark.parametrize("modality,dimensions,depth", [
    ("transaction", 3, 3), ("text", 384, 2), ("image", 512, 2),
])
def test_modality_branch_and_ssl_head(modality: str, dimensions: int, depth: int) -> None:
    normalization = {"price_mean": 0.1, "price_std": 0.2,
                     "log_gap_mean": 1.0, "log_gap_std": 0.5}
    branch = TemporalModalityBranch(modality, depth,
                                    normalization=normalization if modality == "transaction" else None)
    values = torch.randn(3, 4, dimensions)
    if modality == "transaction":
        values[..., 0] = values[..., 0].abs()
        values[..., 1] = torch.tensor([1., 2., 1., 2.])
        values[..., 2] = values[..., 2].abs()
    mask = torch.ones(3, 4, dtype=torch.bool)
    h = branch(values, mask)
    head = SSLProjectionHead()
    z = head(h)
    assert h.shape == (3, 128) and z.shape == (3, 64)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(3))
    loss, positive, negative = nt_xent(z, z.roll(1, 0))
    assert torch.isfinite(torch.stack((loss, positive, negative))).all()
    loss.backward()
    assert branch.temporal.attention.weight.grad is not None
    assert head.network[0].weight.grad is not None
    if modality == "transaction":
        assert branch.adapter.channel.weight.grad is not None
        assert branch.adapter.price.weight.grad is not None
        assert branch.adapter.gap.weight.grad is not None
    else:
        assert branch.adapter.weight.grad is not None


def test_split_and_views_are_stable_and_distinct() -> None:
    ids = [f"c{index:03d}" for index in range(40)]
    train, val, meta = customer_split(ids, 42, 20)
    reverse_train, reverse_val, reverse_meta = customer_split(ids[::-1], 42, 20)
    assert {ids[index] for index in train} == {ids[::-1][index] for index in reverse_train}
    assert {ids[index] for index in val} == {ids[::-1][index] for index in reverse_val}
    assert meta == reverse_meta

    values = np.arange(2 * 6 * 3, dtype="float32").reshape(2, 6, 3)
    mask = np.zeros((2, 6), dtype=bool)
    mask[0, :6] = True
    mask[1, :2] = True
    first, second, metrics = augment_views(values, mask, np.random.default_rng(42))
    assert metrics["identical_view_rate"] == 0
    assert first[1].all(axis=1).tolist() == [False, False]  # neither view fills padding
    for view in (first, second):
        assert view[1].sum(axis=1).min() >= 2
        assert np.all(np.diff(view[0][0, :view[1][0].sum(), 0]) > 0)
    assert first[1][0].sum() <= 5 and second[1][0].sum() <= 5
    assert first[1][0].sum() + second[1][0].sum() >= 2
    assert (values[0, 5, 0] in first[0][0, :, 0] or
            values[0, 5, 0] in second[0][0, :, 0])
    assert not np.array_equal(first[2][1], second[2][1])


def test_synthetic_end_to_end(workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    from multimodal_churn.temporal_ssl_pipeline import build_temporal_ssl_feature_table

    tmp_path = workspace_tmp
    archive_path = tmp_path / "hm.zip"
    lines = ["t_dat,customer_id,article_id,price,sales_channel_id"]
    text_vectors = {}
    image_vectors = {}
    for customer in range(24):
        for event in range(2):
            article = f"{customer * 2 + event + 1:010d}"
            lines.append(f"2019-01-{event + 1:02d},c{customer:03d},{article},0.{customer + 10:02d},{event + 1}")
            rng = np.random.default_rng(customer * 2 + event)
            if customer != 0:
                text_vectors[article] = rng.normal(size=384).astype("float32")
            if customer != 1:
                image_vectors[article] = rng.normal(size=512).astype("float32")
    lines.append("2019-02-01,c000,9999999999,99,2")
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("hm/transactions_train.csv", "\n".join(lines) + "\n")
        archive.writestr("hm/articles.csv", "article_id,prod_name,detail_desc\n")
    settings = {
        "max_history": 32, "batch_size": 8, "export_batch_size": 8,
        "learning_rate": 1e-4, "temperature": 0.1, "max_epochs": 2,
        "validation_percent": 25, "long_history_drop_rate": 0.2,
        "short_history_mask_rate": 0.1, "minilm_revision": "main",
        "clip_revision": "main",
    }
    config = {"dataset": {"name": "hm"}, "temporal_ssl": settings,
              "training": {"seed": 42}}
    persistent = tmp_path / "persistent"
    import multimodal_churn.training.temporal_ssl as training_module

    original_epoch = training_module._epoch
    transaction_train_epochs = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal transaction_train_epochs
        if args[2] == "transaction" and args[5] is not None:
            transaction_train_epochs += 1
            if transaction_train_epochs == 2:
                raise RuntimeError("simulated Colab disconnect")
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(training_module, "_epoch", interrupt_second_epoch)
    with pytest.raises(RuntimeError, match="simulated Colab disconnect"):
        build_temporal_ssl_feature_table(
            config, archive_path, "2019-01-31", tmp_path / "interrupted_output",
            device="cpu", resume_dir=persistent,
            product_vectors={"text": text_vectors, "image": image_vectors})
    assert torch.load(persistent / "checkpoints/latest_transaction_ssl.pt",
                      map_location="cpu", weights_only=False)["epoch"] == 1
    assert (persistent / "checkpoints/best_transaction_ssl.pt").is_file()
    (persistent / "checkpoints/best_transaction_ssl.pt").unlink()
    resumed_transaction_epochs = 0

    def count_resumed_epoch(*args, **kwargs):
        nonlocal resumed_transaction_epochs
        if args[2] == "transaction" and args[5] is not None:
            resumed_transaction_epochs += 1
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(training_module, "_epoch", count_resumed_epoch)

    output = tmp_path / "output"
    path = build_temporal_ssl_feature_table(
        config, archive_path, "2019-01-31", output, device="cpu",
        resume_dir=persistent,
        product_vectors={"text": text_vectors, "image": image_vectors})
    assert resumed_transaction_epochs == 1
    assert (persistent / "checkpoints/best_transaction_ssl.pt").is_file()
    table = pq.read_table(path).to_pandas()
    sidecar = pq.read_table(output / "features_temporal_ssl_metadata.parquet").to_pandas()
    assert table.shape == (24, 385)
    assert table.columns[0] == "customer_id"
    assert np.isfinite(table.iloc[:, 1:].to_numpy()).all()
    assert not sidecar.loc[sidecar.customer_id == "c000", "text_available"].item()
    assert not sidecar.loc[sidecar.customer_id == "c001", "image_available"].item()
    assert table.loc[table.customer_id == "c000", [f"text_{i:03d}" for i in range(128)]].to_numpy().sum() == 0
    assert sidecar.loc[sidecar.customer_id == "c000", "transaction_original_length"].item() == 2
    for modality in ("transaction", "text", "image"):
        assert (output / "checkpoints" / f"best_{modality}_ssl.pt").exists()
        latest = torch.load(persistent / "checkpoints" / f"latest_{modality}_ssl.pt",
                            map_location="cpu", weights_only=False)
        assert latest["epoch"] == 2
        assert len(latest["history"]) == 2
    from multimodal_churn.data.temporal_store import TemporalHistoryStore

    store = TemporalHistoryStore(output / "intermediate")
    manifest = json.loads((output / "manifest_temporal_ssl.json").read_text())
    branch = TemporalModalityBranch("transaction", 3,
                                    normalization=manifest["normalization"]).eval()
    checkpoint = torch.load(output / "checkpoints" / "best_transaction_ssl.pt",
                            map_location="cpu", weights_only=False)
    branch.load_state_dict(checkpoint["branch"])
    values, mask = store.batch("transaction", np.array([2]))
    with torch.inference_mode():
        reloaded = branch(torch.from_numpy(values), torch.from_numpy(mask)).numpy()[0]
    exported = table.loc[table.customer_id == "c002",
                         [f"transaction_{i:03d}" for i in range(128)]].to_numpy()[0]
    np.testing.assert_allclose(reloaded, exported, rtol=1e-5, atol=1e-5)
    store.close()
    from multimodal_churn.utils.artifact_backup import copy_temporal_ssl_run_to_drive

    backup = workspace_tmp / "drive_copy"
    marker = copy_temporal_ssl_run_to_drive(output, backup)
    copy_metadata = json.loads(marker.read_text())
    assert copy_metadata["run_id"] == manifest["run_id"]
    assert (backup / "features_temporal_ssl.parquet").is_file()
    assert (backup / "intermediate" / "transaction_values.npy").is_file()
    assert (backup / "checkpoints" / "best_image_ssl.pt").is_file()
    changed_config = deepcopy(config)
    changed_config["temporal_ssl"]["temperature"] = 0.2
    with pytest.raises(ValueError, match="different run"):
        build_temporal_ssl_feature_table(
            changed_config, archive_path, "2019-01-31", tmp_path / "wrong_config",
            device="cpu", resume_dir=persistent,
            product_vectors={"text": text_vectors, "image": image_vectors})
