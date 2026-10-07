"""Deterministic two-view SSL for modality-specific temporal branches."""

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from multimodal_churn.data.temporal_store import TemporalHistoryStore
from multimodal_churn.models.temporal_ssl import (
    SSLProjectionHead, TemporalModalityBranch, nt_xent,
)
from multimodal_churn.utils.artifact_backup import copy_verified_file
from multimodal_churn.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def _save_checkpoint(path: Path, payload: dict) -> None:
    """Replace a complete local checkpoint without exposing a partial file."""
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict:
    state = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot resume without CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def customer_split(customer_ids: list[str], seed: int, validation_percent: int = 5
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Hash customer IDs, independently of input order or batch boundaries."""
    if not 1 <= validation_percent < 100:
        raise ValueError("validation_percent must be between 1 and 99")
    val = np.array([
        int.from_bytes(hashlib.sha256(f"{seed}:{cid}".encode()).digest()[:8], "big") % 100
        < validation_percent for cid in customer_ids
    ])
    train_rows = np.flatnonzero(~val)
    val_rows = np.flatnonzero(val)
    def checksum(rows: np.ndarray) -> str:
        joined = "\n".join(sorted(customer_ids[int(row)] for row in rows))
        return hashlib.sha256(joined.encode()).hexdigest()
    return train_rows, val_rows, {
        "split_algorithm": "sha256(seed:customer_id) first64 mod100",
        "train_customer_checksum": checksum(train_rows),
        "validation_customer_checksum": checksum(val_rows),
    }


def transaction_normalization(store: TemporalHistoryStore, train_rows: np.ndarray,
                              batch_size: int = 4096) -> dict[str, float]:
    """Fit statistics on observed training customers and valid events only."""
    count = 0
    price_sum = price_square = gap_sum = gap_square = 0.0
    for offset in range(0, len(train_rows), batch_size):
        values, mask = store.batch("transaction", train_rows[offset:offset + batch_size])
        price = values[..., 0][mask].astype("float64")
        log_gap = np.log1p(values[..., 2][mask].astype("float64"))
        count += len(price)
        price_sum += float(price.sum())
        price_square += float(np.square(price).sum())
        gap_sum += float(log_gap.sum())
        gap_square += float(np.square(log_gap).sum())
    if count == 0:
        raise ValueError("No training transactions for normalization")
    price_mean = price_sum / count
    gap_mean = gap_sum / count
    return {
        "price_mean": price_mean,
        "price_std": max(max(price_square / count - price_mean ** 2, 0.0) ** .5, 1e-6),
        "log_gap_mean": gap_mean,
        "log_gap_std": max(max(gap_square / count - gap_mean ** 2, 0.0) ** .5, 1e-6),
    }


def augment_views(values: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                  drop_rate: float = 0.2, short_mask_rate: float = 0.1
                  ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """Long: distinct ordered subsequences. Short: distinct latent feature masks."""
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("values/mask shape mismatch")
    if not 0 < drop_rate < 1 or not 0 < short_mask_rate < 1:
        raise ValueError("augmentation rates must be between 0 and 1")
    batch, width, dimension = values.shape
    views = [(np.zeros_like(values), np.zeros_like(mask),
              np.ones((batch, width, 128), dtype=bool)) for _ in range(2)]
    overlap = retained = identical = 0.0
    for row in range(batch):
        length = int(mask[row].sum())
        if length < 2:
            raise ValueError("SSL needs at least two valid events per customer")
        if length >= 4:
            dropped = max(1, min(length - 1, round(length * drop_rate)))
            first_drop = set(rng.choice(length, dropped, replace=False).tolist())
            second_drop = set(rng.choice(length, dropped, replace=False).tolist())
            if length - 1 in first_drop and length - 1 in second_drop:
                second_drop.remove(length - 1)
                second_drop.add(next(index for index in range(length - 1)
                                     if index not in second_drop))
            if first_drop == second_drop:
                replacement = next(index for index in range(length)
                                   if index not in second_drop)
                second_drop.remove(next(iter(second_drop)))
                second_drop.add(replacement)
            indices = [np.array([i for i in range(length) if i not in removed], dtype=int)
                       for removed in (first_drop, second_drop)]
            overlap += len(set(indices[0]) & set(indices[1])) / len(set(indices[0]) | set(indices[1]))
            for view, kept in zip(views, indices, strict=True):
                view[0][row, :len(kept)] = values[row, kept]
                view[1][row, :len(kept)] = True
                retained += len(kept) / length / 2
            identical += float(np.array_equal(indices[0], indices[1]))
        else:
            selected = []
            masked_count = max(1, round(128 * short_mask_rate))
            for view in views:
                view[0][row, :length] = values[row, :length]
                view[1][row, :length] = True
                event = int(rng.integers(length))
                dimensions = rng.choice(128, masked_count, replace=False)
                view[2][row, event, dimensions] = False
                selected.append((event, set(dimensions.tolist())))
                retained += .5
            if selected[0] == selected[1]:
                other = (selected[1][0] + 1) % length
                views[1][2][row, selected[1][0]] = True
                views[1][2][row, other, list(selected[1][1])] = False
            overlap += 1.0
            identical += float(np.array_equal(views[0][2][row], views[1][2][row]))
        if (np.array_equal(views[0][0][row], views[1][0][row]) and
                np.array_equal(views[0][1][row], views[1][1][row]) and
                np.array_equal(views[0][2][row], views[1][2][row])):
            views[1][2][row, 0, 0] = ~views[1][2][row, 0, 0]
            identical = max(0.0, identical - 1.0)
    return views[0], views[1], {
        "mean_view_overlap": overlap / batch,
        "mean_retained_events": retained / batch,
        "identical_view_rate": identical / batch,
    }


def _tensor_view(view: tuple[np.ndarray, np.ndarray, np.ndarray], device: str
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, mask, latent = view
    return (torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(latent).to(device))


def _epoch(store: TemporalHistoryStore, rows: np.ndarray, modality: str,
           branch: TemporalModalityBranch, head: SSLProjectionHead,
           optimizer: torch.optim.Optimizer | None, batch_size: int,
           temperature: float, rng: np.random.Generator, device: str,
           drop_rate: float, short_mask_rate: float) -> dict[str, float]:
    metrics = {key: 0.0 for key in (
        "ssl_loss", "positive_cosine_similarity", "negative_cosine_similarity",
        "mean_h_norm", "std_h_norm", "mean_feature_std", "mean_view_overlap",
        "mean_retained_events", "identical_view_rate", "positive_negative_margin")}
    total = 0
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset:offset + batch_size]
        if len(batch_rows) < 2:
            continue
        values, mask = store.batch(modality, batch_rows)
        view1, view2, augmentation = augment_views(values, mask, rng,
                                                    drop_rate, short_mask_rate)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        h1 = branch(*_tensor_view(view1, device))
        h2 = branch(*_tensor_view(view2, device))
        loss, positive, negative = nt_xent(head(h1), head(h2), temperature)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite {modality} SSL loss")
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        h = torch.cat((h1.detach(), h2.detach()))
        norms = h.norm(dim=1)
        current = {
            "ssl_loss": float(loss.detach()),
            "positive_cosine_similarity": float(positive),
            "negative_cosine_similarity": float(negative),
            "positive_negative_margin": float(positive - negative),
            "mean_h_norm": float(norms.mean()),
            "std_h_norm": float(norms.std(unbiased=False)),
            "mean_feature_std": float(h.std(dim=0, unbiased=False).mean()),
            **augmentation,
        }
        if not np.isfinite(list(current.values())).all():
            raise ValueError(f"Non-finite {modality} SSL metrics")
        for key, value in current.items():
            metrics[key] += value * len(batch_rows)
        total += len(batch_rows)
    if total == 0:
        raise ValueError(f"{modality} has fewer than two eligible customers in this split")
    return {key: value / total for key, value in metrics.items()}


def _representation_sanity(store: TemporalHistoryStore, rows: np.ndarray,
                           modality: str, branch: TemporalModalityBranch,
                           device: str) -> dict[str, float]:
    sample = rows[:min(512, len(rows))]
    with torch.inference_mode():
        values, mask = store.batch(modality, sample)
        h = branch(torch.from_numpy(values).to(device),
                   torch.from_numpy(mask).to(device)).cpu().numpy()
    norms = np.linalg.norm(h, axis=1)
    normalized = h / np.maximum(norms[:, None], 1e-12)
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(len(sample), k=1)]
    return {
        "feature_dimension_std": float(h.std(axis=0).mean()),
        "pairwise_cosine_mean": float(upper.mean()) if len(upper) else 0.0,
        "pairwise_cosine_std": float(upper.std()) if len(upper) else 0.0,
        "pairwise_cosine_p05": float(np.quantile(upper, .05)) if len(upper) else 0.0,
        "pairwise_cosine_p50": float(np.quantile(upper, .50)) if len(upper) else 0.0,
        "pairwise_cosine_p95": float(np.quantile(upper, .95)) if len(upper) else 0.0,
        "unique_representation_ratio": len(np.unique(h.round(4), axis=0)) / len(h),
    }


def train_branches(store: TemporalHistoryStore, output_dir: Path, settings: dict,
                   normalization: dict[str, float], device: str,
                   resume_dir: Path | None = None,
                   run_identity: dict | None = None) -> tuple[dict, dict]:
    """Train independent branches and resume from verified epoch snapshots."""
    seed = int(settings["seed"])
    set_seed(seed)
    train_rows, val_rows, split_meta = customer_split(
        store.customer_ids, seed, int(settings["validation_percent"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    persistent = resume_dir / "checkpoints" if resume_dir is not None else None
    if persistent is not None:
        if persistent.resolve() == output_dir.resolve():
            raise ValueError("Persistent checkpoints must be separate from local checkpoints")
        persistent.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps({
        "run": run_identity, "settings": settings, "normalization": normalization,
        "split": split_meta, "customer_count": len(store),
    }, sort_keys=True).encode()).hexdigest()
    summaries = {}
    for offset, (modality, depth) in enumerate((
        ("transaction", 3), ("text", 2), ("image", 2))):
        eligible = np.asarray(store.lengths(modality)) >= 2
        train = train_rows[eligible[train_rows]]
        validation = val_rows[eligible[val_rows]]
        if len(train) < 2 or len(validation) < 2:
            raise ValueError(f"{modality} needs at least two eligible train and validation customers")
        branch = TemporalModalityBranch(modality, depth, int(settings["max_history"]),
                                        normalization if modality == "transaction" else None).to(device)
        head = SSLProjectionHead().to(device)
        optimizer = torch.optim.AdamW(list(branch.parameters()) + list(head.parameters()),
                                      lr=float(settings["learning_rate"]))
        best_loss = float("inf")
        best_epoch = 0
        history = []
        metrics_path = output_dir / f"{modality}_ssl_metrics.json"
        checkpoint = output_dir / f"best_{modality}_ssl.pt"
        latest = output_dir / f"latest_{modality}_ssl.pt"
        saved_latest = (torch.load(latest, map_location="cpu", weights_only=False)
                        if latest.exists() else None)
        saved_persistent = (torch.load(persistent / latest.name, map_location="cpu",
                                       weights_only=False)
                            if persistent is not None and (persistent / latest.name).exists()
                            else None)
        for candidate in (saved_latest, saved_persistent):
            if candidate is not None and (candidate.get("identity") != identity or
                                          candidate.get("modality") != modality):
                raise ValueError(f"{modality} checkpoint belongs to a different run")
        if saved_persistent is not None and (saved_latest is None or
                                             int(saved_persistent["epoch"]) >
                                             int(saved_latest["epoch"])):
            copy_verified_file(persistent / latest.name, latest)
            if (persistent / checkpoint.name).is_file():
                copy_verified_file(persistent / checkpoint.name, checkpoint)
            saved_latest = saved_persistent
        elif (saved_persistent is not None and not checkpoint.exists() and
              (persistent / checkpoint.name).is_file()):
            copy_verified_file(persistent / checkpoint.name, checkpoint)
        start_epoch = 0
        if saved_latest is not None:
            start_epoch = int(saved_latest["epoch"])
            if not 0 < start_epoch <= int(settings["max_epochs"]):
                raise ValueError(f"Invalid {modality} checkpoint epoch: {start_epoch}")
            best_epoch = int(saved_latest["best_epoch"])
            saved_best = (torch.load(checkpoint, map_location="cpu", weights_only=False)
                          if checkpoint.is_file() else None)
            if (saved_best is None or saved_best.get("identity") != identity or
                    saved_best.get("epoch") != best_epoch):
                if best_epoch != start_epoch:
                    raise ValueError(f"Missing or inconsistent best checkpoint for {modality}")
                _save_checkpoint(checkpoint, {
                    "branch": saved_latest["branch"],
                    "projector": saved_latest["projector"],
                    "optimizer": saved_latest["optimizer"],
                    "epoch": best_epoch,
                    "validation_metrics": saved_latest["best_validation_metrics"],
                    "normalization": normalization, "settings": settings,
                    "seed": seed, "modality": modality, "identity": identity,
                })
                if persistent is not None:
                    copy_verified_file(checkpoint, persistent / checkpoint.name)
            branch.load_state_dict(saved_latest["branch"])
            head.load_state_dict(saved_latest["projector"])
            optimizer.load_state_dict(saved_latest["optimizer"])
            best_loss = float(saved_latest["best_validation_loss"])
            history = saved_latest["history"]
            if len(history) != start_epoch:
                raise ValueError(f"Incomplete {modality} checkpoint history")
            _restore_rng_state(saved_latest["rng_state"])
            metrics_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            if persistent is not None and (saved_persistent is None or
                                           int(saved_persistent["epoch"]) < start_epoch):
                copy_verified_file(latest, persistent / latest.name)
                copy_verified_file(checkpoint, persistent / checkpoint.name)
                copy_verified_file(metrics_path, persistent / metrics_path.name)
            LOGGER.info("Resumed %s SSL at epoch %d/%d", modality, start_epoch,
                        int(settings["max_epochs"]))
        for epoch in range(start_epoch, int(settings["max_epochs"])):
            branch.train(); head.train()
            shuffled = np.random.default_rng(seed + epoch * 17 + offset).permutation(train)
            train_metrics = _epoch(
                store, shuffled, modality, branch, head, optimizer,
                int(settings["batch_size"]), float(settings["temperature"]),
                np.random.default_rng(seed + epoch * 31 + offset), device,
                float(settings["long_history_drop_rate"]),
                float(settings["short_history_mask_rate"]),
            )
            branch.eval(); head.eval()
            with torch.inference_mode():
                val_metrics = _epoch(
                    store, validation, modality, branch, head, None,
                    int(settings["batch_size"]), float(settings["temperature"]),
                    np.random.default_rng(seed + 100_000 + offset), device,
                    float(settings["long_history_drop_rate"]),
                    float(settings["short_history_mask_rate"]),
                )
            sanity = _representation_sanity(store, validation, modality, branch, device)
            record = {"epoch": epoch + 1, "train": train_metrics,
                      "validation": val_metrics, "representation_sanity": sanity}
            history.append(record)
            metrics_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            LOGGER.info("%s SSL epoch %d/%d: train=%.4f val=%.4f",
                        modality, epoch + 1, int(settings["max_epochs"]),
                        train_metrics["ssl_loss"], val_metrics["ssl_loss"])
            if val_metrics["ssl_loss"] < best_loss:
                best_loss = val_metrics["ssl_loss"]
                best_epoch = epoch + 1
                _save_checkpoint(checkpoint, {
                    "branch": branch.state_dict(), "projector": head.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                    "validation_metrics": val_metrics, "normalization": normalization,
                    "settings": settings, "seed": seed, "modality": modality,
                    "identity": identity,
                })
            _save_checkpoint(latest, {
                "branch": branch.state_dict(), "projector": head.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                "best_validation_loss": best_loss, "best_epoch": best_epoch,
                "best_validation_metrics": history[best_epoch - 1]["validation"],
                "history": history,
                "rng_state": _rng_state(), "normalization": normalization,
                "settings": settings, "seed": seed, "modality": modality,
                "identity": identity,
            })
            if persistent is not None:
                copy_verified_file(latest, persistent / latest.name)
                if best_epoch == epoch + 1:
                    copy_verified_file(checkpoint, persistent / checkpoint.name)
                copy_verified_file(metrics_path, persistent / metrics_path.name)
                LOGGER.info("Backed up %s SSL epoch %d to %s", modality, epoch + 1,
                            persistent)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        branch.load_state_dict(saved["branch"])
        branch.eval()
        sanity = _representation_sanity(store, validation, modality, branch, device)
        if not np.isfinite(list(sanity.values())).all() or sanity["feature_dimension_std"] < 1e-6:
            raise ValueError(f"{modality} representation collapsed or became non-finite")
        summaries[modality] = {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "best_epoch": saved["epoch"], "best_validation_loss": best_loss,
            "eligible_train_customers": len(train),
            "eligible_validation_customers": len(validation),
            "representation_sanity": sanity,
        }
    return summaries, split_meta
