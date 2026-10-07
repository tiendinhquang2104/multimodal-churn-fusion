"""Three independent temporal branches with a shared implementation."""

import torch
from torch import nn
from torch.nn import functional as F


class TemporalHistoryTransformer(nn.Module):
    """Turn a nonempty, padded sequence of 128D events into one raw 128D vector."""

    def __init__(self, num_layers: int, max_history: int = 32) -> None:
        super().__init__()
        if num_layers < 1 or max_history < 1:
            raise ValueError("num_layers and max_history must be positive")
        self.max_history = max_history
        self.position = nn.Embedding(max_history, 128)
        block = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=num_layers)
        self.attention = nn.Linear(128, 1)

    def forward(
        self, values: torch.Tensor, valid_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_sequence(values, valid_mask)
        scores = self.attention(hidden).squeeze(-1).masked_fill(~valid_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        result = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return (result, weights) if return_attention else result

    def encode_sequence(self, values: torch.Tensor,
                        valid_mask: torch.Tensor) -> torch.Tensor:
        """Expose pre-pooling event states for later sequence-level experiments."""
        if values.ndim != 3 or values.shape[-1] != 128:
            raise ValueError("values must have shape [batch, length, 128]")
        if not 1 <= values.shape[1] <= self.max_history:
            raise ValueError("sequence length is outside max_history")
        if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [batch, length]")
        if not valid_mask.any(dim=1).all():
            raise ValueError("all-invalid histories must be handled before the transformer")
        positions = torch.arange(values.shape[1], device=values.device)
        hidden = values + self.position(positions)
        return self.blocks(hidden, src_key_padding_mask=~valid_mask)


class TransactionEventRepresentation(nn.Module):
    """Respect continuous price/gap and categorical H&M sales channel."""

    def __init__(self, price_mean: float, price_std: float,
                 log_gap_mean: float, log_gap_std: float) -> None:
        super().__init__()
        if price_std <= 0 or log_gap_std <= 0:
            raise ValueError("normalization scales must be positive")
        for name, value in (
            ("price_mean", price_mean), ("price_std", price_std),
            ("log_gap_mean", log_gap_mean), ("log_gap_std", log_gap_std),
        ):
            self.register_buffer(name, torch.tensor(float(value)))
        self.price = nn.Linear(1, 32)
        self.channel = nn.Embedding(2, 16)
        self.gap = nn.Linear(1, 32)
        self.combine = nn.Linear(80, 128)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("transaction values must be [batch, length, 3]")
        channels = values[..., 1]
        if ((channels[valid_mask] != 1) & (channels[valid_mask] != 2)).any():
            raise ValueError("valid H&M sales channels must be 1 or 2")
        if (values[..., 2][valid_mask] < 0).any():
            raise ValueError("inter_purchase_days must be nonnegative")
        price = ((values[..., 0:1] - self.price_mean) / self.price_std)
        gap = ((torch.log1p(values[..., 2:3].clamp_min(0)) - self.log_gap_mean)
               / self.log_gap_std)
        channel = self.channel((channels.long() - 1).clamp(0, 1))
        return self.combine(torch.cat((self.price(price), channel, self.gap(gap)), dim=-1))


class TemporalModalityBranch(nn.Module):
    """A modality-specific input adapter followed by its own temporal weights."""

    def __init__(self, modality: str, num_layers: int, max_history: int = 32,
                 normalization: dict[str, float] | None = None) -> None:
        super().__init__()
        if modality == "transaction":
            if normalization is None:
                raise ValueError("transaction normalization is required")
            self.adapter = TransactionEventRepresentation(**normalization)
        elif modality in ("text", "image"):
            self.adapter = nn.Linear(384 if modality == "text" else 512, 128)
        else:
            raise ValueError(f"unsupported modality: {modality}")
        self.modality = modality
        self.temporal = TemporalHistoryTransformer(num_layers, max_history)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor,
                latent_keep_mask: torch.Tensor | None = None,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        projected = self._project(values, valid_mask, latent_keep_mask)
        return self.temporal(projected, valid_mask, return_attention)

    def encode_sequence(self, values: torch.Tensor, valid_mask: torch.Tensor
                        ) -> torch.Tensor:
        """Return pre-pooling states when event-level fusion is explored later."""
        return self.temporal.encode_sequence(self._project(values, valid_mask, None), valid_mask)

    def _project(self, values: torch.Tensor, valid_mask: torch.Tensor,
                 latent_keep_mask: torch.Tensor | None) -> torch.Tensor:
        projected = (self.adapter(values, valid_mask) if self.modality == "transaction"
                     else self.adapter(values))
        if latent_keep_mask is not None:
            if latent_keep_mask.shape != projected.shape or latent_keep_mask.dtype != torch.bool:
                raise ValueError("latent_keep_mask must be boolean [batch, length, 128]")
            projected = projected * latent_keep_mask
        return projected


class SSLProjectionHead(nn.Module):
    """Used for NT-Xent; never used for feature export."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(h), dim=-1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric two-view InfoNCE with all other customers in the batch as negatives."""
    if z1.shape != z2.shape or z1.ndim != 2 or z1.shape[0] < 2:
        raise ValueError("NT-Xent needs two matching views of at least two customers")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = z1.shape[0]
    z = torch.cat((z1, z2), dim=0)
    similarity = z @ z.T
    logits = similarity / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    loss = F.cross_entropy(logits, targets)
    positive = similarity[torch.arange(2 * batch, device=z.device), targets].mean()
    negatives = similarity.masked_fill(torch.eye(2 * batch, device=z.device, dtype=torch.bool), 0)
    negatives[torch.arange(2 * batch, device=z.device), targets] = 0
    negative = negatives.sum() / (2 * batch * (2 * batch - 2))
    return loss, positive.detach(), negative.detach()
