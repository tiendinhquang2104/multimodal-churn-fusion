"""Prepare leakage-controlled H&M transaction sequences."""

import numpy as np
import pandas as pd


def transaction_sequences(
    transactions: pd.DataFrame,
    observation_end: str,
    max_sequence_length: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return IDs, [customer, time, 3] values, and valid-position masks.

    The three inputs are log(1 + price), online-channel indicator, and days
    before the configured observation end divided by 365. The caller must
    filter transactions to the observation window before calling this.
    """
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive.")
    if transactions.empty:
        raise ValueError("No observed transactions were selected.")
    cutoff = pd.Timestamp(observation_end)
    grouped = transactions.groupby("customer_id", sort=True)
    customer_ids = list(grouped.groups)
    values = np.zeros((len(customer_ids), max_sequence_length, 3), dtype=np.float32)
    valid = np.zeros((len(customer_ids), max_sequence_length), dtype=bool)
    for row, customer_id in enumerate(customer_ids):
        history = grouped.get_group(customer_id).sort_values("t_dat", kind="stable")
        history = history.tail(max_sequence_length)
        count = len(history)
        values[row, :count, 0] = np.log1p(
            np.maximum(history["price"].to_numpy(dtype=np.float32), 0)
        )
        values[row, :count, 1] = (
            history["sales_channel_id"].to_numpy() == 2
        ).astype(np.float32)
        days_ago = (cutoff - history["t_dat"]).dt.days.to_numpy(dtype=np.float32)
        values[row, :count, 2] = np.maximum(days_ago, 0) / 365.0
        valid[row, :count] = True
    return customer_ids, values, valid
