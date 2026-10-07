"""Disk-backed, observation-only H&M histories for temporal SSL."""

from pathlib import Path
import logging
import sqlite3
from zipfile import ZipFile

import numpy as np
import pandas as pd


MODALITIES = ("transaction", "text", "image")
LOGGER = logging.getLogger(__name__)


def _archive_member(archive: ZipFile, filename: str) -> str:
    matches = [name for name in archive.namelist()
               if name == filename or name.endswith("/" + filename)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} in the archive")
    return matches[0]


def stage_observed_transactions(archive: ZipFile, db_path: Path,
                                observation_end: str, observation_start: str | None = None,
                                chunk_size: int = 200_000) -> dict[str, int]:
    """Stream observed rows into SQLite; never read post-cutoff events into histories."""
    end = pd.Timestamp(observation_end)
    start = pd.Timestamp(observation_start) if observation_start else None
    if start is not None and start > end:
        raise ValueError("observation_start must not follow observation_end")
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE observed (customer TEXT, day TEXT, article TEXT, "
                       "price REAL, channel INTEGER, ordinal INTEGER)")
    ordinal = 0
    skipped_channels = 0
    with archive.open(_archive_member(archive, "transactions_train.csv")) as stream:
        chunks = pd.read_csv(
            stream, usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            dtype={"customer_id": "string", "article_id": "string"}, chunksize=chunk_size,
        )
        for chunk_number, chunk in enumerate(chunks, start=1):
            dates = pd.to_datetime(chunk["t_dat"], errors="coerce")
            keep = dates.notna() & dates.le(end) & chunk["customer_id"].notna()
            if start is not None:
                keep &= dates.ge(start)
            chunk = chunk.loc[keep].copy()
            if chunk.empty:
                continue
            chunk["t_dat"] = dates.loc[keep].dt.strftime("%Y-%m-%d")
            channels = pd.to_numeric(chunk["sales_channel_id"], errors="coerce")
            valid_channel = channels.isin([1, 2])
            skipped_channels += int((~valid_channel).sum())
            chunk = chunk.loc[valid_channel].copy()
            chunk["sales_channel_id"] = channels.loc[valid_channel].astype(int)
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").fillna(0).clip(lower=0)
            article_ids = chunk["article_id"].fillna("")
            chunk["article_id"] = article_ids.where(article_ids.eq(""), article_ids.str.zfill(10))
            chunk["ordinal"] = np.arange(ordinal, ordinal + len(chunk), dtype=np.int64)
            ordinal += len(chunk)
            records = chunk[["customer_id", "t_dat", "article_id", "price",
                             "sales_channel_id", "ordinal"]].itertuples(index=False, name=None)
            connection.executemany("INSERT INTO observed VALUES (?,?,?,?,?,?)", records)
            connection.commit()
            if chunk_number % 20 == 0:
                LOGGER.info("Staged %d observed transactions", ordinal)
    if ordinal == 0:
        connection.close()
        raise ValueError("No valid transactions in observation window")
    connection.execute("CREATE INDEX observed_customer_order ON observed(customer, day, ordinal)")
    connection.commit()
    connection.close()
    return {"observed_rows": ordinal, "skipped_invalid_channels": skipped_channels}


def observed_article_ids(db_path: Path, max_customers: int | None = None) -> set[str]:
    connection = sqlite3.connect(db_path)
    try:
        if max_customers is None:
            query = "SELECT DISTINCT article FROM observed"
            rows = connection.execute(query)
        else:
            query = ("SELECT DISTINCT article FROM observed WHERE customer IN "
                     "(SELECT DISTINCT customer FROM observed ORDER BY customer LIMIT ?)")
            rows = connection.execute(query, (max_customers,))
        return {row[0] for row in rows if row[0]}
    finally:
        connection.close()


def _flush_customer(row: int, customer: str, events: list[tuple],
                    text_index: dict[str, int], image_index: dict[str, int],
                    arrays: dict[str, np.memmap], ids_file, max_history: int) -> None:
    ids_file.write(customer + "\n")
    arrays["transaction_original"][row] = len(events)
    transaction = events[-max_history:]
    arrays["transaction_used"][row] = len(transaction)
    previous = None
    for position, (day, article, price, channel) in enumerate(transaction):
        date = pd.Timestamp(day)
        gap = 0 if previous is None else (date - previous).days
        arrays["transaction_values"][row, position] = (price, channel, gap)
        previous = date
    for modality, index in (("text", text_index), ("image", image_index)):
        available = [index[article] for _, article, _, _ in events if article in index]
        arrays[f"{modality}_original"][row] = len(available)
        kept = available[-max_history:]
        arrays[f"{modality}_used"][row] = len(kept)
        arrays[f"{modality}_indices"][row, :len(kept)] = kept


def build_history_arrays(db_path: Path, output_dir: Path,
                         text_vectors: dict[str, np.ndarray],
                         image_vectors: dict[str, np.ndarray],
                         max_history: int = 32, max_customers: int | None = None) -> int:
    """Write fixed-width arrays in customer order; keep original counts separately."""
    if max_history < 1 or (max_customers is not None and max_customers < 1):
        raise ValueError("history/customer limits must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    count = int(connection.execute("SELECT COUNT(DISTINCT customer) FROM observed").fetchone()[0])
    count = min(count, max_customers) if max_customers is not None else count
    if count == 0:
        connection.close()
        raise ValueError("No customers to encode")
    shapes = {
        "transaction_values": ((count, max_history, 3), "float32", 0),
        "text_indices": ((count, max_history), "int32", -1),
        "image_indices": ((count, max_history), "int32", -1),
    }
    for modality in MODALITIES:
        for suffix in ("original", "used"):
            shapes[f"{modality}_{suffix}"] = ((count,), "int32", 0)
    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype, fill) in shapes.items():
        array = np.lib.format.open_memmap(output_dir / f"{name}.npy", mode="w+",
                                          dtype=dtype, shape=shape)
        array[:] = fill
        arrays[name] = array
    text_index = {article: index for index, article in enumerate(sorted(text_vectors))}
    image_index = {article: index for index, article in enumerate(sorted(image_vectors))}
    with (output_dir / "customer_ids.txt").open("w", encoding="utf-8") as ids_file:
        cursor = connection.execute(
            "SELECT customer, day, article, price, channel FROM observed "
            "ORDER BY customer, day, ordinal"
        )
        current = None
        events: list[tuple] = []
        row = 0
        for customer, day, article, price, channel in cursor:
            if current is not None and customer != current:
                _flush_customer(row, current, events, text_index, image_index,
                                arrays, ids_file, max_history)
                row += 1
                if row % 100_000 == 0:
                    LOGGER.info("Built histories for %d/%d customers", row, count)
                if row == count:
                    break
                events = []
            current = customer
            events.append((day, article, price, channel))
        else:
            if current is not None and row < count:
                _flush_customer(row, current, events, text_index, image_index,
                                arrays, ids_file, max_history)
                row += 1
    connection.close()
    for array in arrays.values():
        array.flush()
        array._mmap.close()
    np.save(output_dir / "text_products.npy",
            np.stack([text_vectors[key] for key in sorted(text_vectors)]).astype("float32")
            if text_vectors else np.empty((0, 384), dtype="float32"))
    np.save(output_dir / "image_products.npy",
            np.stack([image_vectors[key] for key in sorted(image_vectors)]).astype("float32")
            if image_vectors else np.empty((0, 512), dtype="float32"))
    return row


class TemporalHistoryStore:
    """Memory-map customer histories and materialize only requested batches."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.customer_ids = (directory / "customer_ids.txt").read_text(encoding="utf-8").splitlines()
        self.arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r")
                       for name in ("transaction_values", "text_indices", "image_indices")
                       + tuple(f"{modality}_{suffix}" for modality in MODALITIES
                               for suffix in ("original", "used"))}
        self.products = {modality: np.load(directory / f"{modality}_products.npy", mmap_mode="r")
                         for modality in ("text", "image")}

    def __len__(self) -> int:
        return len(self.customer_ids)

    def close(self) -> None:
        """Release memory maps, especially important before cleanup on Windows."""
        for array in (*self.arrays.values(), *self.products.values()):
            array._mmap.close()

    def lengths(self, modality: str) -> np.ndarray:
        return self.arrays[f"{modality}_used"]

    def batch(self, modality: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = self.lengths(modality)[rows]
        width = self.arrays["transaction_values"].shape[1]
        mask = np.arange(width)[None, :] < length[:, None]
        if modality == "transaction":
            values = np.array(self.arrays["transaction_values"][rows], dtype="float32")
        elif modality in ("text", "image"):
            indices = self.arrays[f"{modality}_indices"][rows]
            products = self.products[modality]
            values = np.zeros((len(rows), width, products.shape[1]), dtype="float32")
            if len(products):
                values[mask] = products[np.asarray(indices[mask], dtype=np.int64)]
        else:
            raise ValueError(f"unsupported modality: {modality}")
        return values, mask
