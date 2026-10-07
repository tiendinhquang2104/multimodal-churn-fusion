"""H&M ZIP reader for a bounded customer sample.

The archive is read in place; it is never extracted into the repository.
"""

from pathlib import PurePosixPath
from zipfile import ZipFile

import pandas as pd
from tqdm.auto import tqdm


def _member(archive: ZipFile, filename: str) -> str:
    matches = [
        name for name in archive.namelist()
        if name == filename or name.endswith("/" + filename)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} in archive.")
    return matches[0]


def read_hm_transactions(
    archive: ZipFile,
    observation_end: str,
    max_customers: int,
    max_sequence_length: int,
    observation_start: str | None = None,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Read complete observed histories for the first N encountered customers.

    The customer cap is for an executable sample, not a statistically valid
    research cohort. The full CSV is scanned once so selected customers retain
    their later observed purchases.
    """
    if max_customers <= 0 or max_sequence_length <= 0 or chunk_size <= 0:
        raise ValueError("Customer, sequence, and chunk limits must be positive.")
    end = pd.Timestamp(observation_end)
    start = pd.Timestamp(observation_start) if observation_start else None
    if start is not None and start > end:
        raise ValueError("observation_start must not follow observation_end.")
    selected: set[str] = set()
    pieces: list[pd.DataFrame] = []
    with archive.open(_member(archive, "transactions_train.csv")) as stream:
        chunks = pd.read_csv(
            stream,
            usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            dtype={"customer_id": "string", "article_id": "string"},
            chunksize=chunk_size,
        )
        for chunk in tqdm(chunks, desc="H&M transaction chunks", unit="chunk"):
            chunk["t_dat"] = pd.to_datetime(chunk["t_dat"], errors="coerce")
            chunk = chunk.loc[chunk["t_dat"].le(end)]
            if start is not None:
                chunk = chunk.loc[chunk["t_dat"].ge(start)]
            if chunk.empty:
                continue
            if len(selected) < max_customers:
                for customer_id in chunk["customer_id"].dropna().unique():
                    if len(selected) >= max_customers:
                        break
                    selected.add(str(customer_id))
            chunk = chunk.loc[chunk["customer_id"].isin(selected)].copy()
            if not chunk.empty:
                pieces.append(chunk)
    if not pieces:
        raise ValueError("No transactions matched the observation window.")
    result = pd.concat(pieces, ignore_index=True)
    result["article_id"] = result["article_id"].str.zfill(10)
    result["price"] = pd.to_numeric(result["price"], errors="coerce").fillna(0)
    result["sales_channel_id"] = (
        pd.to_numeric(result["sales_channel_id"], errors="coerce").fillna(0)
    )
    result = result.sort_values(["customer_id", "t_dat"], kind="stable")
    return result.groupby("customer_id", sort=False).tail(max_sequence_length).reset_index(drop=True)


def read_hm_articles(archive: ZipFile, article_ids: set[str]) -> pd.DataFrame:
    """Read product names and descriptions only for selected articles."""
    with archive.open(_member(archive, "articles.csv")) as stream:
        articles = pd.read_csv(
            stream,
            usecols=["article_id", "prod_name", "detail_desc"],
            dtype="string",
        )
    articles["article_id"] = articles["article_id"].str.zfill(10)
    return articles.loc[articles["article_id"].isin(article_ids)].copy()


def hm_image_members(archive: ZipFile, article_ids: set[str]) -> dict[str, str]:
    """Map selected article IDs to JPG members of the H&M archive."""
    found: dict[str, str] = {}
    for member in archive.namelist():
        path = PurePosixPath(member)
        article_id = path.stem
        if (
            article_id in article_ids
            and path.suffix.lower() in {".jpg", ".jpeg"}
            and "images" in path.parts
        ):
            found[article_id] = member
    return found
