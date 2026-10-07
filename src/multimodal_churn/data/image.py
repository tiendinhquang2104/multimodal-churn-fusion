"""Frozen CLIP product-image embedding extraction from the H&M ZIP."""

from io import BytesIO
import logging
from zipfile import ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def encode_article_images(
    archive: ZipFile,
    image_members: dict[str, str],
    model_name: str,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode available images once per selected article ID."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not image_members:
        return {}
    import torch
    from transformers import AutoProcessor, CLIPModel

    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    results: dict[str, np.ndarray] = {}
    items = sorted(image_members.items())
    for offset in tqdm(range(0, len(items), batch_size), desc="CLIP image batches", unit="batch"):
        images: list[Image.Image] = []
        article_ids: list[str] = []
        for article_id, member in items[offset : offset + batch_size]:
            try:
                with archive.open(member) as stream:
                    with Image.open(BytesIO(stream.read())) as source:
                        image = source.convert("RGB")
                images.append(image)
                article_ids.append(article_id)
            except (OSError, UnidentifiedImageError) as error:
                LOGGER.warning("Skipping unreadable image %s: %s", member, error)
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixels)
        if not isinstance(features, torch.Tensor):
            candidate = getattr(features, "image_embeds", None)
            if candidate is None:
                candidate = getattr(features, "pooler_output", None)
            if not isinstance(candidate, torch.Tensor):
                raise TypeError("CLIP returned an unsupported image-feature result.")
            features = candidate
        vectors = features.float().cpu().numpy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        for article_id, vector in zip(article_ids, vectors, strict=True):
            results[article_id] = vector.astype(np.float32, copy=False)
        for image in images:
            image.close()
    return results
