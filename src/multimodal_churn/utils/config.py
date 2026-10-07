"""Small YAML loader and dataset path resolver."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        content = yaml.safe_load(stream) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return content


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    baseline_path: str | Path,
    experiment_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load baseline YAML, then optional dataset and experiment overrides."""
    config = _read_yaml(Path(baseline_path))
    for override_path in (dataset_path, experiment_path):
        if override_path is not None:
            config = _merge(config, _read_yaml(Path(override_path)))
    return config


def resolve_dataset_paths(
    config: dict[str, Any], datasets_root: str | Path
) -> dict[str, Path]:
    """Resolve dataset-relative folder and optional archive under a chosen root.

    The root is supplied by the runtime (for example a mounted Drive path).
    Dataset YAML files contain only relative paths, so they work on Colab and
    other machines without source changes.
    """
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Configuration needs a dataset mapping.")

    def relative_path(value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"dataset.{field} must be a nonempty relative path.")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"dataset.{field} must stay within datasets_root.")
        return path

    folder = Path(datasets_root).expanduser() / relative_path(
        dataset.get("relative_path"), "relative_path"
    )
    paths = {"folder": folder}
    if dataset.get("archive") is not None:
        paths["archive"] = folder / relative_path(dataset["archive"], "archive")
    return paths
